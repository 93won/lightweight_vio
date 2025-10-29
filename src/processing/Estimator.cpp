/**
 * @file      Estimator.cpp
 * @brief     Implements the main VO estimation logic.
 * @author    Seungwon Choi (csw3575@snu.ac.kr)
 * @date      2025-08-23
 * @copyright Copyright (c) 2025 Seungwon Choi. All rights reserved.
 *
 * @par License
 * This project is released under the MIT License.
 */

#include "processing/Estimator.h"
#include "processing/FeatureTracker.h"
#include "processing/IMUHandler.h"
#include "database/Frame.h"
#include "database/MapPoint.h"
#include "processing/Optimizer.h"
#include "util/Config.h"
#include "util/EurocUtils.h"
#include "database/Feature.h"

#include <opencv2/opencv.hpp>
#include <Eigen/Dense>
#include <spdlog/spdlog.h>
#include <chrono>
#include <cmath>
#include <iostream>
#include <numeric>
#include <algorithm>

namespace lightweight_vio {

Estimator::Estimator()
    : m_frame_id_counter(0)
    , m_frames_since_last_keyframe(0)
    , m_last_keyframe_grid_coverage(0.0)
    , m_current_pose(Eigen::Matrix4f::Identity())
    , m_predicted_pose(Eigen::Matrix4f::Identity())
    , m_transform_from_last(Eigen::Matrix4f::Identity())
    , m_has_initial_gt_pose(false)
    , m_initial_gt_pose(Eigen::Matrix4f::Identity())
    , m_Tgw_init(Eigen::Matrix4f::Identity())  // Initialize as Identity
    , m_sliding_window_thread_running(false)
    , m_keyframes_updated(false) {
    
    // Initialize feature tracker
    m_feature_tracker = std::make_unique<FeatureTracker>();
    
    // Initialize pose optimizer - now uses global Config internally
    m_pose_optimizer = std::make_unique<PnPOptimizer>();
    
    // Initialize sliding window optimizer
    m_sliding_window_optimizer = std::make_unique<SlidingWindowOptimizer>(
        Config::getInstance().m_keyframe_window_size);  // window size only
    
    // Initialize IMU handler
    m_imu_handler = std::make_unique<IMUHandler>();
    
    // Initialize inertial optimizer  
    m_inertial_optimizer = std::make_unique<InertialOptimizer>();
    
    // Start sliding window optimization thread
    m_sliding_window_thread_running = true;
    m_sliding_window_thread = std::make_unique<std::thread>(&Estimator::sliding_window_thread_function, this);
    
    if (Config::getInstance().m_enable_debug_output) {
        spdlog::info("[ESTIMATOR] Sliding window optimization thread started");
    }
}

Estimator::EstimationResult Estimator::process_rgbd_frame(const cv::Mat& rgb_image, const cv::Mat& depth_map, long long timestamp) {
    EstimationResult result;
    auto total_start_time = std::chrono::high_resolution_clock::now();

    // Frame processing starts
    if (Config::getInstance().m_enable_debug_output)
    {
        std::cout << "\n";
        spdlog::info("============================== Frame {} ==============================\n", m_frame_id_counter);
    }

    // Increment frame counter since last keyframe for every new frame
    m_frames_since_last_keyframe++;

    // Initialize timing variables
    double frame_creation_time = 0.0;
    double prediction_time = 0.0;
    double tracking_time = 0.0;
    double optimization_time = 0.0;

    // Create new stereo frame
    auto frame_creation_start = std::chrono::high_resolution_clock::now();
    m_current_frame = create_rgbd_frame(rgb_image, depth_map, timestamp);
    auto frame_creation_end = std::chrono::high_resolution_clock::now();
    frame_creation_time = std::chrono::duration_cast<std::chrono::microseconds>(frame_creation_end - frame_creation_start).count() / 1000.0;
    
    if (!m_current_frame)
    {
        spdlog::error("[Estimator] Failed to create RGBD frame!");
        result.success = false;
        return result;
    }

    if(m_previous_frame)
    {
        // Predict state using motion model
        predict_state();

        // Feature tracking using FeatureTracker
        m_feature_tracker->track_features(m_current_frame, m_previous_frame);

        result.num_features = m_current_frame->get_feature_count();

        // Compute depth using depth map
        m_current_frame->compute_depth();

        int num_tracked_with_map_points = count_features_with_map_points(m_current_frame);

        if(num_tracked_with_map_points >= 5)
        {

            auto opt_result = optimize_pose(m_current_frame);

            result.success = opt_result.success;
            result.num_inliers =  opt_result.num_inliers;
            result.num_outliers = opt_result.num_outliers;

            if(opt_result.success)
            {
                m_current_pose = opt_result.optimized_pose;
                m_current_frame->set_Twb(m_current_pose);

                // Update transform from last frame for velocity estimation
                update_transform_from_last();

                if (Config::getInstance().m_enable_debug_output)
                {
                    spdlog::info("[POSE_OPT] ✅ Optimization successful: {} inliers, {} outliers", opt_result.num_inliers, opt_result.num_outliers);
                }
            }
            else
            {
                if (Config::getInstance().m_enable_debug_output)
                {
                    spdlog::warn("[POSE_OPT] ❌ Optimization failed - keeping previous pose");
                }
            }

        }
        else
        {

            // No tracking, keep previous pose (already set in create_frame)
            m_current_pose = m_current_frame->get_Twb();
            // Update transform from last frame for velocity estimation (even if tracking failed)
            update_transform_from_last();
            result.success = true;

        }

        // Decide whether to create keyframe
        bool is_keyframe_required = should_create_keyframe(m_current_frame);

        if(is_keyframe_required)
        {
            int new_map_points = create_new_map_points(m_current_frame);
            result.num_new_map_points = new_map_points;

            create_keyframe(m_current_frame);
            m_frames_since_last_keyframe = 0;  // Reset to 0 after creating keyframe
        }
        else
        {
            result.num_new_map_points = 0;
        }

        // Count tracked features and features with map points
        result.num_tracked_features = m_current_frame->get_feature_count();
        result.num_features_with_map_points = count_features_with_map_points(m_current_frame);

    }
    else
    {
        // Initial frame
        m_feature_tracker->track_features(m_current_frame, nullptr);
        result.num_features = m_current_frame->get_feature_count();
        // Compute depth using depth map
        m_current_frame->compute_depth();
        // First frame - keep identity pose (already set in create_frame)
        m_current_pose = m_current_frame->get_Twb();
        // Increment frame counter (first frame processing)
        m_frames_since_last_keyframe++;

        // Create initial map points (first frame is always considered keyframe)
        int initial_map_points = create_initial_map_points(m_current_frame);
        result.num_new_map_points = initial_map_points;

        create_keyframe(m_current_frame);

        m_frames_since_last_keyframe = 0;  // Reset after creating first keyframe

        // Count features for first frame
        result.num_tracked_features = m_current_frame->get_feature_count();
        result.num_features_with_map_points = count_features_with_map_points(m_current_frame);
        result.success = true;
    }

    // Add processed frame to all frames vector for trajectory export
    m_all_frames.push_back(m_current_frame);

    // Set reference keyframe for non-keyframe frames (after pose optimization)
    if (!m_current_frame->is_keyframe() && m_last_keyframe)
    {
        m_current_frame->set_reference_keyframe(m_last_keyframe);
    }

    // Update state - release old previous frame's images before updating
    if (m_previous_frame)
    {
        m_previous_frame->release_images();
    }
    m_previous_frame = m_current_frame;

    return result;

}

Estimator::EstimationResult Estimator::process_frame(const cv::Mat& left_image, const cv::Mat& right_image, long long timestamp) {
    EstimationResult result;
    auto total_start_time = std::chrono::high_resolution_clock::now();

    // Frame processing starts
    if (Config::getInstance().m_enable_debug_output) {
        std::cout<<"\n";
        spdlog::info("============================== Frame {} ==============================\n", m_frame_id_counter);
    }

    // Increment frame counter since last keyframe for every new frame
    m_frames_since_last_keyframe++;

    // Initialize timing variables
    double frame_creation_time = 0.0;
    double prediction_time = 0.0;
    double tracking_time = 0.0;
    double optimization_time = 0.0;

    // Create new stereo frame
    auto frame_creation_start = std::chrono::high_resolution_clock::now();
    m_current_frame = create_frame(left_image, right_image, timestamp);
    auto frame_creation_end = std::chrono::high_resolution_clock::now();
    frame_creation_time = std::chrono::duration_cast<std::chrono::microseconds>(frame_creation_end - frame_creation_start).count() / 1000.0;

    if (!m_current_frame)
    {
        spdlog::error("[Estimator] Failed to create frame!");
        result.success = false;
        return result;
    }

    if (m_previous_frame) {
        auto prediction_start = std::chrono::high_resolution_clock::now();
        predict_state();
        auto prediction_end = std::chrono::high_resolution_clock::now();
        prediction_time = std::chrono::duration_cast<std::chrono::microseconds>(prediction_end - prediction_start).count() / 1000.0;
        
        // Track features from previous frame using FeatureTracker
        // FeatureTracker now handles both tracking and map point association/creation
        auto tracking_start = std::chrono::high_resolution_clock::now();
        m_feature_tracker->track_features(m_current_frame, m_previous_frame);
        auto tracking_end = std::chrono::high_resolution_clock::now();
        tracking_time = std::chrono::duration_cast<std::chrono::microseconds>(tracking_end - tracking_start).count() / 1000.0;
        
        result.num_features = m_current_frame->get_feature_count();
        
        // Compute stereo depth for all features
        m_current_frame->compute_stereo_depth();
        
        // Count how many features have associated map points (already done by FeatureTracker)
        int num_tracked_with_map_points = count_features_with_map_points(m_current_frame);
        
        // Log tracking information
        if (Config::getInstance().m_enable_debug_output) {
            spdlog::info("[TRACKING] {} features tracked, {} with map points", 
                        result.num_features, num_tracked_with_map_points);
        }
        
        if (num_tracked_with_map_points > 0) {
            // ✅ ENABLED: Pose optimization re-enabled after fixing coordinate space mismatch!
            if (num_tracked_with_map_points >= 5) {
                auto optimization_start = std::chrono::high_resolution_clock::now();
                auto opt_result = optimize_pose(m_current_frame);
                auto optimization_end = std::chrono::high_resolution_clock::now();
                optimization_time = std::chrono::duration_cast<std::chrono::microseconds>(optimization_end - optimization_start).count() / 1000.0;
                
                result.success = opt_result.success;
                result.num_inliers = opt_result.num_inliers;
                result.num_outliers = opt_result.num_outliers;
                
                if (opt_result.success) {
                    m_current_pose = opt_result.optimized_pose;
                    m_current_frame->set_Twb(m_current_pose);
                    
                    
                    // Update transform from last frame for velocity estimation
                    update_transform_from_last();
                    
                    if (Config::getInstance().m_enable_debug_output) {
                        spdlog::info("[POSE_OPT] ✅ Optimization successful: {} inliers, {} outliers", opt_result.num_inliers, opt_result.num_outliers);
                    }
                } else {
                    if (Config::getInstance().m_enable_debug_output) {
                        spdlog::warn("[POSE_OPT] ❌ Optimization failed - keeping previous pose");
                    }
                }
            } else {
                if (Config::getInstance().m_enable_debug_output) {
                    spdlog::warn("[POSE_OPT] ⚠️ Not enough map point associations for optimization: {} (need ≥5)", num_tracked_with_map_points);
                }
                // Fallback: use current pose as-is
                m_current_pose = m_current_frame->get_Twb();
                
                // Update transform from last frame for velocity estimation
                update_transform_from_last();
                
                result.success = true;
                result.num_inliers = num_tracked_with_map_points;
                result.num_outliers = 0;
            } 
        } else {
            // No tracking, keep previous pose (already set in create_frame)
            m_current_pose = m_current_frame->get_Twb();
            
            // Update transform from last frame for velocity estimation (even if tracking failed)
            update_transform_from_last();
            
            result.success = false;
        }

        // NOTE: FeatureTracker already handles map point association during tracking
        // No need to call associate_tracked_features_with_map_points() again
        
        
        // Decide whether to create keyframe
        auto keyframe_decision_start = std::chrono::high_resolution_clock::now();
        bool is_keyframe = should_create_keyframe(m_current_frame);
        auto keyframe_decision_end = std::chrono::high_resolution_clock::now();
        auto keyframe_decision_time = std::chrono::duration_cast<std::chrono::microseconds>(keyframe_decision_end - keyframe_decision_start).count() / 1000.0;
        
        // Only create new map points for keyframes to avoid trajectory drift
        if (is_keyframe) {
            auto map_points_start = std::chrono::high_resolution_clock::now();
            int new_map_points = create_new_map_points(m_current_frame);
            auto map_points_end = std::chrono::high_resolution_clock::now();
            auto map_points_time = std::chrono::duration_cast<std::chrono::microseconds>(map_points_end - map_points_start).count() / 1000.0;
            
            result.num_new_map_points = new_map_points;
            // spdlog::info("[MAP_POINTS] Created {} new map points by new keyframe insertion", new_map_points);
            
            auto keyframe_creation_start = std::chrono::high_resolution_clock::now();
            create_keyframe(m_current_frame);
            auto keyframe_creation_end = std::chrono::high_resolution_clock::now();
            auto keyframe_creation_time = std::chrono::duration_cast<std::chrono::microseconds>(keyframe_creation_end - keyframe_creation_start).count() / 1000.0;
            
            m_frames_since_last_keyframe = 0;  // Reset to 0 after creating keyframe
            
        } else {
            result.num_new_map_points = 0;
        }
        
        // Count tracked features and features with map points
        result.num_tracked_features = m_current_frame->get_feature_count();
        result.num_features_with_map_points = count_features_with_map_points(m_current_frame);
        
        // Compute reprojection error statistics for keyframes
        if (is_keyframe && count_features_with_map_points(m_current_frame) > 5) {
            compute_reprojection_error_statistics(m_current_frame);
        }
        
      
    } else {
        // First frame - extract features using FeatureTracker
        m_feature_tracker->track_features(m_current_frame, nullptr);
        
        result.num_features = m_current_frame->get_feature_count();
        
        // Compute stereo depth for all features
        auto stereo_start = std::chrono::high_resolution_clock::now();
        m_current_frame->compute_stereo_depth();
        auto stereo_end = std::chrono::high_resolution_clock::now();
        auto stereo_time = std::chrono::duration_cast<std::chrono::microseconds>(stereo_end - stereo_start).count() / 1000.0;
        
        // First frame - keep identity pose (already set in create_frame)
        m_current_pose = m_current_frame->get_Twb();
        
        // Increment frame counter (first frame processing)
        m_frames_since_last_keyframe++;
        
        // Create initial map points (first frame is always considered keyframe)
        auto initial_map_points_start = std::chrono::high_resolution_clock::now();
        int initial_map_points = create_initial_map_points(m_current_frame);
        auto initial_map_points_end = std::chrono::high_resolution_clock::now();
        auto initial_map_points_time = std::chrono::duration_cast<std::chrono::microseconds>(initial_map_points_end - initial_map_points_start).count() / 1000.0;
        
        result.num_new_map_points = initial_map_points;
        
        auto first_keyframe_start = std::chrono::high_resolution_clock::now();
        create_keyframe(m_current_frame);
        auto first_keyframe_end = std::chrono::high_resolution_clock::now();
        auto first_keyframe_time = std::chrono::duration_cast<std::chrono::microseconds>(first_keyframe_end - first_keyframe_start).count() / 1000.0;
        
        m_frames_since_last_keyframe = 0;  // Reset after creating first keyframe
        
        
        // Count features for first frame
        result.num_tracked_features = m_current_frame->get_feature_count();
        result.num_features_with_map_points = count_features_with_map_points(m_current_frame);
        
        result.success = true;
    }
    
    // Add processed frame to all frames vector for trajectory export
    m_all_frames.push_back(m_current_frame);
  
    
    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - total_start_time);
    result.optimization_time_ms = duration.count() / 1000.0;
    
    // Set reference keyframe for non-keyframe frames (after pose optimization)
    if (!m_current_frame->is_keyframe() && m_last_keyframe) {
        m_current_frame->set_reference_keyframe(m_last_keyframe);
    }
    
    // Update state - release old previous frame's images before updating
    if (m_previous_frame) {
        m_previous_frame->release_images();
    }
    m_previous_frame = m_current_frame;
    
    return result;
}

// // IMU process_frame overload
// Estimator::EstimationResult Estimator::process_frame(const cv::Mat& left_image, const cv::Mat& right_image, 
//                                                     long long timestamp, const std::vector<IMUData>& imu_data_from_last_frame) {
//     // ===== IMU-SPECIFIC PROCESSING =====
//     // Accumulate IMU data from last frame
//     for (const auto& imu_data : imu_data_from_last_frame) {
//         m_imu_vec_from_last_keyframe.push_back(imu_data);
//     }
    
//     // Create frame first
//     std::shared_ptr<Frame> frame = create_frame(left_image, right_image, timestamp);
//     if (!frame) {
//         EstimationResult result;
//         result.success = false;
//         return result;
//     }

//     if(m_last_keyframe){
//         frame->set_accel_bias(m_last_keyframe->get_accel_bias());
//         frame->set_gyro_bias(m_last_keyframe->get_gyro_bias());
//     }


//     // Set IMU data to the frame (frame-to-frame data)
//     frame->set_imu_data_from_last_frame(imu_data_from_last_frame);
    
//     // Compute frame-to-frame preintegration if IMU data is available
//     if (!imu_data_from_last_frame.empty() && m_imu_handler) {
//         // Always compute frame-to-frame preintegration, regardless of IMU initialization status
//         // This is useful for state prediction and velocity estimation
//         double first_imu_time = imu_data_from_last_frame.front().timestamp;
//         double last_imu_time = imu_data_from_last_frame.back().timestamp;
        
//         auto frame_to_frame_preint = m_imu_handler->preintegrate(imu_data_from_last_frame, first_imu_time, last_imu_time);
//         if (frame_to_frame_preint && frame_to_frame_preint->is_valid()) {
//             frame->set_imu_preintegration_from_last_frame(frame_to_frame_preint);
//         } else {
//             spdlog::warn("[IMU] Failed to create frame-to-frame preintegration for frame {}", frame->get_frame_id());
//         }
//     }
    
//     // Set as current frame for the rest of the processing
//     m_current_frame = frame;
    
//     // ===== IDENTICAL VO PROCESSING (SAME AS NON-IMU VERSION) =====
//     EstimationResult result;
//     auto total_start_time = std::chrono::high_resolution_clock::now();

//     // Frame processing starts
//     std::cout<<"\n";
//     spdlog::info("============================== Frame {} ==============================\n", m_current_frame->get_frame_id());

//     // Increment frame counter since last keyframe for every new frame
//     m_frames_since_last_keyframe++;

//     // Initialize timing variables
//     double frame_creation_time = 0.0;
//     double prediction_time = 0.0;
//     double tracking_time = 0.0;
//     double optimization_time = 0.0;

//     // IMU data processed (reduced logging)
//     if (!imu_data_from_last_frame.empty() && m_current_frame->get_frame_id() % 10 == 0) {
//         spdlog::info("[IMU] Frame {} processed {} IMU measurements", 
//                     m_current_frame->get_frame_id(), imu_data_from_last_frame.size());
//     }

//     if (m_previous_frame) {
//         auto prediction_start = std::chrono::high_resolution_clock::now();
//         predict_state();
//         auto prediction_end = std::chrono::high_resolution_clock::now();
//         prediction_time = std::chrono::duration_cast<std::chrono::microseconds>(prediction_end - prediction_start).count() / 1000.0;
        
//         // Track features from previous frame using FeatureTracker
//         // FeatureTracker now handles both tracking and map point association/creation
//         auto tracking_start = std::chrono::high_resolution_clock::now();
//         m_feature_tracker->track_features(m_current_frame, m_previous_frame);
//         auto tracking_end = std::chrono::high_resolution_clock::now();
//         tracking_time = std::chrono::duration_cast<std::chrono::microseconds>(tracking_end - tracking_start).count() / 1000.0;
        
//         result.num_features = m_current_frame->get_feature_count();
        
//         // Compute stereo depth for all features
//         m_current_frame->compute_stereo_depth();
        
//         // Count how many features have associated map points (already done by FeatureTracker)
//         int num_tracked_with_map_points = count_features_with_map_points(m_current_frame);
        
//         // Log tracking information
//         spdlog::info("[TRACKING] {} features tracked, {} with map points", 
//                     result.num_features, num_tracked_with_map_points);
        
//         if (num_tracked_with_map_points > 0) {
//             // ✅ ENABLED: Pose optimization re-enabled after fixing coordinate space mismatch!
//             if (num_tracked_with_map_points >= 5) {
//                 auto optimization_start = std::chrono::high_resolution_clock::now();
//                 auto opt_result = optimize_pose(m_current_frame);
//                 auto optimization_end = std::chrono::high_resolution_clock::now();
//                 optimization_time = std::chrono::duration_cast<std::chrono::microseconds>(optimization_end - optimization_start).count() / 1000.0;
                
//                 result.success = opt_result.success;
//                 result.num_inliers = opt_result.num_inliers;
//                 result.num_outliers = opt_result.num_outliers;
                
//                 if (opt_result.success) {
//                     m_current_pose = opt_result.optimized_pose;
//                     m_current_frame->set_Twb(m_current_pose);
                    
//                     // Log comparison between predicted and optimized pose
//                     if (!m_predicted_pose.isApprox(Eigen::Matrix4f::Identity())) {
//                         Eigen::Matrix4f pose_diff = m_current_pose.inverse() * m_predicted_pose;
//                         Eigen::Vector3f translation_diff = pose_diff.block<3,1>(0,3);
//                         Eigen::Matrix3f rotation_diff = pose_diff.block<3,3>(0,0);
                        
//                         // Compute rotation angle difference
//                         float rotation_angle = std::acos(std::min(1.0f, (rotation_diff.trace() - 1.0f) / 2.0f));
//                         rotation_angle = rotation_angle * 180.0f / M_PI;  // Convert to degrees
                        
//                         spdlog::info("[POSE_COMPARE] Frame {}: Translation diff=({:.3f}, {:.3f}, {:.3f})m, Rotation diff={:.2f}°", 
//                                    m_current_frame->get_frame_id(),
//                                    translation_diff.x(), translation_diff.y(), translation_diff.z(),
//                                    rotation_angle);
//                     }
                    
//                     // 🎯 Compare frame-to-frame transformations: VO vs IMU prediction
//                     if (m_previous_frame) {
//                         // 1. VO-based frame-to-frame transform (optimized result)
//                         Eigen::Matrix4f T_vo_prev = m_previous_frame->get_Twb();
//                         Eigen::Matrix4f T_vo_curr = m_current_frame->get_Twb();
//                         Eigen::Matrix4f delta_T_vo = T_vo_prev.inverse() * T_vo_curr;
                        
//                         // 2. IMU-based frame-to-frame transform (predicted)
//                         Eigen::Matrix4f delta_T_imu = T_vo_prev.inverse() * m_predicted_pose;
                        
//                         // 3. Extract relative translations and rotations
//                         Eigen::Vector3f delta_t_vo = delta_T_vo.block<3,1>(0,3);
//                         Eigen::Vector3f delta_t_imu = delta_T_imu.block<3,1>(0,3);
                        
//                         Eigen::Matrix3f delta_R_vo = delta_T_vo.block<3,3>(0,0);
//                         Eigen::Matrix3f delta_R_imu = delta_T_imu.block<3,3>(0,0);
                        
//                         // Compute translation differences
//                         Eigen::Vector3f translation_diff_vo_imu = delta_t_vo - delta_t_imu;
                        
//                         // Compute rotation differences (angle between rotations)
//                         Eigen::Matrix3f R_diff = delta_R_vo.transpose() * delta_R_imu;
//                         float angle_diff = std::acos(std::min(1.0f, std::max(-1.0f, (R_diff.trace() - 1.0f) / 2.0f)));
//                         float angle_diff_deg = angle_diff * 180.0f / M_PI;
//                     }
                    
//                     // Update transform from last frame for velocity estimation
//                     update_transform_from_last();
                    
//                     spdlog::info("[POSE_OPT] ✅ Optimization successful: {} inliers, {} outliers", opt_result.num_inliers, opt_result.num_outliers);
//                 } else {
//                     spdlog::warn("[POSE_OPT] ❌ Optimization failed - keeping previous pose");
//                 }
//             } else {
//                 spdlog::warn("[POSE_OPT] ⚠️ Not enough map point associations for optimization: {} (need ≥5)", num_tracked_with_map_points);
//                 // Fallback: use current pose as-is
//                 m_current_pose = m_current_frame->get_Twb();
                
//                 // Update transform from last frame for velocity estimation
//                 update_transform_from_last();
                
//                 result.success = true;
//                 result.num_inliers = num_tracked_with_map_points;
//                 result.num_outliers = 0;
//             } 
//         } else {
//             // No tracking, keep previous pose (already set in create_frame)
//             m_current_pose = m_current_frame->get_Twb();
            
//             // Update transform from last frame for velocity estimation (even if tracking failed)
//             update_transform_from_last();
            
//             result.success = false;
//         }

//         // NOTE: FeatureTracker already handles map point association during tracking
//         // No need to call associate_tracked_features_with_map_points() again
        
        
//         // Decide whether to create keyframe
//         auto keyframe_decision_start = std::chrono::high_resolution_clock::now();
//         bool is_keyframe = should_create_keyframe(m_current_frame);
//         auto keyframe_decision_end = std::chrono::high_resolution_clock::now();
//         auto keyframe_decision_time = std::chrono::duration_cast<std::chrono::microseconds>(keyframe_decision_end - keyframe_decision_start).count() / 1000.0;
        
//         // Only create new map points for keyframes to avoid trajectory drift
//         if (is_keyframe) {
//             auto map_points_start = std::chrono::high_resolution_clock::now();
//             int new_map_points = create_new_map_points(m_current_frame);
//             auto map_points_end = std::chrono::high_resolution_clock::now();
//             auto map_points_time = std::chrono::duration_cast<std::chrono::microseconds>(map_points_end - map_points_start).count() / 1000.0;
            
//             result.num_new_map_points = new_map_points;
//             // spdlog::info("[MAP_POINTS] Created {} new map points by new keyframe insertion", new_map_points);
            
//             auto keyframe_creation_start = std::chrono::high_resolution_clock::now();
//             create_keyframe(m_current_frame);
//             auto keyframe_creation_end = std::chrono::high_resolution_clock::now();
//             auto keyframe_creation_time = std::chrono::duration_cast<std::chrono::microseconds>(keyframe_creation_end - keyframe_creation_start).count() / 1000.0;
            
//             m_frames_since_last_keyframe = 0;  // Reset to 0 after creating keyframe
            
//         } else {
//             result.num_new_map_points = 0;
//         }
        
//         // Count tracked features and features with map points
//         result.num_tracked_features = m_current_frame->get_feature_count();
//         result.num_features_with_map_points = count_features_with_map_points(m_current_frame);
        
//         // Compute reprojection error statistics for keyframes
//         if (is_keyframe && count_features_with_map_points(m_current_frame) > 5) {
//             compute_reprojection_error_statistics(m_current_frame);
//         }
        
      
//     } else {
//         // First frame - extract features using FeatureTracker
//         m_feature_tracker->track_features(m_current_frame, nullptr);
        
//         result.num_features = m_current_frame->get_feature_count();
        
//         // Compute stereo depth for all features
//         auto stereo_start = std::chrono::high_resolution_clock::now();
//         m_current_frame->compute_stereo_depth();
//         auto stereo_end = std::chrono::high_resolution_clock::now();
//         auto stereo_time = std::chrono::duration_cast<std::chrono::microseconds>(stereo_end - stereo_start).count() / 1000.0;
        
//         // First frame - keep identity pose (already set in create_frame)
//         m_current_pose = m_current_frame->get_Twb();
        
//         // Increment frame counter (first frame processing)
//         m_frames_since_last_keyframe++;
        
//         // Create initial map points (first frame is always considered keyframe)
//         auto initial_map_points_start = std::chrono::high_resolution_clock::now();
//         int initial_map_points = create_initial_map_points(m_current_frame);
//         auto initial_map_points_end = std::chrono::high_resolution_clock::now();
//         auto initial_map_points_time = std::chrono::duration_cast<std::chrono::microseconds>(initial_map_points_end - initial_map_points_start).count() / 1000.0;
        
//         result.num_new_map_points = initial_map_points;
//         spdlog::info("[MAP_POINTS] Created {} initial map points", initial_map_points);
        
//         auto first_keyframe_start = std::chrono::high_resolution_clock::now();
//         create_keyframe(m_current_frame);
//         auto first_keyframe_end = std::chrono::high_resolution_clock::now();
//         auto first_keyframe_time = std::chrono::duration_cast<std::chrono::microseconds>(first_keyframe_end - first_keyframe_start).count() / 1000.0;
        
//         m_frames_since_last_keyframe = 0;  // Reset after creating first keyframe
        
//         spdlog::info("[TIMING] First frame initialization: stereo={:.2f}ms, initial_map_points={:.2f}ms, keyframe={:.2f}ms", 
//                     stereo_time, initial_map_points_time, first_keyframe_time);
        
//         // Count features for first frame
//         result.num_tracked_features = m_current_frame->get_feature_count();
//         result.num_features_with_map_points = count_features_with_map_points(m_current_frame);
        
//         result.success = true;
//     }
    
//     // Update result
//     result.pose = m_current_frame->get_Twb();
    
//     // Add processed frame to all frames vector for trajectory export
//     m_all_frames.push_back(m_current_frame);
    
//     // Total frames processed (reduced logging)
//     if (m_all_frames.size() % 10 == 0 || m_all_frames.size() <= 5) {
//         spdlog::info("[ESTIMATOR] Processed {} frames", m_all_frames.size());
//     }
    
//     auto end_time = std::chrono::high_resolution_clock::now();
//     auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - total_start_time);
//     result.optimization_time_ms = duration.count() / 1000.0;
    
//     // Set reference keyframe for non-keyframe frames (after pose optimization)
//     if (!m_current_frame->is_keyframe() && m_last_keyframe) {
//         m_current_frame->set_reference_keyframe(m_last_keyframe);
//     }
    
//     // Update state
//     m_previous_frame = m_current_frame;
    
//     // ===== IMU-SPECIFIC PROCESSING CONTINUED =====
//     // Increment frame counter for gravity estimation
//     m_frame_count_since_start++;
    
//     // Log bias values after IMU optimization is enabled
//     if (m_enable_imu_optimization && m_imu_handler) {
//         // Get current bias from IMU handler
//         Eigen::Vector3f accel_bias = m_imu_handler->get_accel_bias();
//         Eigen::Vector3f gyro_bias = m_imu_handler->get_gyro_bias();

//         spdlog::info("Frame {}: Accel bias: [{:.10f}, {:.10f}, {:.10f}], Gyro bias: [{:.10f}, {:.10f}, {:.10f}]",
//                      m_frame_count_since_start,
//                      accel_bias[0], accel_bias[1], accel_bias[2],
//                      gyro_bias[0], gyro_bias[1], gyro_bias[2]);
//     }

//     // 🎯 Attempt gravity estimation if conditions are met (already in VIO mode since IMU data is available)
//     const auto& config = Config::getInstance();
//     if (!m_success_imu_init && m_keyframes.size() >= 5) {  // Unified condition: keyframes >= 5

//         spdlog::info("[GRAVITY_EST] Attempting gravity estimation with {} keyframes", m_keyframes.size());
//         m_success_imu_init = try_initialize_imu();

//         if (m_success_imu_init) {
//             m_gravity_initialized = true;
//             m_enable_imu_optimization = true;  // Enable bias logging
//             spdlog::info("✅ [IMU_INIT] IMU initialization successful!");
            
//             // Enable IMU optimization in sliding window optimizer
//             Eigen::Vector3f gravity_vector = m_imu_handler->get_gravity();
//             std::shared_ptr<IMUHandler> shared_imu_handler = std::shared_ptr<IMUHandler>(m_imu_handler.get(), [](IMUHandler*){});
//             m_sliding_window_optimizer->enable_imu_optimization(shared_imu_handler, gravity_vector.cast<double>());
//             spdlog::info("🚀 [SW_IMU] Enabled IMU optimization in sliding window with gravity: ({:.3f}, {:.3f}, {:.3f})",
//                          gravity_vector.x(), gravity_vector.y(), gravity_vector.z());
           
//         } else {
//             spdlog::warn("❌ [IMU_INIT] IMU initialization failed, will retry later");
//         }

//     }
    
//     return result;
// }



// IMU process_frame overload
Estimator::EstimationResult Estimator::process_frame(const cv::Mat& left_image, const cv::Mat& right_image, 
                                                    long long timestamp, const std::vector<IMUData>& imu_data_from_last_frame) {
    // ===== IMU-SPECIFIC PROCESSING =====
    // Accumulate IMU data from last frame
    for (const auto& imu_data : imu_data_from_last_frame) {
        m_imu_vec_from_last_keyframe.push_back(imu_data);
    }
    
    // Create frame first
    std::shared_ptr<Frame> frame = create_frame(left_image, right_image, timestamp);
    if (!frame) {
        EstimationResult result;
        result.success = false;
        return result;
    }

    if(m_last_keyframe){
        frame->set_accel_bias(m_last_keyframe->get_accel_bias());
        frame->set_gyro_bias(m_last_keyframe->get_gyro_bias());
    }


    // Set IMU data to the frame (frame-to-frame data)
    frame->set_imu_data_from_last_frame(imu_data_from_last_frame);
    
    // Compute frame-to-frame preintegration if IMU data is available
    if (!imu_data_from_last_frame.empty() && m_imu_handler) {
        // Always compute frame-to-frame preintegration, regardless of IMU initialization status
        // This is useful for state prediction and velocity estimation
        
        // 🎯 Use FRAME timestamps for dt calculation (not IMU timestamp range)
        // This ensures dt matches the actual frame interval (0.05s)
        double current_frame_time = static_cast<double>(timestamp) / 1e9;
        double previous_frame_time = m_previous_frame ? 
            static_cast<double>(m_previous_frame->get_timestamp()) / 1e9 : current_frame_time;
        
        auto frame_to_frame_preint = m_imu_handler->preintegrate(imu_data_from_last_frame, previous_frame_time, current_frame_time);
        if (frame_to_frame_preint && frame_to_frame_preint->is_valid()) {
            frame->set_imu_preintegration_from_last_frame(frame_to_frame_preint);
        } else {
            spdlog::warn("[IMU] Failed to create frame-to-frame preintegration for frame {}", frame->get_frame_id());
        }
    }
    
    // Compute from-last-keyframe preintegration for more stable state prediction
    if (!m_imu_vec_from_last_keyframe.empty() && m_imu_handler && m_last_keyframe) {
        double current_frame_time = static_cast<double>(timestamp) / 1e9;
        double last_keyframe_time = static_cast<double>(m_last_keyframe->get_timestamp()) / 1e9;
        
        // Create preintegration from last keyframe to current frame using accumulated IMU data
        auto keyframe_to_frame_preint = m_imu_handler->preintegrate(m_imu_vec_from_last_keyframe, last_keyframe_time, current_frame_time);
        if (keyframe_to_frame_preint && keyframe_to_frame_preint->is_valid()) {
            frame->set_imu_preintegration_from_last_keyframe(keyframe_to_frame_preint);
            // spdlog::debug("[IMU] Created keyframe-to-frame preintegration: dt={:.4f}s", keyframe_to_frame_preint->dt_total);
        } else {
            spdlog::warn("[IMU] Failed to create keyframe-to-frame preintegration for frame {}", frame->get_frame_id());
        }
    }
    
    // Set as current frame for the rest of the processing
    m_current_frame = frame;
    
    // ===== IDENTICAL VO PROCESSING (SAME AS NON-IMU VERSION) =====
    EstimationResult result;
    auto total_start_time = std::chrono::high_resolution_clock::now();

     // Frame processing starts
    if (Config::getInstance().m_enable_debug_output) {
        std::cout<<"\n";
        spdlog::info("============================== Frame {} ==============================\n", m_frame_id_counter);
    }
    // Increment frame counter since last keyframe for every new frame
    m_frames_since_last_keyframe++;

    // Initialize timing variables
    double frame_creation_time = 0.0;
    double prediction_time = 0.0;
    double tracking_time = 0.0;
    double optimization_time = 0.0;

    // IMU data processed (reduced logging)
    if (!imu_data_from_last_frame.empty() && m_current_frame->get_frame_id() % 10 == 0) {
        if (Config::getInstance().m_enable_debug_output) {
            spdlog::info("[IMU] Frame {} processed {} IMU measurements", 
                        m_current_frame->get_frame_id(), imu_data_from_last_frame.size());
        }
    }

    if (m_previous_frame) {
        auto prediction_start = std::chrono::high_resolution_clock::now();
        predict_state();
        auto prediction_end = std::chrono::high_resolution_clock::now();
        prediction_time = std::chrono::duration_cast<std::chrono::microseconds>(prediction_end - prediction_start).count() / 1000.0;
        
        // Track features from previous frame using FeatureTracker
        // FeatureTracker now handles both tracking and map point association/creation
        auto tracking_start = std::chrono::high_resolution_clock::now();
        m_feature_tracker->track_features(m_current_frame, m_previous_frame);
        auto tracking_end = std::chrono::high_resolution_clock::now();
        tracking_time = std::chrono::duration_cast<std::chrono::microseconds>(tracking_end - tracking_start).count() / 1000.0;
        
        result.num_features = m_current_frame->get_feature_count();
        
        // Compute stereo depth for all features
        m_current_frame->compute_stereo_depth();
        
        // Count how many features have associated map points (already done by FeatureTracker)
        int num_tracked_with_map_points = count_features_with_map_points(m_current_frame);
        // Log tracking information
        if (Config::getInstance().m_enable_debug_output) {
            spdlog::info("[TRACKING] {} features tracked, {} with map points", 
                        result.num_features, num_tracked_with_map_points);
        }
       
        
        if (num_tracked_with_map_points > 0) {
            // ✅ ENABLED: Pose optimization re-enabled after fixing coordinate space mismatch!
            if (num_tracked_with_map_points >= 5) {
                auto optimization_start = std::chrono::high_resolution_clock::now();
                auto opt_result = optimize_pose(m_current_frame);
                auto optimization_end = std::chrono::high_resolution_clock::now();
                optimization_time = std::chrono::duration_cast<std::chrono::microseconds>(optimization_end - optimization_start).count() / 1000.0;
                
                result.success = opt_result.success;
                result.num_inliers = opt_result.num_inliers;
                result.num_outliers = opt_result.num_outliers;
                
                if (opt_result.success) {
                    m_current_pose = opt_result.optimized_pose;
                    m_current_frame->set_Twb(m_current_pose);
                    
                    // Log comparison between predicted and optimized pose
                    if (!m_predicted_pose.isApprox(Eigen::Matrix4f::Identity())) {
                        Eigen::Matrix4f pose_diff = m_current_pose.inverse() * m_predicted_pose;
                        Eigen::Vector3f translation_diff = pose_diff.block<3,1>(0,3);
                        Eigen::Matrix3f rotation_diff = pose_diff.block<3,3>(0,0);
                        
                        // Compute rotation angle difference
                        float rotation_angle = std::acos(std::min(1.0f, (rotation_diff.trace() - 1.0f) / 2.0f));
                        rotation_angle = rotation_angle * 180.0f / M_PI;  // Convert to degrees
                        
                        if (Config::getInstance().m_enable_debug_output) {
                            spdlog::info("[POSE_COMPARE] Frame {}: Translation diff=({:.3f}, {:.3f}, {:.3f})m, Rotation diff={:.2f}°",
                                        m_frame_id_counter, translation_diff.x(), translation_diff.y(), translation_diff.z(), rotation_angle);
                        }
                    }
                    
                    // 🎯 Compare frame-to-frame transformations: VO vs IMU prediction
                    if (m_previous_frame) {
                        // 1. VO-based frame-to-frame transform (optimized result)
                        Eigen::Matrix4f T_vo_prev = m_previous_frame->get_Twb();
                        Eigen::Matrix4f T_vo_curr = m_current_frame->get_Twb();
                        Eigen::Matrix4f delta_T_vo = T_vo_prev.inverse() * T_vo_curr;
                        
                        // 2. IMU-based frame-to-frame transform (predicted)
                        Eigen::Matrix4f delta_T_imu = T_vo_prev.inverse() * m_predicted_pose;
                        
                        // 3. Extract relative translations and rotations
                        Eigen::Vector3f delta_t_vo = delta_T_vo.block<3,1>(0,3);
                        Eigen::Vector3f delta_t_imu = delta_T_imu.block<3,1>(0,3);
                        
                        Eigen::Matrix3f delta_R_vo = delta_T_vo.block<3,3>(0,0);
                        Eigen::Matrix3f delta_R_imu = delta_T_imu.block<3,3>(0,0);
                        
                        // Compute translation differences
                        Eigen::Vector3f translation_diff_vo_imu = delta_t_vo - delta_t_imu;
                        
                        // Compute rotation differences (angle between rotations)
                        Eigen::Matrix3f R_diff = delta_R_vo.transpose() * delta_R_imu;
                        float angle_diff = std::acos(std::min(1.0f, std::max(-1.0f, (R_diff.trace() - 1.0f) / 2.0f)));
                        float angle_diff_deg = angle_diff * 180.0f / M_PI;
                       
                    }
                    
                    // Update transform from last frame for velocity estimation
                    update_transform_from_last();
                    
                    if (Config::getInstance().m_enable_debug_output) {
                        spdlog::info("[POSE_OPT] ✅ Optimization successful: {} inliers, {} outliers", opt_result.num_inliers, opt_result.num_outliers);
                    }
                } else {
                    if (Config::getInstance().m_enable_debug_output) {
                        spdlog::warn("[POSE_OPT] ❌ Optimization failed - keeping previous pose");
                    }
                }
            } else {
                if (Config::getInstance().m_enable_debug_output) {
                    spdlog::warn("[POSE_OPT] ⚠️ Not enough map point associations for optimization: {} (need ≥5)", num_tracked_with_map_points);
                }
                // Fallback: use current pose as-is
                m_current_pose = m_current_frame->get_Twb();
                
                // Update transform from last frame for velocity estimation
                update_transform_from_last();
                
                result.success = true;
                result.num_inliers = num_tracked_with_map_points;
                result.num_outliers = 0;
            } 
        } else {
            // No tracking, keep previous pose (already set in create_frame)
            m_current_pose = m_current_frame->get_Twb();
            
            // Update transform from last frame for velocity estimation (even if tracking failed)
            update_transform_from_last();
            
            result.success = false;
        }

        // NOTE: FeatureTracker already handles map point association during tracking
        // No need to call associate_tracked_features_with_map_points() again
        
        
        // Decide whether to create keyframe
        auto keyframe_decision_start = std::chrono::high_resolution_clock::now();
        bool is_keyframe = should_create_keyframe(m_current_frame);
        auto keyframe_decision_end = std::chrono::high_resolution_clock::now();
        auto keyframe_decision_time = std::chrono::duration_cast<std::chrono::microseconds>(keyframe_decision_end - keyframe_decision_start).count() / 1000.0;
        
        // Only create new map points for keyframes to avoid trajectory drift
        if (is_keyframe) {
            auto map_points_start = std::chrono::high_resolution_clock::now();
            int new_map_points = create_new_map_points(m_current_frame);
            auto map_points_end = std::chrono::high_resolution_clock::now();
            auto map_points_time = std::chrono::duration_cast<std::chrono::microseconds>(map_points_end - map_points_start).count() / 1000.0;
            
            result.num_new_map_points = new_map_points;
            // spdlog::info("[MAP_POINTS] Created {} new map points by new keyframe insertion", new_map_points);
            
            auto keyframe_creation_start = std::chrono::high_resolution_clock::now();
            create_keyframe(m_current_frame);
            auto keyframe_creation_end = std::chrono::high_resolution_clock::now();
            auto keyframe_creation_time = std::chrono::duration_cast<std::chrono::microseconds>(keyframe_creation_end - keyframe_creation_start).count() / 1000.0;
            
            m_frames_since_last_keyframe = 0;  // Reset to 0 after creating keyframe
            
        } else {
            result.num_new_map_points = 0;
        }
        
        // Count tracked features and features with map points
        result.num_tracked_features = m_current_frame->get_feature_count();
        result.num_features_with_map_points = count_features_with_map_points(m_current_frame);
        
        // Compute reprojection error statistics for keyframes
        if (is_keyframe && count_features_with_map_points(m_current_frame) > 5) {
            compute_reprojection_error_statistics(m_current_frame);
        }
        
      
    } else {
        // First frame - extract features using FeatureTracker
        m_feature_tracker->track_features(m_current_frame, nullptr);
        
        result.num_features = m_current_frame->get_feature_count();
        
        // Compute stereo depth for all features
        auto stereo_start = std::chrono::high_resolution_clock::now();
        m_current_frame->compute_stereo_depth();
        auto stereo_end = std::chrono::high_resolution_clock::now();
        auto stereo_time = std::chrono::duration_cast<std::chrono::microseconds>(stereo_end - stereo_start).count() / 1000.0;
        
        // First frame - keep identity pose (already set in create_frame)
        m_current_pose = m_current_frame->get_Twb();
        
        // Increment frame counter (first frame processing)
        m_frames_since_last_keyframe++;
        
        // Create initial map points (first frame is always considered keyframe)
        auto initial_map_points_start = std::chrono::high_resolution_clock::now();
        int initial_map_points = create_initial_map_points(m_current_frame);
        auto initial_map_points_end = std::chrono::high_resolution_clock::now();
        auto initial_map_points_time = std::chrono::duration_cast<std::chrono::microseconds>(initial_map_points_end - initial_map_points_start).count() / 1000.0;
        
        result.num_new_map_points = initial_map_points;
        spdlog::info("[MAP_POINTS] Created {} initial map points", initial_map_points);
        
        auto first_keyframe_start = std::chrono::high_resolution_clock::now();
        create_keyframe(m_current_frame);
        auto first_keyframe_end = std::chrono::high_resolution_clock::now();
        auto first_keyframe_time = std::chrono::duration_cast<std::chrono::microseconds>(first_keyframe_end - first_keyframe_start).count() / 1000.0;
        
        m_frames_since_last_keyframe = 0;  // Reset after creating first keyframe
        
        
        // Count features for first frame
        result.num_tracked_features = m_current_frame->get_feature_count();
        result.num_features_with_map_points = count_features_with_map_points(m_current_frame);
        
        result.success = true;
    }
    
    // Update result
    result.pose = m_current_frame->get_Twb();
    
    // Add processed frame to all frames vector for trajectory export
    m_all_frames.push_back(m_current_frame);
   
    
    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - total_start_time);
    result.optimization_time_ms = duration.count() / 1000.0;
    
    // Set reference keyframe for non-keyframe frames (after pose optimization)
    if (!m_current_frame->is_keyframe() && m_last_keyframe) {
        m_current_frame->set_reference_keyframe(m_last_keyframe);
    }
    
    // Update state - release old previous frame's images before updating
    if (m_previous_frame) {
        m_previous_frame->release_images();
    }
    m_previous_frame = m_current_frame;
    
    // ===== IMU-SPECIFIC PROCESSING CONTINUED =====
    // Increment frame counter for gravity estimation
    m_frame_count_since_start++;
    
    // Log bias values after IMU optimization is enabled
    if (m_enable_imu_optimization && m_imu_handler) {
        // Get current bias from IMU handler
        Eigen::Vector3f accel_bias = m_imu_handler->get_accel_bias();
        Eigen::Vector3f gyro_bias = m_imu_handler->get_gyro_bias();

        if (Config::getInstance().m_enable_debug_output) {
            spdlog::info("Frame {}: Accel bias: [{:.10f}, {:.10f}, {:.10f}], Gyro bias: [{:.10f}, {:.10f}, {:.10f}]",
                        m_frame_id_counter, accel_bias.x(), accel_bias.y(), accel_bias.z(),
                        gyro_bias.x(), gyro_bias.y(), gyro_bias.z());
        }
    }

    // 🎯 Attempt gravity estimation if conditions are met (already in VIO mode since IMU data is available)
    const auto& config = Config::getInstance();
    if (!m_success_imu_init && m_keyframes.size() >= 5) {  // Unified condition: keyframes >= 5

        m_success_imu_init = try_initialize_imu();

        if (m_success_imu_init) {
            m_gravity_initialized = true;
            m_enable_imu_optimization = true;  // Enable bias logging
            
            // Enable IMU optimization in sliding window optimizer
            Eigen::Vector3f gravity_vector = m_imu_handler->get_gravity();
            std::shared_ptr<IMUHandler> shared_imu_handler = std::shared_ptr<IMUHandler>(m_imu_handler.get(), [](IMUHandler*){});
            m_sliding_window_optimizer->enable_imu_optimization(shared_imu_handler, gravity_vector.cast<double>());
           
        } 

    }
    
    return result;
}


Estimator::~Estimator() {
    // Stop sliding window thread
    if (m_sliding_window_thread_running) {
        m_sliding_window_thread_running = false;
        m_keyframes_cv.notify_one();
        
        if (m_sliding_window_thread && m_sliding_window_thread->joinable()) {
            m_sliding_window_thread->join();
        }
        
    }
}

void Estimator::reset() {
    m_current_frame.reset();
    m_previous_frame.reset();
    m_last_keyframe.reset();
    m_keyframes.clear();
    m_all_frames.clear();
    m_map_points.clear();
    
    m_frame_id_counter = 0;
    m_frames_since_last_keyframe = 0;
    m_last_keyframe_grid_coverage = 0.0;
    m_current_pose = Eigen::Matrix4f::Identity();
}

Eigen::Matrix4f Estimator::get_current_pose() const {
    return m_current_pose;
}

std::vector<std::shared_ptr<Frame>> Estimator::get_keyframes_safe() const {
    std::lock_guard<std::mutex> lock(m_keyframes_mutex);
    return m_keyframes;  // Return a copy
}

std::vector<std::shared_ptr<MapPoint>> Estimator::get_map_points_safe() const {
    std::lock_guard<std::mutex> lock(m_map_points_mutex);  // Use same mutex as keyframes since they're related
    return m_map_points;  // Return a copy
}

std::shared_ptr<Frame> Estimator::create_frame(const cv::Mat& left_image, const cv::Mat& right_image, long long timestamp) {
    if (left_image.empty() || right_image.empty()) {
        return nullptr;
    }
    
    // Convert to grayscale if needed
    cv::Mat gray_left, gray_right;
    if (left_image.channels() == 3) {
        cv::cvtColor(left_image, gray_left, cv::COLOR_BGR2GRAY);
    } else {
        gray_left = left_image.clone();
    }
    
    if (right_image.channels() == 3) {
        cv::cvtColor(right_image, gray_right, cv::COLOR_BGR2GRAY);
    } else {
        gray_right = right_image.clone();
    }
    
    // Create frame with stereo images and camera parameters
    // Get camera parameters from global config
    const auto& global_config = Config::getInstance();
    cv::Mat left_K = global_config.left_camera_matrix();
    
    auto frame = std::make_shared<Frame>(
        timestamp, 
        m_frame_id_counter++,
        gray_left, gray_right,
        left_K.at<double>(0, 0), left_K.at<double>(1, 1), left_K.at<double>(0, 2), left_K.at<double>(1, 2),
        global_config.left_dist_coeffs()
    );
    
    // Set initial pose and velocity
    if (m_previous_frame) {
        // For non-first frames, start with previous frame pose
        // Actual prediction will be done in process_frame() via predict_state()
        frame->set_Twb(m_previous_frame->get_Twb());

      
        // Initialize velocity to zero
        frame->set_velocity(Eigen::Vector3f::Zero());
    } else {
        // First frame - use ground truth pose if available, otherwise identity
        if (m_has_initial_gt_pose) {
            frame->set_Twb(m_initial_gt_pose);
        } else {
            frame->set_Twb(Eigen::Matrix4f::Identity());
        }
        
        // First frame velocity is zero
        frame->set_velocity(Eigen::Vector3f::Zero());
    }
    
    // Inherit IMU bias from the last keyframe (if available)
    if (m_last_keyframe && m_imu_handler) {
        m_imu_handler->inherit_bias_from_keyframe(frame.get(), m_last_keyframe.get());
    }
    
    return frame;
}

std::shared_ptr<Frame> Estimator::create_rgbd_frame(const cv::Mat& rgb_image, const cv::Mat& depth_map, long long timestamp) {
    if (rgb_image.empty() || depth_map.empty()) {
        return nullptr;
    }
    
    // Player already passes preprocessed grayscale image, so just use it directly
    // No need to convert or clone
    const cv::Mat& gray_image = rgb_image;  // Direct reference (no copy)
    
    // Get camera parameters from global config
    const auto& global_config = Config::getInstance();
    cv::Mat K = global_config.left_camera_matrix();
   
    auto frame = std::make_shared<Frame>(
        timestamp,
        m_frame_id_counter++,
        gray_image,
        depth_map,
        K.at<double>(0, 0), K.at<double>(1, 1), K.at<double>(0, 2), K.at<double>(1, 2),
        global_config.left_dist_coeffs(),
        true  // is_rgbd flag
    );

    
    // Set initial pose and velocity
    if (m_previous_frame) {
        // For non-first frames, start with previous frame pose
        // Actual prediction will be done in process_frame() via predict_state()
        frame->set_Twb(m_previous_frame->get_Twb());
        
        // Initialize velocity to zero
        frame->set_velocity(Eigen::Vector3f::Zero());
    } else {
        // First frame - use ground truth pose if available, otherwise identity
        if (m_has_initial_gt_pose) {
            frame->set_Twb(m_initial_gt_pose);
            Eigen::Vector3f gt_pos = m_initial_gt_pose.block<3,1>(0,3);
            spdlog::info("[ESTIMATOR] First RGBD frame initialized with GT pose at [{:.3f}, {:.3f}, {:.3f}]",
                        gt_pos.x(), gt_pos.y(), gt_pos.z());
        } else {
            frame->set_Twb(Eigen::Matrix4f::Identity());
            spdlog::info("[ESTIMATOR] First RGBD frame initialized at origin (Identity)");
        }
        
        // First frame velocity is zero
        frame->set_velocity(Eigen::Vector3f::Zero());
    }
    
    // Inherit IMU bias from the last keyframe (if available)
    if (m_last_keyframe && m_imu_handler) {
        m_imu_handler->inherit_bias_from_keyframe(frame.get(), m_last_keyframe.get());
    }
    
    return frame;
}



int lightweight_vio::Estimator::create_initial_map_points(std::shared_ptr<Frame> frame) {
    if (!frame) {
        return 0;
    }
    
    int num_created = 0;
    const auto& features = frame->get_features();
    
  
    // Create map points from stereo triangulated features
    for (size_t i = 0; i < features.size(); ++i) {
        auto feature = features[i];
        if (feature && feature->is_valid() && frame->has_depth(i)) {
            // Get 3D point in camera frame from stereo triangulation
            Eigen::Vector3f camera_3d_point = feature->get_3d_point();

            if (camera_3d_point.isZero()) {
                continue;  // Skip if no valid 3D point
            }
            
            // Transform to world coordinates using frame pose
            Eigen::Matrix4f T_wb = frame->get_Twb();
            
            // Use Frame's cached T_cb for consistency
            Eigen::Matrix4f T_cb = frame->get_Tcb().cast<float>();  // Body to Camera
            Eigen::Matrix4f T_bc = T_cb.inverse();     // Camera to Body
            
            // Transform: Camera → Body → World
            Eigen::Vector4f camera_point(camera_3d_point.x(), camera_3d_point.y(), camera_3d_point.z(), 1.0);
            Eigen::Vector4f body_point = T_bc * camera_point;  // Camera to Body
            Eigen::Vector4f world_point = T_wb * body_point;    // Body to World
            
            Eigen::Vector3f world_pos = world_point.head<3>();
            
            auto map_point = std::make_shared<MapPoint>(world_pos);

            map_point->add_observation(frame, i);
            
            // // Initialize uncertainty from config
            // if(Config::getInstance().m_uncertainty_enable)
            //     map_point->update_world_uncertainty_with_observations();
            
            m_map_points.push_back(map_point);
            
            // Associate with frame
            frame->set_map_point(i, map_point);
            num_created++;
            
            // Compute reprojection error for verification
            double fx, fy, cx, cy;
            frame->get_camera_intrinsics(fx, fy, cx, cy);
            
            // Project world point back to camera
            Eigen::Vector4f world_pos_h(world_pos.x(), world_pos.y(), world_pos.z(), 1.0f);
            Eigen::Vector4f camera_pos_h = T_cb * (T_wb.inverse() * world_pos_h);
            Eigen::Vector3f camera_pos = camera_pos_h.head<3>();
            
            if (camera_pos.z() > 0) {
                // Project to pixel coordinates using camera intrinsics
                float u_proj = fx * camera_pos.x() / camera_pos.z() + cx;
                float v_proj = fy * camera_pos.y() / camera_pos.z() + cy;
                
                // Get original undistorted pixel coordinates for fair comparison
                cv::Point2f undistorted_pixel = feature->get_undistorted_coord();
                // Convert normalized to undistorted pixel coordinates
                float undist_u = undistorted_pixel.x;
                float undist_v = undistorted_pixel.y;
                
                // Compute reprojection error in undistorted pixel coordinate space
                double error_x = undist_u - u_proj;
                double error_y = undist_v - v_proj;
                double error = std::sqrt(error_x * error_x + error_y * error_y);
            } else {
                // Point behind camera
            }
        }
    }
    
    return num_created;
}



int lightweight_vio::Estimator::create_new_map_points(std::shared_ptr<Frame> frame) {
    if (!frame) {
        return 0;
    }
    
    int num_created = 0;
    const auto& features = frame->get_features();
    
    // Only create map points for features that:
    // 1. Have valid stereo depth
    // 2. Are not already associated with a map point
    // 3. Are valid features
    // 4. NEW: Include outlier features if they have stereo matches
    for (size_t i = 0; i < features.size(); ++i) {
        auto feature = features[i];
        
        if (!feature || !feature->is_valid()) {
            continue;
        }
        
        if (!frame->has_depth(i)) {
            continue;
        }
        
        if (frame->has_map_point(i)) {
            continue;
        }
        
        double depth = frame->get_depth(i);
        
        // Validate depth range using global config parameters
        auto& global_config = lightweight_vio::Config::getInstance();
        if (depth < global_config.m_min_depth || depth > global_config.m_max_depth) {
            continue;  // Skip invalid depths
        }
        
        // Get 3D point in camera frame from stereo triangulation
        Eigen::Vector3f camera_3d_point = feature->get_3d_point();
        if (camera_3d_point.isZero()) {
            continue;  // Skip if no valid 3D point
        }
        
        // Transform to world coordinates using frame pose
        Eigen::Matrix4f T_wb = frame->get_Twb();
        
        // Get actual T_bc from configuration 
        const auto& config = lightweight_vio::Config::getInstance();
        cv::Mat T_bc_cv = config.left_T_BC();  // Get T_BC from config (camera to body)
        
        Eigen::Matrix4f T_bc;
        if (!T_bc_cv.empty()) {
            // Convert T_bc to Eigen
            for (int i = 0; i < 4; ++i) {
                for (int j = 0; j < 4; ++j) {
                    T_bc(i, j) = T_bc_cv.at<double>(i, j);
                }
            }
        } else {
            // Fallback to identity if config not available
            T_bc = Eigen::Matrix4f::Identity();
        }
        
        // Transform: Camera → Body → World
        Eigen::Vector4f camera_point(camera_3d_point.x(), camera_3d_point.y(), camera_3d_point.z(), 1.0);
        Eigen::Vector4f body_point = T_bc * camera_point;
        Eigen::Vector4f world_point = T_wb * body_point;
        
        Eigen::Vector3f world_pos = world_point.head<3>();
        
        // Create new map point
        auto map_point = std::make_shared<MapPoint>(world_pos);
        
        // Initialize uncertainty from config
        
        
        m_map_points.push_back(map_point);
        
        // Associate with current frame
        frame->set_map_point(i, map_point);
        
        // Real-time reprojection verification for new map point
        // spdlog::debug("NEW Map Point {}: world_pos=({:.3f}, {:.3f}, {:.3f})", 
        //              m_map_points.size() - 1, world_pos.x(), world_pos.y(), world_pos.z());
        
        // Verify reprojection in current frame
        Eigen::Matrix4f T_wc = T_wb * T_bc;  // Camera to world
        Eigen::Matrix4f T_cw = T_wc.inverse(); // World to camera (for projection)
        
        // Project back to image
        Eigen::Vector4f world_homogeneous(world_pos.x(), world_pos.y(), world_pos.z(), 1.0);
        Eigen::Vector4f camera_projected = T_cw * world_homogeneous;
        
        if (camera_projected.z() > 0) {  // Valid projection
            auto feature = frame->get_feature(i);
            if (feature) {
                cv::Point2f observed_pt = feature->get_undistorted_coord();
                
                // Get camera intrinsics from frame
                double fx, fy, cx, cy;
                frame->get_camera_intrinsics(fx, fy, cx, cy);
                
                float projected_x = (fx * camera_projected.x() / camera_projected.z()) + cx;
                float projected_y = (fy * camera_projected.y() / camera_projected.z()) + cy;
                
                float reprojection_error = sqrt(pow(observed_pt.x - projected_x, 2) + pow(observed_pt.y - projected_y, 2));
                
                // spdlog::debug("  Reprojection: observed=({:.2f}, {:.2f}), projected=({:.2f}, {:.2f}), error={:.3f}px", 
                //              observed_pt.x, observed_pt.y, projected_x, projected_y, reprojection_error);
            }
        }
        
        num_created++;
    }
    
    return num_created;
}



bool lightweight_vio::Estimator::should_create_keyframe(std::shared_ptr<Frame> frame) {

    if (!frame) {
        return false;
    }
    
    // First frame is always a keyframe
    if (m_keyframes.empty()) {
        return true;
    }
    
    // Time-based keyframe creation policy
    // Force keyframe creation if time since last keyframe exceeds threshold
    if (m_last_keyframe) {
        double current_time = static_cast<double>(frame->get_timestamp()) / 1e9;  // Convert nanoseconds to seconds
        double last_keyframe_time = static_cast<double>(m_last_keyframe->get_timestamp()) / 1e9;
        double time_diff = current_time - last_keyframe_time;
        
        if (time_diff >= Config::getInstance().m_keyframe_time_threshold) {
            return true;
        }
    }
    
    // Grid-based keyframe creation policy
    // Create keyframe when grid coverage drops to configured ratio of last keyframe's coverage
    double current_grid_coverage = calculate_grid_coverage_with_map_points(frame);
    
    // For the first few keyframes, use absolute threshold of 50% to establish baseline
    if (m_keyframes.size() <= 2 || m_last_keyframe_grid_coverage <= 0.0) {
        double absolute_threshold = 0.5;
        if (current_grid_coverage < absolute_threshold) {
            if (Config::getInstance().m_enable_debug_output) {
                spdlog::info("[KEYFRAME] Creating keyframe due to low grid coverage (initial): {:.2f} < {:.2f}", 
                            current_grid_coverage, Config::getInstance().m_grid_coverage_ratio);
            }
            return true;
        }
    } else {
        // Use relative threshold based on last keyframe's coverage
        double relative_threshold = m_last_keyframe_grid_coverage * Config::getInstance().m_grid_coverage_ratio;
        if (current_grid_coverage < relative_threshold) {
            if (Config::getInstance().m_enable_debug_output) {
                spdlog::info("[KEYFRAME] Creating keyframe due to low grid coverage (relative): {:.2f} < {:.2f}", 
                            current_grid_coverage, relative_threshold);
            }
            return true;
        }
    }
    
    return false;
}

void lightweight_vio::Estimator::create_keyframe(std::shared_ptr<Frame> frame) {
    if (!frame) {
        return;
    }
    
    frame->set_keyframe(true);
    
    // Calculate time difference from last keyframe
    double dt_from_last_kf = 0.0;
    if (m_last_keyframe) {
        double current_time = static_cast<double>(frame->get_timestamp()) / 1e9;  // Convert ns to seconds
        double last_kf_time = static_cast<double>(m_last_keyframe->get_timestamp()) / 1e9;
        dt_from_last_kf = current_time - last_kf_time;
        
    } 
    
    
    frame->set_dt_from_last_keyframe(dt_from_last_kf);
    
    // Transfer accumulated IMU data to the new keyframe
    transfer_imu_data_to_keyframe(frame);
    
    // Initialize velocity from preintegration if available
    frame->initialize_velocity_from_preintegration();
    
    // Thread-safe keyframe management
    {
        std::lock_guard<std::mutex> lock(m_keyframes_mutex);
        m_keyframes.push_back(frame);
        
        // Apply sliding window - remove old keyframes if window size exceeded
        const int max_keyframes = Config::getInstance().m_keyframe_window_size;
        if (m_keyframes.size() > static_cast<size_t>(max_keyframes)) {
            // Remove oldest keyframe
            auto oldest_keyframe = m_keyframes.front();
            
            // Clean up observations from map points before removing the keyframe
            int removed_observations = 0;
            const auto& features = oldest_keyframe->get_features();
            for (size_t i = 0; i < features.size(); ++i) {
                auto feature = features[i];
                auto map_point = oldest_keyframe->get_map_point(i);
                
                if (feature && feature->is_valid() && map_point && !map_point->is_bad()) {
                    // Remove observation from map point
                    map_point->remove_observation(oldest_keyframe);
                    removed_observations++;
                    
                    // Check if map point has no more observations after removal
                    if (map_point->get_observation_count() == 0) {
                        // Mark map point as bad if it has no observations
                        map_point->set_bad();
                    }
                }
            }
            
            m_keyframes.erase(m_keyframes.begin());
        }
    } // Release mutex lock here
    
    // 🎯 Update track count only when frame becomes keyframe
    const auto& features = frame->get_features();
    for (auto& feature : features) {
        if (feature && feature->is_valid()) {
            feature->increment_track_count();
        }
    }
    
    // Add observations to map points for this keyframe
    int observations_added = 0;
    // Use the existing features variable from above - no duplicate declaration
    for (size_t i = 0; i < features.size(); ++i) {
        auto feature = features[i];
        auto map_point = frame->get_map_point(i);
        
        if (feature && feature->is_valid() && map_point && !map_point->is_bad()) {
            // Add this keyframe as an observation to the map point
            map_point->add_observation(frame, i);
            observations_added++;
        }
    }



    
    // Store grid coverage of this keyframe for future relative comparisons
    m_last_keyframe_grid_coverage = calculate_grid_coverage_with_map_points(frame);
    

    
    // Update last keyframe reference
    m_last_keyframe = frame;
    
    // Notify sliding window optimization thread
    notify_sliding_window_thread();
}

OptimizationResult lightweight_vio::Estimator::optimize_pose(std::shared_ptr<Frame> frame) {
    // Create pose optimizer - uses global Config internally
    PnPOptimizer optimizer;
    return optimizer.optimize_pose(frame);
}

int lightweight_vio::Estimator::count_features_with_map_points(std::shared_ptr<Frame> frame) {
    if (!frame) {
        return 0;
    }
    
    int count = 0;
    const auto& map_points = frame->get_map_points();
    
    for (const auto& mp : map_points) {
        if (mp && !mp->is_bad()) {
            count++;
        }
    }
    
    return count;
}



void Estimator::set_initial_gt_pose(const Eigen::Matrix4f& gt_pose) {
    m_initial_gt_pose = gt_pose;
    m_has_initial_gt_pose = true;
}

void Estimator::apply_gt_pose_to_current_frame(const Eigen::Matrix4f& gt_pose) {
    if (m_current_frame) {
        m_current_frame->set_Twb(gt_pose);
        m_current_pose = gt_pose;
        spdlog::debug("[GT_APPLY] Applied GT pose to frame {}", m_current_frame->get_frame_id());
    }
}



void lightweight_vio::Estimator::compute_reprojection_error_statistics(std::shared_ptr<Frame> frame) {
    if (!frame) {
        return;
    }
    
    std::vector<double> reprojection_errors;
    const auto& features = frame->get_features();
    const auto& map_points = frame->get_map_points();
    
    // Get camera parameters
    double fx, fy, cx, cy;
    frame->get_camera_intrinsics(fx, fy, cx, cy);
    
    // DEBUG: Print camera parameters
    // spdlog::debug("[REPROJ_DEBUG] Camera params: fx={:.2f}, fy={:.2f}, cx={:.2f}, cy={:.2f}", fx, fy, cx, cy);
    
    // Get current pose
    Eigen::Matrix4f T_wb = frame->get_Twb();
    
    
    // Get T_cb from frame
    const Eigen::Matrix4d& T_cb = frame->get_Tcb();
    Eigen::Matrix4f T_cb_f = T_cb.cast<float>();
   
    
    int valid_projections = 0;
    int behind_camera = 0;
    
    for (size_t i = 0; i < features.size() && i < map_points.size(); ++i) {
        auto feature = features[i];
        auto map_point = map_points[i];
        
        if (!feature || !feature->is_valid() || !map_point || map_point->is_bad()) {
            continue;
        }
        
        // Get 3D world position
        Eigen::Vector3f world_pos = map_point->get_position();
        
        // CORRECTED: Transform from world to camera using correct matrix chain
        // T_wc = T_wb * T_bc (Camera → World)
        Eigen::Matrix4f T_bc = T_cb_f.inverse();  // Camera to Body (T_bc = T_cb^-1)
        Eigen::Matrix4f T_wc = T_wb * T_bc;   // Camera to World transformation
        
        // Transform world point to camera
        Eigen::Vector4f world_pos_h(world_pos.x(), world_pos.y(), world_pos.z(), 1.0f);
        Eigen::Vector4f camera_pos_h = T_wc.inverse() * world_pos_h;
        Eigen::Vector3f camera_pos = camera_pos_h.head<3>();
        
        // Check if point is behind camera
        if (camera_pos.z() <= 0) {
            behind_camera++;
            continue;
        }
        
        // Project to pixel coordinates using camera intrinsics
        float u_proj = fx * camera_pos.x() / camera_pos.z() + cx;
        float v_proj = fy * camera_pos.y() / camera_pos.z() + cy;
        
        // Get original undistorted pixel coordinates for fair comparison
        cv::Point2f undistorted_pixel = feature->get_undistorted_coord();
        // Convert normalized to undistorted pixel coordinates
        float undist_u = undistorted_pixel.x;
        float undist_v = undistorted_pixel.y;

        // Compute reprojection error in undistorted pixel coordinate space
        double error_x = undist_u - u_proj;
        double error_y = undist_v - v_proj;
        double error = std::sqrt(error_x * error_x + error_y * error_y);
        
        reprojection_errors.push_back(error);
        valid_projections++;
        
        // Print core info for all features in one line  
        // spdlog::info("[REPROJ] F{}: obs_undist=({:.1f},{:.1f}) proj_pixel=({:.1f},{:.1f}) err={:.2f}px world=({:.2f},{:.2f},{:.2f})", 
        //              i, undist_u, undist_v, u_proj, v_proj, error,
        //              world_pos.x(), world_pos.y(), world_pos.z());
    }
    
    if (valid_projections > 0) {
        // Compute statistics
        std::sort(reprojection_errors.begin(), reprojection_errors.end());
        
        double mean_error = std::accumulate(reprojection_errors.begin(), reprojection_errors.end(), 0.0) / reprojection_errors.size();
        double median_error = reprojection_errors[reprojection_errors.size() / 2];
        double min_error = reprojection_errors.front();
        double max_error = reprojection_errors.back();
        
        // Count outliers (error > 5.0 pixels in undistorted pixel space)
        int outliers = std::count_if(reprojection_errors.begin(), reprojection_errors.end(), 
                                   [](double error) { return error > 5.0; });
        
        // spdlog::info("[REPROJ_ERROR] Frame {}: {}/{} valid, mean={:.2f}px, median={:.2f}px, min={:.2f}px, max={:.2f}px, outliers={}", 
        //             frame->get_frame_id(), valid_projections, features.size(), 
        //             mean_error, median_error, min_error, max_error, outliers);
        
        if (behind_camera > 0) {
            spdlog::warn("[REPROJ_ERROR] {} points behind camera", behind_camera);
        }
    } else {
        spdlog::warn("[REPROJ_ERROR] Frame {}: No valid projections", frame->get_frame_id());
    }
}

void Estimator::predict_state() {
    if (!m_current_frame || !m_previous_frame) {
        return;
    }
    
    const auto& config = Config::getInstance();
    
    // Check system mode from config
    if (config.m_system_mode == "VIO" && m_success_imu_init) {
        // VIO Mode: Use IMU preintegration for state prediction
        // Only use IMU prediction when IMU is properly initialized

        
        // Get from-last-keyframe IMU preintegration (more stable for longer intervals)
        auto keyframe_to_frame_preint = m_current_frame->get_imu_preintegration_from_last_frame();
        
        if (keyframe_to_frame_preint && keyframe_to_frame_preint->is_valid() && m_last_keyframe) {
            // Use IMU preintegration from last keyframe (more robust)
            
            // Get gravity vector from IMU handler (gravity-aligned coordinate system)
            Eigen::Vector3f Gz = m_imu_handler->get_gravity();
            
            // Get last keyframe state as reference
            const Eigen::Vector3f twb1 = m_last_keyframe->get_Twb().block<3,1>(0,3);     // Position
            const Eigen::Matrix3f Rwb1 = m_last_keyframe->get_Twb().block<3,3>(0,0);     // Rotation  
            const Eigen::Vector3f Vwb1 = m_last_keyframe->get_velocity();                // Velocity
            
            // Get preintegration data and time interval from last keyframe
            const float t12 = keyframe_to_frame_preint->dt_total;
            
            // Get IMU bias from last keyframe
            const Eigen::Vector3f gyro_bias = m_last_keyframe->get_gyro_bias();
            const Eigen::Vector3f accel_bias = m_last_keyframe->get_accel_bias();

            // IMU prediction from last keyframe to current frame
            Eigen::Matrix3f Rwb2 = Rwb1 * keyframe_to_frame_preint->delta_R;  
            Eigen::Vector3f twb2 = twb1 + Vwb1*t12 + 0.5f*t12*t12*Gz + Rwb1*keyframe_to_frame_preint->delta_P;
            Eigen::Vector3f Vwb2 = Vwb1 + t12*Gz + Rwb1*keyframe_to_frame_preint->delta_V;
            
            // Set predicted state
            Eigen::Matrix4f predicted_pose = Eigen::Matrix4f::Identity();
            predicted_pose.block<3,3>(0,0) = Rwb2;
            predicted_pose.block<3,1>(0,3) = twb2;
            
            // Store predicted pose for comparison logging
            m_predicted_pose = predicted_pose;
            m_current_frame->set_Twb(predicted_pose);
            m_current_frame->set_velocity(Vwb2);
            
          
            
            
        } else {
            // Fallback to constant velocity model if no valid preintegration
            
            // VO Mode fallback: Use visual motion model
            Eigen::Matrix4f predicted_pose = m_previous_frame->get_Twb() * m_transform_from_last;
            m_predicted_pose = predicted_pose; // Store for comparison
            
            // Keep velocity zero in fallback mode
            m_current_frame->set_velocity(Eigen::Vector3f::Zero());
        }
        
    } else {
        // VO Mode: Use visual odometry motion model
        Eigen::Matrix4f predicted_pose = m_previous_frame->get_Twb() * m_transform_from_last;
        m_predicted_pose = predicted_pose; // Store for comparison
        
        // Keep velocity zero in VO mode
        m_current_frame->set_velocity(Eigen::Vector3f::Zero());
        
    }
}


void Estimator::update_transform_from_last() {
    // No visual velocity calculation - velocity is handled by IMU or set to zero
    // This function is kept for compatibility but doesn't perform any calculations
}

double lightweight_vio::Estimator::calculate_grid_coverage_with_map_points(std::shared_ptr<Frame> frame) {
    if (!frame || frame->get_feature_count() == 0) {
        return 0.0;
    }
    
    const Config& config = Config::getInstance();
    const int grid_rows = config.m_grid_rows;
    const int grid_cols = config.m_grid_cols;
    const int img_width = config.m_image_width;
    const int img_height = config.m_image_height;
    
    // Initialize grid to track which cells have features with map points
    std::vector<std::vector<bool>> grid_has_map_point(grid_rows, std::vector<bool>(grid_cols, false));
    
    // Check each feature
    const auto& features = frame->get_features();
    for (size_t i = 0; i < features.size(); ++i) {
        const auto& feature = features[i];
        if (!feature || !feature->is_valid()) {
            continue;
        }
        
        // Check if this feature has an associated map point
        auto map_point = frame->get_map_point(i);
        if (!map_point || map_point->is_bad()) {
            continue;
        }
        
        // Calculate grid coordinates
        cv::Point2f pixel_coord = feature->get_pixel_coord();
        
        // Safety check for pixel coordinates
        if (pixel_coord.x < 0 || pixel_coord.x >= img_width || 
            pixel_coord.y < 0 || pixel_coord.y >= img_height) {
            continue;
        }
        
        float cell_width = (float)img_width / grid_cols;
        float cell_height = (float)img_height / grid_rows;
        
        int grid_x = std::min((int)(pixel_coord.x / cell_width), grid_cols - 1);
        int grid_y = std::min((int)(pixel_coord.y / cell_height), grid_rows - 1);
        
        // Additional safety check for grid indices
        if (grid_x >= 0 && grid_x < grid_cols && grid_y >= 0 && grid_y < grid_rows) {
            grid_has_map_point[grid_y][grid_x] = true;
        }
    }
    
    // Count cells with map points
    int cells_with_map_points = 0;
    int total_cells = grid_rows * grid_cols;
    
    for (int row = 0; row < grid_rows; ++row) {
        for (int col = 0; col < grid_cols; ++col) {
            if (grid_has_map_point[row][col]) {
                cells_with_map_points++;
            }
        }
    }
    
    double coverage_ratio = (double)cells_with_map_points / total_cells;
    
    // if (config.m_enable_debug_output) {
    //     // spdlog::debug("Grid coverage: {}/{} cells have features with map points ({:.2f}%)", 
    //     //              cells_with_map_points, total_cells, coverage_ratio * 100.0);
    // }
    
    return coverage_ratio;
}

void lightweight_vio::Estimator::notify_sliding_window_thread() {
    {
        std::lock_guard<std::mutex> lock(m_keyframes_mutex);
        m_keyframes_updated = true;
    }
    m_keyframes_cv.notify_one();
}

void lightweight_vio::Estimator::sliding_window_thread_function() {
    if (Config::getInstance().m_enable_debug_output) {
        spdlog::info("[SW_THREAD] Sliding window optimization thread started");
    }
    
    while (m_sliding_window_thread_running) {
        // Wait for keyframe updates
        std::unique_lock<std::mutex> lock(m_keyframes_mutex);
        m_keyframes_cv.wait(lock, [this] { 
            return !m_sliding_window_thread_running || m_keyframes_updated; 
        });
        
        if (!m_sliding_window_thread_running) {
            break;
        }
        
        if (m_keyframes_updated) {
            m_keyframes_updated = false;
            
            // Copy current keyframes for optimization (thread-safe)
            std::vector<std::shared_ptr<Frame>> keyframes_copy = m_keyframes;
            lock.unlock(); // Release lock early
            
            // Run sliding window bundle adjustment when we have enough keyframes
            if (keyframes_copy.size() >= 2) {
                auto sw_opt_start = std::chrono::high_resolution_clock::now();
                
                auto sw_result = m_sliding_window_optimizer->optimize(keyframes_copy);
                
                auto sw_opt_end = std::chrono::high_resolution_clock::now();
                auto sw_opt_time = std::chrono::duration_cast<std::chrono::microseconds>(sw_opt_end - sw_opt_start).count() / 1000.0;
                
               
            } 
        }
    }
}

void lightweight_vio::Estimator::transfer_imu_data_to_keyframe(std::shared_ptr<Frame> keyframe) {
    if (!keyframe) {
        return;
    }
    
    // Transfer accumulated IMU data since last keyframe to the new keyframe
    if (!m_imu_vec_from_last_keyframe.empty()) {
        keyframe->set_imu_data_since_last_keyframe(m_imu_vec_from_last_keyframe);
        
        
        // Log time range for verification
        double first_time = m_imu_vec_from_last_keyframe.front().timestamp;
        double last_time = m_imu_vec_from_last_keyframe.back().timestamp;
        double frame_time = static_cast<double>(keyframe->get_timestamp()) / 1e9;
        
        // spdlog::debug("[IMU] IMU data range: {:.6f}s to {:.6f}s, Keyframe time: {:.6f}s", first_time, last_time, frame_time);
        
        // Create preintegration for this keyframe interval
        if (m_imu_handler) {
            // Always compute preintegration, regardless of IMU initialization status
            // This allows us to use preintegration for velocity estimation during IMU initialization
            
            auto preint = m_imu_handler->preintegrate(m_imu_vec_from_last_keyframe, first_time, last_time);
            if (preint && preint->is_valid()) {
                // Store preintegration result from last keyframe in keyframe
                keyframe->set_imu_preintegration_from_last_keyframe(preint);
                // spdlog::debug("[IMU] Preintegration from last keyframe completed and stored for keyframe {}: dt={:.3f}s", keyframe->get_frame_id(), preint->dt_total);
            } else {
                spdlog::warn("[IMU] Failed to create preintegration from last keyframe for keyframe {}", keyframe->get_frame_id());
            }
        } else {
            spdlog::warn("[IMU] IMU handler not available for preintegration");
        }
        
        // Clear the buffer for next keyframe interval
        m_imu_vec_from_last_keyframe.clear();
        // spdlog::debug("[IMU] IMU buffer cleared for next keyframe interval");
    }
}

bool lightweight_vio::Estimator::try_initialize_imu() {
    /*
     * 🎯 IMU INITIALIZATION WITH GRAVITY ESTIMATION & BIAS OPTIMIZATION
     * 
     * OBJECTIVE: Initialize IMU parameters (gravity direction, biases) using visual-inertial constraints
     * 
     * TWO-PHASE PROCESS:
     * ===============================================================================
     * PHASE 1: GRAVITY ESTIMATION from visual-inertial comparison
     * - Visual odometry provides true motion: T_visual = T_wb(t1) * T_wb(t0)^-1  
     * - IMU integration without gravity: T_imu = integrate(omega, a_b - g_b)
     * - Gravity effect emerges from difference: Δp_gravity = p_visual - p_imu
     * - Average over multiple intervals to find gravity direction
     * 
     * PHASE 2: IMU PARAMETER OPTIMIZATION using factor graph
     * - InertialGravityFactor: Constrains gravity-aligned accelerometer measurements
     * - Optimize velocities and biases jointly with known gravity direction
     * - Establishes consistent IMU coordinate frame for future VIO
     * ===============================================================================
     * 
     * ===============================================================================
     * MATHEMATICAL FOUNDATION:
     * 
     * Gravity Estimation:
     * - For interval [t0, t1]: Δp_gravity_i = T_visual.translation() - integrate(v_imu_no_gravity)
     * - Gravity vector: g_world = normalize(mean(Δp_gravity_i)) * 9.81
     * 
     * Parameter Optimization:
     * - States: [poses, velocities, accel_bias, gyro_bias] 
     * - Factors: InertialGravityFactor(gravity, accel_measurements)
     * - Result: Consistent IMU biases and initial velocities
     * ===============================================================================
     * 
     * ===============================================================================
     * IMPLEMENTATION FLOW:
     * 1. Collect visual poses from keyframes (≥3 required)
     * 2. Extract corresponding IMU measurements between keyframes
     * 3. Estimate gravity direction from visual-IMU displacement differences
     * 4. Optimize IMU biases and velocities using InertialGravityFactor
     * 5. Initialize IMU handler with estimated parameters for future VIO
     * ===============================================================================
     */
    
    const auto& config = Config::getInstance();
    
    // Check if we have enough keyframes for gravity estimation
    if (m_keyframes.size() < 5) {
        spdlog::debug("[GRAVITY_EST] Not enough keyframes: {} < 5", m_keyframes.size());
        return false;
    }
    
    // Use all keyframes for gravity estimation
    std::vector<Frame*> keyframe_ptrs;
    for (const auto& kf : m_keyframes) {
        keyframe_ptrs.push_back(kf.get());
    }
    
    // Collect all IMU data for the estimation window from keyframes
    std::vector<IMUData> all_imu_data;
    
    // Extract IMU data from all keyframes
    for (const auto& keyframe : m_keyframes) {
        const auto& imu_data_since_last_kf = keyframe->get_imu_data_since_last_keyframe();
        all_imu_data.insert(all_imu_data.end(), 
                           imu_data_since_last_kf.begin(), 
                           imu_data_since_last_kf.end());
    }
    
    // Also add current IMU buffer
    all_imu_data.insert(all_imu_data.end(), 
                       m_imu_vec_from_last_keyframe.begin(), 
                       m_imu_vec_from_last_keyframe.end());
    
    if (all_imu_data.empty()) {
        spdlog::warn("[GRAVITY_EST] No IMU data available for gravity estimation");
        return false;
    }
    
    // Use IMUHandler to estimate gravity
    if (!m_imu_handler) {
        spdlog::error("[GRAVITY_EST] IMU handler not initialized");
        return false;
    }
    
    // First estimate gravity, then debug velocity comparison
    bool gravity_success = m_imu_handler->estimate_gravity_with_stereo_constraints(keyframe_ptrs, all_imu_data);

    m_Rgw_init = m_imu_handler->get_Rgw();

    std::vector<std::shared_ptr<MapPoint>> all_map_points;
    std::set<std::shared_ptr<MapPoint>> unique_map_points;

    Eigen::Matrix4f m_Tgw_init = Eigen::Matrix4f::Identity();
    m_Tgw_init.block<3,3>(0,0) = m_Rgw_init;

    // Get copies for transformation (since function modifies them)
    auto keyframes_copy = get_keyframes_safe();
    auto map_points_copy = get_map_points_safe();
    
    // m_imu_handler->transform_to_gravity_frame(keyframes_copy, map_points_copy, m_Tgw_init);

    // return true;


    // std::cout<<"Estimated initial gravity direction (Rgw):\n"<<m_Rgw_init<<"\n\n\n\n"<<std::endl;

    if (gravity_success) {
        m_imu_handler->debug_velocity_comparison(keyframe_ptrs, all_imu_data);
        
        // 🎯 After gravity estimation, perform IMU initialization optimization
        auto imu_init_result = m_inertial_optimizer->optimize_imu_initialization(
            keyframe_ptrs, 
            std::shared_ptr<IMUHandler>(m_imu_handler.get(), [](IMUHandler*){}) // Non-owning shared_ptr
        );
        
        if (imu_init_result.success) {
            
            // Store Tgw_init for viewer
            m_Tgw_init = imu_init_result.Tgw_init;
            // spdlog::info("  - Stored Tgw_init for viewer\n\n\n\n");
            // std::cout<<m_Tgw_init<<"\n\n\n\n"<<std::endl;

            debug_keyframe_to_keyframe_comparison();
            
            return true;
        } else {
            spdlog::warn("❌ [IMU_INIT] IMU initialization optimization failed");
            return false;
        }
    }

    

    return false;
}


void lightweight_vio::Estimator::debug_keyframe_to_keyframe_comparison()
{

    if (m_keyframes.size() < 2)
    {
        return;
    }

    // Compare all consecutive keyframe pairs
    for (size_t i = 1; i < m_keyframes.size(); ++i)
    {
        std::shared_ptr<Frame> prev_kf = m_keyframes[i - 1];
        std::shared_ptr<Frame> curr_kf = m_keyframes[i];

        if (!prev_kf || !curr_kf)
        {
            continue;
        }

        // Get VO pose change between keyframes
        Eigen::Matrix4f T_wb_prev = prev_kf->get_Twb();
        Eigen::Matrix4f T_wb_curr = curr_kf->get_Twb();

        // Calculate relative pose change from VO (previous to current)
        Eigen::Matrix4f T_rel_vo = T_wb_prev.inverse() * T_wb_curr;

        // Get rotation and translation from VO
        Eigen::Vector3f t_rel_vo = T_rel_vo.block<3, 1>(0, 3);
        Eigen::Matrix3f R_rel_vo = T_rel_vo.block<3, 3>(0, 0);

        // Get preintegrated IMU measurements between keyframes
        if (curr_kf->has_imu_preintegration_from_last_keyframe())
        {
            auto preint_imu = curr_kf->get_imu_preintegration_from_last_keyframe();

            // Get IMU bias and gravity estimates
            Eigen::Vector3f bg, ba;
            m_imu_handler->get_bias(bg, ba); // Get both biases at once
            Eigen::Vector3f gravity = m_imu_handler->get_gravity().cast<float>();

            // Get velocities
            Eigen::Vector3f v_prev = prev_kf->get_velocity();

            // Get time difference
            double dt = curr_kf->get_dt_from_last_keyframe();

            if (dt > 0.0)
            {
                // Compute bias-corrected IMU integration
                // Position: p_curr = p_prev + v_prev*dt + 0.5*gravity*dt^2 + corrected_delta_p
                // where corrected_delta_p = delta_p - J_p_ba * ba - J_p_bg * bg

                Eigen::Vector3f delta_p = preint_imu->delta_P; // Use correct member name
                Eigen::Vector3f delta_v = preint_imu->delta_V; // Use correct member name
                Eigen::Matrix3f delta_R = preint_imu->delta_R; // Already correct

                // Transform delta_p from body frame to world frame
                Eigen::Matrix3f R_wb_prev = prev_kf->get_Twb().block<3, 3>(0, 0); // Previous rotation
                Eigen::Vector3f delta_p_world = R_wb_prev * delta_p;              // Transform to world frame

                // IMU predicted relative position (all in world frame now)
                Eigen::Vector3f t_rel_imu = v_prev * dt + 0.5 * gravity * dt * dt + delta_p_world;

                // IMU predicted relative rotation (already bias-corrected)
                Eigen::Matrix3f R_rel_imu = delta_R;

                // Compare translation differences
                Eigen::Vector3f t_diff = t_rel_vo - t_rel_imu;

                // Compare rotation differences (angle-axis representation)
                Eigen::Matrix3f R_diff = R_rel_vo * R_rel_imu.transpose();
                Eigen::AngleAxisf angle_axis(R_diff);
                float angle_diff_deg = angle_axis.angle() * 180.0f / M_PI;

                if(Config::getInstance().m_enable_debug_output)
                {
                    // Print detailed comparison

                spdlog::info("🔍 [KF_COMPARE] Keyframes - Frame({}) to Frame({}) (dt={:.3f}s):",
                             prev_kf->get_frame_id(), curr_kf->get_frame_id(), dt);
                spdlog::info("  📐 Translation diff: ({:.4f}, {:.4f}, {:.4f}) m, norm: {:.4f} m",
                             t_diff.x(), t_diff.y(), t_diff.z(), t_diff.norm());
                spdlog::info("  🔄 Rotation diff: {:.2f} degrees", angle_diff_deg);
                spdlog::info("  📊 VO translation: ({:.4f}, {:.4f}, {:.4f}) m",
                             t_rel_vo.x(), t_rel_vo.y(), t_rel_vo.z());
                spdlog::info("  📊 IMU translation: ({:.4f}, {:.4f}, {:.4f}) m",
                             t_rel_imu.x(), t_rel_imu.y(), t_rel_imu.z());
                spdlog::info("  🔧 Bias applied: ba=({:.10f}, {:.10f}, {:.10f}), bg=({:.10f}, {:.10f}, {:.10f})",
                             ba.x(), ba.y(), ba.z(), bg.x(), bg.y(), bg.z());
                }
            }
        }
    }
}

} // namespace lightweight_vio
