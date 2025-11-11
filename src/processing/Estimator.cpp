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
#include "processing/MonocularInitializer.h"  // ⭐ Added for monocular initialization
#include "database/Frame.h"
#include "database/MapPoint.h"
#include "processing/Optimizer.h"
#include "camera/Camera.h"
#include "camera/Rectlinear.h"
#include "camera/Fisheye.h"
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
    
    // ⭐ Initialize monocular initializer (will only be used if camera is monocular)
    m_monocular_initializer = std::make_unique<MonocularInitializer>();
    
    // Create camera models based on config
    const Config& config = Config::getInstance();
    
    // Extract camera parameters from cv::Mat
    cv::Mat left_K = config.left_camera_matrix();
    cv::Mat left_dist = config.left_dist_coeffs();
    
    double fx0 = left_K.at<double>(0, 0);
    double fy0 = left_K.at<double>(1, 1);
    double cx0 = left_K.at<double>(0, 2);
    double cy0 = left_K.at<double>(1, 2);
    
    // Convert cv::Mat distortion coefficients to std::vector
    std::vector<double> left_dist_vec;
    
    // Handle both row vector (1xN) and column vector (Nx1)
    int left_dist_count = left_dist.rows * left_dist.cols;
    
    for (int i = 0; i < left_dist_count; ++i) {
        left_dist_vec.push_back(left_dist.at<double>(i));
    }
    
    // OpenCV expects 5 distortion coefficients for pinhole [k1, k2, p1, p2, k3]
    // If only 4 are provided [k1, k2, p1, p2], pad with k3=0
    if (config.get_camera_model() == CameraModel::PINHOLE) {
        if (left_dist_vec.size() == 4) {
            left_dist_vec.push_back(0.0);  // k3 = 0
            spdlog::info("[ESTIMATOR] Padded left distortion coefficients: 4 -> 5");
        }
    }
    
    spdlog::info("[ESTIMATOR] Left dist coeffs size: {}", left_dist_vec.size());
    
    // Create left camera
    if (config.get_camera_model() == CameraModel::FISHEYE) {
        m_left_camera = std::make_shared<Fisheye>(fx0, fy0, cx0, cy0, left_dist_vec);
    } else {
        m_left_camera = std::make_shared<Rectlinear>(fx0, fy0, cx0, cy0, left_dist_vec);
    }
    
    spdlog::info("[ESTIMATOR] Left camera created: fx={}, fy={}, cx={}, cy={}", fx0, fy0, cx0, cy0);
    
    // Create right camera only for stereo
    if (config.get_camera_type() == CameraType::STEREO) {
        cv::Mat right_K = config.right_camera_matrix();
        cv::Mat right_dist = config.right_dist_coeffs();
        
        double fx1 = right_K.at<double>(0, 0);
        double fy1 = right_K.at<double>(1, 1);
        double cx1 = right_K.at<double>(0, 2);
        double cy1 = right_K.at<double>(1, 2);
        
        std::vector<double> right_dist_vec;
        int right_dist_count = right_dist.rows * right_dist.cols;
        for (int i = 0; i < right_dist_count; ++i) {
            right_dist_vec.push_back(right_dist.at<double>(i));
        }
        
        if (config.get_camera_model() == CameraModel::PINHOLE) {
            if (right_dist_vec.size() == 4) {
                right_dist_vec.push_back(0.0);  // k3 = 0
                spdlog::info("[ESTIMATOR] Padded right distortion coefficients: 4 -> 5");
            }
        }
        
        spdlog::info("[ESTIMATOR] Right dist coeffs size: {}", right_dist_vec.size());
        
        if (config.get_camera_model() == CameraModel::FISHEYE) {
            m_right_camera = std::make_shared<Fisheye>(fx1, fy1, cx1, cy1, right_dist_vec);
        } else {
            m_right_camera = std::make_shared<Rectlinear>(fx1, fy1, cx1, cy1, right_dist_vec);
        }
    } else {
        // Monocular or RGBD - no right camera needed
        m_right_camera = nullptr;
    }
    
    // // // Start sliding window optimization thread
    m_sliding_window_thread_running = true;
    m_sliding_window_thread = std::make_unique<std::thread>(&Estimator::sliding_window_thread_function, this);
    
    if (Config::getInstance().m_enable_debug_output) {
        spdlog::info("[ESTIMATOR] Camera models created: {}", 
                     (Config::getInstance().get_camera_model() == CameraModel::PINHOLE) ? "PINHOLE" : "FISHEYE");
        spdlog::info("[ESTIMATOR] Sliding window optimization thread started");
    }
}

Estimator::EstimationResult Estimator::process_rgbd_frame(const cv::Mat& rgb_image, const cv::Mat& depth_map, double timestamp) {
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
                    spdlog::info("[POSE_OPT] Optimization successful: {} inliers, {} outliers", opt_result.num_inliers, opt_result.num_outliers);
                }
            }
            else
            {
                // Use just predicted pose if optimization failed
                m_current_pose = m_current_frame->get_Twb();
                // Update transform from last frame for velocity estimation (even if tracking failed)
                update_transform_from_last();
                if (Config::getInstance().m_enable_debug_output)
                {
                    spdlog::warn("[POSE_OPT] Optimization failed - keeping previous pose");
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
        // ✅ Use unified initialization function
        auto vo_result = initialize_rgbd(m_current_frame);
        
        result.success = vo_result.success;
        result.num_features = vo_result.num_features;
        result.num_new_map_points = vo_result.num_map_points;
        result.num_tracked_features = m_current_frame->get_feature_count();
        result.num_features_with_map_points = count_features_with_map_points(m_current_frame);
    }

    // Add processed frame to all frames vector for trajectory export
    m_all_frames.push_back(m_current_frame);

    // Set reference keyframe for non-keyframe frames (after pose optimization)
    if (!m_current_frame->is_keyframe() && m_last_keyframe)
    {
        m_current_frame->set_reference_keyframe(m_last_keyframe);
    }

    // Update state - only release images if previous frame is NOT a keyframe
    if (m_previous_frame && !m_previous_frame->is_keyframe())
    {
        m_previous_frame->release_images();
    }
    m_previous_frame = m_current_frame;

    return result;

}

Estimator::EstimationResult Estimator::process_monocular_frame(const cv::Mat& image, double timestamp,
                                                              const std::vector<IMUData>& imu_data_from_last_frame) {
    EstimationResult result;
    auto total_start_time = std::chrono::high_resolution_clock::now();

    for (const auto& imu_data : imu_data_from_last_frame) {
        m_imu_vec_from_last_keyframe.push_back(imu_data);
    }
    

    // Increment frame counter since last keyframe for every new frame
    m_frames_since_last_keyframe++;

    // Create new monocular frame
    m_current_frame = create_monocular_frame(image, timestamp);

    if(m_last_keyframe){
        m_current_frame->set_accel_bias(m_last_keyframe->get_accel_bias());
        m_current_frame->set_gyro_bias(m_last_keyframe->get_gyro_bias());
    }


    // Set IMU data to the frame (frame-to-frame data)
    m_current_frame->set_imu_data_from_last_frame(imu_data_from_last_frame);
    
    // Compute frame-to-frame preintegration if IMU data is available
    if (!imu_data_from_last_frame.empty() && m_imu_handler) {
        // Always compute frame-to-frame preintegration, regardless of IMU initialization status
        // This is useful for state prediction and velocity estimation
        
        // 🎯 Use FRAME timestamps for dt calculation (not IMU timestamp range)
        // This ensures dt matches the actual frame interval (0.05s)
        double current_frame_time = timestamp;  // Already in seconds
        double previous_frame_time = m_previous_frame ? 
            m_previous_frame->get_timestamp() : current_frame_time;  // Already in seconds
        
        auto frame_to_frame_preint = m_imu_handler->preintegrate(imu_data_from_last_frame, previous_frame_time, current_frame_time);
        if (frame_to_frame_preint && frame_to_frame_preint->is_valid()) {
            m_current_frame->set_imu_preintegration_from_last_frame(frame_to_frame_preint);
        } else {
            spdlog::warn("[IMU] Failed to create frame-to-frame preintegration for frame {}", m_current_frame->get_frame_id());
        }

        spdlog::debug("[IMU] Frame-to-frame preintegration set for frame {} with {} imu data", m_current_frame->get_frame_id(), imu_data_from_last_frame.size());
    }

    // Compute from-last-keyframe preintegration for more stable state prediction
    if (!m_imu_vec_from_last_keyframe.empty() && m_imu_handler && m_last_keyframe) {
        double current_frame_time = timestamp;  // Already in seconds
        double last_keyframe_time = m_last_keyframe->get_timestamp();  // Already in seconds
        
        // Create preintegration from last keyframe to current frame using accumulated IMU data
        auto keyframe_to_frame_preint = m_imu_handler->preintegrate(m_imu_vec_from_last_keyframe, last_keyframe_time, current_frame_time);
        if (keyframe_to_frame_preint && keyframe_to_frame_preint->is_valid()) {
            m_current_frame->set_imu_preintegration_from_last_keyframe(keyframe_to_frame_preint);
            // spdlog::debug("[IMU] Created keyframe-to-frame preintegration: dt={:.4f}s", keyframe_to_frame_preint->dt_total);
        } else {
            spdlog::warn("[IMU] Failed to create keyframe-to-frame preintegration for frame {}", m_current_frame->get_frame_id());
        }
    }



    // Monocular initialization: If not yet initialized
    if (!m_monocular_initialized) {

        // Set initial pose
        m_current_frame->set_Twb(m_initial_gt_pose);

        // Extract or track features
        if (m_previous_frame) {
            // Track features from previous frame (which is the reference frame)
            m_feature_tracker->track_features(m_current_frame, m_previous_frame);
        } else {
            // Extract features for first frame
            m_feature_tracker->track_features(m_current_frame, nullptr);
            // Set first frame as reference for tracking in subsequent frames
        }
        
        // Undistort features for parallax computation
        m_current_frame->undistort_features();
        
        
        // Add current frame to initializer (may trigger initialization)
        bool init_attempted = false;
        m_monocular_initializer->try_initialize_visual_sfm(m_current_frame, &init_attempted);

        m_previous_frame = m_current_frame;

        // Check if initialization succeeded
        if (m_monocular_initializer->is_initialized()) {
            auto init_result = m_monocular_initializer->get_result();
            
            // Store initialized frames and map points
            for (const auto& frame : init_result.initialized_keyframes) {
                m_keyframes.push_back(frame);
                m_all_frames.push_back(frame);
            }
            
            for (const auto& mp : init_result.initialized_mappoints) {
                m_map_points.push_back(mp);
            }
            
            // 🎯 Compute preintegration for initialized keyframes using accumulated IMU data
            if (m_imu_handler && !m_imu_vec_from_last_keyframe.empty()) {
                spdlog::info("[MONO_INIT] Computing preintegration for {} initialized keyframes", 
                            init_result.initialized_keyframes.size());
                
                // For each keyframe (except first), compute preintegration from previous keyframe
                for (size_t i = 1; i < init_result.initialized_keyframes.size(); ++i) {
                    auto prev_kf = init_result.initialized_keyframes[i-1];
                    auto curr_kf = init_result.initialized_keyframes[i];
                    
                    double t_prev = prev_kf->get_timestamp();
                    double t_curr = curr_kf->get_timestamp();
                    
                    // Extract IMU data in this time interval
                    std::vector<IMUData> imu_interval;
                    for (const auto& imu : m_imu_vec_from_last_keyframe) {
                        if (imu.timestamp >= t_prev && imu.timestamp <= t_curr) {
                            imu_interval.push_back(imu);
                        }
                    }
                    
                    if (!imu_interval.empty()) {
                        // Compute preintegration for this interval
                        auto preint = m_imu_handler->preintegrate(imu_interval, t_prev, t_curr);
                        if (preint && preint->is_valid()) {
                            curr_kf->set_imu_preintegration_from_last_keyframe(preint);

                            // 🎯 Log timestamp difference and preintegration dt
                            double timestamp_diff = t_curr - t_prev;

                            spdlog::info("[MONO_INIT] KF{}: t_prev={:.6f}s, t_curr={:.6f}s, Δt={:.4f}s, preint_dt={:.4f}s, Δp=[{:.4f}, {:.4f}, {:.4f}], Δv=[{:.4f}, {:.4f}, {:.4f}]",
                                        curr_kf->get_frame_id(),
                                        t_prev, t_curr, timestamp_diff,
                                        preint->dt_total,
                                        preint->delta_P.x(), preint->delta_P.y(), preint->delta_P.z(),
                                        preint->delta_V.x(), preint->delta_V.y(), preint->delta_V.z());
                        }
                    }
                }
            }
            
      
            // set both frames as keyframes
            m_current_frame->set_keyframe(true);
            if (m_previous_frame)
                m_previous_frame->set_keyframe(true);

            // set m_last_keyframe to current frame
            m_last_keyframe = m_current_frame;
            m_previous_frame = m_current_frame;
            
            // Mark as initialized
            m_monocular_initialized = true;

            result.success = true;
            result.num_features = m_current_frame->get_feature_count();
            result.num_inliers = init_result.initialized_mappoints.size();

            // Try initialize imu here

            bool imu_init_success = initialize_imu_monocular();

            if (imu_init_success) {
                spdlog::info("[MONO_INIT] IMU initialized successfully");
            } else {
                spdlog::warn("[MONO_INIT] IMU initialization failed");
            }

            return result;
        }
        else
        {
            spdlog::info("[MONO_INIT] Initialization not yet successful - waiting for more frames to accumulate parallax");
        }
        
        // If initialization was attempted but failed (parallax sufficient but too few inliers),
        // reset reference frame to current frame and try again from here
        if (init_attempted) {
            spdlog::warn("[MONO_INIT] Initialization attempted but failed - resetting reference frame");
            m_previous_frame = m_current_frame;
        }
        // Otherwise, keep m_previous_frame as the reference frame for tracking
        
        result.success = false;
        return result;
    }

    if (m_previous_frame) {
        // Predict state using motion model
        auto prediction_start = std::chrono::high_resolution_clock::now();
        predict_state();
        auto prediction_end = std::chrono::high_resolution_clock::now();
        auto prediction_time = std::chrono::duration_cast<std::chrono::microseconds>(prediction_end - prediction_start).count() / 1000.0;
        
        // Track features from previous frame
        auto tracking_start = std::chrono::high_resolution_clock::now();
        m_feature_tracker->track_features(m_current_frame, m_previous_frame);
        auto tracking_end = std::chrono::high_resolution_clock::now();
        auto tracking_time = std::chrono::duration_cast<std::chrono::microseconds>(tracking_end - tracking_start).count() / 1000.0;
        
        result.num_features = m_current_frame->get_feature_count();

        m_current_frame->undistort_features();
        
        // Count how many features have associated map points
        int num_tracked_with_map_points = count_features_with_map_points(m_current_frame);

        if(num_tracked_with_map_points < 100)
        {
            num_tracked_with_map_points += create_temporary_map_points(m_current_frame);
        }

        spdlog::info("Num tracked mp: {}", num_tracked_with_map_points);

        
        // Log tracking information
        if (Config::getInstance().m_enable_debug_output) {
            spdlog::info("[TRACKING] {} features tracked, {} with map points", 
                        result.num_features, num_tracked_with_map_points);
        }
        
        if (num_tracked_with_map_points >= 5) {
            // Pose optimization
            auto opt_result = optimize_pose(m_current_frame);
            

            // // Calculate T_rel
            // Eigen::Matrix4f T_prev = m_previous_frame->get_Twb();
            // Eigen::Matrix4f T_curr = opt_result.optimized_pose;
            // Eigen::Matrix4f T_rel = T_prev.inverse() * T_curr;

            // // Set velocity based on T_rel and frame time difference
            // double dt = m_current_frame->get_timestamp() - m_previous_frame->get_timestamp();
            // if (dt > 0) {
            //     Eigen::Vector3f translation = T_rel.block<3,1>(0,3);
            //     Eigen::Vector3f velocity = translation / dt;
            //     m_current_frame->set_velocity(velocity);
            // }

            result.success = opt_result.success;
            result.num_inliers = opt_result.num_inliers;
            result.num_outliers = opt_result.num_outliers;
        } else 
        {
            spdlog::warn("[POSE_OPT] ⚠️ Not enough map point associations for optimization: {} (need ≥20)", num_tracked_with_map_points);
            // Fallback: use current pose as-is
            m_current_pose = m_current_frame->get_Twb();
            
            result.success = true;
            result.num_inliers = num_tracked_with_map_points;
            result.num_outliers = 0;
        }
        
        // Decide whether to create keyframe
        bool is_keyframe = should_create_keyframe_monocular(m_current_frame);
        
        if (is_keyframe) {
            // ⭐ Use monocular-specific keyframe creation with triangulation
            int new_map_points = create_keyframe_monocular(m_current_frame);
            result.num_new_map_points = new_map_points;
            
            m_frames_since_last_keyframe = 0;
        } 
        
        
        // Count tracked features and features with map points
        result.num_tracked_features = m_current_frame->get_feature_count();
        result.num_features_with_map_points = count_features_with_map_points(m_current_frame);
        
    } else {
        // Should not reach here - initialization should have set m_previous_frame
        spdlog::error("[MONO] No previous frame after initialization!");
        result.success = false;
        return result;
    }

    // Add processed frame to all frames vector for trajectory export
    m_all_frames.push_back(m_current_frame);

    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - total_start_time);
    result.optimization_time_ms = duration.count() / 1000.0;

    // Set reference keyframe for non-keyframe frames
    if (!m_current_frame->is_keyframe() && m_last_keyframe) {
        m_current_frame->set_reference_keyframe(m_last_keyframe);
    }

    // Update state - only release images if previous frame is NOT a keyframe
    if (m_previous_frame && !m_previous_frame->is_keyframe()) {
        m_previous_frame->release_images();
    }


    update_transform_from_last();

    m_previous_frame = m_current_frame;

    return result;
}

Estimator::EstimationResult Estimator::process_frame(const cv::Mat& left_image, const cv::Mat& right_image, double timestamp) {
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
        // ✅ Use unified initialization function
        auto vo_result = initialize_stereo(m_current_frame);
        
        result.success = vo_result.success;
        result.num_features = vo_result.num_features;
        result.num_new_map_points = vo_result.num_map_points;
        result.num_tracked_features = m_current_frame->get_feature_count();
        result.num_features_with_map_points = count_features_with_map_points(m_current_frame);
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
    
    // Update state - only release images if previous frame is NOT a keyframe
    if (m_previous_frame && !m_previous_frame->is_keyframe()) {
        m_previous_frame->release_images();
    }
    m_previous_frame = m_current_frame;
    
    return result;
}



// IMU process_frame overload
Estimator::EstimationResult Estimator::process_frame(const cv::Mat& left_image, const cv::Mat& right_image, 
                                                    double timestamp, const std::vector<IMUData>& imu_data_from_last_frame) {
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
        double current_frame_time = timestamp;  // Already in seconds
        double previous_frame_time = m_previous_frame ? 
            m_previous_frame->get_timestamp() : current_frame_time;  // Already in seconds
        
        auto frame_to_frame_preint = m_imu_handler->preintegrate(imu_data_from_last_frame, previous_frame_time, current_frame_time);
        if (frame_to_frame_preint && frame_to_frame_preint->is_valid()) {
            frame->set_imu_preintegration_from_last_frame(frame_to_frame_preint);
        } else {
            spdlog::warn("[IMU] Failed to create frame-to-frame preintegration for frame {}", frame->get_frame_id());
        }
    }
    
    // Compute from-last-keyframe preintegration for more stable state prediction
    if (!m_imu_vec_from_last_keyframe.empty() && m_imu_handler && m_last_keyframe) {
        double current_frame_time = timestamp;  // Already in seconds
        double last_keyframe_time = m_last_keyframe->get_timestamp();  // Already in seconds
        
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
        // ✅ Use unified initialization function
        auto vo_result = initialize_stereo(m_current_frame);
        
        result.success = vo_result.success;
        result.num_features = vo_result.num_features;
        result.num_new_map_points = vo_result.num_map_points;
        result.num_tracked_features = m_current_frame->get_feature_count();
        result.num_features_with_map_points = count_features_with_map_points(m_current_frame);
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
    
    // Update state - only release images if previous frame is NOT a keyframe
    if (m_previous_frame && !m_previous_frame->is_keyframe()) {
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

    // 🎯 Attempt IMU initialization if conditions are met
    const auto& config = Config::getInstance();
    if (should_initialize_imu()) {
        initialize_imu();
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

std::shared_ptr<Frame> Estimator::create_frame(const cv::Mat& left_image, const cv::Mat& right_image, double timestamp) {
    if (left_image.empty()) {
        return nullptr;
    }
    
    const Config& config = Config::getInstance();
    CameraType camera_type = config.get_camera_type();
    
    // Monocular mode - only use left image
    if (camera_type == CameraType::MONOCULAR) {
        return create_monocular_frame(left_image, timestamp);
    }
    
    // Stereo/RGBD mode - need both images
    if (right_image.empty()) {
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
    auto frame = std::make_shared<Frame>(
        timestamp, 
        m_frame_id_counter++,
        gray_left, gray_right,
        m_left_camera, m_right_camera
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

std::shared_ptr<Frame> Estimator::create_monocular_frame(const cv::Mat& image, double timestamp) {
    if (image.empty()) {
        return nullptr;
    }
    
    // Check if left camera is initialized
    if (!m_left_camera) {
        spdlog::error("[ESTIMATOR] Left camera is nullptr! Cannot create monocular frame.");
        return nullptr;
    }
    

    const cv::Mat& gray_image = image;  // Direct reference (no copy)
    
    auto frame = std::make_shared<Frame>(
        timestamp,
        m_frame_id_counter++,
        gray_image,
        m_left_camera  // Monocular uses left camera parameters
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

std::shared_ptr<Frame> Estimator::create_rgbd_frame(const cv::Mat& rgb_image, const cv::Mat& depth_map, double timestamp) {
    if (rgb_image.empty() || depth_map.empty()) {
        return nullptr;
    }
    
    // Player already passes preprocessed grayscale image, so just use it directly
    // No need to convert or clone
    const cv::Mat& gray_image = rgb_image;  // Direct reference (no copy)
    
    auto frame = std::make_shared<Frame>(
        timestamp,
        m_frame_id_counter++,
        gray_image,
        depth_map,
        m_left_camera
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

// ========================================================================
// VO Initialization Functions
// ========================================================================

Estimator::VOInitializationResult lightweight_vio::Estimator::initialize_rgbd(std::shared_ptr<Frame> frame) {
    VOInitializationResult result;
    
    if (!frame) {
        spdlog::error("[INIT_RGBD] Invalid frame provided");
        result.success = false;
        return result;
    }
    
    // 1. Feature extraction
    m_feature_tracker->track_features(frame, nullptr);
    result.num_features = frame->get_feature_count();
    
    // 2. Depth computation (direct reading from depth map)
    frame->compute_depth();
    
    // 3. Identity pose (first frame starts at origin)
    m_current_pose = frame->get_Twb();
    
    // 4. Create initial map points
    int num_map_points = create_initial_map_points(frame);
    result.num_map_points = num_map_points;
    
    // 5. Create keyframe
    create_keyframe(frame);
    m_frames_since_last_keyframe = 0;
    
    result.success = true;
    
    if (Config::getInstance().m_enable_debug_output) {
        spdlog::info("[INIT_RGBD] ✅ Initialized with {} features, {} map points", 
                    result.num_features, result.num_map_points);
    }
    
    return result;
}

Estimator::VOInitializationResult lightweight_vio::Estimator::initialize_stereo(std::shared_ptr<Frame> frame) {
    VOInitializationResult result;
    
    if (!frame) {
        spdlog::error("[INIT_STEREO] Invalid frame provided");
        result.success = false;
        return result;
    }
    
    // 1. Feature extraction
    m_feature_tracker->track_features(frame, nullptr);
    result.num_features = frame->get_feature_count();
    
    // 2. Stereo matching + triangulation
    frame->compute_stereo_depth();
    
    // 3. Identity pose (first frame starts at origin)
    m_current_pose = frame->get_Twb();
    
    // 4. Create initial map points
    int num_map_points = create_initial_map_points(frame);
    result.num_map_points = num_map_points;
    
    // 5. Create keyframe
    create_keyframe(frame);
    m_frames_since_last_keyframe = 0;
    
    result.success = true;
    
    if (Config::getInstance().m_enable_debug_output) {
        spdlog::info("[INIT_STEREO] ✅ Initialized with {} features, {} map points", 
                    result.num_features, result.num_map_points);
    }
    
    return result;
}

// ========================================================================
// Map Point Creation Functions
// ========================================================================


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
            fx = frame->get_fx(); fy = frame->get_fy(); cx = frame->get_cx(); cy = frame->get_cy();
            
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

// ========================================================================
// IMU Initialization Functions
// ========================================================================

bool lightweight_vio::Estimator::should_initialize_imu() const {
    // IMU initialization conditions:
    // 1. Not already initialized
    // 2. Have at least 5 keyframes for reliable gravity estimation
    return !m_success_imu_init && m_keyframes.size() >= 5;
}

bool lightweight_vio::Estimator::initialize_imu() {
    if (!should_initialize_imu()) {
        return false;
    }
    
    const auto& config = Config::getInstance();
    
    spdlog::info("================================================================================");
    spdlog::info("[INIT_IMU] Starting IMU Initialization");
    spdlog::info("[INIT_IMU] Keyframes available: {}", m_keyframes.size());
    spdlog::info("================================================================================");
    
    // Run optimization (get results only, no frame modification)
    auto imu_init_result = try_initialize_imu();
    
    if (!imu_init_result.success) {
        spdlog::warn("================================================================================");
        spdlog::warn("[INIT_IMU] IMU Initialization FAILED");
        spdlog::warn("================================================================================\n");
        return false;
    }
    
    // ⭐ Apply results in Estimator
    
    // 1. Set gravity in IMU handler (BEFORE transformation!)
    m_imu_handler->set_gravity(imu_init_result.g_world_before_transform);
    
    // 2. Apply velocities and biases to frames
    apply_imu_optimization_results(imu_init_result);
    
    // 3. Set bias in IMU handler
    m_imu_handler->set_bias(
        imu_init_result.optimized_gyro_bias,
        imu_init_result.optimized_accel_bias
    );
    
    // 4. Update preintegrations with new bias
    update_preintegrations_with_new_bias(imu_init_result);
    
    // 6. Visualize gravity direction (BEFORE transformation)
    visualize_gravity_direction(imu_init_result);
    
    // 7. Apply Tgw transformation to all frames and map points
    apply_gravity_alignment_transform(imu_init_result.Tgw_init);

    // imu_init_result.Tgw_init = Eigen::Matrix4f::Identity();  // Reset to identity after application
    
    // 8. Update gravity visualization to use gravity-aligned frame coordinates
    update_gravity_visualization_after_transform();
    
    // 9. Enable IMU optimization in sliding window
    std::shared_ptr<IMUHandler> shared_imu_handler = std::shared_ptr<IMUHandler>(
        m_imu_handler.get(), 
        [](IMUHandler*){}  // Non-owning shared_ptr
    );
    m_sliding_window_optimizer->enable_imu_optimization(
        shared_imu_handler, 
        imu_init_result.g_world_before_transform.cast<double>()
    );
    
    // Update initialization flags
    m_gravity_initialized = true;
    m_enable_imu_optimization = true;
    m_success_imu_init = true;
    
    // Log success with detailed information
    spdlog::info("================================================================================");
    spdlog::info("[INIT_IMU] IMU Initialization SUCCESSFUL!");
    spdlog::info("================================================================================\n");
    
    return true;
}

bool lightweight_vio::Estimator::initialize_imu_monocular() {
    if (should_initialize_imu())
    {
        spdlog::info("[MONO_INIT] IMU initialization started");

        // Gravity direction initialization

        // Use all keyframe for gravity estimation
        std::vector<Frame *> keyframe_ptrs;

        for (const auto &kf : m_keyframes)
            keyframe_ptrs.push_back(kf.get());

        // Collect all IMU data
        std::vector<IMUData> all_imu_data;
        for (const auto &keyframe : m_keyframes)
        {
            const auto &imu_data_since_last_kf = keyframe->get_imu_data_since_last_keyframe();
            all_imu_data.insert(all_imu_data.end(), imu_data_since_last_kf.begin(), imu_data_since_last_kf.end());
        }

        // Also add current IMU buffer
        all_imu_data.insert(all_imu_data.end(),
                            m_imu_vec_from_last_keyframe.begin(),
                            m_imu_vec_from_last_keyframe.end());

        if (all_imu_data.empty() || !m_imu_handler)
        {
            spdlog::error("[MONO_INIT] IMU initialization failed - no IMU data available");
        }

        m_imu_handler->set_initialized(true);

        auto imu_init_result = m_inertial_optimizer->optimize_imu_initialization(
            keyframe_ptrs,                                                        // Frames for optimization
            keyframe_ptrs,                                                        // ALL keyframes for transformation
            std::shared_ptr<IMUHandler>(m_imu_handler.get(), [](IMUHandler *) {}) // Non-owning shared_ptr
        );

        if(!imu_init_result.success)
        {
            spdlog::error("[MONO_INIT] IMU initialization optimization failed");
            return false;
        }

        m_Tgw_init = imu_init_result.Tgw_init;
        double optimized_scale = imu_init_result.optimized_scale;

        spdlog::info("[MONO_INIT] IMU initialization completed - optimized scale: {:.6f}", optimized_scale);

        // Update values after imu initialization

        // Set gravity in IMU Handler

        m_imu_handler->set_gravity(imu_init_result.g_world_before_transform);

        // Apply velocity and bias to keyframes
        apply_imu_optimization_results(imu_init_result);

        // Set bias in IMU Handler
        m_imu_handler->set_bias(imu_init_result.optimized_gyro_bias,
                                imu_init_result.optimized_accel_bias);

        // Update preintegrations with new bias
        update_preintegrations_with_new_bias(imu_init_result);

        // Apply Tgw transformation to all frams and map points
        apply_gravity_alignment_transform(imu_init_result.Tgw_init);

        Eigen::Matrix4f Twc_init = m_current_frame->get_Twc();

        // step 0 - scale factor
        float scale_factor = 1.0f / (static_cast<float>(optimized_scale));

        // step 1 - apply to all keyframes

        Eigen::Matrix4f Twb_0 = m_keyframes[0]->get_Twb();

        for (unsigned int i = 1; i < m_keyframes.size(); i++)
        {
            Eigen::Matrix4f Twb_i = m_keyframes[i]->get_Twb();
            Eigen::Matrix4f T_0_i = Twb_0.inverse() * Twb_i;
            T_0_i.block<3, 1>(0, 3) *= scale_factor; // scale translation
            m_keyframes[i]->set_Twb(Twb_0 * T_0_i);  // update pose
        }

        // velocity scaling
        for (unsigned int i = 0; i < m_keyframes.size(); i++)
        {
            Eigen::Vector3f vel = m_keyframes[i]->get_velocity();
            vel *= scale_factor;
            m_keyframes[i]->set_velocity(vel);
        }

        Eigen::Matrix4f Twc_after_scale = m_current_frame->get_Twc();

        // step 2 - apply to all map points

        // collect unique map points in window
        std::set<std::shared_ptr<MapPoint>> unique_map_points;
        std::vector<std::shared_ptr<MapPoint>> all_map_points;

        for (const auto &kf : m_keyframes)
        {
            for (const auto &mp : kf->get_map_points())
            {
                if (mp && !mp->is_bad())
                {
                    unique_map_points.insert(mp);
                }
            }
        }

        all_map_points.assign(unique_map_points.begin(), unique_map_points.end());

        for (const auto &mp : all_map_points)
        {
            Eigen::Vector3f pos_w = mp->get_position();
            Eigen::Vector3f pos_c = Twc_init.inverse().block<3, 3>(0, 0) * (pos_w - Twc_init.block<3, 1>(0, 3));
            pos_c *= scale_factor; // scale translation
            Eigen::Vector3f pos_w_after = Twc_after_scale.block<3, 3>(0, 0) * pos_c + Twc_after_scale.block<3, 1>(0, 3);
            mp->set_position(pos_w_after);
        }

        // Update initialization flags
        m_gravity_initialized = true;
        m_enable_imu_optimization = true;
        m_success_imu_init = true;

         std::shared_ptr<IMUHandler> shared_imu_handler = std::shared_ptr<IMUHandler>(
        m_imu_handler.get(), 
        [](IMUHandler*){}  // Non-owning shared_ptr
    );

        m_sliding_window_optimizer->enable_imu_optimization(
            shared_imu_handler,
            imu_init_result.g_world_before_transform.cast<double>());

        return true;
    }

    return false;
}

// ========================================================================
// Map Point Creation Functions (continued)
// ========================================================================


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
                fx = frame->get_fx(); fy = frame->get_fy(); cx = frame->get_cx(); cy = frame->get_cy();
                
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


bool lightweight_vio::Estimator::should_create_keyframe_monocular(std::shared_ptr<Frame> frame) {

  if (!frame) {
        return false;
    }
    
    // First frame is always a keyframe
    if (m_keyframes.empty()) {
        return true;
    }


        // I want to skeep stationary or if rotation is dominant compared to translation
    if(m_last_keyframe)
    {
        Eigen::Matrix4f T_last = m_last_keyframe->get_Twb();
        Eigen::Matrix4f T_curr = frame->get_Twb();
        Eigen::Matrix4f T_last_curr = T_last.inverse() * T_curr;
        Eigen::Vector3f translation = T_last_curr.block<3,1>(0,3);

        if(translation.norm() < 0.05) // 10 cm
        {
            spdlog::info("[KEYFRAME_DECISION] Skipping keyframe due to low translation ({:.2f} m)", translation.norm());
            return false;
        }
    }

    // Force keyframe creation if time since last keyframe exceeds threshold
    if (m_last_keyframe) {
        double current_time = frame->get_timestamp();  // Already in seconds
        double last_keyframe_time = m_last_keyframe->get_timestamp();  // Already in seconds
        double time_diff = current_time - last_keyframe_time;
        
        if (time_diff >= Config::getInstance().m_keyframe_time_threshold) {
            return true;
        }
    }




    // Check average parallax since last keyframe

    std::vector<double> parallaxes;

    if(m_last_keyframe)
    {
        auto features = frame->get_features();

        for(auto& feature : features)
        {
            auto observations = feature->get_observations();

            for(auto& obs : observations)
            {
                if(obs.frame == m_last_keyframe)
                {
                    auto feature_last_kf = obs.frame->get_feature(obs.feature_index);

                    auto pt_curr = feature->get_undistorted_coord();
                    auto pt_last = feature_last_kf->get_undistorted_coord();

                    double parallax = cv::norm(pt_curr - pt_last);

                    parallaxes.push_back(parallax);



                }
            }
        }
    }

    double median_parallax = 0.0;
    if(!parallaxes.empty())
    {
        // Sort parallaxes to compute median
        std::sort(parallaxes.begin(), parallaxes.end());
        
        // Compute median
        size_t n = parallaxes.size();
        if (n % 2 == 0) {
            median_parallax = (parallaxes[n/2 - 1] + parallaxes[n/2]) / 2.0;
        } else {
            median_parallax = parallaxes[n/2];
        }
    }


    spdlog::info("[KEYFRAME_DECISION] Median parallax since last keyframe: {:.2f} pixels (from {} matches)", 
                 median_parallax, parallaxes.size());

    if(median_parallax > 20.0) // Threshold in pixels
    {
        return true;
    }

    // // Grid-based keyframe creation policy
    // // Create keyframe when grid coverage drops to configured ratio of last keyframe's coverage
    // double current_grid_coverage = calculate_grid_coverage_with_map_points(frame);

    // spdlog::error("Current grid coverage: {:.2f}", current_grid_coverage);

    // if(current_grid_coverage < 0.2)
    //     return true;


    return false;



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
        double current_time = frame->get_timestamp();  // Already in seconds
        double last_keyframe_time = m_last_keyframe->get_timestamp();  // Already in seconds
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
        double current_time = frame->get_timestamp();  // Already in seconds
        double last_kf_time = m_last_keyframe->get_timestamp();  // Already in seconds
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
            
            // Mark oldest keyframe as inactive (removed from sliding window)
            oldest_keyframe->set_active(false);
            
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
            
            // Release images from oldest keyframe before removing it from window
            oldest_keyframe->release_images();
            
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

bool lightweight_vio::Estimator::multi_view_triangulation(
    const std::vector<std::pair<std::shared_ptr<Frame>, int>>& observations,
    Eigen::Vector3f& P_world) {
    
    if (observations.size() < 2) {
        return false;  // Need at least 2 views for triangulation
    }
    
    // Build SVD matrix A for multi-view triangulation
    int num_observations = observations.size();
    Eigen::MatrixXf svd_A(2 * num_observations, 4);
    
    int row_idx = 0;
    
    for (const auto& obs : observations) {
        auto obs_frame = obs.first;
        int feature_idx = obs.second;
        
        if (!obs_frame || feature_idx < 0) {
            return false;
        }
        
        auto obs_feature = obs_frame->get_feature(feature_idx);
        if (!obs_feature || !obs_feature->is_valid()) {
            return false;
        }
        
        // Get pixel coordinates (undistorted)
        cv::Point2f pixel_coord = obs_feature->get_undistorted_coord();
        float u = pixel_coord.x;
        float v = pixel_coord.y;
        
        // Homogeneous pixel coordinates
        Eigen::Vector3f x_hom(u, v, 1.0f);
        
        // Get camera-to-world transform for this observation
        Eigen::Matrix4f T_wc_obs = obs_frame->get_Twc();
        
        // Build projection matrix P = K * [R | t]
        // Get camera intrinsics
        double fx = obs_frame->get_fx();
        double fy = obs_frame->get_fy();
        double cx = obs_frame->get_cx();
        double cy = obs_frame->get_cy();
        
        Eigen::Matrix3f K;
        K << fx, 0, cx,
             0, fy, cy,
             0, 0, 1;
        
        // World to camera transformation
        Eigen::Matrix4f T_cw = T_wc_obs.inverse();
        
        // Build P = K * [R | t] (world to camera, then project)
        Eigen::Matrix<float, 3, 4> Rt;
        Rt.block<3, 3>(0, 0) = T_cw.block<3, 3>(0, 0);
        Rt.block<3, 1>(0, 3) = T_cw.block<3, 1>(0, 3);
        
        Eigen::Matrix<float, 3, 4> P = K * Rt;
        
        // Add two rows per observation: DLT equations
        // x_hom(0) * P.row(2) - x_hom(2) * P.row(0) = 0
        // x_hom(1) * P.row(2) - x_hom(2) * P.row(1) = 0
        svd_A.row(row_idx++) = x_hom(0) * P.row(2) - x_hom(2) * P.row(0);
        svd_A.row(row_idx++) = x_hom(1) * P.row(2) - x_hom(2) * P.row(1);
    }
    
    // Solve using SVD
    Eigen::JacobiSVD<Eigen::MatrixXf> svd(svd_A, Eigen::ComputeThinV);
    Eigen::Vector4f svd_V = svd.matrixV().rightCols<1>();
    
    // Homogeneous to 3D
    if (std::abs(svd_V(3)) < 1e-6) {
        return false;  // Point at infinity
    }
    
    P_world = svd_V.head<3>() / svd_V(3);
    
    // Validate: check if point is in front of all cameras
    for (const auto& obs : observations) {
        auto obs_frame = obs.first;
        Eigen::Matrix4f T_cw = obs_frame->get_Twc().inverse();
        Eigen::Vector4f P_camera_h = T_cw * Eigen::Vector4f(P_world.x(), P_world.y(), P_world.z(), 1.0f);

        
        if (P_camera_h.z() < 0.0f) {  // Behind camera
            return false;
        }

        // Check reprojection error
        Eigen::Vector3f P_camera = P_camera_h.head<3>();
        double fx = obs_frame->get_fx();
        double fy = obs_frame->get_fy();
        double cx = obs_frame->get_cx();
        double cy = obs_frame->get_cy();    
        float u_proj = fx * P_camera.x() / P_camera.z() + cx;
        float v_proj = fy * P_camera.y() / P_camera.z() + cy;
        auto obs_feature = obs_frame->get_feature(obs.second);
        cv::Point2f observed_pt = obs_feature->get_undistorted_coord();
        double error_x = observed_pt.x - u_proj;
        double error_y = observed_pt.y - v_proj;
        double reproj_error_square = (error_x * error_x + error_y * error_y);

        if(reproj_error_square > 5.991) // 95%
        {   spdlog::warn("Triangulation reprojection error too high: {:.2f}", reproj_error_square);
            return false;
        }
    }

    //
    
    return true;
}



int lightweight_vio::Estimator::create_temporary_map_points(std::shared_ptr<Frame> frame) {

    int num_new_map_points = 0;
    // Let's check feature observations before creating the keyframe
    const auto& features = frame->get_features();
    
    for (auto& feature : features) {
        if (feature && feature->is_valid()) {

            // If already map point associated, skip
            auto curr_idx = feature->get_feature_id();
            auto existing_mp = frame->get_map_point(curr_idx);
            
            if (existing_mp && !existing_mp->is_bad())
                continue;
            
            const auto& observation = feature->get_observations();
            
            // Collect all active keyframe observations
            std::vector<std::pair<std::shared_ptr<Frame>, int>> all_observations;
            all_observations.push_back({frame, curr_idx});  // Add current frame first

            bool is_there_valid_mp = false;
            
            for (const auto &obs : observation)
            {
                if (obs.frame && obs.frame->is_keyframe() && obs.frame->is_active())
                {
                    all_observations.push_back({obs.frame, obs.feature_index});
                }
            }

            // Need at least 5 observations (current + 4 keyframes)
            if (all_observations.size() < 2 || is_there_valid_mp) {
                continue;
            }
            
            // Sort keyframe observations by timestamp (oldest first)
            std::vector<std::pair<double, int>> sorted_kf_indices;
            for (size_t i = 1; i < all_observations.size(); ++i) {
                double obs_timestamp = all_observations[i].first->get_timestamp();
                sorted_kf_indices.push_back({obs_timestamp, i});
            }
            std::sort(sorted_kf_indices.begin(), sorted_kf_indices.end());
            
            // Find a keyframe pair with parallax in range [10, 20]
            std::vector<std::pair<std::shared_ptr<Frame>, int>> selected_pair;
            bool found_good_pair = false;
            
            for (const auto& kf_pair : sorted_kf_indices) {
                int kf_idx = kf_pair.second;
                
                // Check parallax with current frame
                cv::Point2f pt_curr = all_observations[0].first->get_feature(all_observations[0].second)->get_undistorted_coord();
                cv::Point2f pt_kf = all_observations[kf_idx].first->get_feature(all_observations[kf_idx].second)->get_undistorted_coord();
                
                Eigen::Vector2f pt1(pt_curr.x, pt_curr.y);
                Eigen::Vector2f pt2(pt_kf.x, pt_kf.y);
                
                float parallax = (pt1 - pt2).norm();
                
                // Accept if parallax is in desired range [1, 100]
                if (parallax > 1.0f) {
                    selected_pair.push_back(all_observations[0]);  // current frame
                    selected_pair.push_back(all_observations[kf_idx]);  // selected keyframe
                    found_good_pair = true;
                    break;
                }
            }
            
            if (!found_good_pair) {
                continue;  // No keyframe with suitable parallax found
            }

            // Triangulate using selected pair
            if (selected_pair.size() == 2) {
                Eigen::Vector3f P_world;
                if (multi_view_triangulation(selected_pair, P_world)) {
                    // Triangulation successful! Create MapPoint
                    auto new_mp = std::make_shared<MapPoint>(P_world);
                    
                    // Add observations for both frames
                    for (const auto& obs : selected_pair) {
                        Eigen::Matrix4f T_cw = obs.first->get_Twc().inverse();
                        Eigen::Vector4f P_camera = T_cw * Eigen::Vector4f(P_world.x(), P_world.y(), P_world.z(), 1.0f);

                        new_mp->add_observation(obs.first, obs.second);
                        obs.first->set_map_point(obs.second, new_mp);
                        obs.first->get_feature(obs.second)->set_3d_point(P_camera.head<3>());
                    }
                    
                    {
                        std::lock_guard<std::mutex> lock(m_map_points_mutex);
                        m_map_points.push_back(new_mp);
                    }
                    num_new_map_points++;
                }
            }
        }
    }

    return num_new_map_points;

}

int lightweight_vio::Estimator::create_keyframe_monocular(std::shared_ptr<Frame> frame) {
    if (!frame) {
        return 0;
    }
    
    // First, do standard keyframe creation
    frame->set_keyframe(true);
    
    // Calculate time difference from last keyframe
    double dt_from_last_kf = 0.0;
    if (m_last_keyframe) {
        double current_time = frame->get_timestamp();
        double last_kf_time = m_last_keyframe->get_timestamp();
        dt_from_last_kf = current_time - last_kf_time;
    }
    
    frame->set_dt_from_last_keyframe(dt_from_last_kf);
    
    int num_new_map_points = 0;
    int num_reused_map_points = 0;

    // Let's check feature observations before creating the keyframe
    const auto& features = frame->get_features();
    
    for (auto& feature : features) {
        if (feature && feature->is_valid()) {

            // If already map point associated, skip
            auto curr_idx = feature->get_feature_id();
            auto existing_mp = frame->get_map_point(curr_idx);
            
            if (existing_mp && !existing_mp->is_bad())
            {
                num_reused_map_points++;
                continue;
            }
            
            const auto& observation = feature->get_observations();
            
            // Collect all active keyframe observations
            std::vector<std::pair<std::shared_ptr<Frame>, int>> all_observations;
            all_observations.push_back({frame, curr_idx});  // Add current frame first

            bool is_there_valid_mp = false;
            
            for (const auto &obs : observation)
            {
                if (obs.frame && obs.frame->is_keyframe() && obs.frame->is_active())
                {
                    all_observations.push_back({obs.frame, obs.feature_index});
                }
            }

            // Need at least 5 observations (current + 4 keyframes)
            if (all_observations.size() < 5 || is_there_valid_mp) {
                continue;
            }
            
            // Sort keyframe observations by timestamp (oldest first)
            std::vector<std::pair<double, int>> sorted_kf_indices;
            for (size_t i = 1; i < all_observations.size(); ++i) {
                double obs_timestamp = all_observations[i].first->get_timestamp();
                sorted_kf_indices.push_back({obs_timestamp, i});
            }
            std::sort(sorted_kf_indices.begin(), sorted_kf_indices.end());
            
            // Find a keyframe pair with parallax in range [10, 20]
            std::vector<std::pair<std::shared_ptr<Frame>, int>> selected_pair;
            bool found_good_pair = false;
            
            for (const auto& kf_pair : sorted_kf_indices) {
                int kf_idx = kf_pair.second;
                
                // Check parallax with current frame
                cv::Point2f pt_curr = all_observations[0].first->get_feature(all_observations[0].second)->get_undistorted_coord();
                cv::Point2f pt_kf = all_observations[kf_idx].first->get_feature(all_observations[kf_idx].second)->get_undistorted_coord();
                
                Eigen::Vector2f pt1(pt_curr.x, pt_curr.y);
                Eigen::Vector2f pt2(pt_kf.x, pt_kf.y);
                
                float parallax = (pt1 - pt2).norm();
                
                // Accept if parallax is in desired range [1, 100]
                if (parallax > 1.0f && parallax <= 100.0f) {
                    selected_pair.push_back(all_observations[0]);  // current frame
                    selected_pair.push_back(all_observations[kf_idx]);  // selected keyframe
                    found_good_pair = true;
                    break;
                }
            }
            
            if (!found_good_pair) {
                continue;  // No keyframe with suitable parallax found
            }

            // Triangulate using selected pair
            if (selected_pair.size() == 2) {
                Eigen::Vector3f P_world;
                if (multi_view_triangulation(selected_pair, P_world)) {
                    // Triangulation successful! Create MapPoint
                    auto new_mp = std::make_shared<MapPoint>(P_world);
                    
                    // Add observations for both frames
                    for (const auto& obs : selected_pair) {
                        Eigen::Matrix4f T_cw = obs.first->get_Twc().inverse();
                        Eigen::Vector4f P_camera = T_cw * Eigen::Vector4f(P_world.x(), P_world.y(), P_world.z(), 1.0f);

                        new_mp->add_observation(obs.first, obs.second);
                        obs.first->set_map_point(obs.second, new_mp);
                        obs.first->get_feature(obs.second)->set_3d_point(P_camera.head<3>());
                    }
                    
                    {
                        std::lock_guard<std::mutex> lock(m_map_points_mutex);
                        m_map_points.push_back(new_mp);
                    }
                    num_new_map_points++;
                }
            }
        }
    }

    spdlog::info("[MONO_KF] Reused {} existing map points from observations", num_reused_map_points);
    spdlog::info("[MONO_KF] Created {} new map points via triangulation", num_new_map_points);

    // Thread-safe keyframe management
    {
        std::lock_guard<std::mutex> lock(m_keyframes_mutex);
        m_keyframes.push_back(frame);
        
        const int max_keyframes = Config::getInstance().m_keyframe_window_size;
        if (m_keyframes.size() > static_cast<size_t>(max_keyframes)) {
            auto oldest_keyframe = m_keyframes.front();
            
            // Mark oldest keyframe as inactive (removed from sliding window)
            oldest_keyframe->set_active(false);
            
            // Clean up observations from oldest keyframe
            const auto& oldest_features = oldest_keyframe->get_features();
            for (size_t i = 0; i < oldest_features.size(); ++i) {
                auto feature = oldest_features[i];
                auto map_point = oldest_keyframe->get_map_point(i);
                
                if (feature && feature->is_valid() && map_point && !map_point->is_bad()) {
                    map_point->remove_observation(oldest_keyframe);
                    
                    if (map_point->get_observation_count() == 0) {
                        map_point->set_bad();
                    }
                }
            }
            
            oldest_keyframe->release_images();
            m_keyframes.erase(m_keyframes.begin());
        }
    }
    
    // Update track counts for keyframe features
    for (auto& feature : features) {
        if (feature && feature->is_valid()) {
            feature->increment_track_count();
        }
    }
    
    // Update grid coverage and last keyframe reference
    m_last_keyframe_grid_coverage = calculate_grid_coverage_with_map_points(frame);

    spdlog::info("[MONO_KF] Keyframe grid coverage: {:.2f}", m_last_keyframe_grid_coverage);

    m_last_keyframe = frame;
    
    // Notify sliding window optimizer
    notify_sliding_window_thread();
    
    spdlog::info("[MONO_KF] Keyframe {} created: {} reused + {} triangulated = {} total map points",
                frame->get_frame_id(), num_reused_map_points, num_new_map_points, 
                num_reused_map_points + num_new_map_points);
    
    return num_new_map_points;
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
    fx = frame->get_fx(); fy = frame->get_fy(); cx = frame->get_cx(); cy = frame->get_cy();
    
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
        spdlog::warn("[PREDICT] Cannot predict: current_frame={}, previous_frame={}", 
                     (bool)m_current_frame, (bool)m_previous_frame);
        return;
    }
    
    const auto& config = Config::getInstance();
    
    // Check system mode from config
    if (config.m_system_mode == "VIO" && m_success_imu_init) {
        // VIO Mode: Use IMU preintegration for state prediction
        // Only use IMU prediction when IMU is properly initialized

        spdlog::info("[PREDICT] Using IMU preintegration for state prediction");

        
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

            // Check Velocity magnitude for validity

            // spdlog::error("Predicted Velocity: [{:.2f}, {:.2f}, {:.2f}] m/s", Vwb2.x(), Vwb2.y(), Vwb2.z());

            // Store predicted pose for comparison logging
            m_predicted_pose = predicted_pose;
            
            
            // spdlog::error("Trans change : [{:.2f}, {:.2f}, {:.2f}] m", 
            //               predicted_pose(0,3) - m_previous_frame->get_Twb()(0,3),
            //               predicted_pose(1,3) - m_previous_frame->get_Twb()(1,3),
            //               predicted_pose(2,3) - m_previous_frame->get_Twb()(2,3));
            
            m_current_frame->set_Twb(predicted_pose);
            m_current_frame->set_velocity(Vwb2);

            
            
          
            
            
        } 
        else 
        {
            // Fallback to constant velocity model if no valid preintegration
            
            // VO Mode fallback: Use visual motion model
            Eigen::Matrix4f predicted_pose = m_previous_frame->get_Twb() * m_transform_from_last;
            m_predicted_pose = predicted_pose; // Store for comparison
            
            // Keep velocity zero in fallback mode
            m_current_frame->set_velocity(Eigen::Vector3f::Zero());
        }
        
    } 
    else 
    {
        // VO Mode: Use visual odometry motion model
        Eigen::Matrix4f predicted_pose = m_previous_frame->get_Twb() * m_transform_from_last;

        m_predicted_pose = predicted_pose; // Store for comparison
        m_current_frame->set_Twb(predicted_pose);

        // Keep velocity zero in VO mode
        m_current_frame->set_velocity(Eigen::Vector3f::Zero());
        
    }
}


void Estimator::update_transform_from_last() {
    // Update the transform from the previous frame to the current frame
    if (!m_current_frame || !m_previous_frame) {
        spdlog::warn("[TRANSFORM_UPDATE] Cannot update transform: current_frame={}, previous_frame={}", 
                     (bool)m_current_frame, (bool)m_previous_frame);
        return;
    }

    Eigen::Matrix4f Twb_prev = m_previous_frame->get_Twb();
    Eigen::Matrix4f Twb_curr = m_current_frame->get_Twb();
    m_transform_from_last = Twb_prev.inverse() * Twb_curr;

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
    
        spdlog::debug("Grid coverage: {}/{} cells have features with map points ({:.2f}%)", 
                     cells_with_map_points, total_cells, coverage_ratio * 100.0);
    
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
            
            // 🎯 VIO Mode: Skip sliding window optimization until IMU is initialized
            const auto& config = Config::getInstance();
            if (config.m_system_mode == "VIO" && !m_success_imu_init) {

                spdlog::warn("[SW_THREAD] IMU not initialized yet, skipping sliding window optimization");
                if (Config::getInstance().m_enable_debug_output) {
                    spdlog::debug("[SW_THREAD] ⏸️ Waiting for IMU initialization before running sliding window optimization");
                }
                lock.unlock();
                continue;  // Skip optimization, wait for IMU init
            }
            

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
        double frame_time = keyframe->get_timestamp();  // Already in seconds
        
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

InertialOptimizationResult lightweight_vio::Estimator::try_initialize_imu() {
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
    
    InertialOptimizationResult result;  // Default success = false
    
    const auto& config = Config::getInstance();
    
    // Check if we have enough keyframes for gravity estimation
    if (m_keyframes.size() < 2) {
        spdlog::debug("[GRAVITY_EST] Not enough keyframes: {} < 5", m_keyframes.size());
        return result;
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
        return result;
    }
    
    // Use IMUHandler to estimate gravity
    if (!m_imu_handler) {
        spdlog::error("[GRAVITY_EST] IMU handler not initialized");
        return result;
    }
    
    // Variables to capture optimization costs
    double initial_cost = 0.0;
    double final_cost = 0.0;
    
    // First estimate gravity, then debug velocity comparison

    spdlog::info("[GRAVITY_EST] Starting gravity estimation with {} keyframes and {} IMU measurements", 
                 keyframe_ptrs.size(), all_imu_data.size());

    bool gravity_success = m_imu_handler->estimate_gravity_with_stereo_constraints(
        keyframe_ptrs, all_imu_data, 9.81f, &initial_cost, &final_cost);
    
    // Store cost information for logging in initialize_imu()
    if (gravity_success && Config::getInstance().m_enable_debug_output) {
        double cost_reduction = initial_cost - final_cost;
        double cost_reduction_percentage = (initial_cost > 0.0) ? (cost_reduction / initial_cost * 100.0) : 0.0;
        
        spdlog::info("[GRAVITY_EST] 📉 Optimization Cost:");
        spdlog::info("  Initial Cost:  {:.6e}", initial_cost);
        spdlog::info("  Final Cost:    {:.6e}", final_cost);
        spdlog::info("  Cost Reduction: {:.6e} ({:.2f}%)", cost_reduction, cost_reduction_percentage);
    }

    m_Rgw_init = m_imu_handler->get_Rgw();

    std::vector<std::shared_ptr<MapPoint>> all_map_points;
    std::set<std::shared_ptr<MapPoint>> unique_map_points;

    Eigen::Matrix4f m_Tgw_init = Eigen::Matrix4f::Identity();
    m_Tgw_init.block<3,3>(0,0) = m_Rgw_init;

    if (Config::getInstance().m_enable_debug_output) {
        spdlog::info("================================================================================");
        spdlog::info("[ESTIMATOR] 📐 Constructed Tgw_init from Rgw:");
        spdlog::info("[ESTIMATOR]   Rgw rotation angle from vertical: {:.1f} degrees", 
                     std::acos(std::abs(m_Rgw_init(2,2))) * 180.0 / M_PI);
        spdlog::info("================================================================================");
    }

  

    // std::cout<<"Estimated initial gravity direction (Rgw):\n"<<m_Rgw_init<<"\n\n\n\n"<<std::endl;

    spdlog::info("Num all keyframe VS keyframes for optimization: {} VS {}", keyframe_ptrs.size(), m_keyframes.size());

    if (gravity_success) {
        m_imu_handler->debug_velocity_comparison(keyframe_ptrs, all_imu_data);
        
        // 🎯 Prepare ALL keyframes for transformation (not just optimization frames)
        std::vector<Frame*> all_keyframe_ptrs;
        for (const auto& kf : m_keyframes) {
            all_keyframe_ptrs.push_back(kf.get());
        }
        
        // 🎯 After gravity estimation, perform IMU initialization optimization
        auto imu_init_result = m_inertial_optimizer->optimize_imu_initialization(
            keyframe_ptrs,          // Frames for optimization (first 5)
            all_keyframe_ptrs,      // ALL keyframes for transformation
            std::shared_ptr<IMUHandler>(m_imu_handler.get(), [](IMUHandler*){}) // Non-owning shared_ptr
        );
        
        if (imu_init_result.success) {
            
            // Store Tgw_init for viewer
            m_Tgw_init = imu_init_result.Tgw_init;

            std::cout<<"Optimized initial gravity direction (Rgw):\n"<<m_Tgw_init.block<3,3>(0,0)<<"\n\n\n\n"<<std::endl;
            
            // ⭐ Store gravity visualization data
            if (imu_init_result.has_gravity_visualization_data) {
                m_has_gravity_viz_data = true;
                m_g_world_before_transform = imu_init_result.g_world_before_transform;
                m_gravity_arrow_origin = imu_init_result.first_frame_position;
            }

            debug_keyframe_to_keyframe_comparison();
            
          
            return imu_init_result;  // ✅ Return result struct
        } else {
            spdlog::warn("❌ [IMU_INIT] IMU initialization optimization failed");
            return result;  // Return failed result
        }
    }

    

    return result;  // Return failed result
}

// ========================================================================
// Helper Functions for IMU Initialization
// ========================================================================

void lightweight_vio::Estimator::apply_imu_optimization_results(const InertialOptimizationResult& result) {
    // Update Frame[0] velocity from Frame[1]
    if (m_keyframes.size() >= 2 && result.optimized_velocities.size() > 0) {

        spdlog::info("Check Veclocity {} {} {}", 
                     result.optimized_velocities[0].x(),
                     result.optimized_velocities[0].y(),
                     result.optimized_velocities[0].z());
        m_keyframes[0]->set_velocity(result.optimized_velocities[0]);
    }
    
    // Update Frame[1..N] velocities
    for (size_t i = 0; i < result.optimized_velocities.size(); ++i) {
        size_t frame_idx = i + 1;
        if (frame_idx < m_keyframes.size()) {
            m_keyframes[frame_idx]->set_velocity(result.optimized_velocities[i]);
        }
    }
    
    // Apply averaged bias to ALL frames
    for (auto& kf : m_keyframes) {
        kf->set_accel_bias(result.optimized_accel_bias);
        kf->set_gyro_bias(result.optimized_gyro_bias);
    }
    
    spdlog::info("[ESTIMATOR] ✅ Updated velocities and biases for {} frames", m_keyframes.size());
}

void lightweight_vio::Estimator::update_preintegrations_with_new_bias(const InertialOptimizationResult& result) {
    std::vector<std::pair<Eigen::Vector3f, Eigen::Vector3f>> frame_biases;
    std::vector<Frame*> frames_to_update;
    
    for (size_t i = 1; i < m_keyframes.size(); ++i) {
        frame_biases.emplace_back(
            result.optimized_gyro_bias,
            result.optimized_accel_bias
        );
        frames_to_update.push_back(m_keyframes[i].get());
    }
    
    m_imu_handler->update_preintegrations_with_optimized_bias(
        frames_to_update,
        frame_biases
    );
    
    spdlog::info("[ESTIMATOR] ✅ Updated preintegrations with optimized bias");
}

void lightweight_vio::Estimator::apply_imu_based_scale_correction() {
    spdlog::info("================================================================================");
    spdlog::info("[SCALE_CORRECTION] 📏 Starting IMU-Based Scale Correction (Monocular Only)");
    spdlog::info("  Total keyframes: {}", m_keyframes.size());
    spdlog::info("================================================================================");
    
    if (m_keyframes.size() < 2) {
        spdlog::warn("[SCALE_CORRECTION] ⚠️  Need at least 2 keyframes, skipping");
        return;
    }
    
    // Use first two keyframes (from two-view initialization)
    auto kf_prev = m_keyframes[0];
    auto kf_curr = m_keyframes[1];
    
    // Get VO translation (scale-ambiguous)
    Eigen::Matrix4f Twb_prev = kf_prev->get_Twb();
    Eigen::Matrix4f Twb_curr = kf_curr->get_Twb();
    Eigen::Matrix4f T_vo = Twb_prev.inverse() * Twb_curr;

    Eigen::Vector3f t_vo = T_vo.block<3,1>(0,3);
    double t_vo_norm = t_vo.norm();
    
    // Get IMU preintegration (metric scale)
    auto preint = kf_curr->get_imu_preintegration_from_last_keyframe();
    if (!preint) {
        spdlog::error("[SCALE_CORRECTION] ❌ No preintegration for second keyframe");
        return;
    }
    
    // IMU translation in world frame: R_prev * delta_P
    Eigen::Matrix3f R_wb_prev = kf_prev->get_Twb().block<3,3>(0,0);
    Eigen::Vector3f delta_p_world = R_wb_prev * preint->delta_P;
    double t_imu_norm = delta_p_world.norm();
    
    // Compute scale: s = ||t_imu|| / ||t_vo||
    if (t_vo_norm < 1e-6) {
        spdlog::error("[SCALE_CORRECTION] ❌ VO translation too small: {:.6f}m", t_vo_norm);
        return;
    }
    
    double scale = t_imu_norm / t_vo_norm;
    
    spdlog::info("--------------------------------------------------------------------------------");
    spdlog::info("[SCALE_CORRECTION] 📊 Scale Computation:");
    spdlog::info("  VO translation norm:  {:.6f} m", t_vo_norm);
    spdlog::info("  IMU translation norm: {:.6f} m", t_imu_norm);
    spdlog::info("  Computed scale:       {:.6f}x", scale);
    spdlog::info("--------------------------------------------------------------------------------");
    
    T_vo.block<3,1>(0,3) *= scale;
    Eigen::Matrix4f Twb_curr_scaled = Twb_prev * T_vo;
    kf_curr->set_Twb(Twb_curr_scaled);

    
    // // Apply scale to all keyframe velocities
    // for (auto& kf : m_keyframes) {
    //     Eigen::Vector3f vel = kf->get_velocity();
    //     vel *= scale;
    //     kf->set_velocity(vel);
    // }
    
    // Apply scale to all map points
    int num_mp_updated = 0;
    for (auto& mp : m_map_points) {
        if (mp && !mp->is_bad()) {
            Eigen::Vector3f pos = mp->get_position();
            pos *= scale;
            mp->set_position(pos);
            num_mp_updated++;
        }
    }
    
    spdlog::info("[SCALE_CORRECTION] ✅ Applied scale correction:");
    spdlog::info("  Keyframes:  {} updated", m_keyframes.size());
    spdlog::info("  Map points: {} updated", num_mp_updated);
    spdlog::info("  Scale factor: {:.6f}x", scale);
    spdlog::info("================================================================================\n");
}

void lightweight_vio::Estimator::visualize_gravity_direction(const InertialOptimizationResult& result) {
    if (!result.has_gravity_visualization_data) {
        return;
    }
    
    // Store gravity visualization data for viewer
    m_has_gravity_viz_data = true;
    m_g_world_before_transform = result.g_world_before_transform;
    m_gravity_arrow_origin = result.first_frame_position;
    
    spdlog::info("");
    spdlog::info("🎨 [VISUALIZATION] Gravity direction optimized:");
    spdlog::info("   📍 g_world (World frame): [{:.6f}, {:.6f}, {:.6f}] m/s²",
                 result.g_world_before_transform.x(),
                 result.g_world_before_transform.y(),
                 result.g_world_before_transform.z());
    spdlog::info("   📏 Magnitude: {:.6f} m/s²", result.g_world_before_transform.norm());
    spdlog::info("   🔴 RED ARROW will show g_world direction (before transformation)");
}

// void lightweight_vio::Estimator::update_gravity_visualization_after_transform() {
//     if (!m_has_gravity_viz_data || m_keyframes.empty()) {
//         return;
//     }
    
//     // ⭐ Only update arrow origin to follow camera in gravity-aligned frame
//     // Keep gravity direction unchanged (still shows world frame direction)
//     m_gravity_arrow_origin = m_keyframes[0]->get_Twb().block<3,1>(0,3);
    
//     // Get current gravity from IMUHandler for verification only
//     Eigen::Vector3f g_after = m_imu_handler->get_gravity();
    
//     spdlog::info("");
//     spdlog::info("🎨 [VISUALIZATION] Gravity visualization updated:");
//     spdlog::info("   📍 g_world (KEPT from before transform): [{:.6f}, {:.6f}, {:.6f}] m/s²",
//                  m_g_world_before_transform.x(),
//                  m_g_world_before_transform.y(),
//                  m_g_world_before_transform.z());
//     spdlog::info("   📍 g_current (IMU handler, after):      [{:.6f}, {:.6f}, {:.6f}] m/s²",
//                  g_after.x(), g_after.y(), g_after.z());
//     spdlog::info("   📍 Arrow origin (Gravity-aligned frame): [{:.3f}, {:.3f}, {:.3f}]",
//                  m_gravity_arrow_origin.x(),
//                  m_gravity_arrow_origin.y(),
//                  m_gravity_arrow_origin.z());
//     spdlog::info("   � RED ARROW continues showing original g_world direction (before transform)");
// }

void lightweight_vio::Estimator::update_gravity_visualization_after_transform() {
    if (!m_has_gravity_viz_data || m_keyframes.empty()) {
        return;
    }
    
    // ⭐ Store original g_world before transform for logging
    Eigen::Vector3f g_world_original = m_g_world_before_transform;
    
    // ⭐ Transform gravity using Tgw: g_gravity_aligned = Rgw * g_world
    Eigen::Matrix3f Rgw = m_Tgw.block<3,3>(0,0);
    m_g_world_before_transform = Rgw * g_world_original;
    
    // Update arrow origin to current keyframe position in gravity-aligned frame
    m_gravity_arrow_origin = m_keyframes[0]->get_Twb().block<3,1>(0,3);
    
    // Get current gravity from IMUHandler for verification
    Eigen::Vector3f g_imu_handler = m_imu_handler->get_gravity();
}

void lightweight_vio::Estimator::apply_gravity_alignment_transform(const Eigen::Matrix4f& Tgw) {
    spdlog::info("");
    spdlog::info("🔄 ===============================================================================");
    spdlog::info("🔄 [GRAVITY_ALIGN] Starting Coordinate Transformation");
    spdlog::info("🔄 ===============================================================================");
    
    // ⭐ Store gravity BEFORE transformation for logging
    Eigen::Vector3f g_world_before = m_imu_handler->get_gravity();
    
    // ⭐ Store Tgw for gravity visualization update
    m_Tgw = Tgw;
    
    // Log transformation matrix
    spdlog::info("📐 [GRAVITY_ALIGN] Transformation Matrix Tgw (World → Gravity-aligned):");
    spdlog::info("     [{:9.6f} {:9.6f} {:9.6f} | {:9.3f}]", Tgw(0,0), Tgw(0,1), Tgw(0,2), Tgw(0,3));
    spdlog::info("     [{:9.6f} {:9.6f} {:9.6f} | {:9.3f}]", Tgw(1,0), Tgw(1,1), Tgw(1,2), Tgw(1,3));
    spdlog::info("     [{:9.6f} {:9.6f} {:9.6f} | {:9.3f}]", Tgw(2,0), Tgw(2,1), Tgw(2,2), Tgw(2,3));
    spdlog::info("     [{:9.6f} {:9.6f} {:9.6f} | {:9.3f}]", Tgw(3,0), Tgw(3,1), Tgw(3,2), Tgw(3,3));
    
    // Collect all map points
    std::vector<std::shared_ptr<MapPoint>> all_map_points;
    std::set<std::shared_ptr<MapPoint>> unique_map_points;
    
    for (const auto& kf : m_keyframes) {
        for (const auto& mp : kf->get_map_points()) {
            if (mp && !mp->is_bad()) {
                unique_map_points.insert(mp);
            }
        }
    }
    
    all_map_points.assign(unique_map_points.begin(), unique_map_points.end());

    // Let's transform map points here

    for (const auto& mp : all_map_points) {
        Eigen::Vector4f pos_homogeneous;
        pos_homogeneous << mp->get_position(), 1.0f;
        Eigen::Vector4f pos_transformed = Tgw * pos_homogeneous;
        mp->set_position(pos_transformed.head<3>());
    }

    // Let's transform keyframes here
    for (const auto& kf : m_keyframes) {
        Eigen::Matrix4f Twb = kf->get_Twb();
        Eigen::Matrix4f Tgb = Tgw * Twb;
        kf->set_Twb(Tgb);
    }

    // Let's transform keyframe velocities here
    for (const auto& kf : m_keyframes) {
        Eigen::Vector3f vel_w = kf->get_velocity();
        Eigen::Vector3f vel_g = Tgw.block<3,3>(0,0) * vel_w;
        kf->set_velocity(vel_g);
    }

    spdlog::info("🔄 [GRAVITY_ALIGN] Coordinate transformation complete!");
    spdlog::info("🔄 ===============================================================================\n");
    
    // 🎯 Notify sliding window thread that IMU initialization is complete
    // This will wake up the thread to start optimization
    notify_sliding_window_thread();
    spdlog::info("🚀 [SW_THREAD] Sliding window thread notified - optimization will resume");
  
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

bool Estimator::get_gravity_visualization_data(Eigen::Vector3f& g_world, Eigen::Vector3f& origin) const {
    if (!m_has_gravity_viz_data) {
        return false;
    }
    
    g_world = m_g_world_before_transform;
    origin = m_gravity_arrow_origin;
    return true;
}

} // namespace lightweight_vio
