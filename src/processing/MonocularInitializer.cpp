/**
 * @file      MonocularInitializer.cpp
 * @brief     Monocular VIO initialization implementation
 * @author    Seungwon Choi (csw3575@snu.ac.kr)
 * @date      2025-11-09
 * @copyright Copyright (c) 2025 Seungwon Choi. All rights reserved.
 *
 * @par License
 * This project is released under the MIT License.
 */

#include "MonocularInitializer.h"
#include "database/Feature.h"
#include "Optimizer.h"
#include <spdlog/spdlog.h>
#include <opencv2/calib3d.hpp>
#include <algorithm>
#include <numeric>

namespace lightweight_vio {

MonocularInitializer::MonocularInitializer() 
    : m_is_initialized(false)
{
    const Config& config = Config::getInstance();
    
    // Load initialization parameters from config
    m_min_parallax_pixels = config.m_init_min_parallax_pixels;
    m_window_size = config.m_keyframe_window_size;  // Reuse keyframe window size for initialization
    
    // Two-view geometry parameters
    m_ransac_threshold = config.m_init_ransac_threshold;
    m_ransac_confidence = config.m_init_ransac_confidence;
    m_ransac_max_iterations = config.m_init_ransac_max_iterations;
    m_min_inliers = config.m_init_min_inliers;
    
    // Triangulation parameters
    m_min_triangulation_angle = config.m_init_min_triangulation_angle;
    m_max_reprojection_error = config.m_init_max_reprojection_error;
    
    spdlog::info("[MonocularInitializer] Created with parameters:");
    spdlog::info("  Window size: {} frames", m_window_size);
    spdlog::info("  Min parallax: {:.1f} pixels (median)", m_min_parallax_pixels);
    spdlog::info("  Min inliers: {}", m_min_inliers);
}

bool MonocularInitializer::try_initialize_visual_sfm(std::shared_ptr<Frame> frame, bool* init_attempted) {
    if (!frame) {
        spdlog::warn("[MonocularInitializer] Null frame provided");
        return false;
    }
    
    // Reset init_attempted flag
    if (init_attempted) {
        *init_attempted = false;
    }
    
    // Already initialized - no more frames needed
    if (m_is_initialized) {
        return false;
    }
    
    // Add frame to window
    m_frame_window.push_back(frame);
    spdlog::debug("[MonocularInitializer] Added frame to window ({}/{})", m_frame_window.size(), m_window_size);
    
    // Window not full yet - keep collecting
    if (m_frame_window.size() < static_cast<size_t>(m_window_size)) {
        return true;  // Frame added successfully
    }
    
    // Window is full - attempt initialization
    spdlog::info("[MonocularInitializer] Window full ({} frames) - attempting initialization", m_window_size);
    
    if (init_attempted) {
        *init_attempted = true;
    }
    
    // Find best frame pair with maximum parallax
    int ref_idx, cur_idx;
    if (!find_best_frame_pair(ref_idx, cur_idx)) {
        spdlog::warn("[MonocularInitializer] No valid frame pair found with sufficient parallax");
        
        // Slide window: remove oldest frame
        m_frame_window.erase(m_frame_window.begin());
        spdlog::debug("[MonocularInitializer] Slid window - oldest frame removed");
        
        return false;
    }
    
    spdlog::info("[MonocularInitializer] Best frame pair: Frame {} (idx {}) <-> Frame {} (idx {})",
                 m_frame_window[ref_idx]->get_frame_id(), ref_idx,
                 m_frame_window[cur_idx]->get_frame_id(), cur_idx);
    
    // Attempt two-view initialization with best pair
    Eigen::Matrix3f R_21;
    Eigen::Vector3f t_21;
    std::vector<Eigen::Vector3f> points_3d;
    std::vector<int> inlier_indices_1, inlier_indices_2;
    
    if (initialize_two_views(m_frame_window[ref_idx], m_frame_window[cur_idx], 
                            R_21, t_21, points_3d, inlier_indices_1, inlier_indices_2)) {

        // Store initialization result
        m_result.success = true;
        m_result.num_triangulated_points = points_3d.size();
        
        // Get reference and current frames
        auto frame_ref = m_frame_window[ref_idx];
        auto frame_cur = m_frame_window[cur_idx];
        
        frame_ref->set_keyframe(true);
        frame_cur->set_keyframe(true);

        Eigen::Matrix4f Twb_ref = frame_ref->get_Twb();
        Eigen::Matrix4f Twb_cur = frame_cur->get_Twb();

        double dt_total = frame_cur->get_timestamp() - frame_ref->get_timestamp(); 

        m_result.initialized_keyframes.push_back(frame_ref);


        for (size_t i = 1; i < m_frame_window.size()-1; ++i) {

            auto frame_i = m_frame_window[i];
            double dt_i = frame_i->get_timestamp() - frame_ref->get_timestamp();
            double alpha = dt_i / dt_total;

            // Interpolate translation
            Eigen::Vector3f t_i = (1 - alpha) * Twb_ref.block<3, 1>(0, 3) + alpha * Twb_cur.block<3, 1>(0, 3);

            // Interpolate rotation (slerp)
            Eigen::Quaternionf q_ref(Twb_ref.block<3, 3>(0, 0));
            Eigen::Quaternionf q_cur(Twb_cur.block<3, 3>(0, 0));
            Eigen::Quaternionf q_i = q_ref.slerp(alpha, q_cur);
            
            // Construct Twb from interpolated rotation and translation
            Eigen::Matrix4f Twb_i = Eigen::Matrix4f::Identity();
            Twb_i.block<3, 3>(0, 0) = q_i.toRotationMatrix();
            Twb_i.block<3, 1>(0, 3) = t_i;
            
            frame_i->set_Twb(Twb_i);
            frame_i->set_keyframe(true);

            m_result.initialized_keyframes.push_back(frame_i);

        }

        m_result.initialized_keyframes.push_back(frame_cur);

        // Link using feature observations

        // Ref frame
        for (size_t idx = 0; idx < inlier_indices_1.size(); ++idx) {
            int feat_idx_ref = inlier_indices_1[idx];
            int feat_idx_cur = inlier_indices_2[idx];

            auto feature_ref = frame_ref->get_feature(feat_idx_ref);
            auto feature_cur = frame_cur->get_feature(feat_idx_cur);

            if (feature_ref && feature_cur) {
                feature_ref->add_observation(frame_cur, feat_idx_cur);
                feature_cur->add_observation(frame_ref, feat_idx_ref);
            }

    
            spdlog::info("Num obs check for each feature after linking : {} {}", 
                         feature_ref->get_observations().size(),
                         feature_cur->get_observations().size());
            // ref
            const auto& observations_ref = feature_ref->get_observations();
            for (const auto& obs : observations_ref) {
                auto obs_frame = obs.frame;

                if(obs_frame == frame_ref || obs_frame == frame_cur) continue;

                auto mp = frame_ref->get_map_point(feat_idx_ref);
                if (!mp) continue;

                obs_frame->set_map_point(obs.feature_index, mp);
                mp->add_observation(obs_frame, obs.feature_index);
            }

            // cur
            const auto& observations_cur = feature_cur->get_observations();
            for (const auto& obs : observations_cur) {
                auto obs_frame = obs.frame;

                if(obs_frame == frame_cur || obs_frame == frame_ref) continue;

                auto mp = frame_cur->get_map_point(feat_idx_cur);
                if (!mp) continue;

                obs_frame->set_map_point(obs.feature_index, mp);
                mp->add_observation(obs_frame, obs.feature_index);
            }

        }

        // After link, count how many valid map points exist in each keyframe
        for (auto& kf : m_result.initialized_keyframes) {
            int valid_mp_count = 0;
            const auto& map_points = kf->get_map_points();
            for (const auto& mp : map_points) {
                if (mp && !mp->is_bad()) {
                    valid_mp_count++;
                }
            }
            spdlog::info("[MonocularInitializer] Keyframe {} has {} valid map points after initialization",
                         kf->get_frame_id(), valid_mp_count);
        }

        // PnP refinement for each keyframe
        PnPOptimizer pnp_optimizer;

        for(unsigned int i=1; i<m_result.initialized_keyframes.size()-1; ++i) {
            auto frame_i = m_result.initialized_keyframes[i];
            pnp_optimizer.optimize_pose(frame_i);
        }



        m_is_initialized = true;
        return true;
    } 
    else {
        
        m_frame_window.erase(m_frame_window.begin());
        spdlog::debug("[MonocularInitializer] Slid window - will retry with next frame");
        
        return false;
    }
}


void MonocularInitializer::reset() {
    m_frame_window.clear();
    m_is_initialized = false;
    m_result = InitializationResult();
    spdlog::info("[MonocularInitializer] Reset");
}

bool MonocularInitializer::find_best_frame_pair(int& ref_idx, int& cur_idx) {
    // Always use first (oldest) and last (newest) frame pair
    // This maximizes baseline (temporal distance)
    ref_idx = 0;
    cur_idx = m_frame_window.size() - 1;

    // Compute parallax between first and last frame
    ParallaxInfo parallax_info = compute_parallax_normalized(m_frame_window[ref_idx], m_frame_window[cur_idx]);

    spdlog::info("[MonocularInitializer] Frame pair [{}, {}]: parallax={:.2f}px, features={}",
                 m_frame_window[ref_idx]->get_frame_id(), m_frame_window[cur_idx]->get_frame_id(), 
                 parallax_info.average_parallax_pixels, parallax_info.num_tracked_features);

    // Check if parallax and feature count are sufficient
    if (parallax_info.num_tracked_features < 20)
    {
        spdlog::warn("[MonocularInitializer] Insufficient correspondences: {} (need >= 20)", 
                     parallax_info.num_tracked_features);
        return false;
    }

    if (parallax_info.average_parallax_pixels < m_min_parallax_pixels)
    {
        spdlog::warn("[MonocularInitializer] Insufficient parallax: {:.2f}px (threshold={:.2f}px)", 
                     parallax_info.average_parallax_pixels, m_min_parallax_pixels);
        return false;
    }

    spdlog::info("[MonocularInitializer] Valid frame pair with parallax={:.2f}px (threshold={:.2f}px)",
                 parallax_info.average_parallax_pixels, m_min_parallax_pixels);
    return true;
}

MonocularInitializer::ParallaxInfo MonocularInitializer::compute_parallax_normalized(
    const std::shared_ptr<Frame>& frame1,
    const std::shared_ptr<Frame>& frame2) const 
{
    ParallaxInfo info;

    // Find correspondences using feature observations

    const auto& features1 = frame1->get_features();
    
    // Debug: Check feature observations
    int total_features = features1.size();
    int features_with_observations = 0;
    int total_observations = 0;
    
    for (auto &feat1 : features1)
    {
        const auto &observations = feat1->get_observations();
        if (!observations.empty()) {
            features_with_observations++;
            total_observations += observations.size();
        }
    }
    

    for (auto &feat1 : features1)
    {

        const auto &observations = feat1->get_observations();


        for (const auto &obs : observations)
        {

            if (obs.frame->get_frame_id() == frame2->get_frame_id())
            {
                // Compute parallax in normalized coordinates
                Eigen::Vector2f norm1 = feat1->get_normalized_coord();
                Eigen::Vector2f norm2 = obs.frame->get_feature(obs.feature_index)->get_normalized_coord();

                // Pixel displacement in normalized space
                double du = norm2.x() - norm1.x();
                double dv = norm2.y() - norm1.y();
                double parallax_normalized = std::sqrt(du * du + dv * dv);

                // Convert to pixel space: parallax_pixels = parallax_normalized * focal_length
                double fx = frame1->get_fx(), fy = frame1->get_fy();
                double focal_length = (fx + fy) / 2.0;
                double parallax_pixels = parallax_normalized * focal_length;

                info.pixel_displacements.push_back(parallax_pixels);
            }
        }
    }

    if (info.pixel_displacements.empty()) {
        spdlog::warn("[MonocularInitializer] No tracked features between frames {} and {}",
                     frame1->get_frame_id(), frame2->get_frame_id());
        return info;
    }

    // Compute statistics
    info.num_tracked_features = info.pixel_displacements.size();
    info.average_parallax_pixels = std::accumulate(info.pixel_displacements.begin(), 
                                                    info.pixel_displacements.end(), 0.0) 
                                   / info.pixel_displacements.size();
    
    // Compute median (more robust than average)
    std::vector<double> sorted_displacements = info.pixel_displacements;
    std::sort(sorted_displacements.begin(), sorted_displacements.end());
    size_t mid = sorted_displacements.size() / 2;
    if (sorted_displacements.size() % 2 == 0) {
        info.median_parallax_pixels = (sorted_displacements[mid-1] + sorted_displacements[mid]) / 2.0;
    } else {
        info.median_parallax_pixels = sorted_displacements[mid];
    }
    
    info.sufficient_parallax = (info.median_parallax_pixels >= m_min_parallax_pixels);

    
    return info;
}

MonocularInitializer::ParallaxInfo MonocularInitializer::compute_parallax(
    const std::shared_ptr<Frame>& frame1,
    const std::shared_ptr<Frame>& frame2) const 
{
    ParallaxInfo info;
    
    // Get features from both frames
    const auto& features1 = frame1->get_features();
    const auto& features2 = frame2->get_features();
    
    if (features1.empty() || features2.empty()) {
        return info;
    }
    
    // Find tracked features between frames using tracked_feature_id
    std::vector<double> pixel_displacements;
    pixel_displacements.reserve(features2.size());
    
    // Debug: Check how many features have tracked_feature_id in each frame
    int frame1_tracked = 0, frame2_tracked = 0;
    for (const auto& f : features1) {
        if (f->has_tracked_feature()) frame1_tracked++;
    }
    for (const auto& f : features2) {
        if (f->has_tracked_feature()) frame2_tracked++;
    }
    
    // FIXED: Iterate through frame2 (newer frame with tracked_feature_id)
    // instead of frame1 (older reference frame)
    std::vector<cv::Point2f> matched_pts1, matched_pts2;  // For visualization
    int zero_displacement_count = 0;
    
    for (const auto& feat2 : features2) {
        if (!feat2->is_valid() || !feat2->has_tracked_feature()) {
            continue;
        }
        
        // FIXED: Get tracked_feature_id from frame2 and find corresponding feature in frame1
        int tracked_id = feat2->get_tracked_feature_id();
        auto feat1 = frame1->get_feature(tracked_id);
        
        if (!feat1 || !feat1->is_valid()) {
            continue;
        }
        
        // Compute pixel displacement (use undistorted coordinates)
        cv::Point2f pt1 = feat1->get_undistorted_coord();
        cv::Point2f pt2 = feat2->get_undistorted_coord();
        
        
        double dx = pt2.x - pt1.x;
        double dy = pt2.y - pt1.y;
        double displacement = std::sqrt(dx * dx + dy * dy);
        
        if (displacement < 0.01) {
            zero_displacement_count++;
        }
        
        pixel_displacements.push_back(displacement);
        matched_pts1.push_back(feat1->get_pixel_coord());
        matched_pts2.push_back(feat2->get_pixel_coord());
    }
    
    if (pixel_displacements.empty()) {
        return info;
    }
    
    // Compute statistics
    info.num_tracked_features = pixel_displacements.size();
    info.average_parallax_pixels = std::accumulate(pixel_displacements.begin(), 
                                                    pixel_displacements.end(), 0.0) 
                                   / pixel_displacements.size();
    
    // Compute median (more robust than average)
    std::vector<double> sorted_displacements = pixel_displacements;
    std::sort(sorted_displacements.begin(), sorted_displacements.end());
    size_t mid = sorted_displacements.size() / 2;
    if (sorted_displacements.size() % 2 == 0) {
        info.median_parallax_pixels = (sorted_displacements[mid-1] + sorted_displacements[mid]) / 2.0;
    } else {
        info.median_parallax_pixels = sorted_displacements[mid];
    }
    
    info.sufficient_parallax = (info.median_parallax_pixels >= m_min_parallax_pixels);
    
    return info;
}

bool MonocularInitializer::initialize_two_views(
    const std::shared_ptr<Frame>& frame1,
    const std::shared_ptr<Frame>& frame2,
    Eigen::Matrix3f& R_21,
    Eigen::Vector3f& t_21,
    std::vector<Eigen::Vector3f>& points_3d,
    std::vector<int>& inlier_indices_1,
    std::vector<int>& inlier_indices_2)
{
    // Find matched features between two frames
    // NOTE: Iterate frame2 features (current frame) and find correspondences in frame1 (reference)
    std::vector<cv::Point2f> pts1, pts2;
    std::vector<int> feature_indices_1;
    std::vector<int> feature_indices_2;


    const auto& feature_1 = frame1->get_features();

    for (const auto& feat1 : feature_1) {

        const auto& observations_1 = feat1->get_observations();

        for (const auto& obs : observations_1) {

            if (obs.frame->get_frame_id() == frame2->get_frame_id()) {

                // Compute normalized coordinates

                Eigen::Vector2f norm1 = feat1->get_normalized_coord();
                Eigen::Vector2f norm2 = obs.frame->get_feature(obs.feature_index)->get_normalized_coord();

                cv::Point2f cv_norm1(norm1.x(), norm1.y());
                cv::Point2f cv_norm2(norm2.x(), norm2.y());

                pts1.push_back(cv_norm1);
                pts2.push_back(cv_norm2);
                feature_indices_1.push_back(feat1->get_feature_id());
                feature_indices_2.push_back(obs.feature_index);
            }
        }
    }

    if (pts1.size() < 8) {
        spdlog::error("[MonocularInitializer] Insufficient matches: {} (need >= 8)", pts1.size());
        return false;
    }
    
    spdlog::info("[MonocularInitializer] Found {} matched features for two-view geometry", pts1.size());
    
    // Compute Essential matrix directly from normalized coordinates using 8-point algorithm
    // E = [t]_x * R constraint: x2^T * E * x1 = 0
    
    double focal_length = (frame1->get_fx() + frame1->get_fy()) / 2.0;
    double ransac_threshold_normalized = m_ransac_threshold / focal_length;
    
    spdlog::debug("[MonocularInitializer] Estimating Essential matrix with RANSAC: threshold={:.6f} (normalized), {:.2f} (pixels), focal={:.1f}",
                 ransac_threshold_normalized, m_ransac_threshold, focal_length);
    
    // Use OpenCV's findEssentialMat with RANSAC on normalized coordinates
    cv::Mat mask;
    cv::Mat E_cv = cv::findEssentialMat(
        pts1, pts2,
        1.0,  // Focal length = 1 for normalized coordinates
        cv::Point2d(0, 0),  // Principal point at origin for normalized coords
        cv::RANSAC,
        m_ransac_confidence,
        ransac_threshold_normalized,
        mask
    );
    
    if (E_cv.empty()) {
        spdlog::error("[MonocularInitializer] Essential matrix estimation failed");
        return false;
    }

    int inlier_count = cv::countNonZero(mask);

    spdlog::info("[MonocularInitializer] Essential matrix inliers: {}/{}", inlier_count, pts1.size());

    if (inlier_count < m_min_inliers) {
        spdlog::error("[MonocularInitializer] Too few inliers for Essential matrix: {} (threshold: {})", 
                     inlier_count, m_min_inliers);
        return false;
    }
    
    // Convert Essential matrix to Eigen
    Eigen::Matrix3f E;
    E << E_cv.at<double>(0,0), E_cv.at<double>(0,1), E_cv.at<double>(0,2),
         E_cv.at<double>(1,0), E_cv.at<double>(1,1), E_cv.at<double>(1,2),
         E_cv.at<double>(2,0), E_cv.at<double>(2,1), E_cv.at<double>(2,2);
    
    // Extract inlier normalized points for pose recovery
    std::vector<Eigen::Vector2f> inlier_pts1_normalized;
    std::vector<Eigen::Vector2f> inlier_pts2_normalized;

    std::vector<Eigen::Vector2f> inlier_pts1_pixel_undistorted;
    std::vector<Eigen::Vector2f> inlier_pts2_pixel_undistorted;

    std::vector<int> inlier_feature_indices_1;
    std::vector<int> inlier_feature_indices_2;
    
    for (size_t i = 0; i < mask.rows; ++i) {
        if (mask.at<uchar>(i)) {
            inlier_pts1_normalized.push_back(Eigen::Vector2f(pts1[i].x, pts1[i].y));
            inlier_pts2_normalized.push_back(Eigen::Vector2f(pts2[i].x, pts2[i].y));

            cv::Point2f undistorted_pt1 = frame1->get_feature(feature_indices_1[i])->get_undistorted_coord();
            cv::Point2f undistorted_pt2 = frame2->get_feature(feature_indices_2[i])->get_undistorted_coord();

            inlier_pts1_pixel_undistorted.push_back(Eigen::Vector2f(undistorted_pt1.x, undistorted_pt1.y));
            inlier_pts2_pixel_undistorted.push_back(Eigen::Vector2f(undistorted_pt2.x, undistorted_pt2.y));

            inlier_feature_indices_1.push_back(feature_indices_1[i]);
            inlier_feature_indices_2.push_back(feature_indices_2[i]);
        }
    }
    
    // Manual pose recovery from Essential matrix with cheirality check
    // Use intrinsic matrix for pixel-level reprojection error check (ORB_SLAM3 style)
    Eigen::Matrix3f K;
    K << frame1->get_fx(), 0, frame1->get_cx(),
         0, frame1->get_fy(), frame1->get_cy(),
         0, 0, 1;
    
    float reproj_threshold_squared = 5.991f;
    
    std::vector<bool> cheirality_mask;
    std::vector<Eigen::Vector3f> points_in_cam_1;
    std::vector<int> inlier_indices;
    int num_inliers = recover_pose_from_essential(
        E, 
        inlier_pts1_pixel_undistorted, 
        inlier_pts2_pixel_undistorted,
        K,
        R_21, 
        t_21,
        cheirality_mask,
        reproj_threshold_squared,
        points_in_cam_1,
        inlier_indices
    );
    
    spdlog::info("[MonocularInitializer] Manual pose recovery: {}/{} inliers passed cheirality check", 
                 num_inliers, inlier_count);
    
    if (num_inliers < m_min_inliers) {
        spdlog::error("[MonocularInitializer] Too few inliers after pose recovery: {} (threshold: {})", 
                     num_inliers, m_min_inliers);
        return false;
    }

    Eigen::Matrix4f T_21 = Eigen::Matrix4f::Identity();
    T_21.block<3, 3>(0, 0) = R_21;
    T_21.block<3, 1>(0, 3) = t_21;

    Eigen::Matrix4f Twc_1 = frame1->get_Twc();
    Eigen::Matrix4f Twc_2 = Twc_1 * T_21.inverse();

    frame2->set_Twc(Twc_2);

    for(unsigned int i=0; i<inlier_indices.size(); ++i)
    {
        Eigen::Vector3f point_cam_1 = points_in_cam_1[i];
        Eigen::Vector3f point_world = Twc_1.block<3,3>(0,0) * point_cam_1 + Twc_1.block<3,1>(0,3);

        int feat_idx_1 = inlier_feature_indices_1[inlier_indices[i]];
        int feat_idx_2 = inlier_feature_indices_2[inlier_indices[i]];

        // make new map point(shared_ptr)
        auto mp = std::make_shared<MapPoint>(point_world);
        frame1->set_map_point(feat_idx_1, mp);
        frame2->set_map_point(feat_idx_2, mp);

        mp->add_observation(frame1, feat_idx_1);
        mp->add_observation(frame2, feat_idx_2);
        
    }

    inlier_indices_1 = inlier_feature_indices_1;
    inlier_indices_2 = inlier_feature_indices_2;

    return (static_cast<int>(inlier_indices.size()) >= m_min_inliers);
}

int MonocularInitializer::triangulate_two_views(
    const std::shared_ptr<Frame>& frame1,
    const std::shared_ptr<Frame>& frame2,
    const Eigen::Matrix3f& R_21,
    const Eigen::Vector3f& t_21,
    std::vector<Eigen::Vector3f>& points_3d,
    std::vector<int>& inlier_indices_1,
    std::vector<int>& inlier_indices_2)
{
    points_3d.clear();
    
    // Projection matrices
    // P1 = K * [I | 0]
    // P2 = K * [R_21 | t_21]
    Eigen::Matrix<float, 3, 4> P1, P2;
    P1.setZero();
    P1.block<3,3>(0,0) = Eigen::Matrix3f::Identity();
    
    P2.setZero();
    P2.block<3,3>(0,0) = R_21;
    P2.block<3,1>(0,3) = t_21;
    
    const auto& features1 = frame1->get_features();
    const auto& features2 = frame2->get_features();
    
    int num_valid = 0;
    std::vector<int> valid_inlier_indices_1;
    std::vector<int> valid_inlier_indices_2;

    spdlog::info("[MonocularInitializer] Triangulating {} inlier points...", inlier_indices_1.size());

    for(unsigned int idx = 0; idx < inlier_indices_1.size(); ++idx) {
        // idx is the index in Frame 1 (reference frame)
        const auto& feat1 = features1[inlier_indices_1[idx]];
        const auto& feat2 = features2[inlier_indices_2[idx]];
        
        // Get normalized coordinates (bearing vectors)
        Eigen::Vector2f norm1 = feat1->get_normalized_coord();
        Eigen::Vector2f norm2 = feat2->get_normalized_coord();
        
        // Convert to 3D homogeneous coordinates
        Eigen::Vector3f bearing1(norm1.x(), norm1.y(), 1.0f);
        Eigen::Vector3f bearing2(norm2.x(), norm2.y(), 1.0f);
        bearing1.normalize();
        bearing2.normalize();
        
        // DLT triangulation
        Eigen::Matrix4f A;
        A.row(0) = bearing1[0] * P1.row(2) - P1.row(0);
        A.row(1) = bearing1[1] * P1.row(2) - P1.row(1);
        A.row(2) = bearing2[0] * P2.row(2) - P2.row(0);
        A.row(3) = bearing2[1] * P2.row(2) - P2.row(1);
        
        Eigen::JacobiSVD<Eigen::Matrix4f> svd(A, Eigen::ComputeFullV);
        Eigen::Vector4f X_h = svd.matrixV().col(3);
        
        if (std::abs(X_h[3]) < 1e-6) 
        {
            // spdlog::debug("[MonocularInitializer] Triangulation failed for feature idx {}: homogeneous w ~ 0", idx);
            continue;
        }
        X_h /= X_h[3];
        Eigen::Vector3f X = X_h.head<3>();
        Eigen::Vector3f X2 = R_21 * X + t_21;

        // Check if point is in front of both cameras
        float depth1 = X[2];
        float depth2 = X2[2];
        
        if (depth1 <= 0.0f || depth2 <= 0.0f)
        { 
            // spdlog::debug("[MonocularInitializer] Triangulated point behind camera for feature idx {}", idx);
            continue;
        }
        
        // Check reprojection error
        Eigen::Vector3f proj1 = P1 * X_h;
        proj1 /= proj1[2];
        
        Eigen::Vector3f proj2 = P2 * X_h;
        proj2 /= proj2[2];
        
        // float error1 = (proj1.head<2>() - bearing1.head<2>()).norm();
        // float error2 = (proj2.head<2>() - bearing2.head<2>()).norm();
        
        // if (error1 > m_max_reprojection_error || error2 > m_max_reprojection_error) {
        //     continue;
        // }

        // Let's check reprojection error in pixel space for logging
        float fx = frame1->get_fx();
        float fy = frame1->get_fy();
        float cx = frame1->get_cx();
        float cy = frame1->get_cy();

        cv::Point2f meas1_cv = feat1->get_undistorted_coord();
        cv::Point2f meas2_cv = feat2->get_undistorted_coord();

        Eigen::Vector2f meas1(meas1_cv.x, meas1_cv.y);
        Eigen::Vector2f meas2(meas2_cv.x, meas2_cv.y);

        Eigen::Vector2f proj1_pixel;
        Eigen::Vector2f proj2_pixel;

        proj1_pixel.x() = (proj1.x() * fx) + cx;
        proj1_pixel.y() = (proj1.y() * fy) + cy;

        proj2_pixel.x() = (proj2.x() * fx) + cx;
        proj2_pixel.y() = (proj2.y() * fy) + cy;

        float reproj_error1_pixel = (proj1_pixel - meas1).norm();
        float reproj_error2_pixel = (proj2_pixel - meas2).norm();

        if(reproj_error1_pixel > m_max_reprojection_error || reproj_error2_pixel > m_max_reprojection_error) {
            // spdlog::debug("[MonocularInitializer] Triangulated point idx {} reproj error too high: frame1={:.2f}px, frame2={:.2f}px",
                        //  idx, reproj_error1_pixel, reproj_error2_pixel);
            continue;
        }


        valid_inlier_indices_1.push_back(inlier_indices_1[idx]);
        valid_inlier_indices_2.push_back(inlier_indices_2[idx]);

        points_3d.push_back(X);
        num_valid++;
    }

    inlier_indices_1 = valid_inlier_indices_1;
    inlier_indices_2 = valid_inlier_indices_2;
    
    return num_valid;
}

bool MonocularInitializer::check_cheirality(
    const Eigen::Vector3f& X,
    const Eigen::Matrix3f& R_21,
    const Eigen::Vector3f& t_21) const
{
    // Check if point has positive depth in camera 1 (depth = Z)
    if (X(2) <= 0) {
        return false;
    }
    
    // Transform point to camera 2 frame: X2 = R_21 * X + t_21
    Eigen::Vector3f X2 = R_21 * X + t_21;
    
    // Check if point has positive depth in camera 2
    if (X2(2) <= 0) {
        return false;
    }
    
    return true;
}

int MonocularInitializer::recover_pose_from_essential(
    const Eigen::Matrix3f &E,
    const std::vector<Eigen::Vector2f> &pts1_pixel,
    const std::vector<Eigen::Vector2f> &pts2_pixel,
    const Eigen::Matrix3f &K,
    Eigen::Matrix3f &R_21,
    Eigen::Vector3f &t_21,
    std::vector<bool> &mask,
    float reproj_threshold,
    std::vector<Eigen::Vector3f> &points_cam_in_1,
    std::vector<int> &inlier_feature_indices)
{
    // Extract camera parameters
    float fx = K(0, 0);
    float fy = K(1, 1);
    float cx = K(0, 2);
    float cy = K(1, 2);
    
    // SVD decomposition of Essential matrix: E = U * diag(sigma, sigma, 0) * V^T
    Eigen::JacobiSVD<Eigen::Matrix3f> svd(E, Eigen::ComputeFullU | Eigen::ComputeFullV);
    Eigen::Matrix3f U = svd.matrixU();
    Eigen::Matrix3f Vt = svd.matrixV().transpose();
    
    // Extract translation (third column of U) and normalize
    Eigen::Vector3f t = U.col(2);
    t = t / t.norm();
    
    // W matrix for extracting rotation
    Eigen::Matrix3f W;
    W.setZero();
    W(0,1) = -1;
    W(1,0) = 1;
    W(2,2) = 1;
    
    // Two possible rotations
    Eigen::Matrix3f R1 = U * W * Vt;
    if (R1.determinant() < 0) {
        R1 = -R1;
    }
    
    Eigen::Matrix3f R2 = U * W.transpose() * Vt;
    if (R2.determinant() < 0) {
        R2 = -R2;
    }
    
    // Four possible solutions:
    // (R1, +t), (R1, -t), (R2, +t), (R2, -t)
    std::vector<Eigen::Matrix3f> R_candidates(4);
    std::vector<Eigen::Vector3f> t_candidates(4);
    
    R_candidates[0] = R1;
    R_candidates[1] = R1;
    R_candidates[2] = R2;
    R_candidates[3] = R2;
    
    t_candidates[0] = t;
    t_candidates[1] = -t;
    t_candidates[2] = t;
    t_candidates[3] = -t;
    
    // Perform cheirality check for all 4 solutions
    int best_good_count = 0;
    int best_idx = -1;
    std::vector<bool> best_mask;
    
    for (int sol_idx = 0; sol_idx < 4; ++sol_idx) {

        std::vector<Eigen::Vector3f> current_points_cam_in_1;

        std::vector<int> current_inlier_feature_indices;

        const Eigen::Matrix3f& R_test = R_candidates[sol_idx];
        const Eigen::Vector3f& t_test = t_candidates[sol_idx];
        
        // Projection matrices: P1 = K[I | 0], P2 = K[R | t]
        Eigen::Matrix<float, 3, 4> P1, P2;
        P1.setZero();
        P1.block<3,3>(0,0) = K;
        
        Eigen::Matrix<float, 3, 4> Rt;
        Rt.block<3,3>(0,0) = R_test;
        Rt.block<3,1>(0,3) = t_test;
        P2 = K * Rt;
        
        int good_count = 0;
        std::vector<bool> current_mask(pts1_pixel.size(), false);
        
        // Triangulate all points and check cheirality + reprojection error
        for (size_t i = 0; i < pts1_pixel.size(); ++i) {
            // DLT triangulation: A * X = 0
            // pts1_pixel and pts2_pixel are pixel coordinates (undistorted)
            // P1 and P2 include the intrinsics K (P = K*[R|t])
            float meas1_u = pts1_pixel[i](0);
            float meas1_v = pts1_pixel[i](1);
            float meas2_u = pts2_pixel[i](0);
            float meas2_v = pts2_pixel[i](1);

            Eigen::Matrix4f A;
            Eigen::Vector3f x1_hom(meas1_u, meas1_v, 1.0f);
            Eigen::Vector3f x2_hom(meas2_u, meas2_v, 1.0f);

            A.row(0) = x1_hom(0) * P1.row(2) - x1_hom(2) * P1.row(0);
            A.row(1) = x1_hom(1) * P1.row(2) - x1_hom(2) * P1.row(1);
            A.row(2) = x2_hom(0) * P2.row(2) - x2_hom(2) * P2.row(0);
            A.row(3) = x2_hom(1) * P2.row(2) - x2_hom(2) * P2.row(1);
            
            // Solve using SVD
            Eigen::JacobiSVD<Eigen::Matrix4f> svd_triangulate(A, Eigen::ComputeFullV);
            Eigen::Vector4f X_hom = svd_triangulate.matrixV().col(3);
            
            // Convert to 3D point
            if (std::abs(X_hom(3)) < 1e-6) {
                continue;  // Point at infinity
            }
            
            Eigen::Vector3f X = X_hom.head<3>() / X_hom(3);
            
            // Check cheirality (depth > 0 in both cameras)
            if (!check_cheirality(X, R_test, t_test)) {
                continue;
            }
            
            // Camera 1: X is already in camera 1 frame
            float proj1_x = fx * X(0) / X(2) + cx;
            float proj1_y = fy * X(1) / X(2) + cy;
            
            // pts1_pixel are pixel coordinates (undistorted)
            float meas1_x = pts1_pixel[i](0);
            float meas1_y = pts1_pixel[i](1);
            
            float dx1 = proj1_x - meas1_x;
            float dy1 = proj1_y - meas1_y;
            float squared_error1 = dx1 * dx1 + dy1 * dy1;
            
            if (squared_error1 > reproj_threshold) {

                spdlog::warn("[MonocularInitializer] Point {} reproj error in camera 1 too high: {:.2f} (threshold: {:.2f})",
                             i, squared_error1, reproj_threshold);
                continue;
            }
            
            // Camera 2: Transform point to camera 2 frame

            // Transform from camera 1 to camera 2
            Eigen::Vector3f X2 = R_test * X + t_test;
            float proj2_x = fx * X2(0) / X2(2) + cx;
            float proj2_y = fy * X2(1) / X2(2) + cy;
            
            float meas2_x = pts2_pixel[i](0);
            float meas2_y = pts2_pixel[i](1);
            
            float dx2 = proj2_x - meas2_x;
            float dy2 = proj2_y - meas2_y;
            float squared_error2 = dx2 * dx2 + dy2 * dy2;
            
            if (squared_error2 > reproj_threshold) {
                spdlog::warn    ("[MonocularInitializer] Point {} reproj error in camera 2 too high: {:.2f} (threshold: {:.2f})",
                             i, squared_error2, reproj_threshold);
                continue;
            }

            current_points_cam_in_1.push_back(X);
            current_inlier_feature_indices.push_back(i);

        
            // Point passes all checks
            good_count++;
            current_mask[i] = true;
        }
        
        // Keep track of best solution
        if (good_count > best_good_count) {
            best_good_count = good_count;
            best_idx = sol_idx;
            best_mask = current_mask;

            points_cam_in_1 = current_points_cam_in_1;
            inlier_feature_indices = current_inlier_feature_indices;
        }
    }
    
    if (best_idx < 0) {
        spdlog::error("[MonocularInitializer] No valid pose solution found");
        return 0;
    }
    
    // Normalize depth: compute median depth and scale points and translation
    if (!points_cam_in_1.empty()) {
        std::vector<float> depths;
        depths.reserve(points_cam_in_1.size());
        for (const auto& pt : points_cam_in_1) {
            depths.push_back(pt(2));  // Z coordinate is depth
        }
        
        // Compute median depth
        std::sort(depths.begin(), depths.end());
        float median_depth;
        size_t mid = depths.size() / 2;
        if (depths.size() % 2 == 0) {
            median_depth = (depths[mid - 1] + depths[mid]) / 2.0f;
        } else {
            median_depth = depths[mid];
        }
        
        // Scale factor to normalize median depth to 1.0
        float scale_factor = 1.0f / median_depth;
        
        spdlog::info("[MonocularInitializer] Normalizing depth: median_depth={:.3f}, scale_factor={:.3f}", 
                     median_depth, scale_factor);
        
        // Scale all 3D points
        for (auto& pt : points_cam_in_1) {
            pt *= scale_factor;
        }
        
        // Scale translation vector
        t_candidates[best_idx] *= scale_factor;
    }
    
    // Set output
    R_21 = R_candidates[best_idx];
    t_21 = t_candidates[best_idx];
    mask = best_mask;
    
    spdlog::info("[MonocularInitializer] Manual pose recovery: solution {} selected with {} inliers", best_idx, best_good_count);
    
    return best_good_count;
}

} // namespace lightweight_vio
