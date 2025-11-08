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
#include "../database/Feature.h"
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
    m_required_keyframes = config.m_init_required_keyframes;
    m_min_parallax_degrees = config.m_init_min_parallax_degrees;
    m_min_average_parallax = config.m_init_min_average_parallax;
    
    // Two-view geometry parameters
    m_ransac_threshold = config.m_init_ransac_threshold;
    m_ransac_confidence = config.m_init_ransac_confidence;
    m_ransac_max_iterations = config.m_init_ransac_max_iterations;
    
    // Triangulation parameters
    m_min_triangulation_angle = config.m_init_min_triangulation_angle;
    m_max_reprojection_error = config.m_init_max_reprojection_error;
    
    spdlog::info("[MonocularInitializer] Created with parameters:");
    spdlog::info("  Required keyframes: {}", m_required_keyframes);
    spdlog::info("  Min parallax: {:.1f} degrees", m_min_parallax_degrees);
    spdlog::info("  Min average parallax: {:.1f} degrees", m_min_average_parallax);
}

bool MonocularInitializer::add_frame(std::shared_ptr<Frame> frame) {
    if (!frame) {
        spdlog::warn("[MonocularInitializer] Null frame provided");
        return false;
    }
    
    // Check if this is the first frame
    if (m_candidate_frames.empty()) {
        m_candidate_frames.push_back(frame);
        spdlog::info("[MonocularInitializer] Added first candidate frame {}", frame->get_frame_id());
        return true;
    }
    
    // Check parallax with last candidate frame
    auto last_frame = m_candidate_frames.back();
    ParallaxInfo parallax_info = compute_parallax(last_frame, frame);
    
    spdlog::debug("[MonocularInitializer] Frame {} parallax with last candidate: avg={:.2f}°, median={:.2f}°, tracked={}",
                 frame->get_frame_id(), 
                 parallax_info.average_parallax,
                 parallax_info.median_parallax,
                 parallax_info.num_tracked_features);
    
    // Only add if parallax is sufficient
    if (parallax_info.average_parallax >= m_min_average_parallax) {
        m_candidate_frames.push_back(frame);
        spdlog::info("[MonocularInitializer] Added candidate frame {} ({}/{} collected, avg_parallax={:.2f}°)",
                    frame->get_frame_id(),
                    m_candidate_frames.size(),
                    m_required_keyframes,
                    parallax_info.average_parallax);
        return true;
    } else {
        spdlog::debug("[MonocularInitializer] Frame {} rejected: insufficient parallax ({:.2f}° < {:.2f}°)",
                     frame->get_frame_id(),
                     parallax_info.average_parallax,
                     m_min_average_parallax);
        return false;
    }
}

bool MonocularInitializer::has_sufficient_frames() const {
    return m_candidate_frames.size() >= static_cast<size_t>(m_required_keyframes);
}

void MonocularInitializer::reset() {
    m_candidate_frames.clear();
    m_is_initialized = false;
    spdlog::info("[MonocularInitializer] Reset");
}

MonocularInitializer::InitializationResult MonocularInitializer::try_initialize() {
    InitializationResult result;
    
    if (!has_sufficient_frames()) {
        result.failure_reason = "Insufficient candidate frames";
        spdlog::warn("[MonocularInitializer] Cannot initialize: only {}/{} frames collected",
                    m_candidate_frames.size(), m_required_keyframes);
        return result;
    }
    
    spdlog::info("[MonocularInitializer] Starting initialization with {} candidate frames",
                m_candidate_frames.size());
    
    // Stage 1: Consecutive SFM
    if (!consecutive_sfm(result)) {
        spdlog::error("[MonocularInitializer] Stage 1 (Consecutive SFM) failed: {}",
                     result.failure_reason);
        return result;
    }
    
    spdlog::info("[MonocularInitializer] ✓ Stage 1 (Consecutive SFM) succeeded:");
    spdlog::info("  - Initialized {} keyframes", result.initialized_keyframes.size());
    spdlog::info("  - Triangulated {} 3D points", result.num_triangulated_points);
    
    // TODO: Stage 2 - Gravity Initialization
    // TODO: Stage 3 - IMU Initialization (if VIO mode)
    
    result.success = true;
    m_is_initialized = true;
    
    return result;
}

// ============================================================================
// Stage 1: Consecutive SFM
// ============================================================================

bool MonocularInitializer::consecutive_sfm(InitializationResult& result) {
    if (m_candidate_frames.size() < 2) {
        result.failure_reason = "Need at least 2 frames for SFM";
        return false;
    }
    
    // Step 1: Initialize first two frames using two-view geometry
    auto frame1 = m_candidate_frames[0];
    auto frame2 = m_candidate_frames[1];
    
    Eigen::Matrix3f R_21;
    Eigen::Vector3f t_21;
    std::vector<Eigen::Vector3f> points_3d;
    std::vector<int> inlier_indices;
    
    if (!initialize_two_views(frame1, frame2, R_21, t_21, points_3d, inlier_indices)) {
        result.failure_reason = "Two-view initialization failed";
        return false;
    }
    
    spdlog::info("[MonocularInitializer] Two-view initialization succeeded:");
    spdlog::info("  - Frames: {} → {}", frame1->get_frame_id(), frame2->get_frame_id());
    spdlog::info("  - Triangulated points: {}", points_3d.size());
    
    // Set poses for first two frames (frame1 at origin, frame2 relative)
    frame1->set_Twb(Eigen::Matrix4f::Identity());
    
    Eigen::Matrix4f T_21 = Eigen::Matrix4f::Identity();
    T_21.block<3,3>(0,0) = R_21;
    T_21.block<3,1>(0,3) = t_21;
    frame2->set_Twb(T_21);
    
    // Create MapPoints from triangulated points
    std::vector<std::shared_ptr<MapPoint>> mappoints;
    mappoints.reserve(points_3d.size());
    
    for (size_t i = 0; i < points_3d.size(); ++i) {
        auto mappoint = std::make_shared<MapPoint>(points_3d[i]);
        mappoints.push_back(mappoint);
        
        // Associate with features in both frames
        int feature_idx1 = inlier_indices[i];
        // Note: Need to find corresponding feature in frame2
        // This is simplified - in practice need proper feature association
    }
    
    result.initialized_keyframes.push_back(frame1);
    result.initialized_keyframes.push_back(frame2);
    result.initialized_mappoints = mappoints;
    result.num_triangulated_points = points_3d.size();
    
    // TODO: Step 2: Process remaining frames (if any)
    // For each additional frame:
    //   - Use PnP to estimate pose
    //   - Triangulate new points
    //   - Add to result
    
    return true;
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
    std::vector<double> parallax_angles;
    parallax_angles.reserve(features1.size());
    
    for (const auto& feat1 : features1) {
        if (!feat1->is_valid() || !feat1->has_tracked_feature()) {
            continue;
        }
        
        // Find corresponding feature in frame2
        int tracked_id = feat1->get_tracked_feature_id();
        auto feat2 = frame2->get_feature(tracked_id);
        
        if (!feat2 || !feat2->is_valid()) {
            continue;
        }
        
        // Compute parallax angle using normalized coordinates (bearing vectors)
        Eigen::Vector2f norm1 = feat1->get_normalized_coord();
        Eigen::Vector2f norm2 = feat2->get_normalized_coord();
        
        // Convert to 3D bearing vectors
        Eigen::Vector3f bearing1(norm1.x(), norm1.y(), 1.0f);
        Eigen::Vector3f bearing2(norm2.x(), norm2.y(), 1.0f);
        bearing1.normalize();
        bearing2.normalize();
        
        // Compute angle between bearing vectors
        double cos_angle = bearing1.dot(bearing2);
        cos_angle = std::max(-1.0, std::min(1.0, static_cast<double>(cos_angle))); // Clamp to [-1,1]
        double angle_rad = std::acos(cos_angle);
        double angle_deg = angle_rad * 180.0 / M_PI;
        
        parallax_angles.push_back(angle_deg);
    }
    
    if (parallax_angles.empty()) {
        return info;
    }
    
    // Compute statistics
    info.num_tracked_features = parallax_angles.size();
    info.average_parallax = std::accumulate(parallax_angles.begin(), parallax_angles.end(), 0.0) 
                           / parallax_angles.size();
    
    // Compute median
    std::vector<double> sorted_angles = parallax_angles;
    std::sort(sorted_angles.begin(), sorted_angles.end());
    size_t mid = sorted_angles.size() / 2;
    if (sorted_angles.size() % 2 == 0) {
        info.median_parallax = (sorted_angles[mid-1] + sorted_angles[mid]) / 2.0;
    } else {
        info.median_parallax = sorted_angles[mid];
    }
    
    info.sufficient_parallax = (info.average_parallax >= m_min_average_parallax);
    
    return info;
}

bool MonocularInitializer::initialize_two_views(
    const std::shared_ptr<Frame>& frame1,
    const std::shared_ptr<Frame>& frame2,
    Eigen::Matrix3f& R_21,
    Eigen::Vector3f& t_21,
    std::vector<Eigen::Vector3f>& points_3d,
    std::vector<int>& inlier_indices)
{
    // Find matched features between two frames
    std::vector<cv::Point2f> pts1, pts2;
    std::vector<int> feature_indices;
    
    const auto& features1 = frame1->get_features();
    const auto& features2 = frame2->get_features();
    
    for (size_t i = 0; i < features1.size(); ++i) {
        const auto& feat1 = features1[i];
        if (!feat1->is_valid() || !feat1->has_tracked_feature()) {
            continue;
        }
        
        int tracked_id = feat1->get_tracked_feature_id();
        auto feat2 = frame2->get_feature(tracked_id);
        
        if (!feat2 || !feat2->is_valid()) {
            continue;
        }
        
        // Use undistorted coordinates for two-view geometry
        pts1.push_back(feat1->get_undistorted_coord());
        pts2.push_back(feat2->get_undistorted_coord());
        feature_indices.push_back(i);
    }
    
    if (pts1.size() < 8) {
        spdlog::error("[MonocularInitializer] Insufficient matches: {} (need >= 8)", pts1.size());
        return false;
    }
    
    spdlog::info("[MonocularInitializer] Found {} matched features for two-view geometry", pts1.size());
    
    // Get camera intrinsics
    cv::Mat K = cv::Mat::eye(3, 3, CV_64F);
    K.at<double>(0, 0) = frame1->get_fx();
    K.at<double>(1, 1) = frame1->get_fy();
    K.at<double>(0, 2) = frame1->get_cx();
    K.at<double>(1, 2) = frame1->get_cy();
    
    // Compute Essential matrix using RANSAC
    cv::Mat mask;
    cv::Mat E = cv::findEssentialMat(
        pts1, pts2, K,
        cv::RANSAC,
        m_ransac_confidence,
        m_ransac_threshold,
        mask
    );
    
    if (E.empty()) {
        spdlog::error("[MonocularInitializer] Essential matrix estimation failed");
        return false;
    }
    
    // Recover pose from Essential matrix
    cv::Mat R_cv, t_cv;
    int num_inliers = cv::recoverPose(E, pts1, pts2, K, R_cv, t_cv, mask);
    
    if (num_inliers < 50) {
        spdlog::error("[MonocularInitializer] Too few inliers after pose recovery: {}", num_inliers);
        return false;
    }
    
    spdlog::info("[MonocularInitializer] Pose recovery: {}/{} inliers", num_inliers, pts1.size());
    
    // Convert to Eigen
    R_21 << R_cv.at<double>(0,0), R_cv.at<double>(0,1), R_cv.at<double>(0,2),
            R_cv.at<double>(1,0), R_cv.at<double>(1,1), R_cv.at<double>(1,2),
            R_cv.at<double>(2,0), R_cv.at<double>(2,1), R_cv.at<double>(2,2);
    
    t_21 << t_cv.at<double>(0), t_cv.at<double>(1), t_cv.at<double>(2);
    
    // Triangulate inlier points
    inlier_indices.clear();
    for (size_t i = 0; i < mask.rows; ++i) {
        if (mask.at<uchar>(i)) {
            inlier_indices.push_back(feature_indices[i]);
        }
    }
    
    int num_triangulated = triangulate_two_views(frame1, frame2, R_21, t_21, points_3d, inlier_indices);
    
    spdlog::info("[MonocularInitializer] Triangulation: {}/{} successful", 
                num_triangulated, inlier_indices.size());
    
    return (num_triangulated >= 50);
}

int MonocularInitializer::triangulate_two_views(
    const std::shared_ptr<Frame>& frame1,
    const std::shared_ptr<Frame>& frame2,
    const Eigen::Matrix3f& R_21,
    const Eigen::Vector3f& t_21,
    std::vector<Eigen::Vector3f>& points_3d,
    std::vector<int>& inlier_indices)
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
    std::vector<int> valid_inlier_indices;
    
    for (int idx : inlier_indices) {
        const auto& feat1 = features1[idx];
        
        // Find corresponding feature in frame2
        if (!feat1->has_tracked_feature()) continue;
        
        int tracked_id = feat1->get_tracked_feature_id();
        auto feat2 = frame2->get_feature(tracked_id);
        if (!feat2 || !feat2->is_valid()) continue;
        
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
        
        if (std::abs(X_h[3]) < 1e-6) continue;
        
        X_h /= X_h[3];
        Eigen::Vector3f X = X_h.head<3>();
        
        // Check if point is in front of both cameras
        float depth1 = X[2];
        Eigen::Vector3f X2 = R_21 * X + t_21;
        float depth2 = X2[2];
        
        if (depth1 <= 0.0f || depth2 <= 0.0f) continue;
        
        // Check reprojection error
        Eigen::Vector3f proj1 = P1 * X_h;
        proj1 /= proj1[2];
        
        Eigen::Vector3f proj2 = P2 * X_h;
        proj2 /= proj2[2];
        
        float error1 = (proj1.head<2>() - bearing1.head<2>()).norm();
        float error2 = (proj2.head<2>() - bearing2.head<2>()).norm();
        
        if (error1 > m_max_reprojection_error || error2 > m_max_reprojection_error) {
            continue;
        }
        
        points_3d.push_back(X);
        valid_inlier_indices.push_back(idx);
        num_valid++;
    }
    
    inlier_indices = valid_inlier_indices;
    
    return num_valid;
}

bool MonocularInitializer::recover_pose_pnp(
    const std::shared_ptr<Frame>& frame,
    const std::vector<std::shared_ptr<MapPoint>>& mappoints,
    Eigen::Matrix3f& R_new,
    Eigen::Vector3f& t_new)
{
    // TODO: Implement PnP for additional frames
    spdlog::warn("[MonocularInitializer] PnP not yet implemented");
    return false;
}

int MonocularInitializer::triangulate_new_points(
    const std::shared_ptr<Frame>& existing_frame,
    const std::shared_ptr<Frame>& new_frame,
    std::vector<std::shared_ptr<MapPoint>>& existing_mappoints)
{
    // TODO: Implement new point triangulation
    spdlog::warn("[MonocularInitializer] New point triangulation not yet implemented");
    return 0;
}

} // namespace lightweight_vio
