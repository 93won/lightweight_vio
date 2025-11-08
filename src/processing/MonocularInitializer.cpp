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
    , m_reference_frame(nullptr)
{
    const Config& config = Config::getInstance();
    
    // Load initialization parameters from config
    m_min_parallax_pixels = config.m_init_min_parallax_pixels;
    
    // Two-view geometry parameters
    m_ransac_threshold = config.m_init_ransac_threshold;
    m_ransac_confidence = config.m_init_ransac_confidence;
    m_ransac_max_iterations = config.m_init_ransac_max_iterations;
    
    // Triangulation parameters
    m_min_triangulation_angle = config.m_init_min_triangulation_angle;
    m_max_reprojection_error = config.m_init_max_reprojection_error;
    
    spdlog::info("[MonocularInitializer] Created with parameters:");
    spdlog::info("  Min parallax: {:.1f} pixels (median)", m_min_parallax_pixels);
}

bool MonocularInitializer::add_frame(std::shared_ptr<Frame> frame) {
    if (!frame) {
        spdlog::warn("[MonocularInitializer] Null frame provided");
        return false;
    }
    
    // Already initialized - no more frames needed
    if (m_is_initialized) {
        return false;
    }
    
    // First frame - keep as reference
    if (!m_reference_frame) {
        m_reference_frame = frame;
        spdlog::info("[MONO_INIT] Frame {}: Set as reference frame", frame->get_frame_id());
        return true;
    }
    
    // Check parallax with reference frame
    ParallaxInfo parallax_info = compute_parallax(m_reference_frame, frame);
    
    spdlog::debug("[MONO_INIT] Frame {} parallax with reference {}: median={:.2f}px, avg={:.2f}px, tracked={}",
                 frame->get_frame_id(), 
                 m_reference_frame->get_frame_id(),
                 parallax_info.median_parallax_pixels,
                 parallax_info.average_parallax_pixels,
                 parallax_info.num_tracked_features);
    
    // Sufficient parallax - attempt initialization
    if (parallax_info.median_parallax_pixels >= m_min_parallax_pixels) {
        spdlog::info("[MONO_INIT] Sufficient parallax ({:.2f}px >= {:.2f}px), attempting initialization...",
                    parallax_info.median_parallax_pixels, m_min_parallax_pixels);
        
        // Attempt two-view initialization
        Eigen::Matrix3f R_21;
        Eigen::Vector3f t_21;
        std::vector<Eigen::Vector3f> points_3d;
        std::vector<int> inlier_indices;
        
        if (initialize_two_views(m_reference_frame, frame, R_21, t_21, points_3d, inlier_indices)) {
            // Success - store result
            // Set camera poses (Twc = camera-to-world transform)
            m_reference_frame->set_Twc(Eigen::Matrix4f::Identity());
            
            Eigen::Matrix4f T_21 = Eigen::Matrix4f::Identity();
            T_21.block<3,3>(0,0) = R_21;
            T_21.block<3,1>(0,3) = t_21;
            frame->set_Twc(T_21);
            
            // Create MapPoints
            m_result.initialized_keyframes.clear();
            m_result.initialized_keyframes.push_back(m_reference_frame);
            m_result.initialized_keyframes.push_back(frame);
            
            m_result.initialized_mappoints.clear();
            m_result.initialized_mappoints.reserve(points_3d.size());
            for (const auto& pt : points_3d) {
                auto mappoint = std::make_shared<MapPoint>(pt);
                m_result.initialized_mappoints.push_back(mappoint);
            }
            
            m_result.num_triangulated_points = points_3d.size();
            m_result.success = true;
            m_is_initialized = true;
            
            spdlog::info("[MONO_INIT] ✅ Initialization successful!");
            spdlog::info("  - Frame {} → {}", m_reference_frame->get_frame_id(), frame->get_frame_id());
            spdlog::info("  - Map points created: {}", m_result.initialized_mappoints.size());
            return true;
        } else {
            // Initialization failed - discard reference, keep current as new reference
            spdlog::warn("[MONO_INIT] ❌ Two-view initialization failed");
            spdlog::info("[MONO_INIT] Discarding frame {}, using frame {} as new reference",
                        m_reference_frame->get_frame_id(), frame->get_frame_id());
            m_reference_frame = frame;
            return true;
        }
    } else {
        // Insufficient parallax - discard reference, keep current as new reference
        spdlog::debug("[MONO_INIT] Insufficient parallax ({:.2f}px < {:.2f}px)",
                     parallax_info.median_parallax_pixels, m_min_parallax_pixels);
        spdlog::debug("[MONO_INIT] Discarding frame {}, using frame {} as new reference",
                     m_reference_frame->get_frame_id(), frame->get_frame_id());
        m_reference_frame = frame;
        return true;
    }
}

void MonocularInitializer::reset() {
    m_reference_frame.reset();
    m_is_initialized = false;
    m_result = InitializationResult();
    spdlog::info("[MonocularInitializer] Reset");
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
    pixel_displacements.reserve(features1.size());
    
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
        
        // Compute pixel displacement (use undistorted coordinates)
        cv::Point2f pt1 = feat1->get_undistorted_coord();
        cv::Point2f pt2 = feat2->get_undistorted_coord();
        
        double dx = pt2.x - pt1.x;
        double dy = pt2.y - pt1.y;
        double displacement = std::sqrt(dx * dx + dy * dy);
        
        pixel_displacements.push_back(displacement);
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

} // namespace lightweight_vio
