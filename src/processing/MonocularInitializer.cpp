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

bool MonocularInitializer::add_frame(std::shared_ptr<Frame> frame, bool* init_attempted) {
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
    
    // First frame - keep as reference
    if (!m_reference_frame) {
        m_reference_frame = frame;
        return true;
    }
    
    // Check parallax with reference frame
    ParallaxInfo parallax_info = compute_parallax(m_reference_frame, frame);
    
    
    // Sufficient parallax - attempt initialization
    if (parallax_info.median_parallax_pixels >= m_min_parallax_pixels) {
        
        // Set flag indicating initialization was attempted
        if (init_attempted) {
            *init_attempted = true;
        }
        
        // Attempt two-view initialization
        Eigen::Matrix3f R_21;
        Eigen::Vector3f t_21;
        std::vector<Eigen::Vector3f> points_3d;
        std::vector<int> inlier_indices;
        
        if (initialize_two_views(m_reference_frame, frame, R_21, t_21, points_3d, inlier_indices)) {
            // Success - store result
            // Set camera poses (Twc = camera-to-world transform)
            // Frame1 (reference): Set as world origin
            
            // Compute median depth for scale normalization
            std::vector<float> depths;
            depths.reserve(points_3d.size());
            for (size_t i = 0; i < points_3d.size(); ++i) {
                depths.push_back(points_3d[i].z());
            }
            std::sort(depths.begin(), depths.end());
            float median_depth = depths[depths.size() / 2];
            
            spdlog::info("[MONO_INIT] Median depth before normalization: {:.3f}", median_depth);
            
            // Scale t_21 and all 3D points to make median depth = 1.0
            float scale = 1.0f / median_depth;
            t_21 *= scale;
            
            for (auto& pt : points_3d) {
                pt *= scale;
            }
            
            spdlog::info("[MONO_INIT] Applied scale {:.3f} to normalize median depth to 1.0", scale);
            
            // Frame2: Transform from frame1 to frame2 (in frame1's coordinate system)
            // T_21 = [R_21 | t_21] where:
            //   R_21: Rotation from cam1 to cam2
            //   t_21: Translation from cam1 to cam2 (SCALED to make median depth = 1.0)
            // X_cam2 = R_21 * X_cam1 + t_21
            Eigen::Matrix4f T_21 = Eigen::Matrix4f::Identity();
            T_21.block<3,3>(0,0) = R_21;
            T_21.block<3,1>(0,3) = t_21;

     

            Eigen::Matrix4f Twc_1 = m_reference_frame->get_Twc();
            Eigen::Matrix4f Tc1_w = Twc_1.inverse();
            Eigen::Matrix4f Tc2_w = T_21 * Tc1_w;
            Eigen::Matrix4f Twc_2 = Tc2_w.inverse();

            frame->set_Twc(Twc_2);
            
            // Create MapPoints and associate with features
            m_result.initialized_keyframes.clear();
            m_result.initialized_keyframes.push_back(m_reference_frame);
            m_result.initialized_keyframes.push_back(frame);
            
            m_result.initialized_mappoints.clear();
            m_result.initialized_mappoints.reserve(points_3d.size());
            
            // inlier_indices contains Frame 2's feature indices
            for (size_t i = 0; i < points_3d.size(); ++i) {
                int feat2_idx = inlier_indices[i];
                
                // Create MapPoint in world frame
                // Transform point from camera1 frame to world frame
                Eigen::Vector4f point_cam1(points_3d[i].x(), points_3d[i].y(), points_3d[i].z(), 1.0f);
                Eigen::Vector4f point_world = Twc_1 * point_cam1;
                Eigen::Vector3f world_pos = point_world.head<3>();
                
                auto mappoint = std::make_shared<MapPoint>(world_pos);
                
                // Get feature from frame2 and find corresponding feature in frame1
                auto feat2 = frame->get_features()[feat2_idx];
                if (!feat2->has_tracked_feature()) continue;
                
                int tracked_id = feat2->get_tracked_feature_id();
                auto feat1 = m_reference_frame->get_feature(tracked_id);
                if (!feat1) continue;
                
                // Add observations (frame, feature_index)
                mappoint->add_observation(m_reference_frame, tracked_id);
                mappoint->add_observation(frame, feat2_idx);
                
                // Associate map point with features in both frames
                m_reference_frame->set_map_point(tracked_id, mappoint);
                frame->set_map_point(feat2_idx, mappoint);
                
                m_result.initialized_mappoints.push_back(mappoint);
            }
            
            m_result.num_triangulated_points = m_result.initialized_mappoints.size();
            m_result.success = true;
            m_is_initialized = true;
            
            spdlog::info("[MONO_INIT] ✅ Initialization successful!");
            spdlog::info("  - Frame {} → {}", m_reference_frame->get_frame_id(), frame->get_frame_id());
            spdlog::info("  - Map points created: {}", m_result.initialized_mappoints.size());
            return true;
        } else {
            // Initialization failed - Estimator will reset reference frame
            spdlog::warn("[MONO_INIT] ❌ Two-view initialization failed (parallax sufficient but inliers insufficient)");
            return false;
        }
    } else {
        return false;  // Frame not used for initialization
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
    
    
    // === VISUAL DEBUG: Show parallax computation (DISABLED) ===
    // if (!matched_pts1.empty() && !frame1->get_image().empty() && !frame2->get_image().empty()) {
    //     cv::Mat debug_img;
    //     cv::cvtColor(frame2->get_image(), debug_img, cv::COLOR_GRAY2BGR);
    //     
    //     for (size_t i = 0; i < matched_pts1.size(); ++i) {
    //         // Draw arrow showing displacement
    //         cv::arrowedLine(debug_img, matched_pts1[i], matched_pts2[i], 
    //                       cv::Scalar(255, 0, 0), 1, cv::LINE_AA, 0, 0.3);
    //         cv::circle(debug_img, matched_pts2[i], 2, cv::Scalar(255, 0, 0), -1);
    //         
    //         // Highlight if displacement > 0
    //         double dx = matched_pts2[i].x - matched_pts1[i].x;
    //         double dy = matched_pts2[i].y - matched_pts1[i].y;
    //         double disp = std::sqrt(dx*dx + dy*dy);
    //         if (disp > 0.5) {
    //             cv::circle(debug_img, matched_pts2[i], 5, cv::Scalar(0, 255, 0), 1);
    //         }
    //     }
    //     
    //     std::string info = cv::format("Parallax F%d->F%d: matched=%d, median=%.2fpx", 
    //                                   frame1->get_frame_id(), frame2->get_frame_id(),
    //                                   (int)matched_pts1.size(), 
    //                                   pixel_displacements.empty() ? 0.0 : pixel_displacements[pixel_displacements.size()/2]);
    //     cv::putText(debug_img, info, cv::Point(10, 30), cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(0, 255, 255), 2);
    //     
    //     cv::imshow("Parallax Computation", debug_img);
    //     cv::waitKey(1);
    // }
    // === END VISUAL DEBUG ===
    
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
    // NOTE: Iterate frame2 features (current frame) and find correspondences in frame1 (reference)
    std::vector<cv::Point2f> pts1, pts2;
    std::vector<int> feature_indices;
    
    const auto& features2 = frame2->get_features();
    
    for (size_t i = 0; i < features2.size(); ++i) {
        const auto& feat2 = features2[i];
        if (!feat2->is_valid() || !feat2->has_tracked_feature()) {
            continue;
        }
        
        int tracked_id = feat2->get_tracked_feature_id();
        auto feat1 = frame1->get_feature(tracked_id);
        
        if (!feat1 || !feat1->is_valid()) {
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

    spdlog::debug("[MonocularInitializer] Estimating Essential matrix with RANSAC: threshold={}, confidence={}",
                 m_ransac_threshold, m_ransac_confidence);
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

    // I want to check num of inliers here
    int inlier_count = cv::countNonZero(mask);
    spdlog::info("[MonocularInitializer] Essential matrix inliers: {}/{}", inlier_count, pts1.size());
    if (inlier_count < 50) {
        spdlog::error("[MonocularInitializer] Too few inliers for Essential matrix: {}", inlier_count);
        return false;
    }
    
    // Manually recover pose from Essential matrix
    // R_cv, t_cv: Rotation and translation from camera1 to camera2
    // X_cam2 = R * X_cam1 + t
    // Note: t is a UNIT VECTOR (normalized), so scale is ambiguous
    
    // SVD decomposition of Essential matrix: E = U * Diag(1,1,0) * Vt
    cv::Mat U, S, Vt;
    cv::SVD::compute(E, S, U, Vt);
    
    // Check determinants (should be +1)
    if (cv::determinant(U) < 0) U *= -1;
    if (cv::determinant(Vt) < 0) Vt *= -1;
    
    // W matrix for rotation extraction
    cv::Mat W = (cv::Mat_<double>(3,3) << 0, -1, 0,
                                           1,  0, 0,
                                           0,  0, 1);
    
    // Two possible rotations: R1 = U*W*Vt, R2 = U*W'*Vt
    cv::Mat R1 = U * W * Vt;
    cv::Mat R2 = U * W.t() * Vt;
    
    // Translation (up to scale): t = u3 (third column of U)
    cv::Mat t = U.col(2);
    
    // Four possible solutions: (R1,t), (R1,-t), (R2,t), (R2,-t)
    std::vector<cv::Mat> R_solutions = {R1, R1, R2, R2};
    std::vector<cv::Mat> t_solutions = {t, -t, t, -t};
    
    int best_num_inliers = 0;
    int best_solution_idx = -1;
    cv::Mat R_cv, t_cv;
    
    spdlog::debug("[MonocularInitializer] Testing 4 pose solutions...");
    
    for (int sol = 0; sol < 4; ++sol) {
        cv::Mat R_test = R_solutions[sol];
        cv::Mat t_test = t_solutions[sol];
        
        // Check how many points are in front of both cameras
        int num_good = 0;
        
        // Projection matrices: P1 = K*[I|0], P2 = K*[R|t]
        cv::Mat P1 = K * (cv::Mat_<double>(3,4) << 1,0,0,0, 0,1,0,0, 0,0,1,0);
        cv::Mat P2_temp = (cv::Mat_<double>(3,4) << 
            R_test.at<double>(0,0), R_test.at<double>(0,1), R_test.at<double>(0,2), t_test.at<double>(0),
            R_test.at<double>(1,0), R_test.at<double>(1,1), R_test.at<double>(1,2), t_test.at<double>(1),
            R_test.at<double>(2,0), R_test.at<double>(2,1), R_test.at<double>(2,2), t_test.at<double>(2));
        cv::Mat P2 = K * P2_temp;
        
        for (size_t i = 0; i < mask.rows; ++i) {
            if (!mask.at<uchar>(i)) continue;  // Only check RANSAC inliers
            
            // Triangulate point
            cv::Mat pt1_h = (cv::Mat_<double>(3,1) << pts1[i].x, pts1[i].y, 1.0);
            cv::Mat pt2_h = (cv::Mat_<double>(3,1) << pts2[i].x, pts2[i].y, 1.0);
            
            // Build linear system: A * X = 0
            cv::Mat A(4, 4, CV_64F);
            A.row(0) = pt1_h.at<double>(0) * P1.row(2) - P1.row(0);
            A.row(1) = pt1_h.at<double>(1) * P1.row(2) - P1.row(1);
            A.row(2) = pt2_h.at<double>(0) * P2.row(2) - P2.row(0);
            A.row(3) = pt2_h.at<double>(1) * P2.row(2) - P2.row(1);
            
            cv::Mat u, w, vt;
            cv::SVD::compute(A, w, u, vt);
            cv::Mat X = vt.row(3).t();
            X /= X.at<double>(3);  // Normalize homogeneous coordinate
            
            // Check depth in camera 1
            double z1 = X.at<double>(2);
            
            // Check depth in camera 2: z2 = R.row(2) * X + t(2)
            cv::Mat X_cam2 = R_test * X.rowRange(0,3) + t_test;
            double z2 = X_cam2.at<double>(2);
            
            // Both depths should be positive
            if (z1 > 0 && z2 > 0) {
                num_good++;
            }
        }
        
        if (num_good > best_num_inliers) {
            best_num_inliers = num_good;
            best_solution_idx = sol;
            R_cv = R_test.clone();
            t_cv = t_test.clone();
        }
    }
    
    // Update mask with best solution - mark points that pass cheirality check
    cv::Mat R_best = R_solutions[best_solution_idx];
    cv::Mat t_best = t_solutions[best_solution_idx];
    cv::Mat P1 = K * (cv::Mat_<double>(3,4) << 1,0,0,0, 0,1,0,0, 0,0,1,0);
    cv::Mat P2_temp = (cv::Mat_<double>(3,4) << 
        R_best.at<double>(0,0), R_best.at<double>(0,1), R_best.at<double>(0,2), t_best.at<double>(0),
        R_best.at<double>(1,0), R_best.at<double>(1,1), R_best.at<double>(1,2), t_best.at<double>(1),
        R_best.at<double>(2,0), R_best.at<double>(2,1), R_best.at<double>(2,2), t_best.at<double>(2));
    cv::Mat P2 = K * P2_temp;



    
    for (size_t i = 0; i < mask.rows; ++i) {
        if (!mask.at<uchar>(i)) continue;
        
        // Triangulate point
        cv::Mat pt1_h = (cv::Mat_<double>(3,1) << pts1[i].x, pts1[i].y, 1.0);
        cv::Mat pt2_h = (cv::Mat_<double>(3,1) << pts2[i].x, pts2[i].y, 1.0);
        
        cv::Mat A(4, 4, CV_64F);
        A.row(0) = pt1_h.at<double>(0) * P1.row(2) - P1.row(0);
        A.row(1) = pt1_h.at<double>(1) * P1.row(2) - P1.row(1);
        A.row(2) = pt2_h.at<double>(0) * P2.row(2) - P2.row(0);
        A.row(3) = pt2_h.at<double>(1) * P2.row(2) - P2.row(1);
        
        cv::Mat u, w, vt;
        cv::SVD::compute(A, w, u, vt);
        cv::Mat X = vt.row(3).t();
        X /= X.at<double>(3);
        
        double z1 = X.at<double>(2);
        cv::Mat X_cam2 = R_best * X.rowRange(0,3) + t_best;
        double z2 = X_cam2.at<double>(2);
        
        // Update mask: only keep points in front of both cameras
        if (z1 <= 0 || z2 <= 0) {
            mask.at<uchar>(i) = 0;
        }
    }
    
    int num_inliers = best_num_inliers;
    
    if (num_inliers < 50) {
        spdlog::error("[MonocularInitializer] Too few inliers after pose recovery: {}", num_inliers);
        return false;
    }
    
    spdlog::info("[MonocularInitializer] Pose recovery: {}/{} inliers (best solution)", num_inliers, inlier_count);
    
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

    spdlog::info("[MonocularInitializer] Triangulating {} inlier points...", inlier_indices.size());
    
    for (int idx : inlier_indices) {
        // idx is the index in Frame 2 (current frame)
        const auto& feat2 = features2[idx];
        
        // Find corresponding feature in frame1 using tracked_feature_id
        if (!feat2->has_tracked_feature()) 
        {
            spdlog::debug("[MonocularInitializer] Feature idx {} has no tracked_feature_id", idx);
            continue;
        }
        int tracked_id = feat2->get_tracked_feature_id();
        auto feat1 = frame1->get_feature(tracked_id);
        if (!feat1 || !feat1->is_valid()) 
        {
            spdlog::debug("[MonocularInitializer] Corresponding feature for idx {} not found or invalid", idx);
            continue;
        }
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
            spdlog::debug("[MonocularInitializer] Triangulation failed for feature idx {}: homogeneous w ~ 0", idx);
            continue;
        }
        X_h /= X_h[3];
        Eigen::Vector3f X = X_h.head<3>();
        
        // Check if point is in front of both cameras
        float depth1 = X[2];
        Eigen::Vector3f X2 = R_21 * X + t_21;
        float depth2 = X2[2];
        
        if (depth1 <= 0.0f || depth2 <= 0.0f)
        { 
            spdlog::debug("[MonocularInitializer] Triangulated point behind camera for feature idx {}", idx);
            continue;
        }
        
        // Check reprojection error
        Eigen::Vector3f proj1 = P1 * X_h;
        proj1 /= proj1[2];
        
        Eigen::Vector3f proj2 = P2 * X_h;
        proj2 /= proj2[2];
        
        float error1 = (proj1.head<2>() - bearing1.head<2>()).norm();
        float error2 = (proj2.head<2>() - bearing2.head<2>()).norm();
        
        if (error1 > m_max_reprojection_error || error2 > m_max_reprojection_error) {
            spdlog::debug("[MonocularInitializer] High reprojection error for feature idx {}: error1={:.3f}, error2={:.3f} / max: {:.3f}", 
                         idx, error1, error2, m_max_reprojection_error);
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
