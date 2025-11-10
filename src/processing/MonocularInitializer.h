/**
 * @file      MonocularInitializer.h
 * @brief     Monocular VIO initialization using consecutive SFM
 * @author    Seungwon Choi (csw3575@snu.ac.kr)
 * @date      2025-11-09
 * @copyright Copyright (c) 2025 Seungwon Choi. All rights reserved.
 *
 * @par License
 * This project is released under the MIT License.
 */

#pragma once

#include <vector>
#include <memory>
#include <Eigen/Dense>
#include <opencv2/opencv.hpp>

#include "../database/Frame.h"
#include "../database/MapPoint.h"
#include "../util/Config.h"

namespace lightweight_vio {

/**
 * @brief Monocular VIO Initializer with window-based frame collection
 * 
 * Window-based initialization process:
 * 1. Collect frames until window is full (e.g., 5 frames)
 * 2. Find best frame pair with maximum parallax
 * 3. Attempt two-view initialization with best pair
 * 4. If failed: slide window (remove oldest) and continue collecting
 */
class MonocularInitializer {
public:
    struct InitializationResult {
        bool success = false;
        
        // Two-view geometry results
        std::vector<std::shared_ptr<Frame>> initialized_keyframes;  // Always 2 frames
        std::vector<std::shared_ptr<MapPoint>> initialized_mappoints;
        int num_triangulated_points = 0;
        
        std::string failure_reason;
    };
    
    struct ParallaxInfo {
        double median_parallax_pixels = 0.0;    // Median pixel displacement
        double average_parallax_pixels = 0.0;   // Average pixel displacement
        int num_tracked_features = 0;          // Number of features used for parallax computation
        bool sufficient_parallax = false;       // Whether parallax exceeds threshold
        std::vector<double> pixel_displacements;  // Raw pixel displacement values
    };

public:
    MonocularInitializer();
    ~MonocularInitializer() = default;
    
    /**
     * @brief Try to initialize visual SFM with a new frame
     * @param frame New frame to consider
     * @param init_attempted Optional output parameter - set to true if initialization was attempted
     * @return true if initialization succeeded or frame was added to window
     * 
     * Process:
     * 1. Collect frames in window until full
     * 2. Two-view initialization: find best frame pair (first & last)
     * 3. Multi-view SFM: estimate poses for middle frames using PnP
     * 4. Triangulate new points between frames
     * 5. Bundle adjustment (optional)
     */
    bool try_initialize_visual_sfm(std::shared_ptr<Frame> frame, bool* init_attempted = nullptr);
    
    /**
     * @brief Check if initialization succeeded
     * @return true if two-view initialization completed successfully
     */
    bool is_initialized() const { return m_is_initialized; }
    
    /**
     * @brief Get initialization result (only valid if is_initialized() == true)
     */
    InitializationResult get_result() const { return m_result; }
    
    /**
     * @brief Reset the initializer state
     */
    void reset();

private:
    /**
     * @brief Find best frame pair in window based on parallax
     * @param ref_idx Output: index of reference frame in window
     * @param cur_idx Output: index of current frame in window (usually last)
     * @return true if found valid pair with sufficient parallax
     */
    bool find_best_frame_pair(int& ref_idx, int& cur_idx);
    
    /**
     * @brief Attempt two-view initialization with reference and current frame
     * @return true if initialization succeeded
     */
    bool try_initialize();
    
    /**
     * @brief Compute parallax between two frames in normalized coordinates
     * @param frame1 First frame
     * @param frame2 Second frame
     * @return Parallax information
     */
    ParallaxInfo compute_parallax_normalized(
        const std::shared_ptr<Frame>& frame1,
        const std::shared_ptr<Frame>& frame2) const;
    
    /**
     * @brief Compute parallax between consecutive frames
     * @param frame1 First frame
     * @param frame2 Second frame
     * @return Parallax information
     */
    ParallaxInfo compute_parallax(
        const std::shared_ptr<Frame>& frame1,
        const std::shared_ptr<Frame>& frame2) const;
    
    /**
     * @brief Initialize two frames using two-view geometry
     * @param frame1 First frame (will be at origin)
     * @param frame2 Second frame
     * @param R_21 Output rotation from frame1 to frame2
     * @param t_21 Output translation from frame1 to frame2 (unit length)
     * @param points_3d Output triangulated 3D points in frame1 coordinate
     * @param inlier_indices_1 Output indices of inlier features in frame1
     * @param inlier_indices_2 Output indices of inlier features in frame2
     * @return true if initialization succeeded
     */
    bool initialize_two_views(
        const std::shared_ptr<Frame>& frame1,
        const std::shared_ptr<Frame>& frame2,
        Eigen::Matrix3f& R_21,
        Eigen::Vector3f& t_21,
        std::vector<Eigen::Vector3f>& points_3d,
        std::vector<int>& inlier_indices_1,
        std::vector<int>& inlier_indices_2);
    
    /**
     * @brief Triangulate 3D points from two frames
     * @param frame1 First frame
     * @param frame2 Second frame
     * @param R_21 Rotation from frame1 to frame2
     * @param t_21 Translation from frame1 to frame2
     * @param points_3d Output triangulated points in frame1 coordinate
     * @param inlier_indices_1 Feature indices that were successfully triangulated
     * @param inlier_indices_2 Feature indices that were successfully triangulated
     * @return Number of successfully triangulated points
     */
    int triangulate_two_views(
        const std::shared_ptr<Frame>& frame1,
        const std::shared_ptr<Frame>& frame2,
        const Eigen::Matrix3f& R_21,
        const Eigen::Vector3f& t_21,
        std::vector<Eigen::Vector3f>& points_3d,
        std::vector<int>& inlier_indices_1,
        std::vector<int>& inlier_indices_2);
    
    /**
     * @brief Set the initial body-to-world transform
     * @param T_wb Initial body-to-world transformation matrix
     */
    void set_Twb_init(const Eigen::Matrix4f& T_wb) { m_Twb_init = T_wb; }
    
    /**
     * @brief Manually recover relative camera pose from Essential matrix
     * @param E Essential matrix (3x3)
     * @param pts1_pixel Matched points in frame 1 (pixel coordinates, undistorted)
     * @param pts2_pixel Matched points in frame 2 (pixel coordinates, undistorted)
     * @param K Camera intrinsic matrix (3x3)
     * @param R_21 Output rotation from frame 1 to frame 2
     * @param t_21 Output translation from frame 1 to frame 2 (unit norm)
     * @param mask Inlier mask
     * @param reproj_threshold Reprojection error threshold in pixels (squared)
     * @return Number of inliers that pass cheirality check
     */
    int recover_pose_from_essential(
        const Eigen::Matrix3f& E,
        const std::vector<Eigen::Vector2f>& pts1_pixel,
        const std::vector<Eigen::Vector2f>& pts2_pixel,
        const Eigen::Matrix3f& K,
        Eigen::Matrix3f& R_21,
        Eigen::Vector3f& t_21,
        std::vector<bool>& mask,
        float reproj_threshold,
        std::vector<Eigen::Vector3f>& points_cam_in_1,
        std::vector<int>& inlier_feature_indices
    );
    
    /**
     * @brief Check if a 3D point is in front of both cameras (cheirality check)
     * @param X 3D point in frame 1 coordinate
     * @param R_21 Rotation from frame 1 to frame 2
     * @param t_21 Translation from frame 1 to frame 2
     * @return true if point has positive depth in both cameras
     */
    bool check_cheirality(
        const Eigen::Vector3f& X,
        const Eigen::Matrix3f& R_21,
        const Eigen::Vector3f& t_21) const;

private:
    // Configuration
    double m_min_parallax_pixels;       // Minimum median pixel displacement for initialization (pixels)
    int m_window_size;                  // Number of frames to collect before attempting initialization
    
    // State
    std::vector<std::shared_ptr<Frame>> m_frame_window;  // Sliding window of candidate frames
    bool m_is_initialized;
    InitializationResult m_result;
    
    // Two-view geometry parameters
    double m_ransac_threshold;          // RANSAC inlier threshold (pixels)
    double m_ransac_confidence;         // RANSAC confidence level
    int m_ransac_max_iterations;        // Maximum RANSAC iterations
    int m_min_inliers;                  // Minimum inliers for F-matrix and pose recovery
    
    // Triangulation parameters
    double m_min_triangulation_angle;   // Minimum angle for triangulation (degrees)
    double m_max_reprojection_error;    // Maximum reprojection error (pixels)

    Eigen::Matrix4f m_Twb_init;

};

} // namespace lightweight_vio
