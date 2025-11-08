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
 * @brief Monocular VO Initializer
 * 
 * Two-frame initialization process:
 * 1. Keep only the last frame as reference
 * 2. When new frame arrives, check parallax
 * 3. If parallax sufficient: initialize with two-view geometry
 * 4. If parallax insufficient: discard reference and use current as new reference
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
    };

public:
    MonocularInitializer();
    ~MonocularInitializer() = default;
    
    /**
     * @brief Add a candidate frame and attempt initialization
     * @param frame New frame to consider
     * @return true if frame was kept as reference (not necessarily initialized)
     * 
     * Behavior:
     * - If no reference: keep as reference
     * - If parallax sufficient: attempt initialization
     * - If parallax insufficient: discard old reference, keep new as reference
     */
    bool add_frame(std::shared_ptr<Frame> frame);
    
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
     * @brief Attempt two-view initialization with reference and current frame
     * @return true if initialization succeeded
     */
    bool try_initialize();
    
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
     * @param inlier_indices Output indices of inlier features
     * @return true if initialization succeeded
     */
    bool initialize_two_views(
        const std::shared_ptr<Frame>& frame1,
        const std::shared_ptr<Frame>& frame2,
        Eigen::Matrix3f& R_21,
        Eigen::Vector3f& t_21,
        std::vector<Eigen::Vector3f>& points_3d,
        std::vector<int>& inlier_indices);
    
    /**
     * @brief Triangulate 3D points from two frames
     * @param frame1 First frame
     * @param frame2 Second frame
     * @param R_21 Rotation from frame1 to frame2
     * @param t_21 Translation from frame1 to frame2
     * @param points_3d Output triangulated points in frame1 coordinate
     * @param inlier_indices Feature indices that were successfully triangulated
     * @return Number of successfully triangulated points
     */
    int triangulate_two_views(
        const std::shared_ptr<Frame>& frame1,
        const std::shared_ptr<Frame>& frame2,
        const Eigen::Matrix3f& R_21,
        const Eigen::Vector3f& t_21,
        std::vector<Eigen::Vector3f>& points_3d,
        std::vector<int>& inlier_indices);

private:
    // Configuration
    double m_min_parallax_pixels;       // Minimum median pixel displacement for initialization (pixels)
    
    // State
    std::shared_ptr<Frame> m_reference_frame;  // Reference frame (most recent candidate)
    bool m_is_initialized;
    InitializationResult m_result;
    
    // Two-view geometry parameters
    double m_ransac_threshold;          // RANSAC inlier threshold (pixels)
    double m_ransac_confidence;         // RANSAC confidence level
    int m_ransac_max_iterations;        // Maximum RANSAC iterations
    
    // Triangulation parameters
    double m_min_triangulation_angle;   // Minimum angle for triangulation (degrees)
    double m_max_reprojection_error;    // Maximum reprojection error (pixels)
};

} // namespace lightweight_vio
