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
 * @brief Monocular VIO Initializer
 * 
 * Three-stage initialization process:
 * 1. Consecutive SFM: Recover structure from N keyframes with sufficient parallax
 * 2. Gravity Initialization: Estimate gravity direction (scale-free)
 * 3. IMU Initialization: Joint optimization with scale, bias estimation
 */
class MonocularInitializer {
public:
    struct InitializationResult {
        bool success = false;
        
        // Stage 1: SFM results
        std::vector<std::shared_ptr<Frame>> initialized_keyframes;
        std::vector<std::shared_ptr<MapPoint>> initialized_mappoints;
        int num_triangulated_points = 0;
        
        // Stage 2: Gravity results
        Eigen::Vector3f gravity_direction;  // Unit vector
        Eigen::Matrix3f Rwg;                 // Rotation from world to gravity-aligned frame
        
        // Stage 3: IMU results (if VIO mode)
        double scale = 1.0;
        Eigen::Vector3f gyro_bias = Eigen::Vector3f::Zero();
        Eigen::Vector3f accel_bias = Eigen::Vector3f::Zero();
        
        std::string failure_reason;
    };
    
    struct ParallaxInfo {
        double average_parallax = 0.0;      // Average parallax in degrees
        double median_parallax = 0.0;       // Median parallax in degrees
        int num_tracked_features = 0;      // Number of features used for parallax computation
        bool sufficient_parallax = false;   // Whether parallax exceeds threshold
    };

public:
    MonocularInitializer();
    ~MonocularInitializer() = default;
    
    /**
     * @brief Add a candidate frame for initialization
     * @param frame New frame to consider
     * @return true if frame was added to candidate list
     */
    bool add_frame(std::shared_ptr<Frame> frame);
    
    /**
     * @brief Attempt to initialize the monocular VIO system
     * @return Initialization result with success flag and recovered structure
     */
    InitializationResult try_initialize();
    
    /**
     * @brief Reset the initializer state
     */
    void reset();
    
    /**
     * @brief Get current number of candidate frames
     */
    int get_num_candidates() const { return m_candidate_frames.size(); }
    
    /**
     * @brief Check if sufficient frames collected
     */
    bool has_sufficient_frames() const;

private:
    // ========================================================================
    // Stage 1: Consecutive SFM
    // ========================================================================
    
    /**
     * @brief Perform consecutive structure from motion on candidate frames
     * @param result Output initialization result
     * @return true if SFM succeeded
     */
    bool consecutive_sfm(InitializationResult& result);
    
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
     * @brief Initialize first two frames using two-view geometry
     * @param frame1 First frame
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
    
    /**
     * @brief Recover pose for a new frame using PnP with known 3D points
     * @param frame New frame
     * @param mappoints 3D map points visible in this frame
     * @param R_new Output rotation for new frame
     * @param t_new Output translation for new frame
     * @return true if PnP succeeded
     */
    bool recover_pose_pnp(
        const std::shared_ptr<Frame>& frame,
        const std::vector<std::shared_ptr<MapPoint>>& mappoints,
        Eigen::Matrix3f& R_new,
        Eigen::Vector3f& t_new);
    
    /**
     * @brief Triangulate new points between an existing frame and new frame
     * @param existing_frame Frame with known pose
     * @param new_frame New frame with estimated pose
     * @param existing_mappoints Existing map points (will be updated)
     * @return Number of newly triangulated points
     */
    int triangulate_new_points(
        const std::shared_ptr<Frame>& existing_frame,
        const std::shared_ptr<Frame>& new_frame,
        std::vector<std::shared_ptr<MapPoint>>& existing_mappoints);
    
    // ========================================================================
    // Stage 2: Gravity Initialization (TODO)
    // ========================================================================
    
    // ========================================================================
    // Stage 3: IMU Initialization (TODO)
    // ========================================================================

private:
    // Configuration
    int m_required_keyframes;           // Number of keyframes needed for initialization
    double m_min_parallax_degrees;      // Minimum parallax threshold (degrees)
    double m_min_average_parallax;      // Minimum average parallax for keyframe selection
    
    // State
    std::vector<std::shared_ptr<Frame>> m_candidate_frames;  // Frames collected for initialization
    bool m_is_initialized;
    
    // Two-view geometry parameters
    double m_ransac_threshold;          // RANSAC inlier threshold (pixels)
    double m_ransac_confidence;         // RANSAC confidence level
    int m_ransac_max_iterations;        // Maximum RANSAC iterations
    
    // Triangulation parameters
    double m_min_triangulation_angle;   // Minimum angle for triangulation (degrees)
    double m_max_reprojection_error;    // Maximum reprojection error (pixels)
};

} // namespace lightweight_vio
