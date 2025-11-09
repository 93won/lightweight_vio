/**
 * @file      rgbd_player.h
 * @brief     RGBD dataset player for VO and VIO pipelines
 * @author    Seungwon Choi (csw3575@snu.ac.kr)
 * @date      2025-10-28
 * @copyright Copyright (c) 2025 Seungwon Choi. All rights reserved.
 *
 * @par License
 * This project is released under the MIT License.
 */

#pragma once

#include <opencv2/opencv.hpp>
#include <spdlog/spdlog.h>
#include <string>
#include <vector>
#include <memory>
#include <chrono>
#include <Eigen/Dense>
#include "util/StringUtils.h"

// Forward declarations to avoid heavy includes
namespace lightweight_vio {
    class Frame;
    class Estimator;
    class PangolinViewer;
    class Feature;
    class MapPoint;
    class Config;
    struct IMUData;
}

namespace lightweight_vio {

/**
 * @brief RGBD image data structure containing timestamp and filenames for RGB and Depth
 */
struct RGBDImageData {
    long long timestamp;
    std::string rgb_filename;
    std::string depth_filename;
};

/**
 * @brief Configuration for RGBD player
 */
struct RGBDPlayerConfig {
    std::string config_path;
    std::string dataset_path;
    bool enable_viewer = false;
    bool enable_statistics = true;          // Enable file statistics output
    bool enable_console_statistics = true;  // Enable console statistics output
    bool use_vio_mode = false;  // true for VIO, false for VO (default VO for RGBD)
    bool step_mode = false;
    int viewer_width = 1920;
    int viewer_height = 1080;
};

/**
 * @brief Result structure containing processing statistics
 */
struct RGBDPlayerResult {
    bool success = false;
    size_t processed_frames = 0;
    double average_processing_time_ms = 0.0;
    std::vector<double> frame_processing_times;
    std::string error_message;
    
    // Error analysis results
    struct ErrorStats {
        bool available = false;
        size_t total_frame_pairs = 0;
        size_t total_frames = 0;
        size_t gt_poses_count = 0;
        // Rotation error statistics (degrees)
        double rotation_rmse = 0.0;
        double rotation_mean = 0.0;
        double rotation_median = 0.0;
        double rotation_min = 0.0;
        double rotation_max = 0.0;
        // Translation error statistics (meters)
        double translation_rmse = 0.0;
        double translation_mean = 0.0;
        double translation_median = 0.0;
        double translation_min = 0.0;
        double translation_max = 0.0;
    } error_stats;
    
    // Velocity analysis results
    struct VelocityStats {
        bool available = false;
        // Linear velocity statistics (m/s)
        double linear_vel_mean = 0.0;
        double linear_vel_median = 0.0;
        double linear_vel_min = 0.0;
        double linear_vel_max = 0.0;
        // Angular velocity statistics (rad/s)
        double angular_vel_mean = 0.0;
        double angular_vel_median = 0.0;
        double angular_vel_min = 0.0;
        double angular_vel_max = 0.0;
    } velocity_stats;
};

/**
 * @brief Frame processing context
 */
struct RGBDFrameContext {
    size_t current_idx = 0;
    size_t processed_frames = 0;
    double previous_frame_timestamp = 0.0;  // Changed to seconds
    std::vector<Eigen::Matrix4f> gt_poses;
    
    // UI control
    bool auto_play = true;
    bool step_mode = false;
    bool advance_frame = false;
};

/**
 * @brief RGBD Dataset Player class
 * 
 * Handles both Visual Odometry (VO) and Visual-Inertial Odometry (VIO) modes
 * for RGBD dataset processing with optional 3D visualization.
 * Supports TUM RGB-D dataset format and similar formats.
 */
class RGBDPlayer {
public:
    /**
     * @brief Constructor
     */
    RGBDPlayer() = default;
    
    /**
     * @brief Destructor
     */
    ~RGBDPlayer() = default;

    /**
     * @brief Run the RGBD player with given configuration
     * @param config Player configuration
     * @return Processing result with statistics
     */
    RGBDPlayerResult run(const RGBDPlayerConfig& config);

private:
    // === Data Loading ===
    
    /**
     * @brief Load RGBD image timestamps from dataset
     * @param dataset_path Path to RGBD dataset
     * @return Vector of RGBD image data with timestamps and filenames
     * 
     * Expected dataset structure:
     * - rgb.txt: timestamp filename pairs for RGB images
     * - depth.txt: timestamp filename pairs for depth images
     * - rgb/ folder with RGB images
     * - depth/ folder with depth images
     */
    std::vector<RGBDImageData> load_rgbd_timestamps(const std::string& dataset_path);
    
    /**
     * @brief Load single RGB image from dataset
     * @param dataset_path Path to dataset
     * @param filename RGB image filename
     * @return Loaded RGB image (color or grayscale depending on config)
     */
    cv::Mat load_rgb_image(const std::string& dataset_path, const std::string& filename);
    
    /**
     * @brief Load single depth image from dataset
     * @param dataset_path Path to dataset
     * @param filename Depth image filename
     * @return Loaded depth image (16UC1 or 32FC1)
     */
    cv::Mat load_depth_image(const std::string& dataset_path, const std::string& filename);
    
    /**
     * @brief Setup ground truth matching and frame range
     * @param dataset_path Path to dataset
     * @param image_data RGBD image data vector
     * @param start_frame_idx Output start frame index
     * @param end_frame_idx Output end frame index
     * @return Success status
     */
    bool setup_ground_truth_matching(const std::string& dataset_path, 
                                    const std::vector<RGBDImageData>& image_data,
                                    size_t& start_frame_idx, 
                                    size_t& end_frame_idx);
    
    /**
     * @brief Load ground truth poses from EuRoC format CSV file (gt.csv in dataset root)
     * @param dataset_path Path to dataset
     * @return Success status
     */
    bool load_ground_truth_euroc_format(const std::string& dataset_path);
    
    /**
     * @brief Load ground truth poses from TUM format file (ground_truth.txt in dataset root)
     * @param dataset_path Path to dataset
     * @return Success status
     */
    bool load_ground_truth_tum_format(const std::string& dataset_path);
    
    /**
     * @brief Match image timestamps with loaded ground truth
     * @param image_timestamps Vector of image timestamps
     * @return True if matching was successful
     */
    bool match_image_timestamps_with_gt(const std::vector<long long>& image_timestamps);

    // === System Initialization ===
    
    /**
     * @brief Initialize viewer if enabled
     * @param config Player configuration
     * @return Unique pointer to viewer (nullptr if disabled)
     */
    std::unique_ptr<PangolinViewer> initialize_viewer(const RGBDPlayerConfig& config);
    
    /**
     * @brief Initialize estimator with ground truth pose if available
     * @param estimator Reference to estimator
     * @param context Frame processing context
     */
    void initialize_estimator(Estimator& estimator, const RGBDFrameContext& context);

    // === Frame Processing ===
    
    /**
     * @brief Process single RGBD frame through VO/VIO pipeline
     * @param estimator Reference to estimator
     * @param context Frame processing context
     * @param image_data RGBD image data vector
     * @param dataset_path Path to dataset
     * @param use_vio_mode Whether to use VIO mode
     * @return Processing time in milliseconds
     */
    double process_single_frame(Estimator& estimator,
                               RGBDFrameContext& context,
                               const std::vector<RGBDImageData>& image_data,
                               const std::string& dataset_path,
                               bool use_vio_mode);
    
    /**
     * @brief Preprocess image with illumination enhancement
     * @param input_image Input grayscale/color image
     * @return Processed image
     */
    cv::Mat preprocess_image(const cv::Mat& input_image);

    // === Viewer Updates ===
    
    /**
     * @brief Update viewer with current frame data
     * @param viewer Reference to viewer
     * @param estimator Reference to estimator
     * @param context Frame processing context
     */
    void update_viewer(PangolinViewer& viewer,
                      const Estimator& estimator,
                      const RGBDFrameContext& context);
    
    /**
     * @brief Handle viewer UI controls
     * @param viewer Reference to viewer
     * @param context Frame processing context
     * @return True if should continue processing
     */
    bool handle_viewer_controls(PangolinViewer& viewer, RGBDFrameContext& context);

    // === Result Saving ===
    
    /**
     * @brief Save trajectory results in TUM format
     * @param estimator Reference to estimator
     * @param context Frame processing context
     * @param dataset_path Path to dataset
     * @param use_vio_mode Whether VIO mode was used
     */
    void save_trajectories(const Estimator& estimator,
                          const RGBDFrameContext& context,
                          const std::string& dataset_path,
                          bool use_vio_mode);
    
    /**
     * @brief Analyze frame-to-frame transform errors
     * @param estimator Reference to estimator
     * @param gt_poses Ground truth poses
     * @param use_vio_mode Whether VIO mode was used
     * @return Error statistics
     */
    RGBDPlayerResult::ErrorStats analyze_transform_errors(const Estimator& estimator,
                                                          const std::vector<Eigen::Matrix4f>& gt_poses,
                                                          bool use_vio_mode);
    
    /**
     * @brief Analyze velocity statistics from trajectory
     * @param estimator Reference to estimator
     * @param gt_poses Ground truth poses for timing
     * @return Velocity statistics
     */
    RGBDPlayerResult::VelocityStats analyze_velocity_statistics(const Estimator& estimator,
                                                                const std::vector<Eigen::Matrix4f>& gt_poses);
    
    /**
     * @brief Save comprehensive statistics to file
     * @param result Player result with statistics
     * @param dataset_path Path to dataset
     * @param use_vio_mode Whether VIO mode was used
     */
    void save_statistics(const RGBDPlayerResult& result,
                        const std::string& dataset_path,
                        bool use_vio_mode);

    // === Utility Functions ===
    
    /**
     * @brief Extract positions from pose matrices
     * @param poses Vector of 4x4 pose matrices
     * @return Vector of 3D positions
     */
    std::vector<Eigen::Vector3f> extract_positions_from_poses(const std::vector<Eigen::Matrix4f>& poses);

private:
    // Member variables for state management
    bool gravity_transformation_sent_ = false;
    
    // Ground truth data storage (EuRoC format)
    struct GroundTruthPose {
        long long timestamp;
        Eigen::Matrix4f pose;
        Eigen::Vector3f velocity;
        Eigen::Vector3f bias_gyro;
        Eigen::Vector3f bias_accel;
    };
    
    std::vector<GroundTruthPose> gt_data_;
    std::vector<Eigen::Matrix4f> matched_gt_poses_;
    std::vector<long long> matched_image_timestamps_;
    
    Eigen::Matrix4f first_matched_gt_pose_ = Eigen::Matrix4f::Identity();
    bool has_first_matched_gt_pose_ = false;
};

} // namespace lightweight_vio
