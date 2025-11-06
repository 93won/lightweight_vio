/**
 * @file      tum_player.h
 * @brief     TUM dataset player for VO and VIO pipelines
 * @author    Seungwon Choi (csw3575@snu.ac.kr)
 * @date      2025-09-27
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
#include "../../src/util/StringUtils.h"

// Forward declarations to avoid heavy includes
namespace lightweight_vio {
    class Frame;
    class Estimator;
    class PangolinViewer;
    class TUMUtils;
    class Feature;
    class MapPoint;
    class Config;
    struct IMUData;
}

namespace lightweight_vio {

/**
 * @brief Image data structure containing timestamp and filename
 */
struct ImageData {
    long long timestamp;
    std::string filename;
};

/**
 * @brief Configuration for TUM player
 */
struct TUMPlayerConfig {
    std::string config_path;
    std::string dataset_path;
    bool enable_viewer = false;
    bool enable_statistics = true;          // Enable file statistics output
    bool enable_console_statistics = true;  // Enable console statistics output
    bool use_vio_mode = true;  // true for VIO, false for VO
    bool step_mode = false;
    int viewer_width = 1920;
    int viewer_height = 1080;
};

/**
 * @brief Result structure containing processing statistics
 */
struct TUMPlayerResult {
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
struct FrameContext {
    size_t current_idx = 0;
    size_t processed_frames = 0;
    long long previous_frame_timestamp = 0;
    
    // GT data (only for frames that have GT matches)
    std::vector<size_t> gt_frame_indices;  // Which frame indices have GT
    std::vector<Eigen::Matrix4f> gt_poses;  // GT poses for those frames
    
    // All estimated poses
    std::vector<Eigen::Matrix4f> estimated_poses;
    
    // UI control
    bool auto_play = true;
    bool step_mode = false;
    bool advance_frame = false;
};

/**
 * @brief TUM Dataset Player class
 * 
 * Handles both Visual Odometry (VO) and Visual-Inertial Odometry (VIO) modes
 * for TUM dataset processing with optional 3D visualization.
 */
class TUMPlayer {
public:
    /**
     * @brief Constructor
     */
    TUMPlayer() = default;
    
    /**
     * @brief Destructor
     */
    ~TUMPlayer() = default;

    /**
     * @brief Run the TUM player with given configuration
     * @param config Player configuration
     * @return Processing result with statistics
     */
    TUMPlayerResult run(const TUMPlayerConfig& config);

private:
    // === Data Loading ===
    
    /**
     * @brief Load image timestamps from TUM dataset
     * @param dataset_path Path to TUM dataset
     * @return Vector of image data with timestamps and filenames
     */
    std::vector<ImageData> load_image_timestamps(const std::string& dataset_path);
    
    /**
     * @brief Load single image from dataset
     * @param dataset_path Path to dataset
     * @param filename Image filename
     * @param cam_id Camera ID (0=left, 1=right)
     * @return Loaded grayscale image
     */
    cv::Mat load_image(const std::string& dataset_path, const std::string& filename, int cam_id = 0);
    
    /**
     * @brief Load stereo images with caching
     * @param dataset_path Path to dataset
     * @param filename Image filename
     * @param frame_idx Frame index for cache key
     * @param left_image Output left image
     * @param right_image Output right image
     * @return True if loaded successfully
     */
    bool load_stereo_images_cached(const std::string& dataset_path, 
                                  const std::string& filename, 
                                  size_t frame_idx,
                                  cv::Mat& left_image, 
                                  cv::Mat& right_image);
    
    /**
     * @brief Setup ground truth matching and frame range
     * @param dataset_path Path to dataset
     * @param image_data Image data vector
     * @param start_frame_idx Output start frame index
     * @param end_frame_idx Output end frame index
     * @return Success status
     */
    bool setup_ground_truth_matching(const std::string& dataset_path, 
                                    const std::vector<ImageData>& image_data,
                                    size_t& start_frame_idx, 
                                    size_t& end_frame_idx);
    
    /**
     * @brief Load IMU data for VIO mode
     * @param dataset_path Path to dataset
     * @param image_data Image data for time range
     * @param start_frame_idx Start frame index
     * @param end_frame_idx End frame index
     * @return Success status
     */
    bool load_imu_data(const std::string& dataset_path,
                      const std::vector<ImageData>& image_data,
                      size_t start_frame_idx,
                      size_t end_frame_idx);

    // === System Initialization ===
    
    /**
     * @brief Initialize viewer if enabled
     * @param config Player configuration
     * @return Unique pointer to viewer (nullptr if disabled)
     */
    std::unique_ptr<PangolinViewer> initialize_viewer(const TUMPlayerConfig& config);
    
    /**
     * @brief Initialize estimator with ground truth pose if available
     * @param estimator Reference to estimator
     * @param image_data Image data vector
     */
    void initialize_estimator(Estimator& estimator, const std::vector<ImageData>& image_data);

    // === Frame Processing ===
    
    /**
     * @brief Process single frame through VO/VIO pipeline
     * @param estimator Reference to estimator
     * @param context Frame processing context
     * @param image_data Image data vector
     * @param dataset_path Path to dataset
     * @param use_vio_mode Whether to use VIO mode
     * @return Processing time in milliseconds
     */
    double process_single_frame(Estimator& estimator,
                               FrameContext& context,
                               const std::vector<ImageData>& image_data,
                               const std::string& dataset_path,
                               bool use_vio_mode);
    
    /**
     * @brief Preprocess image with illumination enhancement
     * @param input_image Input grayscale image
     * @return Processed image
     */
    cv::Mat preprocess_image(const cv::Mat& input_image);
    
    /**
     * @brief Get IMU data between timestamps for VIO mode
     * @param previous_timestamp Previous frame timestamp
     * @param current_timestamp Current frame timestamp
     * @return Vector of IMU data
     */
    std::vector<IMUData> get_imu_data_between_frames(long long previous_timestamp, 
                                                    long long current_timestamp);

    // === Viewer Updates ===
    
    /**
     * @brief Update viewer with current frame data
     * @param viewer Reference to viewer
     * @param estimator Reference to estimator
     * @param context Frame processing context
     */
    void update_viewer(PangolinViewer& viewer,
                      const Estimator& estimator,
                      const FrameContext& context);
    
    /**
     * @brief Handle viewer UI controls
     * @param viewer Reference to viewer
     * @param context Frame processing context
     * @return True if should continue processing
     */
    bool handle_viewer_controls(PangolinViewer& viewer, FrameContext& context);

    // === Result Saving ===
    
    /**
     * @brief Save trajectory results in TUM format
     * @param estimator Reference to estimator
     * @param context Frame processing context
     * @param dataset_path Path to dataset
     * @param use_vio_mode Whether VIO mode was used
     */
    void save_trajectories(const Estimator& estimator,
                          const FrameContext& context,
                          const std::string& dataset_path,
                          bool use_vio_mode);
    
    /**
     * @brief Analyze frame-to-frame transform errors
     * @param estimated_poses All estimated poses from trajectory
     * @param gt_poses Ground truth poses (only for frames with GT)
     * @param gt_frame_indices Frame indices that have GT poses
     * @param use_vio_mode Whether VIO mode was used
     * @return Error statistics
     */
    TUMPlayerResult::ErrorStats analyze_transform_errors(const std::vector<Eigen::Matrix4f>& estimated_poses,
                                                          const std::vector<Eigen::Matrix4f>& gt_poses,
                                                          const std::vector<size_t>& gt_frame_indices,
                                                          bool use_vio_mode);
    
    /**
     * @brief Analyze velocity statistics from trajectory
     * @param estimator Reference to estimator
     * @param gt_poses Ground truth poses for timing
     * @return Velocity statistics
     */
    TUMPlayerResult::VelocityStats analyze_velocity_statistics(const Estimator& estimator,
                                                                const std::vector<Eigen::Matrix4f>& gt_poses);
    
    /**
     * @brief Save comprehensive statistics to file
     * @param result Player result with statistics
     * @param dataset_path Path to dataset
     * @param use_vio_mode Whether VIO mode was used
     */
    void save_statistics(const TUMPlayerResult& result,
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
    
    // Image cache for performance optimization
    struct CachedImage {
        cv::Mat left_image;
        cv::Mat right_image;
        std::string filename;
        size_t frame_idx;
    };
    
    mutable std::vector<CachedImage> image_cache_;
    static constexpr size_t MAX_CACHE_SIZE = 10; // Cache last 10 frames
};

} // namespace lightweight_vio
