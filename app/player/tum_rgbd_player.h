/**
 * @file      tum_rgbd_player.h
 * @brief     TUM RGB-D dataset player for RGB-D Visual Odometry
 * @author    Seungwon Choi (csw3575@snu.ac.kr)
 * @date      2025-10-02
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
#include <Eigen/Dense>

namespace lightweight_vio {

// Forward declarations
class Frame;
class Estimator;
class PangolinViewer;

/**
 * @brief RGB-D image pair with synchronized timestamps
 */
struct RGBDImagePair {
    double rgb_timestamp;
    std::string rgb_path;
    double depth_timestamp;
    std::string depth_path;
    double time_diff;
    
    RGBDImagePair(double rgb_ts, const std::string& rgb_p, 
                  double depth_ts, const std::string& depth_p)
        : rgb_timestamp(rgb_ts), rgb_path(rgb_p), 
          depth_timestamp(depth_ts), depth_path(depth_p),
          time_diff(std::abs(rgb_ts - depth_ts)) {}
};

/**
 * @brief Configuration for TUM RGB-D player
 */
struct TUMRGBDPlayerConfig {
    std::string config_path;
    std::string dataset_path;
    bool enable_viewer = true;
    bool enable_statistics = true;
    bool enable_console_statistics = true;
    bool step_mode = false;
    int viewer_width = 1920;
    int viewer_height = 1080;
    double max_time_diff = 0.02;  // 20ms max sync difference
};

/**
 * @brief Result structure for TUM RGB-D processing
 */
struct TUMRGBDPlayerResult {
    bool success = false;
    size_t processed_frames = 0;
    double average_processing_time_ms = 0.0;
    std::vector<double> frame_processing_times;
    std::string error_message;
};

/**
 * @brief Frame processing context
 */
struct FrameContext {
    size_t current_idx = 0;
    size_t processed_frames = 0;
    
    // Control variables
    bool step_mode = false;
    bool auto_play = true;
    bool advance_frame = false;
    
    // Statistics
    std::vector<Eigen::Matrix4f> trajectory_poses;
};

/**
 * @brief TUM RGB-D dataset player class
 */
class TUMRGBDPlayer {
public:
    TUMRGBDPlayer() = default;
    ~TUMRGBDPlayer() = default;

    /**
     * @brief Run the TUM RGB-D processing pipeline
     */
    TUMRGBDPlayerResult run(const TUMRGBDPlayerConfig& config);

private:
    // Dataset initialization and loading
    bool initialize_dataset(const std::string& dataset_path, double max_time_diff);
    bool load_rgb_file_list(const std::string& dataset_path);
    bool load_depth_file_list(const std::string& dataset_path);
    bool parse_association_file(const std::string& filename, 
                               std::vector<std::pair<double, std::string>>& data);
    void synchronize_rgb_depth();
    
    // Image loading
    cv::Mat load_rgb_image(const std::string& dataset_path, const std::string& filename);
    cv::Mat load_depth_image(const std::string& dataset_path, const std::string& filename);
    std::string build_full_path(const std::string& dataset_path, const std::string& relative_path);
    
    // Frame creation
    std::shared_ptr<Frame> get_frame(size_t index, const std::string& dataset_path, double depth_scale_factor);
    
    // System initialization
    std::unique_ptr<PangolinViewer> initialize_viewer(const TUMRGBDPlayerConfig& config);
    void initialize_estimator(Estimator& estimator);
    
    // Frame processing
    double process_single_frame(Estimator& estimator, FrameContext& context,
                               const std::string& dataset_path, double depth_scale_factor);
    
    // Viewer management
    bool handle_viewer_controls(PangolinViewer& viewer, FrameContext& context);
    void update_viewer(PangolinViewer& viewer, const Estimator& estimator, const FrameContext& context);
    
    // Statistics and output
    void print_sync_stats() const;
    void save_trajectory(const Estimator& estimator, const std::string& dataset_path);
    void save_statistics(const TUMRGBDPlayerResult& result, const std::string& dataset_path);
    
    // Utilities
    std::string trim(const std::string& str);
    std::vector<Eigen::Vector3f> extract_positions_from_poses(const std::vector<Eigen::Matrix4f>& poses);
    
    // Data members
    std::vector<std::pair<double, std::string>> m_rgb_data;      // RGB timestamp and filename pairs
    std::vector<std::pair<double, std::string>> m_depth_data;    // Depth timestamp and filename pairs
    std::vector<RGBDImagePair> m_synchronized_pairs;             // Synchronized RGB-D pairs
    double m_max_time_diff = 0.02;                               // Maximum time difference for sync
    double m_depth_scale_factor = 5000.0;                       // TUM RGB-D depth scale factor
};

} // namespace lightweight_vio