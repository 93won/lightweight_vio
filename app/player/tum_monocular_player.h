/**
 * @file      tum_monocular_player.h
 * @brief     TUM RGB-D dataset player for Monocular Visual Odometry (RGB only)
 * @author    Seungwon Choi (csw3575@snu.ac.kr)
 * @date      2025-11-10
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
 * @brief RGB image data with timestamp
 */
struct RGBImage {
    double timestamp;
    std::string path;
    
    RGBImage(double ts, const std::string& p)
        : timestamp(ts), path(p) {}
};

/**
 * @brief Configuration for TUM Monocular player
 */
struct TUMMonocularPlayerConfig {
    std::string config_path;
    std::string dataset_path;
    bool enable_viewer = true;
    bool enable_statistics = true;
    bool enable_console_statistics = true;
    bool step_mode = false;
    int viewer_width = 1920;
    int viewer_height = 1080;
};

/**
 * @brief Result structure for TUM Monocular processing
 */
struct TUMMonocularPlayerResult {
    bool success = false;
    size_t processed_frames = 0;
    double average_processing_time_ms = 0.0;
    std::vector<double> frame_processing_times;
    std::string error_message;
};

/**
 * @brief Frame processing context
 */
struct MonocularFrameContext {
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
 * @brief TUM Monocular dataset player class (RGB only from RGB-D dataset)
 */
class TUMMonocularPlayer {
public:
    TUMMonocularPlayer() = default;
    ~TUMMonocularPlayer() = default;

    /**
     * @brief Run the TUM Monocular processing pipeline
     */
    TUMMonocularPlayerResult run(const TUMMonocularPlayerConfig& config);

private:
    // Dataset initialization and loading
    bool initialize_dataset(const std::string& dataset_path);
    bool load_rgb_file_list(const std::string& dataset_path);
    bool parse_association_file(const std::string& filename, 
                               std::vector<std::pair<double, std::string>>& data);
    
    // Image loading
    cv::Mat load_rgb_image(const std::string& dataset_path, const std::string& filename);
    std::string build_full_path(const std::string& dataset_path, const std::string& relative_path);
    
    // Frame creation
    std::shared_ptr<Frame> get_frame(size_t index, const std::string& dataset_path);
    
    // System initialization
    std::unique_ptr<PangolinViewer> initialize_viewer(const TUMMonocularPlayerConfig& config);
    void initialize_estimator(Estimator& estimator);
    
    // Frame processing
    double process_single_frame(Estimator& estimator, MonocularFrameContext& context,
                               const std::string& dataset_path);
    
    // Viewer management
    bool handle_viewer_controls(PangolinViewer& viewer, MonocularFrameContext& context);
    void update_viewer(PangolinViewer& viewer, const Estimator& estimator, const MonocularFrameContext& context);
    
    // Statistics and output
    void save_trajectory(const Estimator& estimator, const std::string& dataset_path);
    void save_statistics(const TUMMonocularPlayerResult& result, const std::string& dataset_path);
    
    // Utilities
    std::string trim(const std::string& str);
    std::vector<Eigen::Vector3f> extract_positions_from_poses(const std::vector<Eigen::Matrix4f>& poses);
    
    // Data members
    std::vector<RGBImage> m_rgb_images;  // RGB image list with timestamps
};

} // namespace lightweight_vio
