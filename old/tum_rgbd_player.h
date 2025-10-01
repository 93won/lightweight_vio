/**
 * @file      tum_rgbd_player.h
 * @brief     TUM RGB-D dataset player for RGB-D Visual Odometry pipeline
 * @author    Seungwon Choi (csw3575@snu.ac.kr)
 * @date      2025-09-30
 * @copyright Copyright (c) 2025 Seungwon Choi. All rights reserved.
 *
 * @par License
 * This project is released under the MIT License.
 */

#pragma once

#include <string>
#include <vector>
#include <memory>
#include <opencv2/opencv.hpp>

namespace lightweight_vio {

class Frame;

struct RGBDImagePair {
    double rgb_timestamp;
    double depth_timestamp;
    std::string rgb_path;
    std::string depth_path;
    double time_diff;  // Absolute time difference between RGB and depth
    
    RGBDImagePair(double rgb_ts, const std::string& rgb_p, 
                  double depth_ts, const std::string& depth_p)
        : rgb_timestamp(rgb_ts), depth_timestamp(depth_ts),
          rgb_path(rgb_p), depth_path(depth_p),
          time_diff(std::abs(rgb_ts - depth_ts)) {}
};

class TUMRGBDPlayer {
public:
    TUMRGBDPlayer(const std::string& dataset_path, double max_time_diff = 0.02);
    ~TUMRGBDPlayer() = default;

    // Load and parse RGB and depth file lists
    bool initialize();
    
    // Get total number of synchronized frames
    size_t get_frame_count() const { return m_synchronized_pairs.size(); }
    
    // Get specific frame by index
    std::shared_ptr<Frame> get_frame(size_t index, double depth_scale_factor = 5000.0);
    
    // Get all synchronized pairs for processing
    const std::vector<RGBDImagePair>& get_synchronized_pairs() const { return m_synchronized_pairs; }
    
    // Print synchronization statistics
    void print_sync_stats() const;

private:
    std::string m_dataset_path;
    double m_max_time_diff;
    
    // Raw data from files
    std::vector<std::pair<double, std::string>> m_rgb_data;    // timestamp, filename
    std::vector<std::pair<double, std::string>> m_depth_data;  // timestamp, filename
    
    // Synchronized pairs
    std::vector<RGBDImagePair> m_synchronized_pairs;
    
    // Helper methods
    bool load_rgb_file_list();
    bool load_depth_file_list();
    bool parse_association_file(const std::string& filename, 
                               std::vector<std::pair<double, std::string>>& data);
    void synchronize_rgb_depth();
    std::string build_full_path(const std::string& relative_path) const;
};

} // namespace lightweight_vio