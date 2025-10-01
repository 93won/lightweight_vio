/**
 * @file      tum_rgbd_player.cpp
 * @brief     Implementation of TUM RGB-D dataset player for RGB-D Visual Odometry pipeline
 * @author    Seungwon Choi (csw3575@snu.ac.kr)
 * @date      2025-09-30
 * @copyright Copyright (c) 2025 Seungwon Choi. All rights reserved.
 *
 * @par License
 * This project is released under the MIT License.
 */

#include "tum_rgbd_player.h"
#include "database/Frame.h"
#include "util/Config.h"
#include <spdlog/spdlog.h>
#include <fstream>
#include <sstream>
#include <algorithm>
#include <iomanip>

namespace lightweight_vio {

TUMRGBDPlayer::TUMRGBDPlayer(const std::string& dataset_path, double max_time_diff)
    : m_dataset_path(dataset_path), m_max_time_diff(max_time_diff) {
    spdlog::info("[TUM-RGBD] Initializing player for dataset: {}", dataset_path);
    spdlog::info("[TUM-RGBD] Maximum time difference for synchronization: {:.3f}s", max_time_diff);
}

bool TUMRGBDPlayer::initialize() {
    // Load RGB and depth file lists
    if (!load_rgb_file_list()) {
        spdlog::error("[TUM-RGBD] Failed to load RGB file list");
        return false;
    }
    
    if (!load_depth_file_list()) {
        spdlog::error("[TUM-RGBD] Failed to load depth file list");
        return false;
    }
    
    spdlog::info("[TUM-RGBD] Loaded {} RGB images and {} depth images", 
                 m_rgb_data.size(), m_depth_data.size());
    
    // Synchronize RGB and depth images
    synchronize_rgb_depth();
    
    spdlog::info("[TUM-RGBD] Successfully synchronized {} RGB-D pairs", 
                 m_synchronized_pairs.size());
    
    return !m_synchronized_pairs.empty();
}

bool TUMRGBDPlayer::load_rgb_file_list() {
    std::string rgb_file = m_dataset_path + "/rgb.txt";
    return parse_association_file(rgb_file, m_rgb_data);
}

bool TUMRGBDPlayer::load_depth_file_list() {
    std::string depth_file = m_dataset_path + "/depth.txt";
    return parse_association_file(depth_file, m_depth_data);
}

bool TUMRGBDPlayer::parse_association_file(const std::string& filename, 
                                          std::vector<std::pair<double, std::string>>& data) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        spdlog::error("[TUM-RGBD] Cannot open file: {}", filename);
        return false;
    }
    
    data.clear();
    std::string line;
    int line_count = 0;
    int valid_lines = 0;
    
    while (std::getline(file, line)) {
        line_count++;
        
        // Skip comments and empty lines
        if (line.empty() || line[0] == '#') {
            continue;
        }
        
        std::istringstream iss(line);
        double timestamp;
        std::string image_path;
        
        if (iss >> timestamp >> image_path) {
            data.emplace_back(timestamp, image_path);
            valid_lines++;
        } else {
            spdlog::warn("[TUM-RGBD] Invalid line {} in {}: {}", line_count, filename, line);
        }
    }
    
    file.close();
    
    // Sort by timestamp
    std::sort(data.begin(), data.end(), 
              [](const auto& a, const auto& b) { return a.first < b.first; });
    
    spdlog::info("[TUM-RGBD] Parsed {} valid entries from {} (total lines: {})", 
                 valid_lines, filename, line_count);
    
    if (!data.empty()) {
        spdlog::info("[TUM-RGBD] Timestamp range: {:.6f} - {:.6f} ({:.3f}s duration)", 
                     data.front().first, data.back().first, 
                     data.back().first - data.front().first);
    }
    
    return !data.empty();
}

void TUMRGBDPlayer::synchronize_rgb_depth() {
    m_synchronized_pairs.clear();
    
    if (m_rgb_data.empty() || m_depth_data.empty()) {
        spdlog::error("[TUM-RGBD] No RGB or depth data to synchronize");
        return;
    }
    
    size_t depth_idx = 0;
    int synchronized_count = 0;
    int rgb_without_depth = 0;
    
    std::vector<double> time_differences;
    
    for (const auto& rgb_entry : m_rgb_data) {
        double rgb_timestamp = rgb_entry.first;
        const std::string& rgb_path = rgb_entry.second;
        
        // Find closest depth image
        double min_time_diff = std::numeric_limits<double>::max();
        size_t best_depth_idx = 0;
        bool found_match = false;
        
        // Search around current depth index for efficiency
        size_t search_start = (depth_idx > 10) ? depth_idx - 10 : 0;
        size_t search_end = std::min(depth_idx + 20, m_depth_data.size());
        
        for (size_t i = search_start; i < search_end; ++i) {
            double depth_timestamp = m_depth_data[i].first;
            double time_diff = std::abs(rgb_timestamp - depth_timestamp);
            
            if (time_diff < min_time_diff) {
                min_time_diff = time_diff;
                best_depth_idx = i;
                found_match = (time_diff <= m_max_time_diff);
            }
        }
        
        if (found_match) {
            const std::string& depth_path = m_depth_data[best_depth_idx].second;
            double depth_timestamp = m_depth_data[best_depth_idx].first;
            
            m_synchronized_pairs.emplace_back(rgb_timestamp, rgb_path, 
                                            depth_timestamp, depth_path);
            
            time_differences.push_back(min_time_diff);
            synchronized_count++;
            
            // Update depth index for next search
            depth_idx = best_depth_idx;
        } else {
            rgb_without_depth++;
        }
    }
    
    // Calculate statistics
    if (!time_differences.empty()) {
        std::sort(time_differences.begin(), time_differences.end());
        double avg_diff = std::accumulate(time_differences.begin(), time_differences.end(), 0.0) / time_differences.size();
        double median_diff = time_differences[time_differences.size() / 2];
        double max_diff = *std::max_element(time_differences.begin(), time_differences.end());
        
        spdlog::info("[TUM-RGBD] Synchronization statistics:");
        spdlog::info("  - Synchronized pairs: {}/{} ({:.1f}%)", 
                     synchronized_count, m_rgb_data.size(), 
                     (synchronized_count * 100.0) / m_rgb_data.size());
        spdlog::info("  - RGB without depth: {}", rgb_without_depth);
        spdlog::info("  - Time difference - Avg: {:.4f}s, Median: {:.4f}s, Max: {:.4f}s", 
                     avg_diff, median_diff, max_diff);
    }
}

std::shared_ptr<Frame> TUMRGBDPlayer::get_frame(size_t index, double depth_scale_factor) {
    if (index >= m_synchronized_pairs.size()) {
        spdlog::error("[TUM-RGBD] Frame index {} out of range (max: {})", 
                      index, m_synchronized_pairs.size() - 1);
        return nullptr;
    }
    
    const auto& pair = m_synchronized_pairs[index];
    
    // Build full paths
    std::string rgb_full_path = build_full_path(pair.rgb_path);
    std::string depth_full_path = build_full_path(pair.depth_path);
    
    // Load images
    cv::Mat rgb_image = cv::imread(rgb_full_path, cv::IMREAD_COLOR);
    cv::Mat depth_image = cv::imread(depth_full_path, cv::IMREAD_ANYDEPTH); // uint16
    
    if (rgb_image.empty()) {
        spdlog::error("[TUM-RGBD] Failed to load RGB image: {}", rgb_full_path);
        return nullptr;
    }
    
    if (depth_image.empty()) {
        spdlog::error("[TUM-RGBD] Failed to load depth image: {}", depth_full_path);
        return nullptr;
    }
    
    // Check image dimensions match
    if (rgb_image.cols != depth_image.cols || rgb_image.rows != depth_image.rows) {
        spdlog::error("[TUM-RGBD] Image size mismatch - RGB: {}x{}, Depth: {}x{}", 
                      rgb_image.cols, rgb_image.rows, depth_image.cols, depth_image.rows);
        return nullptr;
    }
    
    // Convert timestamp to nanoseconds for Frame constructor
    long long timestamp_ns = static_cast<long long>(pair.rgb_timestamp * 1e9);
    
    // Create RGB-D frame
    auto frame = std::make_shared<Frame>(timestamp_ns, static_cast<int>(index), 
                                       rgb_image, depth_image, depth_scale_factor);
    
    spdlog::debug("[TUM-RGBD] Created frame {} at timestamp {:.6f}s (time_diff: {:.4f}s)", 
                  index, pair.rgb_timestamp, pair.time_diff);
    
    return frame;
}

void TUMRGBDPlayer::print_sync_stats() const {
    if (m_synchronized_pairs.empty()) {
        spdlog::info("[TUM-RGBD] No synchronized pairs available");
        return;
    }
    
    spdlog::info("[TUM-RGBD] ===== Synchronization Statistics =====");
    spdlog::info("Dataset path: {}", m_dataset_path);
    spdlog::info("Max time difference: {:.3f}s", m_max_time_diff);
    spdlog::info("Total RGB images: {}", m_rgb_data.size());
    spdlog::info("Total depth images: {}", m_depth_data.size());
    spdlog::info("Synchronized pairs: {}", m_synchronized_pairs.size());
    
    // Calculate time difference statistics
    std::vector<double> time_diffs;
    for (const auto& pair : m_synchronized_pairs) {
        time_diffs.push_back(pair.time_diff);
    }
    
    if (!time_diffs.empty()) {
        std::sort(time_diffs.begin(), time_diffs.end());
        double avg_diff = std::accumulate(time_diffs.begin(), time_diffs.end(), 0.0) / time_diffs.size();
        double median_diff = time_diffs[time_diffs.size() / 2];
        double min_diff = time_diffs.front();
        double max_diff = time_diffs.back();
        
        spdlog::info("Time differences - Min: {:.4f}s, Avg: {:.4f}s, Median: {:.4f}s, Max: {:.4f}s", 
                     min_diff, avg_diff, median_diff, max_diff);
        
        // Show first and last synchronized pairs
        const auto& first_pair = m_synchronized_pairs.front();
        const auto& last_pair = m_synchronized_pairs.back();
        
        spdlog::info("First pair: RGB {:.6f}s, Depth {:.6f}s (diff: {:.4f}s)", 
                     first_pair.rgb_timestamp, first_pair.depth_timestamp, first_pair.time_diff);
        spdlog::info("Last pair:  RGB {:.6f}s, Depth {:.6f}s (diff: {:.4f}s)", 
                     last_pair.rgb_timestamp, last_pair.depth_timestamp, last_pair.time_diff);
        
        double total_duration = last_pair.rgb_timestamp - first_pair.rgb_timestamp;
        spdlog::info("Total duration: {:.3f}s ({:.1f} fps average)", 
                     total_duration, m_synchronized_pairs.size() / total_duration);
    }
    
    spdlog::info("[TUM-RGBD] =======================================");
}

std::string TUMRGBDPlayer::build_full_path(const std::string& relative_path) const {
    return m_dataset_path + "/" + relative_path;
}

} // namespace lightweight_vio