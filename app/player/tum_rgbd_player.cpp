/**
 * @file      tum_rgbd_player.cpp
 * @brief     TUM RGB-D dataset player implementation
 * @author    Seungwon Choi (csw3575@snu.ac.kr)
 * @date      2025-10-02
 * @copyright Copyright (c) 2025 Seungwon Choi. All rights reserved.
 *
 * @par License
 * This project is released under the MIT License.
 */

#include "tum_rgbd_player.h"

#include <spdlog/sinks/stdout_color_sinks.h>
#include <iostream>
#include <fstream>
#include <sstream>
#include <algorithm>
#include <chrono>
#include <iomanip>
#include <thread>
#include <numeric>
#include <set>

#include <util/Config.h>
#include <processing/Estimator.h>
#include <viewer/PangolinViewer.h>
#include <database/Frame.h>
#include <database/Feature.h>
#include <database/MapPoint.h>

namespace lightweight_vio {

TUMRGBDPlayerResult TUMRGBDPlayer::run(const TUMRGBDPlayerConfig& config) {
    TUMRGBDPlayerResult result;
    
    try {
        // 1. Load configuration
        Config::getInstance().load(config.config_path);
        spdlog::info("[TUMRGBDPlayer] Successfully loaded configuration from: {}", config.config_path);
        
        // Override viewer settings
        Config::getInstance().m_viewer_enable = config.enable_viewer;
        Config::getInstance().m_viewer_width = config.viewer_width;
        Config::getInstance().m_viewer_height = config.viewer_height;
        
        // 2. Initialize dataset
        if (!initialize_dataset(config.dataset_path, config.max_time_diff)) {
            result.error_message = "Failed to initialize RGB-D dataset";
            return result;
        }
        
        spdlog::info("[TUMRGBDPlayer] Loaded {} synchronized RGB-D pairs", m_synchronized_pairs.size());
        
        // 3. Initialize systems
        auto viewer = initialize_viewer(config);
        Estimator estimator;
        initialize_estimator(estimator);
        
        // 4. Process frames
        FrameContext context;
        context.step_mode = config.step_mode;
        context.auto_play = !config.step_mode;
        
        spdlog::info("[TUMRGBDPlayer] Processing {} RGB-D frames (VO mode)", m_synchronized_pairs.size());
        
        context.current_idx = 0;
        while (context.current_idx < m_synchronized_pairs.size()) {
            // Handle viewer controls first
            if (viewer && !handle_viewer_controls(*viewer, context)) {
                break;
            }
            
            bool should_process_frame = false;
            
            // Check processing conditions based on mode
            if (context.auto_play) {
                should_process_frame = true;
            } else {
                if (context.advance_frame) {
                    should_process_frame = true;
                    context.advance_frame = false;
                } else {
                    if (viewer) {
                        viewer->render();
                    }
                    std::this_thread::sleep_for(std::chrono::milliseconds(10));
                    continue;
                }
            }
            
            if (should_process_frame) {
                // Process single frame
                auto frame_start = std::chrono::high_resolution_clock::now();
                double processing_time = process_single_frame(estimator, context, config.dataset_path, m_depth_scale_factor);
                auto frame_end = std::chrono::high_resolution_clock::now();
                
                auto frame_duration = std::chrono::duration_cast<std::chrono::microseconds>(frame_end - frame_start);
                double total_time_ms = frame_duration.count() / 1000.0;
                result.frame_processing_times.push_back(total_time_ms);
                
                // Update viewer
                if (viewer) {
                    update_viewer(*viewer, estimator, context);
                }
                
                // Progress logging
                if (context.processed_frames % 50 == 0) {
                    spdlog::info("[TUMRGBDPlayer] Processed {} / {} frames", 
                                context.processed_frames, m_synchronized_pairs.size());
                }
                
                ++context.current_idx;
                ++context.processed_frames;
                
                // Add delay for auto mode
                if (context.auto_play) {
                    std::this_thread::sleep_for(std::chrono::milliseconds(30));
                }
            }
        }
        
        // 5. Save results
        if (config.enable_statistics) {
            save_trajectory(estimator, config.dataset_path);
            save_statistics(result, config.dataset_path);
        }
        
        // 6. Calculate final statistics
        result.success = true;
        result.processed_frames = context.processed_frames;
        if (!result.frame_processing_times.empty()) {
            result.average_processing_time_ms = std::accumulate(
                result.frame_processing_times.begin(), 
                result.frame_processing_times.end(), 0.0) / result.frame_processing_times.size();
        }
        
        spdlog::info("[TUMRGBDPlayer] Successfully processed {} frames", result.processed_frames);
        
        // Display statistics
        if (config.enable_console_statistics && result.success) {
            spdlog::info("════════════════════════════════════════════════════════════════════");
            spdlog::info("                      TUM RGB-D VO STATISTICS                       ");
            spdlog::info("════════════════════════════════════════════════════════════════════");
            spdlog::info(" Total Frames Processed: {}", result.processed_frames);
            spdlog::info(" Average Processing Time: {:.2f}ms", result.average_processing_time_ms);
            double fps = 1000.0 / result.average_processing_time_ms;
            spdlog::info(" Average Frame Rate: {:.1f}fps", fps);
            spdlog::info("════════════════════════════════════════════════════════════════════");
        }
        
        // Wait for viewer finish if enabled
        if (viewer) {
            spdlog::info("[TUMRGBDPlayer] Processing completed! Click 'Finish & Exit' to close.");
            while (!viewer->should_close() && !viewer->is_finish_requested()) {
                viewer->render();
                std::this_thread::sleep_for(std::chrono::milliseconds(30));
            }
        }
        
    } catch (const std::exception& e) {
        result.error_message = e.what();
        spdlog::error("[TUMRGBDPlayer] Exception occurred: {}", e.what());
    }
    
    return result;
}

bool TUMRGBDPlayer::initialize_dataset(const std::string& dataset_path, double max_time_diff) {
    m_max_time_diff = max_time_diff;
    
    // Load RGB and depth file lists
    if (!load_rgb_file_list(dataset_path)) {
        spdlog::error("[TUMRGBDPlayer] Failed to load RGB file list");
        return false;
    }
    
    if (!load_depth_file_list(dataset_path)) {
        spdlog::error("[TUMRGBDPlayer] Failed to load depth file list");
        return false;
    }
    
    spdlog::info("[TUMRGBDPlayer] Loaded {} RGB images and {} depth images", 
                 m_rgb_data.size(), m_depth_data.size());
    
    // Synchronize RGB and depth images
    synchronize_rgb_depth();
    
    spdlog::info("[TUMRGBDPlayer] Successfully synchronized {} RGB-D pairs", 
                 m_synchronized_pairs.size());
    
    return !m_synchronized_pairs.empty();
}

bool TUMRGBDPlayer::load_rgb_file_list(const std::string& dataset_path) {
    std::string rgb_file = dataset_path + "/rgb.txt";
    return parse_association_file(rgb_file, m_rgb_data);
}

bool TUMRGBDPlayer::load_depth_file_list(const std::string& dataset_path) {
    std::string depth_file = dataset_path + "/depth.txt";
    return parse_association_file(depth_file, m_depth_data);
}

bool TUMRGBDPlayer::parse_association_file(const std::string& filename, 
                                          std::vector<std::pair<double, std::string>>& data) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        spdlog::error("[TUMRGBDPlayer] Cannot open file: {}", filename);
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
            spdlog::warn("[TUMRGBDPlayer] Invalid line {} in {}: {}", line_count, filename, line);
        }
    }
    
    file.close();
    
    // Sort by timestamp
    std::sort(data.begin(), data.end(), 
              [](const auto& a, const auto& b) { return a.first < b.first; });
    
    spdlog::info("[TUMRGBDPlayer] Parsed {} valid entries from {} (total lines: {})", 
                 valid_lines, filename, line_count);
    
    if (!data.empty()) {
        spdlog::info("[TUMRGBDPlayer] Timestamp range: {:.6f} - {:.6f} ({:.3f}s duration)", 
                     data.front().first, data.back().first, 
                     data.back().first - data.front().first);
    }
    
    return !data.empty();
}

void TUMRGBDPlayer::synchronize_rgb_depth() {
    m_synchronized_pairs.clear();
    
    if (m_rgb_data.empty() || m_depth_data.empty()) {
        spdlog::error("[TUMRGBDPlayer] No RGB or depth data to synchronize");
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
        
        spdlog::info("[TUMRGBDPlayer] Synchronization statistics:");
        spdlog::info("  - Synchronized pairs: {}/{} ({:.1f}%)", 
                     synchronized_count, m_rgb_data.size(), 
                     (synchronized_count * 100.0) / m_rgb_data.size());
        spdlog::info("  - RGB without depth: {}", rgb_without_depth);
        spdlog::info("  - Time difference - Avg: {:.4f}s, Median: {:.4f}s, Max: {:.4f}s", 
                     avg_diff, median_diff, max_diff);
    }
}

cv::Mat TUMRGBDPlayer::load_rgb_image(const std::string& dataset_path, const std::string& filename) {
    std::string full_path = build_full_path(dataset_path, filename);
    cv::Mat image = cv::imread(full_path, cv::IMREAD_COLOR);
    
    if (image.empty()) {
        spdlog::error("[TUMRGBDPlayer] Cannot load RGB image: {}", full_path);
    }
    
    return image;
}

cv::Mat TUMRGBDPlayer::load_depth_image(const std::string& dataset_path, const std::string& filename) {
    std::string full_path = build_full_path(dataset_path, filename);
    cv::Mat image = cv::imread(full_path, cv::IMREAD_UNCHANGED);  // 16-bit depth
    
    if (image.empty()) {
        spdlog::error("[TUMRGBDPlayer] Cannot load depth image: {}", full_path);
    }
    
    return image;
}

std::string TUMRGBDPlayer::build_full_path(const std::string& dataset_path, const std::string& relative_path) {
    return dataset_path + "/" + relative_path;
}

std::shared_ptr<Frame> TUMRGBDPlayer::get_frame(size_t index, const std::string& dataset_path, double depth_scale_factor) {
    if (index >= m_synchronized_pairs.size()) {
        spdlog::error("[TUMRGBDPlayer] Frame index {} out of range (max: {})", 
                      index, m_synchronized_pairs.size() - 1);
        return nullptr;
    }
    
    const auto& pair = m_synchronized_pairs[index];
    
    // Load images
    cv::Mat rgb_image = load_rgb_image(dataset_path, pair.rgb_path);
    cv::Mat depth_image = load_depth_image(dataset_path, pair.depth_path);
    
    if (rgb_image.empty()) {
        spdlog::error("[TUMRGBDPlayer] Failed to load RGB image: {}", pair.rgb_path);
        return nullptr;
    }
    
    if (depth_image.empty()) {
        spdlog::error("[TUMRGBDPlayer] Failed to load depth image: {}", pair.depth_path);
        return nullptr;
    }
    
    // Check image dimensions match
    if (rgb_image.cols != depth_image.cols || rgb_image.rows != depth_image.rows) {
        spdlog::error("[TUMRGBDPlayer] Image size mismatch - RGB: {}x{}, Depth: {}x{}", 
                      rgb_image.cols, rgb_image.rows, depth_image.cols, depth_image.rows);
        return nullptr;
    }
    
    // Convert timestamp to nanoseconds for Frame constructor
    long long timestamp_ns = static_cast<long long>(pair.rgb_timestamp * 1e9);
    
    // Create RGB-D frame
    auto frame = std::make_shared<Frame>(timestamp_ns, static_cast<int>(index), 
                                       rgb_image, depth_image, depth_scale_factor);
    
    spdlog::debug("[TUMRGBDPlayer] Created frame {} at timestamp {:.6f}s (time_diff: {:.4f}s)", 
                  index, pair.rgb_timestamp, pair.time_diff);
    
    return frame;
}

std::unique_ptr<PangolinViewer> TUMRGBDPlayer::initialize_viewer(const TUMRGBDPlayerConfig& config) {
    if (!config.enable_viewer) {
        return nullptr;
    }
    
    auto viewer = std::make_unique<PangolinViewer>();
    if (viewer->initialize(config.viewer_width, config.viewer_height)) {
        spdlog::info("[TUMRGBDPlayer] Viewer initialized successfully");
        
        // Wait for viewer to be ready
        while (!viewer->is_ready()) {
            viewer->render();
            std::this_thread::sleep_for(std::chrono::milliseconds(16));
        }
        spdlog::info("[TUMRGBDPlayer] Viewer is ready!");
        return viewer;
    } else {
        spdlog::warn("[TUMRGBDPlayer] Failed to initialize viewer");
        return nullptr;
    }
}

void TUMRGBDPlayer::initialize_estimator(Estimator& estimator) {
    spdlog::info("[TUMRGBDPlayer] Estimator initialized for RGB-D VO");
}

double TUMRGBDPlayer::process_single_frame(Estimator& estimator,
                                          FrameContext& context,
                                          const std::string& dataset_path,
                                          double depth_scale_factor) {
    auto start_time = std::chrono::high_resolution_clock::now();
    
    if (context.current_idx >= m_synchronized_pairs.size()) {
        return 0.0;
    }
    
    const auto& pair = m_synchronized_pairs[context.current_idx];
    
    // Load RGB and depth images
    cv::Mat rgb_image = load_rgb_image(dataset_path, pair.rgb_path);
    cv::Mat depth_image = load_depth_image(dataset_path, pair.depth_path);
    
    if (rgb_image.empty() || depth_image.empty()) {
        spdlog::warn("[TUMRGBDPlayer] Skipping frame {} due to empty images", context.current_idx);
        return 0.0;
    }
    
    // Convert timestamp to nanoseconds
    long long timestamp_ns = static_cast<long long>(pair.rgb_timestamp * 1e9);
    
    // Process RGB-D frame through estimator with proper depth scale factor
    try {
        auto estimation_result = estimator.process_frame(rgb_image, depth_image, timestamp_ns, depth_scale_factor);
        
        if (!estimation_result.success) {
            spdlog::warn("[TUMRGBDPlayer] Estimation failed for frame {}", context.current_idx);
            return 0.0;
        }
        
        // Store pose for trajectory
        auto current_frame = estimator.get_current_frame();
        if (current_frame) {
            context.trajectory_poses.push_back(current_frame->get_Twb());
        } else {
            spdlog::warn("[TUMRGBDPlayer] No current frame from estimator for frame {}", context.current_idx);
        }
        
    } catch (const std::exception& e) {
        spdlog::error("[TUMRGBDPlayer] Failed to process RGB-D frame {}: {}", 
                      context.current_idx, e.what());
        return 0.0;
    }
    
    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);
    return duration.count() / 1000.0; // Return milliseconds
}

bool TUMRGBDPlayer::handle_viewer_controls(PangolinViewer& viewer, FrameContext& context) {
    // Check for exit conditions
    if (viewer.should_close() || viewer.is_finish_requested()) {
        spdlog::info("[TUMRGBDPlayer] User requested exit");
        return false;
    }
    
    // Process keyboard input and sync UI state
    viewer.process_keyboard_input(context.auto_play, context.step_mode, context.advance_frame);
    viewer.sync_ui_state(context.auto_play, context.step_mode);
    
    return true;
}

void TUMRGBDPlayer::update_viewer(PangolinViewer& viewer,
                                 const Estimator& estimator,
                                 const FrameContext& context) {
    auto current_frame = estimator.get_current_frame();
    if (!current_frame) return;
    
    // Update poses
    Eigen::Matrix4f current_pose = current_frame->get_Twb();
    viewer.update_pose(current_pose);
    viewer.update_camera_pose(current_frame->get_Twc());
    
    // Update trajectory
    std::vector<Eigen::Vector3f> positions = extract_positions_from_poses(context.trajectory_poses);
    viewer.update_trajectory(positions);
    
    // Update frame and keyframes
    viewer.add_frame(current_frame);
    const auto keyframes = estimator.get_keyframes_safe();
    viewer.update_keyframe_window(keyframes);
    
    // Update map points
    std::vector<std::shared_ptr<MapPoint>> all_map_points;
    std::set<std::shared_ptr<MapPoint>> unique_map_points;
    
    for (const auto& kf : keyframes) {
        const auto& kf_map_points = kf->get_map_points();
        for (const auto& mp : kf_map_points) {
            if (mp && !mp->is_bad()) {
                unique_map_points.insert(mp);
            }
        }
    }
    
    all_map_points.assign(unique_map_points.begin(), unique_map_points.end());
    viewer.update_all_map_points(all_map_points);
    
    // Update tracking statistics
    int total_features = current_frame->get_feature_count();
    int map_points_count = 0;
    const auto& map_points = current_frame->get_map_points();
    for (const auto& mp : map_points) {
        if (mp && !mp->is_bad()) {
            map_points_count++;
        }
    }
    
    float success_rate = (total_features > 0) ? 
        (static_cast<float>(map_points_count) / static_cast<float>(total_features)) * 100.0f : 0.0f;
    
    viewer.update_tracking_stats(context.processed_frames + 1, total_features, 
                               map_points_count, map_points_count, success_rate, 0.0f);
    
    // Update tracking images
    cv::Mat tracking_image = current_frame->draw_features();
    const auto& features = current_frame->get_features();
    const auto& frame_map_points = current_frame->get_map_points();
    viewer.update_tracking_image_with_map_points(tracking_image, features, frame_map_points);
    
    viewer.render();
}

void TUMRGBDPlayer::print_sync_stats() const {
    if (m_synchronized_pairs.empty()) {
        spdlog::info("[TUMRGBDPlayer] No synchronized pairs available");
        return;
    }
    
    spdlog::info("[TUMRGBDPlayer] ===== Synchronization Statistics =====");
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
    
    spdlog::info("[TUMRGBDPlayer] =======================================");
}

void TUMRGBDPlayer::save_trajectory(const Estimator& estimator, const std::string& dataset_path) {
    std::string traj_file = dataset_path + "/estimated_trajectory_rgbd_vo.txt";
    std::ofstream traj_out(traj_file);
    
    if (traj_out.is_open()) {
        const auto& all_frames = estimator.get_all_frames();
        spdlog::info("[TUMRGBDPlayer] Saving {} frames to trajectory", all_frames.size());
        
        for (const auto& frame : all_frames) {
            if (!frame) continue;
            
            Eigen::Matrix4f T_wb = frame->get_Twb();
            Eigen::Vector3f translation = T_wb.block<3, 1>(0, 3);
            Eigen::Matrix3f rotation = T_wb.block<3, 3>(0, 0);
            Eigen::Quaternionf quat(rotation);
            
            // Use frame timestamp
            double timestamp_sec = frame->get_timestamp() / 1e9;
            
            traj_out << std::fixed << std::setprecision(6) << timestamp_sec << " "
                     << std::setprecision(8)
                     << translation.x() << " " << translation.y() << " " << translation.z() << " "
                     << quat.x() << " " << quat.y() << " " << quat.z() << " " << quat.w() << std::endl;
        }
        
        traj_out.close();
        spdlog::info("[TUMRGBDPlayer] Saved trajectory to: {}", traj_file);
    }
}

void TUMRGBDPlayer::save_statistics(const TUMRGBDPlayerResult& result, const std::string& dataset_path) {
    std::string stats_file = dataset_path + "/statistics_rgbd_vo.txt";
    
    std::ofstream stats_out(stats_file);
    if (stats_out.is_open()) {
        stats_out << "════════════════════════════════════════════════════════════════════\n";
        stats_out << "                      TUM RGB-D VO STATISTICS                       \n";
        stats_out << "════════════════════════════════════════════════════════════════════\n\n";
        
        stats_out << "                          TIMING ANALYSIS                           \n";
        stats_out << "════════════════════════════════════════════════════════════════════\n";
        stats_out << " Total Frames Processed: " << result.processed_frames << "\n";
        stats_out << " Average Processing Time: " << std::fixed << std::setprecision(2) 
                  << result.average_processing_time_ms << "ms\n";
        double fps = 1000.0 / result.average_processing_time_ms;
        stats_out << " Average Frame Rate: " << std::fixed << std::setprecision(1) << fps << "fps\n\n";
        
        stats_out << "════════════════════════════════════════════════════════════════════\n";
        stats_out.close();
        spdlog::info("[TUMRGBDPlayer] Saved statistics to: {}", stats_file);
    }
}

std::string TUMRGBDPlayer::trim(const std::string& str) {
    size_t first = str.find_first_not_of(" \t\r\n");
    if (first == std::string::npos) return "";
    size_t last = str.find_last_not_of(" \t\r\n");
    return str.substr(first, (last - first + 1));
}

std::vector<Eigen::Vector3f> TUMRGBDPlayer::extract_positions_from_poses(const std::vector<Eigen::Matrix4f>& poses) {
    std::vector<Eigen::Vector3f> positions;
    positions.reserve(poses.size());
    for (const auto& pose : poses) {
        positions.push_back(pose.block<3, 1>(0, 3));
    }
    return positions;
}

} // namespace lightweight_vio