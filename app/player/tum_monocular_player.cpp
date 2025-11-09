/**
 * @file      tum_monocular_player.cpp
 * @brief     TUM Monocular dataset player implementation (RGB only)
 * @author    Seungwon Choi (csw3575@snu.ac.kr)
 * @date      2025-11-10
 * @copyright Copyright (c) 2025 Seungwon Choi. All rights reserved.
 *
 * @par License
 * This project is released under the MIT License.
 */

#include "tum_monocular_player.h"

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
#include <database/MapPoint.h>

namespace lightweight_vio {

TUMMonocularPlayerResult TUMMonocularPlayer::run(const TUMMonocularPlayerConfig& config) {
    TUMMonocularPlayerResult result;
    
    try {
        // 1. Load configuration
        Config::getInstance().load(config.config_path);
        spdlog::info("[TUMMonocularPlayer] Successfully loaded configuration from: {}", config.config_path);
        
        // Override viewer settings
        Config::getInstance().m_viewer_enable = config.enable_viewer;
        Config::getInstance().m_viewer_width = config.viewer_width;
        Config::getInstance().m_viewer_height = config.viewer_height;
        
        // 2. Initialize dataset
        if (!initialize_dataset(config.dataset_path)) {
            result.error_message = "Failed to initialize monocular dataset";
            return result;
        }
        
        spdlog::info("[TUMMonocularPlayer] Loaded {} RGB images", m_rgb_images.size());
        
        // 3. Initialize systems
        auto viewer = initialize_viewer(config);
        Estimator estimator;
        initialize_estimator(estimator);
        
        // 4. Process frames
        MonocularFrameContext context;
        context.step_mode = config.step_mode;
        context.auto_play = !config.step_mode;
        
        spdlog::info("[TUMMonocularPlayer] Processing {} monocular frames", m_rgb_images.size());
        
        context.current_idx = 0;
        while (context.current_idx < m_rgb_images.size()) {
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
                double processing_time = process_single_frame(estimator, context, config.dataset_path);
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
                    spdlog::info("[TUMMonocularPlayer] Processed {} / {} frames", 
                                context.processed_frames, m_rgb_images.size());
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
        
        spdlog::info("[TUMMonocularPlayer] Successfully processed {} frames", result.processed_frames);
        
        // Display statistics
        if (config.enable_console_statistics && result.success) {
            spdlog::info("════════════════════════════════════════════════════════════════════");
            spdlog::info("                  TUM MONOCULAR VO STATISTICS                       ");
            spdlog::info("════════════════════════════════════════════════════════════════════");
            spdlog::info(" Total Frames Processed: {}", result.processed_frames);
            spdlog::info(" Average Processing Time: {:.2f}ms", result.average_processing_time_ms);
            double fps = 1000.0 / result.average_processing_time_ms;
            spdlog::info(" Average Frame Rate: {:.1f}fps", fps);
            spdlog::info("════════════════════════════════════════════════════════════════════");
        }
        
    } catch (const std::exception& e) {
        result.error_message = std::string("Exception occurred: ") + e.what();
        spdlog::error("[TUMMonocularPlayer] {}", result.error_message);
    }
    
    return result;
}

bool TUMMonocularPlayer::initialize_dataset(const std::string& dataset_path) {
    spdlog::info("[TUMMonocularPlayer] Initializing dataset from: {}", dataset_path);
    
    // Load RGB file list
    if (!load_rgb_file_list(dataset_path)) {
        spdlog::error("[TUMMonocularPlayer] Failed to load RGB file list");
        return false;
    }
    
    spdlog::info("[TUMMonocularPlayer] Loaded {} RGB images", m_rgb_images.size());
    return !m_rgb_images.empty();
}

bool TUMMonocularPlayer::load_rgb_file_list(const std::string& dataset_path) {
    std::string rgb_file = dataset_path + "/rgb.txt";
    
    std::vector<std::pair<double, std::string>> rgb_data;
    if (!parse_association_file(rgb_file, rgb_data)) {
        spdlog::error("[TUMMonocularPlayer] Failed to parse rgb.txt");
        return false;
    }
    
    // Convert to RGBImage list
    m_rgb_images.clear();
    for (const auto& [timestamp, path] : rgb_data) {
        m_rgb_images.emplace_back(timestamp, path);
    }
    
    return true;
}

bool TUMMonocularPlayer::parse_association_file(const std::string& filename, 
                                                 std::vector<std::pair<double, std::string>>& data) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        spdlog::error("[TUMMonocularPlayer] Cannot open file: {}", filename);
        return false;
    }
    
    std::string line;
    while (std::getline(file, line)) {
        // Skip empty lines and comments
        line = trim(line);
        if (line.empty() || line[0] == '#') {
            continue;
        }
        
        std::istringstream iss(line);
        double timestamp;
        std::string path;
        
        if (iss >> timestamp >> path) {
            data.emplace_back(timestamp, path);
        }
    }
    
    file.close();
    spdlog::info("[TUMMonocularPlayer] Parsed {} entries from {}", data.size(), filename);
    return !data.empty();
}

cv::Mat TUMMonocularPlayer::load_rgb_image(const std::string& dataset_path, const std::string& filename) {
    std::string full_path = build_full_path(dataset_path, filename);
    cv::Mat image = cv::imread(full_path, cv::IMREAD_GRAYSCALE);  // Load as grayscale for monocular
    
    if (image.empty()) {
        spdlog::error("[TUMMonocularPlayer] Failed to load image: {}", full_path);
    }
    
    return image;
}

std::string TUMMonocularPlayer::build_full_path(const std::string& dataset_path, const std::string& relative_path) {
    // Handle both absolute and relative paths
    if (relative_path[0] == '/') {
        return relative_path;
    }
    return dataset_path + "/" + relative_path;
}

std::shared_ptr<Frame> TUMMonocularPlayer::get_frame(size_t index, const std::string& dataset_path) {
    if (index >= m_rgb_images.size()) {
        return nullptr;
    }
    
    const auto& rgb_img = m_rgb_images[index];
    
    // Load RGB image
    cv::Mat image = load_rgb_image(dataset_path, rgb_img.path);
    if (image.empty()) {
        spdlog::error("[TUMMonocularPlayer] Failed to load RGB image at index {}", index);
        return nullptr;
    }
    
    // Create monocular frame (no depth)
    // Note: Camera object will be created inside Frame constructor from Config
    auto frame = std::make_shared<Frame>(
        rgb_img.timestamp,
        static_cast<int>(index),
        image,
        nullptr  // Camera will be initialized from Config in Frame constructor
    );
    
    return frame;
}

std::unique_ptr<PangolinViewer> TUMMonocularPlayer::initialize_viewer(const TUMMonocularPlayerConfig& config) {
    if (!config.enable_viewer) {
        return nullptr;
    }
    
    auto viewer = std::make_unique<PangolinViewer>();
    if (!viewer->initialize(config.viewer_width, config.viewer_height)) {
        spdlog::error("[TUMMonocularPlayer] Failed to initialize Pangolin viewer");
        return nullptr;
    }
    
    spdlog::info("[TUMMonocularPlayer] Initialized Pangolin viewer: {}x{}", 
                 config.viewer_width, config.viewer_height);
    
    return viewer;
}

void TUMMonocularPlayer::initialize_estimator(Estimator& estimator) {


    Eigen::Vector3f t_init(-0.1357f, -1.4217f, 1.4764f);
    Eigen::Quaternionf q_init(0.6453f, -0.5498f, 0.3363f, -0.4101f);  // (qw, qx, qy, qz)
    
    // Note: Eigen::Quaternionf constructor is (w, x, y, z)
    // But TUM format is (x, y, z, w), so reorder:
    q_init = Eigen::Quaternionf(-0.4101f, 0.6453f, -0.5498f, 0.3363f);  // (w, x, y, z)
    q_init.normalize();
    
    Eigen::Matrix4f Twb_init = Eigen::Matrix4f::Identity();
    Twb_init.block<3, 3>(0, 0) = q_init.toRotationMatrix();
    Twb_init.block<3, 1>(0, 3) = t_init;
    
    estimator.set_initial_gt_pose(Twb_init);
    
}

double TUMMonocularPlayer::process_single_frame(Estimator& estimator, MonocularFrameContext& context,
                                                 const std::string& dataset_path) {
    auto frame_start = std::chrono::high_resolution_clock::now();
    
    // Get RGB image
    if (context.current_idx >= m_rgb_images.size()) {
        spdlog::error("[TUMMonocularPlayer] Frame index out of range: {}", context.current_idx);
        return 0.0;
    }
    
    const auto& rgb_img = m_rgb_images[context.current_idx];
    
    // Load RGB image as grayscale
    cv::Mat image = load_rgb_image(dataset_path, rgb_img.path);
    if (image.empty()) {
        spdlog::error("[TUMMonocularPlayer] Failed to load RGB image at index {}", context.current_idx);
        return 0.0;
    }
    
    // Process frame (monocular VO) - use dedicated monocular API
    auto result = estimator.process_monocular_frame(image, rgb_img.timestamp);
    
    // Store pose for trajectory
    context.trajectory_poses.push_back(result.pose);
    
    auto frame_end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(frame_end - frame_start);
    
    return duration.count() / 1000.0;  // Return time in milliseconds
}

bool TUMMonocularPlayer::handle_viewer_controls(PangolinViewer& viewer, MonocularFrameContext& context) {
    // Check for exit conditions
    if (viewer.should_close() || viewer.is_finish_requested()) {
        spdlog::info("[TUMMonocularPlayer] User requested exit");
        return false;
    }
    
    // Process keyboard input and sync UI state
    viewer.process_keyboard_input(context.auto_play, context.step_mode, context.advance_frame);
    viewer.sync_ui_state(context.auto_play, context.step_mode);
    
    // Always return true, let main loop handle mode logic
    return true;
}

void TUMMonocularPlayer::update_viewer(PangolinViewer& viewer, const Estimator& estimator, const MonocularFrameContext& context) {
    auto current_frame = estimator.get_current_frame();
    if (!current_frame) return;
    
    // Update poses
    Eigen::Matrix4f current_pose = current_frame->get_Twb();
    viewer.update_pose(current_pose);
    viewer.update_camera_pose(current_frame->get_Twc());
    
    // Update trajectory
    auto trajectory = extract_positions_from_poses(context.trajectory_poses);
    viewer.update_trajectory(trajectory);
    
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
    
    // Update tracking view with frame directly
    viewer.update_tracking_with_frame(current_frame);
    
    // Render
    viewer.render();
}

void TUMMonocularPlayer::save_trajectory(const Estimator& estimator, const std::string& dataset_path) {
    std::string output_file = dataset_path + "/trajectory_monocular.txt";
    
    // Get all frames and save trajectory manually
    auto all_frames = estimator.get_all_frames();
    std::ofstream file(output_file);
    
    if (!file.is_open()) {
        spdlog::error("[TUMMonocularPlayer] Failed to open trajectory file: {}", output_file);
        return;
    }
    
    // Write TUM format: timestamp tx ty tz qx qy qz qw
    for (const auto& frame : all_frames) {
        Eigen::Matrix4f Twb = frame->get_Twb();
        Eigen::Vector3f t = Twb.block<3, 1>(0, 3);
        Eigen::Matrix3f R = Twb.block<3, 3>(0, 0);
        Eigen::Quaternionf q(R);
        
        file << std::fixed << std::setprecision(6) << frame->get_timestamp() << " "
             << std::setprecision(9) << t.x() << " " << t.y() << " " << t.z() << " "
             << q.x() << " " << q.y() << " " << q.z() << " " << q.w() << "\n";
    }
    
    file.close();
    spdlog::info("[TUMMonocularPlayer] Saved trajectory to: {}", output_file);
}

void TUMMonocularPlayer::save_statistics(const TUMMonocularPlayerResult& result, const std::string& dataset_path) {
    std::string stats_file = dataset_path + "/statistics_monocular.txt";
    std::ofstream file(stats_file);
    
    if (!file.is_open()) {
        spdlog::error("[TUMMonocularPlayer] Failed to open statistics file: {}", stats_file);
        return;
    }
    
    file << "TUM Monocular VO Statistics\n";
    file << "============================\n";
    file << "Total Frames: " << result.processed_frames << "\n";
    file << "Average Processing Time: " << result.average_processing_time_ms << " ms\n";
    file << "Average FPS: " << (1000.0 / result.average_processing_time_ms) << "\n";
    
    file.close();
    spdlog::info("[TUMMonocularPlayer] Saved statistics to: {}", stats_file);
}

std::string TUMMonocularPlayer::trim(const std::string& str) {
    size_t first = str.find_first_not_of(" \t\r\n");
    if (first == std::string::npos) {
        return "";
    }
    size_t last = str.find_last_not_of(" \t\r\n");
    return str.substr(first, last - first + 1);
}

std::vector<Eigen::Vector3f> TUMMonocularPlayer::extract_positions_from_poses(const std::vector<Eigen::Matrix4f>& poses) {
    std::vector<Eigen::Vector3f> positions;
    positions.reserve(poses.size());
    
    for (const auto& pose : poses) {
        positions.push_back(pose.block<3, 1>(0, 3));
    }
    
    return positions;
}

} // namespace lightweight_vio
