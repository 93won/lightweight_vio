/**
 * @file      rgbd_player.cpp
 * @brief     RGBD dataset player implementation
 * @author    Seungwon Choi (csw3575@snu.ac.kr)
 * @date      2025-10-28
 * @copyright Copyright (c) 2025 Seungwon Choi. All rights reserved.
 *
 * @par License
 * This project is released under the MIT License.
 */

#include "rgbd_player.h"

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
#include <cmath>

#include <util/Config.h>
#include <processing/Estimator.h>
#include <processing/FeatureTracker.h>
#include <viewer/PangolinViewer.h>
#include <database/Frame.h>
#include <database/Feature.h>
#include <database/MapPoint.h>

namespace lightweight_vio {

RGBDPlayerResult RGBDPlayer::run(const RGBDPlayerConfig& config) {
    RGBDPlayerResult result;
    
    try {
        // Note: Configuration is already loaded in main(), so we skip loading here
        // Config::getInstance().load(config.config_path);
        
        // Verify RGBD camera type
        if (Config::getInstance().m_camera_type != CameraType::RGBD) {
            spdlog::warn("[RGBDPlayer] Config camera type is not RGBD, setting to RGBD");
            Config::getInstance().m_camera_type = CameraType::RGBD;
        }
        
        // Override viewer settings with config values
        Config::getInstance().m_viewer_enable = config.enable_viewer;
        Config::getInstance().m_viewer_width = config.viewer_width;
        Config::getInstance().m_viewer_height = config.viewer_height;
        
        // 2. Load RGBD dataset
        auto image_data = load_rgbd_timestamps(config.dataset_path);
        if (image_data.empty()) {
            result.error_message = "No RGBD images found in dataset";
            return result;
        }
        
        size_t start_frame_idx = 0;
        size_t end_frame_idx = image_data.size();
        
        // Try to setup ground truth matching (optional)
        setup_ground_truth_matching(config.dataset_path, image_data, start_frame_idx, end_frame_idx);
        
        // 3. Initialize systems
        auto viewer = initialize_viewer(config);
        Estimator estimator;
        
        // 4. Process frames
        RGBDFrameContext context;
        context.step_mode = config.step_mode;
        context.auto_play = !config.step_mode;  // auto_play is opposite of step_mode
        context.gt_poses = matched_gt_poses_;  // ⭐ Pass matched GT poses to context
        
        initialize_estimator(estimator, context);
        
        // Initial render to make sure window is visible
        if (viewer) {
            viewer->render();
            std::this_thread::sleep_for(std::chrono::milliseconds(500));
        }
        
        context.current_idx = start_frame_idx;
        while (context.current_idx < end_frame_idx) {
            // Handle viewer controls first
            if (viewer && !handle_viewer_controls(*viewer, context)) {
                break;
            }
            
            bool should_process_frame = false;
            
            // Check processing conditions based on mode
            if (context.auto_play) {
                // Auto mode: process frame
                should_process_frame = true;
            } else {
                // Step mode: only process if advance_frame is set
                if (context.advance_frame) {
                    should_process_frame = true;
                    // Reset advance_frame after processing
                    context.advance_frame = false;
                } else {
                    // In step mode with no advance request, just render and wait
                    if (viewer) {
                        viewer->render();
                    }
                    std::this_thread::sleep_for(std::chrono::milliseconds(30));
                    continue;
                }
            }
            
            if (should_process_frame) {
                // Process single frame
                auto frame_start = std::chrono::high_resolution_clock::now();
                double processing_time = process_single_frame(estimator, context, image_data, 
                                                            config.dataset_path, config.use_vio_mode);
                auto frame_end = std::chrono::high_resolution_clock::now();
                
                auto frame_duration = std::chrono::duration_cast<std::chrono::microseconds>(frame_end - frame_start);
                double total_time_ms = frame_duration.count() / 1000.0;
                result.frame_processing_times.push_back(total_time_ms);
                
                // Update viewer
                if (viewer) {
                    update_viewer(*viewer, estimator, context);
                }
                
                ++context.current_idx;
                ++context.processed_frames;
                
                // Calculate sleep time based on actual frame intervals (only in auto mode)
                if (context.auto_play && context.current_idx < end_frame_idx) {
                    double sleep_time_ms = 50 - total_time_ms;
                    if (sleep_time_ms > 0) {
                        std::this_thread::sleep_for(std::chrono::milliseconds(static_cast<int>(sleep_time_ms)));
                    }
                }
            }
        }
        
        // 6. Save results
        save_trajectories(estimator, context, config.dataset_path, config.use_vio_mode);
        
        // 7. Calculate final statistics
        result.success = true;
        result.processed_frames = context.processed_frames;
        if (!result.frame_processing_times.empty()) {
            result.average_processing_time_ms = std::accumulate(
                result.frame_processing_times.begin(), 
                result.frame_processing_times.end(), 0.0) / result.frame_processing_times.size();
        }
        
        spdlog::info("[RGBDPlayer] Successfully processed {} frames", result.processed_frames);
        
        // Display final statistics summary
        if (config.enable_console_statistics && result.success) {
            spdlog::info("════════════════════════════════════════════════════════════════════");
            spdlog::info("                        RGBD STATISTICS (VO)                        ");
            spdlog::info("════════════════════════════════════════════════════════════════════");
            spdlog::info("");
            spdlog::info("                          TIMING ANALYSIS                           ");
            spdlog::info("════════════════════════════════════════════════════════════════════");
            spdlog::info(" Total Frames Processed: {}", result.processed_frames);
            spdlog::info(" Average Processing Time: {:.2f}ms", result.average_processing_time_ms);
            double fps = 1000.0 / result.average_processing_time_ms;
            spdlog::info(" Average Frame Rate: {:.1f}fps", fps);
            spdlog::info("════════════════════════════════════════════════════════════════════");
        }
        
        // Wait for viewer finish if enabled
        if (viewer) {
            spdlog::info("[RGBDPlayer] Processing completed! Click 'Finish & Exit' to close.");
            while (!viewer->should_close() && !viewer->is_finish_requested()) {
                viewer->render();
                std::this_thread::sleep_for(std::chrono::milliseconds(30));
            }
        }
        
    } catch (const std::exception& e) {
        result.error_message = e.what();
        spdlog::error("[RGBDPlayer] Exception occurred: {}", e.what());
    }
    
    return result;
}

std::vector<RGBDImageData> RGBDPlayer::load_rgbd_timestamps(const std::string& dataset_path) {
    std::vector<RGBDImageData> image_data;
    
    // Try to load timestamps.txt (HWASHIN format)
    std::string timestamps_file = dataset_path + "/timestamps.txt";
    std::ifstream ts_stream(timestamps_file);
    
    if (ts_stream.is_open()) {
        std::string line;
        int frame_idx = 0;
        while (std::getline(ts_stream, line)) {
            if (line.empty() || line[0] == '#') continue;
            
            double timestamp_sec;
            std::stringstream ss(line);
            if (ss >> timestamp_sec) {
                RGBDImageData data;
                data.timestamp = static_cast<long long>(timestamp_sec * 1e9);
                
                // Generate filenames based on frame index
                std::ostringstream rgb_fn, depth_fn;
                rgb_fn << "images/" << std::setfill('0') << std::setw(6) << frame_idx << ".png";
                depth_fn << "depths/" << std::setfill('0') << std::setw(6) << frame_idx << ".png";
                
                data.rgb_filename = rgb_fn.str();
                data.depth_filename = depth_fn.str();
                
                image_data.push_back(data);
                frame_idx++;
            }
        }
        ts_stream.close();
        spdlog::info("Loaded {} RGBD frames", image_data.size());
        return image_data;
    }
    
    // Fallback to TUM RGB-D format (rgb.txt and depth.txt)
    spdlog::info("[RGBDPlayer] Loading TUM RGB-D format (rgb.txt and depth.txt)");
    
    // Load RGB timestamps
    std::string rgb_file = dataset_path + "/rgb.txt";
    std::map<long long, std::string> rgb_map;
    
    std::ifstream rgb_stream(rgb_file);
    if (!rgb_stream.is_open()) {
        spdlog::error("[RGBDPlayer] Cannot open rgb.txt or timestamps.txt file");
        return image_data;
    }
    
    std::string line;
    while (std::getline(rgb_stream, line)) {
        if (line.empty() || line[0] == '#') continue;
        
        std::stringstream ss(line);
        double timestamp_sec;
        std::string filename;
        
        if (ss >> timestamp_sec >> filename) {
            long long timestamp_ns = static_cast<long long>(timestamp_sec * 1e9);
            rgb_map[timestamp_ns] = utils::trim(filename);
        }
    }
    rgb_stream.close();
    
    // Load Depth timestamps
    std::string depth_file = dataset_path + "/depth.txt";
    std::map<long long, std::string> depth_map;
    
    std::ifstream depth_stream(depth_file);
    if (!depth_stream.is_open()) {
        spdlog::error("[RGBDPlayer] Cannot open depth.txt file: {}", depth_file);
        return image_data;
    }
    
    while (std::getline(depth_stream, line)) {
        if (line.empty() || line[0] == '#') continue;
        
        std::stringstream ss(line);
        double timestamp_sec;
        std::string filename;
        
        if (ss >> timestamp_sec >> filename) {
            long long timestamp_ns = static_cast<long long>(timestamp_sec * 1e9);
            depth_map[timestamp_ns] = utils::trim(filename);
        }
    }
    depth_stream.close();
    
    // Match RGB and Depth timestamps (tolerance: 1ms = 1e6 ns)
    const long long tolerance_ns = 1000000; // 1ms
    
    for (const auto& rgb_entry : rgb_map) {
        long long rgb_ts = rgb_entry.first;
        const std::string& rgb_fn = rgb_entry.second;
        
        // Find closest depth timestamp
        long long best_depth_ts = -1;
        long long min_diff = std::numeric_limits<long long>::max();
        
        for (const auto& depth_entry : depth_map) {
            long long depth_ts = depth_entry.first;
            long long diff = std::abs(rgb_ts - depth_ts);
            
            if (diff < min_diff) {
                min_diff = diff;
                best_depth_ts = depth_ts;
            }
        }
        
        // Check if match is within tolerance
        if (best_depth_ts != -1 && min_diff < tolerance_ns) {
            RGBDImageData data;
            data.timestamp = rgb_ts;
            data.rgb_filename = rgb_fn;
            data.depth_filename = depth_map[best_depth_ts];
            image_data.push_back(data);
        }
    }
    
    spdlog::info("[RGBDPlayer] Loaded {} RGBD image pairs (TUM format)", image_data.size());
    return image_data;
}

cv::Mat RGBDPlayer::load_rgb_image(const std::string& dataset_path, const std::string& filename) {
    std::string full_path = dataset_path + "/" + filename;
    
    // Load as color image for dense point cloud visualization
    // But convert to grayscale for feature tracking if needed
    cv::Mat image = cv::imread(full_path, cv::IMREAD_COLOR);
    
    if (image.empty()) {
        spdlog::error("[RGBDPlayer] Cannot load RGB image: {}", full_path);
    }
    
    return image;
}

cv::Mat RGBDPlayer::load_depth_image(const std::string& dataset_path, const std::string& filename) {
    std::string full_path = dataset_path + "/" + filename;
    cv::Mat depth = cv::imread(full_path, cv::IMREAD_ANYDEPTH);
    
    if (depth.empty()) {
        spdlog::error("[RGBDPlayer] Cannot load depth image: {}", full_path);
    }
    
    return depth;
}

bool RGBDPlayer::setup_ground_truth_matching(const std::string& dataset_path,
                                            const std::vector<RGBDImageData>& image_data,
                                            size_t& start_frame_idx,
                                            size_t& end_frame_idx) {
    // Try TUM format first (ground_truth.txt)
    if (load_ground_truth_tum_format(dataset_path)) {
        spdlog::info("[RGBDPlayer] Using TUM format ground truth (ground_truth.txt)");
    }
    // Fallback to EuRoC format (gt.csv)
    else if (load_ground_truth_euroc_format(dataset_path)) {
        spdlog::info("[RGBDPlayer] Using EuRoC format ground truth (gt.csv)");
    }
    else {
        return false;
    }
    
    if (gt_data_.empty()) {
        return false;
    }
    
    // Extract timestamps for matching
    std::vector<long long> image_timestamps;
    image_timestamps.reserve(image_data.size());
    for (const auto& img : image_data) {
        image_timestamps.push_back(img.timestamp);
    }
    
    // Match with ground truth
    if (!match_image_timestamps_with_gt(image_timestamps)) {
        spdlog::debug("[RGBDPlayer] Failed to match image timestamps with ground truth");
        return false;
    }
    
    size_t matched_count = matched_gt_poses_.size();
    if (matched_count == 0) {
        spdlog::debug("[RGBDPlayer] No matched timestamps found");
        return false;
    }
    
    // Find valid frame range
    long long first_matched_ts = matched_image_timestamps_[0];
    long long last_matched_ts = matched_image_timestamps_[matched_count - 1];
    
    // Find corresponding indices
    for (size_t i = 0; i < image_data.size(); ++i) {
        if (image_data[i].timestamp == first_matched_ts) {
            start_frame_idx = i;
            break;
        }
    }
    
    for (size_t i = image_data.size(); i > 0; --i) {
        if (image_data[i-1].timestamp == last_matched_ts) {
            end_frame_idx = i;
            break;
        }
    }
    
    // Get first matched GT pose for initialization
    if (!matched_gt_poses_.empty()) {
        first_matched_gt_pose_ = matched_gt_poses_[0];
        has_first_matched_gt_pose_ = true;
        spdlog::info("[RGBDPlayer] Found first matched GT pose at frame index {}", start_frame_idx);
    }
    
    spdlog::info("[RGBDPlayer] Ground truth matched: {} frames, range {} to {}", 
                matched_count, start_frame_idx, end_frame_idx);
    return true;
}

bool RGBDPlayer::load_ground_truth_euroc_format(const std::string& dataset_path) {
    std::string gt_file_path = dataset_path + "/gt.csv";
    
    std::ifstream file(gt_file_path);
    if (!file.is_open()) {
        return false;
    }
    
    std::string line;
    // Skip header line
    if (!std::getline(file, line)) {
        spdlog::debug("[RGBDPlayer] Ground truth file is empty");
        return false;
    }
    
    gt_data_.clear();
    
    int line_count = 0;
    while (std::getline(file, line)) {
        line_count++;
        
        std::istringstream iss(line);
        std::string token;
        std::vector<double> values;
        
        // Parse CSV values
        while (std::getline(iss, token, ',')) {
            try {
                values.push_back(std::stod(token));
            } catch (const std::exception& e) {
                spdlog::error("[RGBDPlayer] Failed to parse value '{}' in line {}: {}", token, line_count, e.what());
                continue;
            }
        }
        
        if (values.size() < 17) {  // timestamp + 3 position + 4 quaternion + 3 velocity + 3 bg + 3 ba
            spdlog::warn("[RGBDPlayer] Incomplete data in line {}, got {} values, expected 17", line_count, values.size());
            continue;
        }
        
        GroundTruthPose gt_pose;
        
        // Extract data: timestamp, p_RS_R (position), q_RS (quaternion)
        gt_pose.timestamp = static_cast<long long>(values[0]);
        Eigen::Vector3f position(values[1], values[2], values[3]);
        Eigen::Quaternionf quaternion(values[4], values[5], values[6], values[7]);  // w,x,y,z format
        
        // Create 4x4 transformation matrix (T_WS - world to sensor)
        Eigen::Matrix4f T_WS = Eigen::Matrix4f::Identity();
        T_WS.block<3,3>(0,0) = quaternion.toRotationMatrix();
        T_WS.block<3,1>(0,3) = position;
        
        // Inverse to get T_SW (sensor to world), which is what we use as Twb
        gt_pose.pose = T_WS.inverse();
        
        // Extract additional data
        gt_pose.velocity = Eigen::Vector3f(values[8], values[9], values[10]);
        gt_pose.bias_gyro = Eigen::Vector3f(values[11], values[12], values[13]);
        gt_pose.bias_accel = Eigen::Vector3f(values[14], values[15], values[16]);
        
        gt_data_.push_back(gt_pose);
    }
    
    file.close();
    spdlog::info("[RGBDPlayer] Loaded {} ground truth poses from {}", gt_data_.size(), gt_file_path);
    return !gt_data_.empty();
}

bool RGBDPlayer::load_ground_truth_tum_format(const std::string& dataset_path) {
    std::string gt_file_path = dataset_path + "/ground_truth.txt";
    
    std::ifstream file(gt_file_path);
    if (!file.is_open()) {
        // Not an error, just no TUM format GT available
        return false;
    }
    
    std::string line;
    gt_data_.clear();
    
    int line_count = 0;
    while (std::getline(file, line)) {
        line_count++;
        
        // Skip comments and empty lines
        if (line.empty() || line[0] == '#') {
            continue;
        }
        
        std::istringstream iss(line);
        std::vector<double> values;
        double value;
        
        // Parse space-separated values
        while (iss >> value) {
            values.push_back(value);
        }
        
        // TUM format: timestamp tx ty tz qx qy qz qw (8 values)
        if (values.size() < 8) {
            spdlog::warn("[RGBDPlayer] Incomplete data in line {}, got {} values, expected 8", line_count, values.size());
            continue;
        }
        
        GroundTruthPose gt_pose;
        
        // Extract timestamp (convert seconds to nanoseconds)
        gt_pose.timestamp = static_cast<long long>(values[0] * 1e9);
        
        // Extract position
        Eigen::Vector3f position(values[1], values[2], values[3]);
        
        // Extract quaternion (TUM format: qx qy qz qw)
        Eigen::Quaternionf quaternion(values[7], values[4], values[5], values[6]);  // w,x,y,z format
        
        // Create 4x4 transformation matrix (T_WB - world to body)
        Eigen::Matrix4f T_WB = Eigen::Matrix4f::Identity();
        T_WB.block<3,3>(0,0) = quaternion.toRotationMatrix();
        T_WB.block<3,1>(0,3) = position;
        
        gt_pose.pose = T_WB;
        
        // No velocity/bias data in TUM format
        gt_pose.velocity = Eigen::Vector3f::Zero();
        gt_pose.bias_gyro = Eigen::Vector3f::Zero();
        gt_pose.bias_accel = Eigen::Vector3f::Zero();
        
        gt_data_.push_back(gt_pose);
    }
    
    file.close();
    spdlog::info("[RGBDPlayer] Loaded {} ground truth poses from {} (TUM format)", gt_data_.size(), gt_file_path);
    return !gt_data_.empty();
}

bool RGBDPlayer::match_image_timestamps_with_gt(const std::vector<long long>& image_timestamps) {
    if (gt_data_.empty() || image_timestamps.empty()) {
        return false;
    }
    
    matched_gt_poses_.clear();
    matched_image_timestamps_.clear();
    
    const long long time_tolerance_ns = 10000000LL;  // 10ms tolerance
    
    for (long long img_ts : image_timestamps) {
        // Find closest GT timestamp
        long long min_diff = std::numeric_limits<long long>::max();
        size_t best_idx = 0;
        bool found = false;
        
        for (size_t i = 0; i < gt_data_.size(); ++i) {
            long long diff = std::abs(img_ts - gt_data_[i].timestamp);
            if (diff < min_diff) {
                min_diff = diff;
                best_idx = i;
                found = true;
            }
            
            // Early exit if we're moving away from the target
            if (found && diff > min_diff * 2) {
                break;
            }
        }
        
        // Only add if within tolerance
        if (found && min_diff < time_tolerance_ns) {
            matched_gt_poses_.push_back(gt_data_[best_idx].pose);
            matched_image_timestamps_.push_back(img_ts);
        }
    }
    
    spdlog::info("[RGBDPlayer] Matched {}/{} image timestamps with ground truth", 
                matched_gt_poses_.size(), image_timestamps.size());
    
    // ⭐ Set first matched GT pose for initialization
    if (!matched_gt_poses_.empty()) {
        first_matched_gt_pose_ = matched_gt_poses_[0];
        has_first_matched_gt_pose_ = true;
        
        Eigen::Vector3f first_pos = first_matched_gt_pose_.block<3,1>(0,3);
        spdlog::info("[RGBDPlayer] First matched GT pose at: [{:.3f}, {:.3f}, {:.3f}]",
                    first_pos.x(), first_pos.y(), first_pos.z());
    }
    
    return !matched_gt_poses_.empty();
}

std::unique_ptr<PangolinViewer> RGBDPlayer::initialize_viewer(const RGBDPlayerConfig& config) {
    if (!config.enable_viewer) {
        return nullptr;
    }
    
    auto viewer = std::make_unique<PangolinViewer>();
    if (viewer->initialize(config.viewer_width, config.viewer_height)) {
        // Wait for viewer to be ready
        while (!viewer->is_ready()) {
            viewer->render();
            std::this_thread::sleep_for(std::chrono::milliseconds(16));
        }
        return viewer;
    } else {
        spdlog::warn("Failed to initialize viewer");
        return nullptr;
    }
}

void RGBDPlayer::initialize_estimator(Estimator& estimator, const RGBDFrameContext& context) {
    // Set initial ground truth pose if available
    if (has_first_matched_gt_pose_) {
        estimator.set_initial_gt_pose(first_matched_gt_pose_);
        
        // Log the GT pose for debugging
        Eigen::Vector3f gt_pos = first_matched_gt_pose_.block<3,1>(0,3);
        spdlog::info("[RGBDPlayer] Set initial ground truth pose at position: [{:.3f}, {:.3f}, {:.3f}]", 
                    gt_pos.x(), gt_pos.y(), gt_pos.z());
    }
}

double RGBDPlayer::process_single_frame(Estimator& estimator,
                                       RGBDFrameContext& context,
                                       const std::vector<RGBDImageData>& image_data,
                                       const std::string& dataset_path,
                                       bool use_vio_mode) {
    auto start_time = std::chrono::high_resolution_clock::now();
    
    // Load RGBD images (rgb_image is now in COLOR format)
    cv::Mat rgb_image = load_rgb_image(dataset_path, image_data[context.current_idx].rgb_filename);
    cv::Mat depth_image = load_depth_image(dataset_path, image_data[context.current_idx].depth_filename);
    
    if (rgb_image.empty() || depth_image.empty()) {
        spdlog::warn("[RGBDPlayer] Skipping frame {} due to empty image", context.current_idx);
        return 0.0;
    }
    
    // Convert RGB to grayscale for feature tracking
    cv::Mat gray_image;
    if (rgb_image.channels() == 3) {
        cv::cvtColor(rgb_image, gray_image, cv::COLOR_BGR2GRAY);
    } else {
        gray_image = rgb_image.clone();
    }
    
    // Preprocess grayscale image for feature tracking
    cv::Mat processed_gray = preprocess_image(gray_image);
    
    // Process RGBD frame through estimator
    // Pass processed grayscale for tracking, but keep original RGB for visualization
    // Convert nanosecond timestamp to seconds
    double timestamp_sec = image_data[context.current_idx].timestamp * 1e-9;
    estimator.process_rgbd_frame(processed_gray, depth_image, timestamp_sec);
    
    // IMPORTANT: Store original RGB image in the frame (for dense point cloud coloring)
    // Use std::move to transfer ownership and avoid extra copy
    auto current_frame = estimator.get_current_frame();
    if (current_frame && current_frame->is_rgbd()) {
        // Move RGB image to frame (no clone, ownership transferred)
        current_frame->set_rgb_image(std::move(rgb_image));
    }
    
    // Release player's local image memory after passing to estimator
    // Note: rgb_image is already moved, but still call release for safety
    rgb_image.release();
    depth_image.release();
    gray_image.release();
    processed_gray.release();
    
    // Handle ground truth pose matching
    if (context.processed_frames < matched_gt_poses_.size()) {
        // Add matched GT pose for current frame
        context.gt_poses.push_back(matched_gt_poses_[context.processed_frames]);
    }
    
    // Update frame timestamp
    context.previous_frame_timestamp = image_data[context.current_idx].timestamp;
    
    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);
    return duration.count() / 1000.0; // Return milliseconds
}

cv::Mat RGBDPlayer::preprocess_image(const cv::Mat& input_image) {
    cv::Mat downsampled, equalized_small, clahe_applied, upsampled;
    
    // Downsample to reduce computation (0.5x scale = 1/4 pixels)
    cv::resize(input_image, downsampled, cv::Size(), 0.5, 0.5, cv::INTER_LINEAR);
    
    // Global histogram equalization on smaller image
    cv::equalizeHist(downsampled, equalized_small);
    
    // CLAHE for local contrast enhancement on smaller image
    cv::Ptr<cv::CLAHE> clahe = cv::createCLAHE(2.0, cv::Size(8, 8));
    clahe->apply(equalized_small, clahe_applied);
    
    // Upsample back to original size
    cv::resize(clahe_applied, upsampled, input_image.size(), 0, 0, cv::INTER_LINEAR);
    
    return upsampled;
}

bool RGBDPlayer::handle_viewer_controls(PangolinViewer& viewer, RGBDFrameContext& context) {
    // Check for exit conditions
    if (viewer.should_close() || viewer.is_finish_requested()) {
        spdlog::info("[RGBDPlayer] User requested exit");
        return false;
    }
    
    // Process keyboard input and sync UI state
    viewer.process_keyboard_input(context.auto_play, context.step_mode, context.advance_frame);
    viewer.sync_ui_state(context.auto_play, context.step_mode);
    
    return true;
}

void RGBDPlayer::update_viewer(PangolinViewer& viewer,
                              const Estimator& estimator,
                              const RGBDFrameContext& context) {
    // Try to get current frame, if null try to get from all frames or keyframes
    auto current_frame = estimator.get_current_frame();
    if (!current_frame) {
        const auto& all_frames = estimator.get_all_frames();
        if (!all_frames.empty()) {
            current_frame = all_frames.back();
        }
    }
    
    if (!current_frame) {
        const auto keyframes = estimator.get_keyframes_safe();
        if (!keyframes.empty()) {
            current_frame = keyframes.back();
        }
    }
    
    if (!current_frame) {
        spdlog::error("[RGBDPlayer] Cannot get any frame! Skipping viewer update");
        return;
    }
    
    // Update poses
    Eigen::Matrix4f current_pose = current_frame->get_Twb();
    viewer.update_pose(current_pose);
    viewer.update_camera_pose(current_frame->get_Twc());
    
    // Update trajectory
    static std::vector<Eigen::Matrix4f> trajectory_poses;
    trajectory_poses.push_back(current_pose);
    auto trajectory_positions = extract_positions_from_poses(trajectory_poses);
    viewer.update_trajectory(trajectory_positions);
    
    // ⭐ Update ground truth trajectory (only once or when changed)
    static bool gt_trajectory_sent = false;
    if (!gt_trajectory_sent && !matched_gt_poses_.empty()) {
        auto gt_positions = extract_positions_from_poses(matched_gt_poses_);
        viewer.update_ground_truth_trajectory(gt_positions);
        gt_trajectory_sent = true;
        spdlog::info("[RGBDPlayer] Sent {} GT trajectory points to viewer", gt_positions.size());
    }
    
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
    
    // Calculate position error if GT available
    float position_error = 0.0f;
    if (context.processed_frames < context.gt_poses.size()) {
        Eigen::Vector3f gt_pos = context.gt_poses[context.processed_frames].block<3,1>(0,3);
        Eigen::Vector3f est_pos = current_pose.block<3,1>(0,3);
        position_error = (gt_pos - est_pos).norm();
    }
    
    viewer.update_tracking_stats(context.processed_frames + 1, total_features, 
                               map_points_count, map_points_count, success_rate, position_error);
    
    // Update tracking view with frame directly
    viewer.update_tracking_with_frame(current_frame);
    
    // ⭐ Update dense point cloud (RGBD only)
    if (current_frame->is_rgbd()) {
        viewer.update_dense_point_cloud(current_frame);
        viewer.update_depth_image(current_frame);  // ⭐ Add depth heatmap
    }
    
    viewer.render();
}

void RGBDPlayer::save_trajectories(const Estimator& estimator,
                                  const RGBDFrameContext& context,
                                  const std::string& dataset_path,
                                  bool use_vio_mode) {
    std::string mode_suffix = use_vio_mode ? "vio" : "vo";
    
    // Save estimated trajectory
    std::string est_file = dataset_path + "/estimated_trajectory_rgbd_" + mode_suffix + ".txt";
    std::ofstream est_out(est_file);
    if (est_out.is_open()) {
        const auto& all_frames = estimator.get_all_frames();
        spdlog::info("[RGBDPlayer] Saving {} frames to estimated trajectory", all_frames.size());
        
        for (const auto& frame : all_frames) {
            if (!frame) continue;
            
            Eigen::Matrix4f T_wb = frame->get_Twb();
            Eigen::Vector3f translation = T_wb.block<3, 1>(0, 3);
            Eigen::Matrix3f rotation = T_wb.block<3, 3>(0, 0);
            Eigen::Quaternionf quat(rotation);
            
            double timestamp_sec = static_cast<double>(frame->get_timestamp()) / 1e9;
            
            est_out << std::fixed << std::setprecision(6) << timestamp_sec << " "
                    << std::setprecision(8)
                    << translation.x() << " " << translation.y() << " " << translation.z() << " "
                    << quat.x() << " " << quat.y() << " " << quat.z() << " " << quat.w() << std::endl;
        }
        est_out.close();
        spdlog::info("[RGBDPlayer] Saved estimated trajectory to: {}", est_file);
    }
    
    // Save ground truth trajectory
    if (!context.gt_poses.empty()) {
        std::string gt_file = dataset_path + "/ground_truth_rgbd_" + mode_suffix + ".txt";
        std::ofstream gt_out(gt_file);
        if (gt_out.is_open()) {
            for (size_t i = 0; i < context.gt_poses.size(); ++i) {
                const auto& gt_pose = context.gt_poses[i];
                
                Eigen::Vector3f translation = gt_pose.block<3, 1>(0, 3);
                Eigen::Matrix3f rotation = gt_pose.block<3, 3>(0, 0);
                Eigen::Quaternionf quat(rotation);
                
                // Find matching timestamp from matched timestamps
                if (i < matched_image_timestamps_.size()) {
                    double timestamp_sec = static_cast<double>(matched_image_timestamps_[i]) / 1e9;
                    
                    gt_out << std::fixed << std::setprecision(6) << timestamp_sec << " "
                           << std::setprecision(8)
                           << translation.x() << " " << translation.y() << " " << translation.z() << " "
                           << quat.x() << " " << quat.y() << " " << quat.z() << " " << quat.w() << std::endl;
                }
            }
            gt_out.close();
            spdlog::info("[RGBDPlayer] Saved ground truth trajectory to: {}", gt_file);
        }
    }
}

RGBDPlayerResult::ErrorStats RGBDPlayer::analyze_transform_errors(const Estimator& estimator,
                                                                  const std::vector<Eigen::Matrix4f>& gt_poses,
                                                                  bool use_vio_mode) {
    RGBDPlayerResult::ErrorStats stats;
    
    if (gt_poses.empty()) {
        spdlog::warn("[RGBDPlayer] No ground truth data for error analysis");
        return stats;
    }
    
    const auto& all_frames = estimator.get_all_frames();
    std::vector<double> rotation_errors;
    std::vector<double> translation_errors;
    
    for (size_t i = 1; i < all_frames.size() && i < gt_poses.size(); ++i) {
        if (!all_frames[i-1] || !all_frames[i]) continue;
        
        // Calculate frame-to-frame transforms
        Eigen::Matrix4f T_est_prev = all_frames[i-1]->get_Twb();
        Eigen::Matrix4f T_est_curr = all_frames[i]->get_Twb();
        Eigen::Matrix4f T_est_rel = T_est_prev.inverse() * T_est_curr;
        
        Eigen::Matrix4f T_gt_prev = gt_poses[i-1];
        Eigen::Matrix4f T_gt_curr = gt_poses[i];
        Eigen::Matrix4f T_gt_rel = T_gt_prev.inverse() * T_gt_curr;
        
        // Calculate relative error
        Eigen::Matrix4f T_error = T_gt_rel.inverse() * T_est_rel;
        
        // Extract rotation error (in degrees)
        Eigen::Matrix3f R_error = T_error.block<3,3>(0,0);
        Eigen::AngleAxisf angle_axis(R_error);
        double rotation_error_deg = std::abs(angle_axis.angle()) * 180.0 / M_PI;
        
        // Extract translation error (in meters)
        Eigen::Vector3f t_error = T_error.block<3,1>(0,3);
        double translation_error_m = t_error.norm();
        
        rotation_errors.push_back(rotation_error_deg);
        translation_errors.push_back(translation_error_m);
    }
    
    if (!rotation_errors.empty()) {
        stats.available = true;
        stats.total_frame_pairs = rotation_errors.size();
        stats.total_frames = all_frames.size();
        stats.gt_poses_count = gt_poses.size();
        
        // Sort for median calculation
        std::vector<double> rot_sorted = rotation_errors;
        std::vector<double> trans_sorted = translation_errors;
        std::sort(rot_sorted.begin(), rot_sorted.end());
        std::sort(trans_sorted.begin(), trans_sorted.end());
        
        // Rotation statistics
        stats.rotation_mean = std::accumulate(rotation_errors.begin(), rotation_errors.end(), 0.0) / rotation_errors.size();
        stats.rotation_rmse = std::sqrt(std::accumulate(rotation_errors.begin(), rotation_errors.end(), 0.0, 
            [](double sum, double err) { return sum + err * err; }) / rotation_errors.size());
        stats.rotation_median = rot_sorted[rot_sorted.size() / 2];
        stats.rotation_min = *std::min_element(rotation_errors.begin(), rotation_errors.end());
        stats.rotation_max = *std::max_element(rotation_errors.begin(), rotation_errors.end());
        
        // Translation statistics
        stats.translation_mean = std::accumulate(translation_errors.begin(), translation_errors.end(), 0.0) / translation_errors.size();
        stats.translation_rmse = std::sqrt(std::accumulate(translation_errors.begin(), translation_errors.end(), 0.0,
            [](double sum, double err) { return sum + err * err; }) / translation_errors.size());
        stats.translation_median = trans_sorted[trans_sorted.size() / 2];
        stats.translation_min = *std::min_element(translation_errors.begin(), translation_errors.end());
        stats.translation_max = *std::max_element(translation_errors.begin(), translation_errors.end());
    }
    
    return stats;
}

RGBDPlayerResult::VelocityStats RGBDPlayer::analyze_velocity_statistics(const Estimator& estimator,
                                                                        const std::vector<Eigen::Matrix4f>& gt_poses) {
    RGBDPlayerResult::VelocityStats stats;
    
    // Velocity analysis implementation (similar to EuRoC player)
    // For now, return empty stats
    return stats;
}

void RGBDPlayer::save_statistics(const RGBDPlayerResult& result,
                                const std::string& dataset_path,
                                bool use_vio_mode) {
    std::string mode_suffix = use_vio_mode ? "vio" : "vo";
    std::string stats_file = dataset_path + "/statistics_rgbd_" + mode_suffix + ".txt";
    
    std::ofstream out(stats_file);
    if (!out.is_open()) {
        spdlog::warn("[RGBDPlayer] Failed to open statistics file: {}", stats_file);
        return;
    }
    
    out << "RGBD Player Statistics (" << (use_vio_mode ? "VIO" : "VO") << ")" << std::endl;
    out << "===========================================" << std::endl;
    out << "Processed Frames: " << result.processed_frames << std::endl;
    out << "Average Processing Time: " << result.average_processing_time_ms << " ms" << std::endl;
    out << "Average FPS: " << (1000.0 / result.average_processing_time_ms) << std::endl;
    out << std::endl;
    
    if (result.error_stats.available) {
        out << "Error Statistics:" << std::endl;
        out << "  Rotation RMSE: " << result.error_stats.rotation_rmse << " deg" << std::endl;
        out << "  Translation RMSE: " << result.error_stats.translation_rmse << " m" << std::endl;
    }
    
    out.close();
    spdlog::info("[RGBDPlayer] Saved statistics to: {}", stats_file);
}

std::vector<Eigen::Vector3f> RGBDPlayer::extract_positions_from_poses(const std::vector<Eigen::Matrix4f>& poses) {
    std::vector<Eigen::Vector3f> positions;
    positions.reserve(poses.size());
    
    for (const auto& pose : poses) {
        positions.push_back(pose.block<3,1>(0,3));
    }
    
    return positions;
}

} // namespace lightweight_vio
