/**
 * @file      tum_player.cpp
 * @brief     TUM dataset player implementation
 * @author    Seungwon Choi (csw3575@snu.ac.kr)
 * @date      2025-09-27
 * @copyright Copyright (c) 2025 Seungwon Choi. All rights reserved.
 *
 * @par License
 * This project is released under the MIT License.
 */

#include "tum_player.h"

#include <spdlog/sinks/stdout_color_sinks.h>
#include <iostream>
#include <fstream>
#include <sstream>
#include <algorithm>
#include <chrono>
#include <iomanip>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <thread>
#include <numeric>
#include <algorithm>
#include <set>

#include <util/Config.h>
#include <util/TUMUtils.h>
#include <processing/Estimator.h>
#include <processing/FeatureTracker.h>
#include <viewer/PangolinViewer.h>
#include <database/Frame.h>
#include <database/Feature.h>
#include <database/MapPoint.h>

namespace lightweight_vio {

TUMPlayerResult TUMPlayer::run(const TUMPlayerConfig& config) {
    TUMPlayerResult result;
    
    try {
        // 1. Load configuration
        Config::getInstance().load(config.config_path);
        spdlog::info("[TUMPlayer] Successfully loaded configuration from: {}", config.config_path);
        
        // Override viewer settings with config values (to respect caller's settings)
        Config::getInstance().m_viewer_enable = config.enable_viewer;
        Config::getInstance().m_viewer_width = config.viewer_width;
        Config::getInstance().m_viewer_height = config.viewer_height;
        
        // 2. Load dataset and setup ground truth
        auto image_data = load_image_timestamps(config.dataset_path);
        if (image_data.empty()) {
            result.error_message = "No images found in dataset";
            return result;
        }
        
        size_t start_frame_idx = 0;
        size_t end_frame_idx = image_data.size();
        
        if (!setup_ground_truth_matching(config.dataset_path, image_data, start_frame_idx, end_frame_idx)) {
            spdlog::warn("[TUMPlayer] Failed to setup ground truth matching, using all frames");
        }
        
        // 3. Load IMU data if VIO mode
        if (config.use_vio_mode) {
            if (!load_imu_data(config.dataset_path, image_data, start_frame_idx, end_frame_idx)) {
                result.error_message = "Failed to load IMU data for VIO mode";
                return result;
            }
        }
        
        // 4. Initialize systems
        auto viewer = initialize_viewer(config);
        Estimator estimator;
        initialize_estimator(estimator, image_data);
        
        // 5. Process frames
        FrameContext context;
        context.step_mode = config.step_mode;
        context.auto_play = !config.step_mode;  // auto_play is opposite of step_mode
        
        spdlog::info("[TUMPlayer] Processing frames {} to {} ({} mode)", 
                    start_frame_idx, end_frame_idx, config.use_vio_mode ? "VIO" : "VO");
        
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
                
                // Progress logging
                if (context.processed_frames % 100 == 0) {
                    spdlog::info("[TUMPlayer] Processed {} / {} frames", 
                                context.processed_frames, end_frame_idx - start_frame_idx);
                }
                
                ++context.current_idx;
                ++context.processed_frames;
                
                // Calculate sleep time based on actual frame intervals (only in auto mode)
                if (context.auto_play && context.current_idx < end_frame_idx) {
                    long long current_timestamp = image_data[context.current_idx - 1].timestamp;
                    long long next_timestamp = image_data[context.current_idx].timestamp;
                    double frame_interval_ms = (next_timestamp - current_timestamp) / 1e6; // nanoseconds to milliseconds
                    
                    double sleep_time_ms = frame_interval_ms - total_time_ms;
                    if (sleep_time_ms > 0) {
                        std::this_thread::sleep_for(std::chrono::milliseconds(static_cast<int>(sleep_time_ms)));
                    }
                }
            }
        }
        
        // 6. Save results
        if (config.enable_statistics) {
            save_trajectories(estimator, context, config.dataset_path, config.use_vio_mode);
            result.error_stats = analyze_transform_errors(estimator, context.gt_poses, config.use_vio_mode);
            result.velocity_stats = analyze_velocity_statistics(estimator, context.gt_poses);
            save_statistics(result, config.dataset_path, config.use_vio_mode);
        }
        
        // 7. Calculate final statistics
        result.success = true;
        result.processed_frames = context.processed_frames;
        if (!result.frame_processing_times.empty()) {
            result.average_processing_time_ms = std::accumulate(
                result.frame_processing_times.begin(), 
                result.frame_processing_times.end(), 0.0) / result.frame_processing_times.size();
        }
        
        spdlog::info("[TUMPlayer] Successfully processed {} frames", result.processed_frames);
        
        // Display final statistics summary
        if (config.enable_console_statistics && result.success) {
            spdlog::info("════════════════════════════════════════════════════════════════════");
            spdlog::info("                          STATISTICS ({})                          ", config.use_vio_mode ? "VIO" : "VO");
            spdlog::info("════════════════════════════════════════════════════════════════════");
            spdlog::info("");
            spdlog::info("                          TIMING ANALYSIS                           ");
            spdlog::info("════════════════════════════════════════════════════════════════════");
            spdlog::info(" Total Frames Processed: {}", result.processed_frames);
            spdlog::info(" Average Processing Time: {:.2f}ms", result.average_processing_time_ms);
            double fps = 1000.0 / result.average_processing_time_ms;
            spdlog::info(" Average Frame Rate: {:.1f}fps", fps);
            spdlog::info("");
            
            if (result.velocity_stats.available) {
                spdlog::info("                          VELOCITY ANALYSIS                         ");
                spdlog::info("════════════════════════════════════════════════════════════════════");
                spdlog::info("                        LINEAR VELOCITY (m/s)                       ");
                spdlog::info(" Mean      : {:>10.4f}m/s", result.velocity_stats.linear_vel_mean);
                spdlog::info(" Median    : {:>10.4f}m/s", result.velocity_stats.linear_vel_median);
                spdlog::info(" Minimum   : {:>10.4f}m/s", result.velocity_stats.linear_vel_min);
                spdlog::info(" Maximum   : {:>10.4f}m/s", result.velocity_stats.linear_vel_max);
                spdlog::info("");
                spdlog::info("                       ANGULAR VELOCITY (rad/s)                     ");
                spdlog::info(" Mean      : {:>10.4f}rad/s", result.velocity_stats.angular_vel_mean);
                spdlog::info(" Median    : {:>10.4f}rad/s", result.velocity_stats.angular_vel_median);
                spdlog::info(" Minimum   : {:>10.4f}rad/s", result.velocity_stats.angular_vel_min);
                spdlog::info(" Maximum   : {:>10.4f}rad/s", result.velocity_stats.angular_vel_max);
                spdlog::info("");
            }
            
            if (result.error_stats.available) {
                spdlog::info("               FRAME-TO-FRAME TRANSFORM ERROR ANALYSIS              ");
                spdlog::info("════════════════════════════════════════════════════════════════════");
                spdlog::info(" Total Frame Pairs Analyzed: {} (all_frames: {}, gt_poses: {})", 
                            result.error_stats.total_frame_pairs, result.error_stats.total_frames, result.error_stats.gt_poses_count);
                spdlog::info(" Frame precision: 32 bit floats");
                spdlog::info("");
                spdlog::info("                     ROTATION ERROR STATISTICS                    ");
                spdlog::info(" Mean      : {:>10.4f}°", result.error_stats.rotation_mean);
                spdlog::info(" Median    : {:>10.4f}°", result.error_stats.rotation_median);
                spdlog::info(" Minimum   : {:>10.4f}°", result.error_stats.rotation_min);
                spdlog::info(" Maximum   : {:>10.4f}°", result.error_stats.rotation_max);
                spdlog::info(" RMSE      : {:>10.4f}°", result.error_stats.rotation_rmse);
                spdlog::info("");
                spdlog::info("                   TRANSLATION ERROR STATISTICS                   ");
                spdlog::info(" Mean      : {:>10.6f}m", result.error_stats.translation_mean);
                spdlog::info(" Median    : {:>10.6f}m", result.error_stats.translation_median);
                spdlog::info(" Minimum   : {:>10.6f}m", result.error_stats.translation_min);
                spdlog::info(" Maximum   : {:>10.6f}m", result.error_stats.translation_max);
                spdlog::info(" RMSE      : {:>10.6f}m", result.error_stats.translation_rmse);
            }
            
            spdlog::info("════════════════════════════════════════════════════════════════════");
        }
        
        // Wait for viewer finish if enabled
        if (viewer) {
            spdlog::info("[TUMPlayer] Processing completed! Click 'Finish & Exit' to close.");
            while (!viewer->should_close() && !viewer->is_finish_requested()) {
                viewer->render();
                std::this_thread::sleep_for(std::chrono::milliseconds(30));
            }
        }
        
    } catch (const std::exception& e) {
        result.error_message = e.what();
        spdlog::error("[TUMPlayer] Exception occurred: {}", e.what());
    }
    
    return result;
}

std::vector<ImageData> TUMPlayer::load_image_timestamps(const std::string& dataset_path) {
    std::vector<ImageData> image_data;
    std::string data_file = dataset_path + "/mav0/cam0/data.csv";
    
    std::ifstream file(data_file);
    if (!file.is_open()) {
        spdlog::error("[TUMPlayer] Cannot open data.csv file: {}", data_file);
        return image_data;
    }
    
    std::string line;
    std::getline(file, line); // Skip header
    
    while (std::getline(file, line)) {
        if (line.empty() || line[0] == '#') continue;
        
        std::stringstream ss(line);
        std::string timestamp_str, filename;
        
        if (std::getline(ss, timestamp_str, ',') && std::getline(ss, filename)) {
            ImageData data;
            data.timestamp = std::stoll(trim(timestamp_str));
            data.filename = trim(filename);
            image_data.push_back(data);
        }
    }
    
    spdlog::info("[TUMPlayer] Loaded {} image timestamps", image_data.size());
    return image_data;
}

cv::Mat TUMPlayer::load_image(const std::string& dataset_path, const std::string& filename, int cam_id) {
    std::string cam_folder = (cam_id == 0) ? "cam0" : "cam1";
    std::string full_path = dataset_path + "/mav0/" + cam_folder + "/data/" + filename;
    cv::Mat image = cv::imread(full_path, cv::IMREAD_GRAYSCALE);
    
    if (image.empty()) {
        spdlog::error("[TUMPlayer] Cannot load image: {}", full_path);
    }
    
    return image;
}

bool TUMPlayer::setup_ground_truth_matching(const std::string& dataset_path, 
                                             const std::vector<ImageData>& image_data,
                                             size_t& start_frame_idx, 
                                             size_t& end_frame_idx) {
    // Check if this is a room sequence (only room sequences have ground truth)
    std::string dataset_name = dataset_path;
    size_t last_slash = dataset_name.find_last_of("/\\");
    if (last_slash != std::string::npos) {
        dataset_name = dataset_name.substr(last_slash + 1);
    }
    
    // Only room sequences have ground truth data in TUM VI dataset
    if (dataset_name.find("room") == std::string::npos) {
        spdlog::info("[TUMPlayer] Non-room sequence detected ({}), skipping ground truth loading", dataset_name);
        return false;
    }
    
    spdlog::info("[TUMPlayer] Room sequence detected ({}), attempting to load ground truth", dataset_name);
    
    // Load ground truth data
    if (!TUMUtils::load_ground_truth(dataset_path)) {
        spdlog::warn("[TUMPlayer] Failed to load ground truth data, continuing without it");
        return false;
    }
    
    if (!TUMUtils::has_ground_truth()) {
        return false;
    }
    
    // Extract timestamps for matching
    std::vector<long long> image_timestamps;
    image_timestamps.reserve(image_data.size());
    for (const auto& img : image_data) {
        image_timestamps.push_back(img.timestamp);
    }
    
    // Match with ground truth
    if (!TUMUtils::match_image_timestamps(image_timestamps)) {
        spdlog::warn("[TUMPlayer] Failed to match image timestamps with ground truth");
        return false;
    }
    
    size_t matched_count = TUMUtils::get_matched_count();
    if (matched_count == 0) {
        spdlog::warn("[TUMPlayer] No matched timestamps found");
        return false;
    }
    
    // Find valid frame range
    long long first_matched_ts = TUMUtils::get_matched_timestamp(0);
    long long last_matched_ts = TUMUtils::get_matched_timestamp(matched_count - 1);
    
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
    
    spdlog::info("[TUMPlayer] Ground truth matched: {} frames, range {} to {}", 
                matched_count, start_frame_idx, end_frame_idx);
    return true;
}

bool TUMPlayer::load_imu_data(const std::string& dataset_path,
                               const std::vector<ImageData>& image_data,
                               size_t start_frame_idx,
                               size_t end_frame_idx) {
    if (start_frame_idx < end_frame_idx && TUMUtils::has_ground_truth()) {
        // Load IMU data in time range with buffer
        long long start_timestamp_ns = image_data[start_frame_idx].timestamp;
        long long end_timestamp_ns = image_data[end_frame_idx - 1].timestamp;
        
        // Add 1 second buffer
        long long buffer_ns = 1000000000LL;
        start_timestamp_ns -= buffer_ns;
        end_timestamp_ns += buffer_ns;
        
        if (!TUMUtils::load_imu_data_in_range(dataset_path, start_timestamp_ns, end_timestamp_ns)) {
            spdlog::error("[TUMPlayer] Failed to load IMU data in range");
            return false;
        }
    } else {
        // Load all IMU data
        if (!TUMUtils::load_imu_data(dataset_path)) {
            spdlog::error("[TUMPlayer] Failed to load IMU data");
            return false;
        }
    }
    
    TUMUtils::print_imu_stats();
    return true;
}

std::unique_ptr<PangolinViewer> TUMPlayer::initialize_viewer(const TUMPlayerConfig& config) {
    if (!config.enable_viewer) {
        return nullptr;
    }
    
    auto viewer = std::make_unique<PangolinViewer>();
    if (viewer->initialize(config.viewer_width, config.viewer_height)) {
        spdlog::info("[TUMPlayer] Viewer initialized successfully");
        
        // Wait for viewer to be ready
        while (!viewer->is_ready()) {
            viewer->render();
            std::this_thread::sleep_for(std::chrono::milliseconds(16));
        }
        spdlog::info("[TUMPlayer] Viewer is ready!");
        return viewer;
    } else {
        spdlog::warn("[TUMPlayer] Failed to initialize viewer");
        return nullptr;
    }
}

void TUMPlayer::initialize_estimator(Estimator& estimator, const std::vector<ImageData>& image_data) {
    // Set initial ground truth pose if available
    if (TUMUtils::has_ground_truth() && !image_data.empty()) {
        auto first_gt_pose = TUMUtils::get_matched_pose(0);
        if (first_gt_pose.has_value()) {
            estimator.set_initial_gt_pose(first_gt_pose.value());
            spdlog::info("[TUMPlayer] Set initial ground truth pose");
        }
    }
}

double TUMPlayer::process_single_frame(Estimator& estimator,
                                        FrameContext& context,
                                        const std::vector<ImageData>& image_data,
                                        const std::string& dataset_path,
                                        bool use_vio_mode) {
    auto start_time = std::chrono::high_resolution_clock::now();
    
    // Load stereo images
    cv::Mat left_image = load_image(dataset_path, image_data[context.current_idx].filename, 0);
    cv::Mat right_image = load_image(dataset_path, image_data[context.current_idx].filename, 1);
    
    if (left_image.empty()) {
        spdlog::warn("[TUMPlayer] Skipping frame {} due to empty image", context.current_idx);
        return 0.0;
    }
    
    // Preprocess images
    cv::Mat processed_left = preprocess_image(left_image);
    cv::Mat processed_right = right_image.empty() ? cv::Mat() : preprocess_image(right_image);
    
    // Process frame
    Estimator::EstimationResult result;
    
    if (use_vio_mode && context.processed_frames > 0) {
        // VIO mode with IMU data
        auto imu_data = get_imu_data_between_frames(context.previous_frame_timestamp, 
                                                   image_data[context.current_idx].timestamp);
        
        if (!imu_data.empty()) {
            result = estimator.process_frame(processed_left, processed_right, 
                                           image_data[context.current_idx].timestamp, imu_data);
        } else {
            // Fallback to VO mode if no IMU data
            result = estimator.process_frame(processed_left, processed_right, 
                                           image_data[context.current_idx].timestamp);
        }
    } else {
        // VO mode
        result = estimator.process_frame(processed_left, processed_right, 
                                       image_data[context.current_idx].timestamp);
    }
    
    // Handle ground truth pose
    if (TUMUtils::get_matched_count() > context.processed_frames) {
        auto gt_pose_opt = TUMUtils::get_matched_pose(context.processed_frames);
        if (gt_pose_opt.has_value()) {
            context.gt_poses.push_back(gt_pose_opt.value());
        }
    }
    
    // Update frame timestamp
    context.previous_frame_timestamp = image_data[context.current_idx].timestamp;
    
    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);
    return duration.count() / 1000.0; // Return milliseconds
}

cv::Mat TUMPlayer::preprocess_image(const cv::Mat& input_image) {
    cv::Mat equalized_image, processed_image;
    
    // Global histogram equalization
    cv::equalizeHist(input_image, equalized_image);
    
    // CLAHE for local contrast enhancement
    cv::Ptr<cv::CLAHE> clahe = cv::createCLAHE(2.0, cv::Size(8, 8));
    clahe->apply(equalized_image, processed_image);
    
    return processed_image;
}

std::vector<IMUData> TUMPlayer::get_imu_data_between_frames(long long previous_timestamp, 
                                                             long long current_timestamp) {
    return TUMUtils::get_imu_between_timestamps(previous_timestamp, current_timestamp);
}

bool TUMPlayer::handle_viewer_controls(PangolinViewer& viewer, FrameContext& context) {
    // Check for exit conditions
    if (viewer.should_close() || viewer.is_finish_requested()) {
        spdlog::info("[TUMPlayer] User requested exit");
        return false;
    }
    
    // Process keyboard input and sync UI state
    viewer.process_keyboard_input(context.auto_play, context.step_mode, context.advance_frame);
    viewer.sync_ui_state(context.auto_play, context.step_mode);
    
    // Always return true, let main loop handle mode logic
    return true;
}

void TUMPlayer::update_viewer(PangolinViewer& viewer,
                               const Estimator& estimator,
                               const FrameContext& context) {
    auto current_frame = estimator.get_current_frame();
    if (!current_frame) return;
    
    // Update poses
    Eigen::Matrix4f current_pose = current_frame->get_Twb();
    viewer.update_pose(current_pose);
    viewer.update_camera_pose(current_frame->get_Twc());
    
    // Update trajectory
    static std::vector<Eigen::Matrix4f> trajectory_poses;
    static std::vector<Eigen::Matrix4f> gt_trajectory_poses;
    
    trajectory_poses.push_back(current_pose);
    
    if (context.processed_frames < context.gt_poses.size()) {
        gt_trajectory_poses.push_back(context.gt_poses[context.processed_frames]);
        viewer.update_trajectory_with_gt(trajectory_poses, gt_trajectory_poses);
    } else {
        viewer.update_trajectory(extract_positions_from_poses(trajectory_poses));
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
    
    // Update tracking images
    cv::Mat tracking_image = current_frame->draw_features();
    const auto& features = current_frame->get_features();
    const auto& frame_map_points = current_frame->get_map_points();
    viewer.update_tracking_image_with_map_points(tracking_image, features, frame_map_points);
    
    if (current_frame->is_stereo()) {
        cv::Mat stereo_image = current_frame->draw_stereo_matches();
        viewer.update_stereo_image(stereo_image);
    }
    
    viewer.render();
}

void TUMPlayer::save_trajectories(const Estimator& estimator,
                                   const FrameContext& context,
                                   const std::string& dataset_path,
                                   bool use_vio_mode) {
    std::string mode_suffix = use_vio_mode ? "vio" : "vo";
    
    // Save estimated trajectory
    std::string est_file = dataset_path + "/estimated_trajectory_" + mode_suffix + ".txt";
    std::ofstream est_out(est_file);
    if (est_out.is_open()) {
        const auto& all_frames = estimator.get_all_frames();
        spdlog::info("[TUMPlayer] Saving {} frames to estimated trajectory", all_frames.size());
        
        for (size_t i = 0; i < all_frames.size(); ++i) {
            const auto& frame = all_frames[i];
            if (!frame) continue;
            
            Eigen::Matrix4f T_wb = frame->get_Twb();
            Eigen::Vector3f translation = T_wb.block<3, 1>(0, 3);
            Eigen::Matrix3f rotation = T_wb.block<3, 3>(0, 0);
            Eigen::Quaternionf quat(rotation);
            
            if (i < TUMUtils::get_matched_count()) {
                long long matched_timestamp = TUMUtils::get_matched_timestamp(i);
                double timestamp_sec = static_cast<double>(matched_timestamp) / 1e9;
                
                est_out << std::fixed << std::setprecision(6) << timestamp_sec << " "
                        << std::setprecision(8)
                        << translation.x() << " " << translation.y() << " " << translation.z() << " "
                        << quat.x() << " " << quat.y() << " " << quat.z() << " " << quat.w() << std::endl;
            }
        }
        est_out.close();
        spdlog::info("[TUMPlayer] Saved estimated trajectory to: {}", est_file);
    }
    
    // Save ground truth trajectory
    if (!context.gt_poses.empty()) {
        std::string gt_file = dataset_path + "/ground_truth_" + mode_suffix + ".txt";
        std::ofstream gt_out(gt_file);
        if (gt_out.is_open()) {
            for (size_t i = 0; i < context.gt_poses.size(); ++i) {
                const auto& gt_pose = context.gt_poses[i];
                
                Eigen::Vector3f translation = gt_pose.block<3, 1>(0, 3);
                Eigen::Matrix3f rotation = gt_pose.block<3, 3>(0, 0);
                Eigen::Quaternionf quat(rotation);
                
                long long matched_timestamp = TUMUtils::get_matched_timestamp(i);
                double timestamp_sec = static_cast<double>(matched_timestamp) / 1e9;
                
                gt_out << std::fixed << std::setprecision(6) << timestamp_sec << " "
                       << std::setprecision(8)
                       << translation.x() << " " << translation.y() << " " << translation.z() << " "
                       << quat.x() << " " << quat.y() << " " << quat.z() << " " << quat.w() << std::endl;
            }
            gt_out.close();
            spdlog::info("[TUMPlayer] Saved ground truth trajectory to: {}", gt_file);
        }
    }
}

TUMPlayerResult::ErrorStats TUMPlayer::analyze_transform_errors(const Estimator& estimator,
                                                                   const std::vector<Eigen::Matrix4f>& gt_poses,
                                                                   bool use_vio_mode) {
    TUMPlayerResult::ErrorStats stats;
    
    if (gt_poses.empty()) {
        spdlog::warn("[TUMPlayer] No ground truth data for error analysis");
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
        // Calculate statistics
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

TUMPlayerResult::VelocityStats TUMPlayer::analyze_velocity_statistics(const Estimator& estimator,
                                                                         const std::vector<Eigen::Matrix4f>& gt_poses) {
    TUMPlayerResult::VelocityStats stats;
    
    const auto& all_frames = estimator.get_all_frames();
    std::vector<double> linear_velocities;
    std::vector<double> angular_velocities;
    
    if (all_frames.size() < 2 || TUMUtils::get_matched_count() < all_frames.size()) {
        spdlog::warn("[TUMPlayer] Insufficient data for velocity analysis");
        return stats;
    }
    
    for (size_t i = 1; i < all_frames.size() && i < TUMUtils::get_matched_count(); ++i) {
        if (!all_frames[i-1] || !all_frames[i]) continue;
        
        // Get timestamps
        long long ts_prev = TUMUtils::get_matched_timestamp(i-1);
        long long ts_curr = TUMUtils::get_matched_timestamp(i);
        double dt = (ts_curr - ts_prev) / 1e9; // Convert nanoseconds to seconds
        
        if (dt <= 0) continue;
        
        // Get poses
        Eigen::Matrix4f T_prev = all_frames[i-1]->get_Twb();
        Eigen::Matrix4f T_curr = all_frames[i]->get_Twb();
        
        // Calculate linear velocity
        Eigen::Vector3f pos_prev = T_prev.block<3,1>(0,3);
        Eigen::Vector3f pos_curr = T_curr.block<3,1>(0,3);
        double linear_vel = (pos_curr - pos_prev).norm() / dt;
        
        // Calculate angular velocity
        Eigen::Matrix3f R_prev = T_prev.block<3,3>(0,0);
        Eigen::Matrix3f R_curr = T_curr.block<3,3>(0,0);
        Eigen::Matrix3f R_rel = R_prev.transpose() * R_curr;
        Eigen::AngleAxisf angle_axis(R_rel);
        double angular_vel = std::abs(angle_axis.angle()) / dt;
        
        linear_velocities.push_back(linear_vel);
        angular_velocities.push_back(angular_vel);
    }
    
    if (!linear_velocities.empty()) {
        stats.available = true;
        
        // Sort for median calculation
        std::vector<double> linear_sorted = linear_velocities;
        std::vector<double> angular_sorted = angular_velocities;
        std::sort(linear_sorted.begin(), linear_sorted.end());
        std::sort(angular_sorted.begin(), angular_sorted.end());
        
        // Linear velocity statistics
        stats.linear_vel_mean = std::accumulate(linear_velocities.begin(), linear_velocities.end(), 0.0) / linear_velocities.size();
        stats.linear_vel_median = linear_sorted[linear_sorted.size() / 2];
        stats.linear_vel_min = *std::min_element(linear_velocities.begin(), linear_velocities.end());
        stats.linear_vel_max = *std::max_element(linear_velocities.begin(), linear_velocities.end());
        
        // Angular velocity statistics
        stats.angular_vel_mean = std::accumulate(angular_velocities.begin(), angular_velocities.end(), 0.0) / angular_velocities.size();
        stats.angular_vel_median = angular_sorted[angular_sorted.size() / 2];
        stats.angular_vel_min = *std::min_element(angular_velocities.begin(), angular_velocities.end());
        stats.angular_vel_max = *std::max_element(angular_velocities.begin(), angular_velocities.end());
        
        spdlog::info("[TUMPlayer] Velocity analysis:");
        spdlog::info("  Linear vel - Mean: {:.4f}m/s, Median: {:.4f}m/s, Range: {:.4f}-{:.4f}m/s", 
                    stats.linear_vel_mean, stats.linear_vel_median, stats.linear_vel_min, stats.linear_vel_max);
        spdlog::info("  Angular vel - Mean: {:.4f}rad/s, Median: {:.4f}rad/s, Range: {:.4f}-{:.4f}rad/s",
                    stats.angular_vel_mean, stats.angular_vel_median, stats.angular_vel_min, stats.angular_vel_max);
                    
        // Velocity Statistics output
        spdlog::info("══════════════════════════════════════════════════════════════════");
        spdlog::info("                          VELOCITY ANALYSIS                         ");
        spdlog::info("══════════════════════════════════════════════════════════════════");
        spdlog::info("                        LINEAR VELOCITY (m/s)                       ");
        spdlog::info(" Mean      : {:>10.4f}m/s", stats.linear_vel_mean);
        spdlog::info(" Median    : {:>10.4f}m/s", stats.linear_vel_median);
        spdlog::info(" Minimum   : {:>10.4f}m/s", stats.linear_vel_min);
        spdlog::info(" Maximum   : {:>10.4f}m/s", stats.linear_vel_max);
        spdlog::info("");
        spdlog::info("                       ANGULAR VELOCITY (rad/s)                     ");
        spdlog::info(" Mean      : {:>10.4f}rad/s", stats.angular_vel_mean);
        spdlog::info(" Median    : {:>10.4f}rad/s", stats.angular_vel_median);
        spdlog::info(" Minimum   : {:>10.4f}rad/s", stats.angular_vel_min);
        spdlog::info(" Maximum   : {:>10.4f}rad/s", stats.angular_vel_max);
        spdlog::info("══════════════════════════════════════════════════════════════════");
    }
    
    return stats;
}

void TUMPlayer::save_statistics(const TUMPlayerResult& result,
                                 const std::string& dataset_path,
                                 bool use_vio_mode) {
    std::string mode_suffix = use_vio_mode ? "vio" : "vo";
    std::string stats_file = dataset_path + "/statistics_" + mode_suffix + ".txt";
    
    std::ofstream stats_out(stats_file);
    if (stats_out.is_open()) {
        stats_out << "════════════════════════════════════════════════════════════════════\n";
        stats_out << "                          STATISTICS (" << (use_vio_mode ? "VIO" : "VO") << ")                          \n";
        stats_out << "════════════════════════════════════════════════════════════════════\n\n";
        
        // Timing statistics
        stats_out << "                          TIMING ANALYSIS                           \n";
        stats_out << "════════════════════════════════════════════════════════════════════\n";
        stats_out << " Total Frames Processed: " << result.processed_frames << "\n";
        stats_out << " Average Processing Time: " << std::fixed << std::setprecision(2) 
                  << result.average_processing_time_ms << "ms\n";
        double fps = 1000.0 / result.average_processing_time_ms;
        stats_out << " Average Frame Rate: " << std::fixed << std::setprecision(1) << fps << "fps\n\n";
        
        // Velocity statistics
        if (result.velocity_stats.available) {
            stats_out << "                          VELOCITY ANALYSIS                         \n";
            stats_out << "════════════════════════════════════════════════════════════════════\n";
            stats_out << "                        LINEAR VELOCITY (m/s)                       \n";
            stats_out << " Mean      :" << std::setw(10) << std::fixed << std::setprecision(4) 
                      << result.velocity_stats.linear_vel_mean << "m/s\n";
            stats_out << " Median    :" << std::setw(10) << std::fixed << std::setprecision(4) 
                      << result.velocity_stats.linear_vel_median << "m/s\n";
            stats_out << " Minimum   :" << std::setw(10) << std::fixed << std::setprecision(4) 
                      << result.velocity_stats.linear_vel_min << "m/s\n";
            stats_out << " Maximum   :" << std::setw(10) << std::fixed << std::setprecision(4) 
                      << result.velocity_stats.linear_vel_max << "m/s\n\n";
            stats_out << "                       ANGULAR VELOCITY (rad/s)                     \n";
            stats_out << " Mean      :" << std::setw(10) << std::fixed << std::setprecision(4) 
                      << result.velocity_stats.angular_vel_mean << "rad/s\n";
            stats_out << " Median    :" << std::setw(10) << std::fixed << std::setprecision(4) 
                      << result.velocity_stats.angular_vel_median << "rad/s\n";
            stats_out << " Minimum   :" << std::setw(10) << std::fixed << std::setprecision(4) 
                      << result.velocity_stats.angular_vel_min << "rad/s\n";
            stats_out << " Maximum   :" << std::setw(10) << std::fixed << std::setprecision(4) 
                      << result.velocity_stats.angular_vel_max << "rad/s\n\n";
        }
        
        // Error statistics
        if (result.error_stats.available) {
            stats_out << "               FRAME-TO-FRAME TRANSFORM ERROR ANALYSIS              \n";
            stats_out << "════════════════════════════════════════════════════════════════════\n";
            stats_out << " Total Frame Pairs Analyzed: " << result.error_stats.total_frame_pairs 
                      << " (all_frames: " << result.error_stats.total_frames 
                      << ", gt_poses: " << result.error_stats.gt_poses_count << ")\n";
            stats_out << " Frame precision: 32 bit floats\n\n";
            
            stats_out << "                     ROTATION ERROR STATISTICS                    \n";
            stats_out << " Mean      :" << std::setw(10) << std::fixed << std::setprecision(4) 
                      << result.error_stats.rotation_mean << "°\n";
            stats_out << " Median    :" << std::setw(10) << std::fixed << std::setprecision(4) 
                      << result.error_stats.rotation_median << "°\n";
            stats_out << " Minimum   :" << std::setw(10) << std::fixed << std::setprecision(4) 
                      << result.error_stats.rotation_min << "°\n";
            stats_out << " Maximum   :" << std::setw(10) << std::fixed << std::setprecision(4) 
                      << result.error_stats.rotation_max << "°\n";
            stats_out << " RMSE      :" << std::setw(10) << std::fixed << std::setprecision(4) 
                      << result.error_stats.rotation_rmse << "°\n\n";
            
            stats_out << "                   TRANSLATION ERROR STATISTICS                   \n";
            stats_out << " Mean      :" << std::setw(10) << std::fixed << std::setprecision(6) 
                      << result.error_stats.translation_mean << "m\n";
            stats_out << " Median    :" << std::setw(10) << std::fixed << std::setprecision(6) 
                      << result.error_stats.translation_median << "m\n";
            stats_out << " Minimum   :" << std::setw(10) << std::fixed << std::setprecision(6) 
                      << result.error_stats.translation_min << "m\n";
            stats_out << " Maximum   :" << std::setw(10) << std::fixed << std::setprecision(6) 
                      << result.error_stats.translation_max << "m\n";
            stats_out << " RMSE      :" << std::setw(10) << std::fixed << std::setprecision(6) 
                      << result.error_stats.translation_rmse << "m\n";
        } else {
            stats_out << "               FRAME-TO-FRAME TRANSFORM ERROR ANALYSIS              \n";
            stats_out << "════════════════════════════════════════════════════════════════════\n";
            stats_out << " No ground truth data available for error analysis\n";
        }
        
        stats_out << "\n════════════════════════════════════════════════════════════════════\n";
        stats_out.close();
        spdlog::info("[TUMPlayer] Saved statistics to: {}", stats_file);
    }
}

std::string TUMPlayer::trim(const std::string& str) {
    size_t first = str.find_first_not_of(" \t\r\n");
    if (first == std::string::npos) return "";
    size_t last = str.find_last_not_of(" \t\r\n");
    return str.substr(first, (last - first + 1));
}

std::vector<Eigen::Vector3f> TUMPlayer::extract_positions_from_poses(const std::vector<Eigen::Matrix4f>& poses) {
    std::vector<Eigen::Vector3f> positions;
    positions.reserve(poses.size());
    for (const auto& pose : poses) {
        positions.push_back(pose.block<3, 1>(0, 3));
    }
    return positions;
}

} // namespace lightweight_vio
