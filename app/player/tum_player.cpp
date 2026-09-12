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
        // Note: Configuration is already loaded in main(), so we skip loading here
        // Config::getInstance().load(config.config_path);
        
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
                
                // Log frame processing time
                // spdlog::info("[TUMPlayer] Frame {}: {:.2f} ms", context.current_idx, total_time_ms);
                
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

                double frame_interval_ms = 50;

                // spdlog::info("Frame Interval: {:.2f} ms, Processing Time: {:.2f} ms", frame_interval_ms, total_time_ms);

                std::this_thread::sleep_for(std::chrono::milliseconds(static_cast<int>(frame_interval_ms-total_time_ms)));
            }
        }
        
        // 6. Save results
        if (config.enable_statistics) {
            save_trajectories(estimator, context, config.dataset_path, config.use_vio_mode);
            result.error_stats = analyze_transform_errors(context.estimated_poses, context.gt_poses, 
                                                          context.gt_frame_indices, config.use_vio_mode);
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
            
            // Log average processing time
            spdlog::info("[TUMPlayer] Average processing time: {:.2f} ms ({:.1f} fps)", 
                        result.average_processing_time_ms, 
                        1000.0 / result.average_processing_time_ms);
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
            data.timestamp = std::stoll(utils::trim(timestamp_str));
            data.filename = utils::trim(filename);
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
    
    // Don't change start_frame_idx or end_frame_idx - process all frames
    // GT matching is only used for error analysis
    spdlog::info("[TUMPlayer] Ground truth matched: {} frames out of {} total images", 
                matched_count, image_data.size());
    return true;
}

bool TUMPlayer::load_imu_data(const std::string& dataset_path,
                               const std::vector<ImageData>& image_data,
                               size_t start_frame_idx,
                               size_t end_frame_idx) {
    if (start_frame_idx < end_frame_idx) {
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
    // // Set initial ground truth pose if available
    // if (TUMUtils::has_ground_truth() && !image_data.empty()) {
    //     auto first_gt_pose = TUMUtils::get_matched_pose(0);
    //     if (first_gt_pose.has_value()) {
    //         estimator.set_initial_gt_pose(first_gt_pose.value());
    //         spdlog::info("[TUMPlayer] Set initial ground truth pose");
    //     }
    // }
}

double TUMPlayer::process_single_frame(Estimator& estimator,
                                        FrameContext& context,
                                        const std::vector<ImageData>& image_data,
                                        const std::string& dataset_path,
                                        bool use_vio_mode) {
    auto start_time = std::chrono::high_resolution_clock::now();
    
    // Use current_idx directly to access image_data
    size_t image_idx = context.current_idx;
    
    // Load stereo images with caching
    auto load_start = std::chrono::high_resolution_clock::now();
    cv::Mat left_image, right_image;
    bool cache_hit = load_stereo_images_cached(dataset_path, image_data[image_idx].filename, image_idx, left_image, right_image);
    auto load_end = std::chrono::high_resolution_clock::now();
    
    auto load_duration = std::chrono::duration_cast<std::chrono::microseconds>(load_end - load_start);
    double load_time_ms = load_duration.count() / 1000.0;
    
    // // Log slow image loading
    // if (load_time_ms > 5.0) {
    //     spdlog::warn("[TUMPlayer] Slow image loading for frame {}: {:.2f}ms (file: {}, cache_hit: {})", 
    //                  image_idx, load_time_ms, image_data[image_idx].filename, cache_hit ? "true" : "false");
    // }
    
    if (left_image.empty()) {
        spdlog::warn("[TUMPlayer] Skipping frame {} due to empty image", image_idx);
        return 0.0;
    }
    
    // Preprocess images
    auto preprocess_start = std::chrono::high_resolution_clock::now();
    cv::Mat processed_left = preprocess_image(left_image);
    cv::Mat processed_right = right_image.empty() ? cv::Mat() : preprocess_image(right_image);
    auto preprocess_end = std::chrono::high_resolution_clock::now();
    
    auto preprocess_duration = std::chrono::duration_cast<std::chrono::microseconds>(preprocess_end - preprocess_start);
    double preprocess_time_ms = preprocess_duration.count() / 1000.0;
    
    // Process frame
    auto estimator_start = std::chrono::high_resolution_clock::now();
    Estimator::EstimationResult result;
    
    if (use_vio_mode && context.processed_frames > 0) {
        // VIO mode with IMU data
        auto imu_data = get_imu_data_between_frames(context.previous_frame_timestamp, 
                                                   image_data[image_idx].timestamp);
        
        if (!imu_data.empty()) {
            result = estimator.process_frame(processed_left, processed_right, 
                                           image_data[image_idx].timestamp, imu_data);
        } else {
            // Fallback to VO mode if no IMU data
            result = estimator.process_frame(processed_left, processed_right, 
                                           image_data[image_idx].timestamp);
        }
    } else {
        // VO mode
        result = estimator.process_frame(processed_left, processed_right, 
                                       image_data[image_idx].timestamp);
    }
    auto estimator_end = std::chrono::high_resolution_clock::now();
    
    auto estimator_duration = std::chrono::duration_cast<std::chrono::microseconds>(estimator_end - estimator_start);
    double estimator_time_ms = estimator_duration.count() / 1000.0;
    
    // Store estimated pose for all frames
    context.estimated_poses.push_back(estimator.get_current_pose());
    
    // If this frame has GT, find and store it
    // Search through matched indices to see if current image_idx matches
    if (TUMUtils::has_ground_truth()) {
        size_t matched_count = TUMUtils::get_matched_count();
        for (size_t matched_idx = 0; matched_idx < matched_count; ++matched_idx) {
            int matched_image_idx = TUMUtils::get_matched_image_index(matched_idx);
            if (matched_image_idx == static_cast<int>(image_idx)) {
                // Found GT for this frame
                auto gt_pose_opt = TUMUtils::get_matched_pose(matched_idx);
                if (gt_pose_opt.has_value()) {
                    // Store the GT pose with the frame index
                    context.gt_frame_indices.push_back(context.processed_frames);
                    context.gt_poses.push_back(gt_pose_opt.value());
                }
                break;
            }
        }
    }
    
    // Release player's local image memory after passing to estimator
    left_image.release();
    right_image.release();
    processed_left.release();
    processed_right.release();
    
    // Update frame timestamp
    context.previous_frame_timestamp = image_data[image_idx].timestamp;
    
    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);
    double total_time_ms = duration.count() / 1000.0;
    
    // // Log detailed timing breakdown (only for frames that take longer than 15ms)
    // if (total_time_ms > 15.0) {
    //     spdlog::warn("[TUMPlayer] Frame {} breakdown: Total={:.2f}ms, Load={:.2f}ms, Preprocess={:.2f}ms, Estimator={:.2f}ms", 
    //                  image_idx, total_time_ms, load_time_ms, preprocess_time_ms, estimator_time_ms);
    // }
    
    return total_time_ms; // Return milliseconds
}

cv::Mat TUMPlayer::preprocess_image(const cv::Mat& input_image) {
    cv::Mat downsampled, equalized_small, upsampled;
    
    // Downsample to reduce computation (0.5x scale = 1/4 pixels)
    cv::resize(input_image, downsampled, cv::Size(), 0.5, 0.5, cv::INTER_LINEAR);
    
    // Global histogram equalization on smaller image
    cv::equalizeHist(downsampled, equalized_small);
    
    // Upsample back to original size
    cv::resize(equalized_small, upsampled, input_image.size(), 0, 0, cv::INTER_LINEAR);
    
    return upsampled;
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
    trajectory_poses.push_back(current_pose);
    viewer.update_trajectory(extract_positions_from_poses(trajectory_poses));
    
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
    
    // ⭐ Update gravity arrow visualization at stored origin (same pattern as EuRoC player)
    Eigen::Vector3f g_world, gravity_arrow_origin;
    if (estimator.get_gravity_visualization_data(g_world, gravity_arrow_origin)) {
        viewer.set_gravity_arrow(gravity_arrow_origin, g_world, "g_world (DOWN)");
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
            
            double timestamp_sec = static_cast<double>(frame->get_timestamp()) / 1e9;
            est_out << std::fixed << std::setprecision(6) << timestamp_sec << " "
                    << std::setprecision(8)
                    << translation.x() << " " << translation.y() << " " << translation.z() << " "
                    << quat.x() << " " << quat.y() << " " << quat.z() << " " << quat.w() << std::endl;
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

TUMPlayerResult::ErrorStats TUMPlayer::analyze_transform_errors(
    const std::vector<Eigen::Matrix4f>& estimated_poses,
    const std::vector<Eigen::Matrix4f>& gt_poses,
    const std::vector<size_t>& gt_frame_indices,
    bool use_vio_mode) {
    
    TUMPlayerResult::ErrorStats stats;
    
    if (gt_poses.empty() || gt_frame_indices.empty()) {
        spdlog::warn("[TUMPlayer] No ground truth data for error analysis");
        return stats;
    }
    
    if (gt_poses.size() != gt_frame_indices.size()) {
        spdlog::error("[TUMPlayer] Mismatch between GT poses ({}) and frame indices ({})",
                     gt_poses.size(), gt_frame_indices.size());
        return stats;
    }
    
    std::vector<double> rotation_errors;
    std::vector<double> translation_errors;
    
    // Iterate through consecutive GT frames only
    for (size_t i = 1; i < gt_frame_indices.size(); ++i) {
        size_t prev_frame_idx = gt_frame_indices[i-1];
        size_t curr_frame_idx = gt_frame_indices[i];
        
        if (prev_frame_idx >= estimated_poses.size() || curr_frame_idx >= estimated_poses.size()) {
            continue;
        }
        
        // Calculate frame-to-frame transforms for estimated poses
        Eigen::Matrix4f T_est_prev = estimated_poses[prev_frame_idx];
        Eigen::Matrix4f T_est_curr = estimated_poses[curr_frame_idx];
        Eigen::Matrix4f T_est_rel = T_est_prev.inverse() * T_est_curr;
        
        // Calculate frame-to-frame transforms for GT poses
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
        stats.total_frames = estimated_poses.size();
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
        
        // Print detailed analysis
        spdlog::info("[TUMPlayer] Transform Error Analysis:");
        spdlog::info("  Total Frame Pairs: {} (estimated_poses: {}, gt_poses: {}, gt_frames: {})", 
                    stats.total_frame_pairs, estimated_poses.size(), gt_poses.size(), gt_frame_indices.size());
        spdlog::info("  Rotation errors - Mean: {:.4f}°, Median: {:.4f}°, RMSE: {:.4f}°, Range: {:.4f}°-{:.4f}°",
                    stats.rotation_mean, stats.rotation_median, stats.rotation_rmse, 
                    stats.rotation_min, stats.rotation_max);
        spdlog::info("  Translation errors - Mean: {:.6f}m, Median: {:.6f}m, RMSE: {:.6f}m, Range: {:.6f}m-{:.6f}m",
                    stats.translation_mean, stats.translation_median, stats.translation_rmse,
                    stats.translation_min, stats.translation_max);
        
        spdlog::info("══════════════════════════════════════════════════════════════════");
        spdlog::info("               FRAME-TO-FRAME TRANSFORM ERROR ANALYSIS              ");
        spdlog::info("══════════════════════════════════════════════════════════════════");
        spdlog::info(" Total Frame Pairs Analyzed: {} (estimated_poses: {}, gt_poses: {}, gt_frames: {})", 
                    stats.total_frame_pairs, estimated_poses.size(), gt_poses.size(), gt_frame_indices.size());
        spdlog::info(" Frame precision: 32 bit floats");
        spdlog::info("");
        spdlog::info("                     ROTATION ERROR STATISTICS                    ");
        spdlog::info(" Mean      : {:>10.4f}°", stats.rotation_mean);
        spdlog::info(" Median    : {:>10.4f}°", stats.rotation_median);
        spdlog::info(" Minimum   : {:>10.4f}°", stats.rotation_min);
        spdlog::info(" Maximum   : {:>10.4f}°", stats.rotation_max);
        spdlog::info(" RMSE      : {:>10.4f}°", stats.rotation_rmse);
        spdlog::info("");
        spdlog::info("                   TRANSLATION ERROR STATISTICS                   ");
        spdlog::info(" Mean      : {:>10.6f}m", stats.translation_mean);
        spdlog::info(" Median    : {:>10.6f}m", stats.translation_median);
        spdlog::info(" Minimum   : {:>10.6f}m", stats.translation_min);
        spdlog::info(" Maximum   : {:>10.6f}m", stats.translation_max);
        spdlog::info(" RMSE      : {:>10.6f}m", stats.translation_rmse);
        spdlog::info("══════════════════════════════════════════════════════════════════");
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

std::vector<Eigen::Vector3f> TUMPlayer::extract_positions_from_poses(const std::vector<Eigen::Matrix4f>& poses) {
    std::vector<Eigen::Vector3f> positions;
    positions.reserve(poses.size());
    for (const auto& pose : poses) {
        positions.push_back(pose.block<3, 1>(0, 3));
    }
    return positions;
}

bool TUMPlayer::load_stereo_images_cached(const std::string& dataset_path, 
                                         const std::string& filename, 
                                         size_t frame_idx,
                                         cv::Mat& left_image, 
                                         cv::Mat& right_image) {
    // Check cache first
    for (const auto& cached : image_cache_) {
        if (cached.filename == filename && cached.frame_idx == frame_idx) {
            left_image = cached.left_image;  // Direct assignment instead of clone()
            right_image = cached.right_image;
            return true;
        }
    }
    
    // Not in cache, load from disk
    left_image = load_image(dataset_path, filename, 0);
    right_image = load_image(dataset_path, filename, 1);
    
    if (left_image.empty()) {
        return false;
    }
    
    // Add to cache (implement simple FIFO replacement)
    CachedImage new_cache_entry;
    new_cache_entry.left_image = left_image;  // Direct assignment instead of clone()
    new_cache_entry.right_image = right_image;
    new_cache_entry.filename = filename;
    new_cache_entry.frame_idx = frame_idx;
    
    // Remove oldest if cache is full
    if (image_cache_.size() >= MAX_CACHE_SIZE) {
        image_cache_.erase(image_cache_.begin());
    }
    
    image_cache_.push_back(new_cache_entry);
    return true;
}

} // namespace lightweight_vio
