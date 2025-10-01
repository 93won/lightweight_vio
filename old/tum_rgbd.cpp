/**
 * @file      tum_rgbd.cpp
 * @brief     TUM RGB-D Visual Odometry application for RGB-D sensor data
 * @author    Seungwon Choi (csw3575@snu.ac.kr)
 * @date      2025-09-30
 * @copyright Copyright (c) 2025 Seungwon Choi. All rights reserved.
 *
 * @par License
 * This project is released under the MIT License.
 */

#include <iostream>
#include <memory>
#include <chrono>
#include <thread>

#include "player/tum_rgbd_player.h"
#include "database/Frame.h"
#include "processing/Estimator.h"
#include "viewer/PangolinViewer.h"
#include "util/Config.h"
#include <spdlog/spdlog.h>
#include <glog/logging.h>

using namespace lightweight_vio;

void print_usage(const char* program_name) {
    std::cout << "Usage: " << program_name << " <config_file> <dataset_path> [options]\n";
    std::cout << "\nArguments:\n";
    std::cout << "  config_file     Path to TUM RGB-D VO configuration file\n";
    std::cout << "  dataset_path    Path to TUM RGB-D dataset directory\n";
    std::cout << "\nOptions:\n";
    std::cout << "  --max-frames N     Process only first N frames (default: all)\n";
    std::cout << "  --max-time-diff T  Maximum time difference for RGB-D sync in seconds (default: 0.02)\n";
    std::cout << "  --no-viewer        Disable 3D visualization\n";
    std::cout << "  --save-trajectory  Save trajectory to file\n";
    std::cout << "\nNote:\n";
    std::cout << "  Depth scale factor is read from the config file (camera.pixel_to_meter_scalefactor)\n";
    std::cout << "\nExample:\n";
    std::cout << "  " << program_name << " config/tum_rgbd_vo.yaml /path/to/rgbd_dataset_freiburg2_desk\n";
    std::cout << "  " << program_name << " config/tum_rgbd_vo.yaml /path/to/dataset --max-frames 500\n";
}

int main(int argc, char** argv) {
    if (argc < 3) {
        print_usage(argv[0]);
        return -1;
    }

    // Suppress Google logging (Ceres) error messages
    google::InitGoogleLogging(argv[0]);
    FLAGS_logtostderr = false;
    FLAGS_minloglevel = 3;  // Only fatal messages
    FLAGS_stderrthreshold = 3;

    // Parse command line arguments
    std::string config_file = argv[1];
    std::string dataset_path = argv[2];
    
    // Default parameters
    int max_frames = -1;  // Process all frames by default
    double max_time_diff = 0.02;  // 20ms maximum time difference
    bool enable_viewer = true;
    bool save_trajectory = false;
    
    // Parse optional arguments
    for (int i = 3; i < argc; ++i) {
        std::string arg = argv[i];
        
        if (arg == "--max-frames" && i + 1 < argc) {
            max_frames = std::atoi(argv[++i]);
        } else if (arg == "--max-time-diff" && i + 1 < argc) {
            max_time_diff = std::atof(argv[++i]);
        } else if (arg == "--no-viewer") {
            enable_viewer = false;
        } else if (arg == "--save-trajectory") {
            save_trajectory = true;
        } else {
            std::cerr << "Unknown option: " << arg << std::endl;
            print_usage(argv[0]);
            return -1;
        }
    }

    // Initialize spdlog
    spdlog::set_level(spdlog::level::info);
    spdlog::set_pattern("[%H:%M:%S.%e] [%^%l%$] %v");

    spdlog::info("=== TUM RGB-D Visual Odometry ===");
    spdlog::info("Config file: {}", config_file);
    spdlog::info("Dataset path: {}", dataset_path);
    spdlog::info("Max frames: {}", max_frames == -1 ? "all" : std::to_string(max_frames));
    spdlog::info("Max time diff: {:.3f}s", max_time_diff);
    spdlog::info("Viewer enabled: {}", enable_viewer);

    // Load configuration
    Config& config = Config::getInstance();
    if (!config.load(config_file)) {
        spdlog::error("Failed to load configuration file: {}", config_file);
        return -1;
    }

    // Override viewer setting from config if not explicitly disabled by command line
    if (enable_viewer) {
        enable_viewer = config.m_viewer_enable;
    }
    
    // Debug output to verify settings
    spdlog::info("[Main] System settings from YAML:");
    spdlog::info("  system_mode: {}", config.m_system_mode);
    spdlog::info("  enable_viewer: {}", enable_viewer);
    spdlog::info("  viewer_width: {}", config.m_viewer_width);
    spdlog::info("  viewer_height: {}", config.m_viewer_height);

    // Initialize RGB-D player
    TUMRGBDPlayer player(dataset_path, max_time_diff);
    if (!player.initialize()) {
        spdlog::error("Failed to initialize TUM RGB-D player");
        return -1;
    }

    // Print synchronization statistics
    player.print_sync_stats();

    // Initialize Estimator for actual tracking (like EuRoC player)
    Estimator estimator;
    spdlog::info("Estimator initialized for RGB-D tracking");

    // Initialize viewer if enabled
    std::unique_ptr<PangolinViewer> viewer = nullptr;
    if (enable_viewer) {
        viewer = std::make_unique<PangolinViewer>();
        // Initialize viewer with config dimensions
        if (viewer->initialize(config.m_viewer_width, config.m_viewer_height)) {
            spdlog::info("Viewer initialized with size {}x{}", config.m_viewer_width, config.m_viewer_height);
            
            // Wait for viewer to be ready (like EuRoC player)
            while (!viewer->is_ready()) {
                viewer->render();
                std::this_thread::sleep_for(std::chrono::milliseconds(16));
            }
            spdlog::info("Viewer is ready!");
        } else {
            spdlog::warn("Failed to initialize viewer, continuing without visualization");
            viewer.reset();
        }
    }

    // Processing loop with step mode support (like EuRoC player)
    size_t total_frames = player.get_frame_count();
    size_t frames_to_process = (max_frames > 0) ? std::min(static_cast<size_t>(max_frames), total_frames) : total_frames;
    
    spdlog::info("Processing {} frames...", frames_to_process);
    
    // Default TUM RGB-D depth scale factor (typical value is 5000.0)
    double depth_scale_factor = 5000.0;
    
    auto start_time = std::chrono::high_resolution_clock::now();
    int processed_frames = 0;
    int successful_frames = 0;
    
    // Frame processing context (like EuRoC player)
    bool step_mode = false;         // Start in auto mode
    bool auto_play = true;          // Auto play by default
    bool advance_frame = false;     // For step mode control
    size_t current_idx = 0;
    
    while (current_idx < frames_to_process) {
        // Handle viewer controls first (like EuRoC player)
        if (viewer) {
            // Check for exit conditions
            if (viewer->should_close() || viewer->is_finish_requested()) {
                spdlog::info("User requested exit");
                break;
            }
            
            // Process keyboard input and sync UI state
            viewer->process_keyboard_input(auto_play, step_mode, advance_frame);
            viewer->sync_ui_state(auto_play, step_mode);
        }
        
        bool should_process_frame = false;
        
        // Check processing conditions based on mode (like EuRoC player)
        if (auto_play) {
            // Auto mode: process frame
            should_process_frame = true;
        } else {
            // Step mode: only process if advance_frame is set
            if (advance_frame) {
                should_process_frame = true;
                // Reset advance_frame after processing
                advance_frame = false;
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
            // Load frame
            auto frame = player.get_frame(current_idx, depth_scale_factor);
            if (!frame) {
                spdlog::error("Failed to load frame {}", current_idx);
                ++current_idx;
                continue;
            }

            // Process frame through Estimator for actual tracking (like EuRoC player)
            cv::Mat left_image = frame->get_image();  // This is grayscale RGB image
            cv::Mat right_image = cv::Mat();  // RGB-D doesn't have right image
            
            // Get timestamp from player frame
            long long timestamp_ns = static_cast<long long>(current_idx * 33333333LL); // ~30fps assumption
            
            // Process frame through estimator for actual VO tracking
            auto estimation_result = estimator.process_frame(left_image, right_image, timestamp_ns);
            
            // Get current frame from estimator for display
            auto current_frame = estimator.get_current_frame();
            if (!current_frame) {
                spdlog::warn("Estimator didn't produce current frame for index {}", current_idx);
                ++current_idx;
                continue;
            }

            // Count valid depth measurements from estimator frame
            int valid_depth_count = 0;
            const auto& features = current_frame->get_features();
            for (size_t j = 0; j < features.size(); ++j) {
                if (current_frame->has_depth(j)) {
                    valid_depth_count++;
                }
            }
            
            if (valid_depth_count > 10) {  // Need minimum features with depth
                successful_frames++;
            }

            processed_frames++;

            // Update viewer if enabled (like EuRoC player)
            if (viewer) {
                // Update with actual pose from estimator
                Eigen::Matrix4f current_pose = current_frame->get_Twb();
                viewer->update_pose(current_pose);
                viewer->update_camera_pose(current_frame->get_Twc());
                
                // Update tracking image with actual features
                cv::Mat tracking_image = current_frame->draw_features();
                viewer->update_tracking_image(tracking_image);
                
                // Update trajectory
                static std::vector<Eigen::Matrix4f> trajectory_poses;
                trajectory_poses.push_back(current_pose);
                
                // Extract positions for trajectory
                std::vector<Eigen::Vector3f> positions;
                positions.reserve(trajectory_poses.size());
                for (const auto& pose : trajectory_poses) {
                    positions.push_back(pose.block<3, 1>(0, 3));
                }
                viewer->update_trajectory(positions);
                
                // Update map points from estimator
                const auto keyframes = estimator.get_keyframes_safe();
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
                viewer->update_all_map_points(all_map_points);
                
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
                
                viewer->update_tracking_stats(processed_frames, total_features, 
                                            map_points_count, map_points_count, success_rate, 0.0f);
                
                // Render the viewer
                viewer->render();
            }

            // Progress update
            if (processed_frames % 50 == 0 || processed_frames == static_cast<int>(frames_to_process)) {
                auto current_time = std::chrono::high_resolution_clock::now();
                auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(current_time - start_time);
                double fps = processed_frames * 1000.0 / elapsed.count();
                
                spdlog::info("Progress: {}/{} frames ({:.1f}%) - {:.1f} fps - {}/{} successful", 
                             processed_frames, frames_to_process, 
                             (processed_frames * 100.0) / frames_to_process,
                             fps, successful_frames, processed_frames);
            }
            
            ++current_idx;
            
            // Add small delay for auto mode (like EuRoC player)
            if (auto_play) {
                std::this_thread::sleep_for(std::chrono::milliseconds(30));
            }
        }
    }

    auto end_time = std::chrono::high_resolution_clock::now();
    auto total_duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
    double avg_fps = processed_frames * 1000.0 / total_duration.count();

    spdlog::info("=== Processing Complete ===");
    spdlog::info("Total frames processed: {}/{}", processed_frames, frames_to_process);
    spdlog::info("Successful frames: {}/{} ({:.1f}%)", 
                 successful_frames, processed_frames,
                 (successful_frames * 100.0) / std::max(1, processed_frames));
    spdlog::info("Total time: {:.3f}s", total_duration.count() / 1000.0);
    spdlog::info("Average FPS: {:.1f}", avg_fps);

    // Save trajectory if requested
    if (save_trajectory) {
        spdlog::info("Trajectory saving not implemented in this RGB-D demo");
    }

    // Keep viewer open if enabled (like EuRoC player)
    if (viewer) {
        spdlog::info("Processing completed! Click 'Finish & Exit' to close or press 'q' in viewer window...");
        while (!viewer->should_close() && !viewer->is_finish_requested()) {
            viewer->render();
            std::this_thread::sleep_for(std::chrono::milliseconds(30));
        }
    }

    spdlog::info("TUM RGB-D VO finished.");
    return 0;
}