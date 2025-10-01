/**
 * @file      tum_rgbd.cpp
 * @brief     Main application entry point for the TUM RGB-D pipeline (VO mode)
 * @author    Seungwon Choi (csw3575@snu.ac.kr)
 * @date      2025-10-02
 * @copyright Copyright (c) 2025 Seungwon Choi. All rights reserved.
 *
 * @par License
 * This project is released under the MIT License.
 */

#include <iostream>
#include <memory>
#include <chrono>
#include <thread>

#include <spdlog/spdlog.h>
#include <spdlog/sinks/stdout_color_sinks.h>
#include <glog/logging.h>

#include "player/tum_rgbd_player.h"
#include "util/Config.h"

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
    spdlog::set_level(spdlog::level::debug);  // Enable debug logs
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

    // Setup TUM RGB-D player configuration
    TUMRGBDPlayerConfig player_config;
    player_config.config_path = config_file;
    player_config.dataset_path = dataset_path;
    player_config.enable_viewer = enable_viewer;
    player_config.viewer_width = config.m_viewer_width;
    player_config.viewer_height = config.m_viewer_height;
    player_config.max_time_diff = max_time_diff;
    player_config.enable_statistics = true;
    player_config.enable_console_statistics = true;
    player_config.step_mode = false;

    // Create and run TUM RGB-D player
    TUMRGBDPlayer player;
    auto result = player.run(player_config);

    if (result.success) {
        spdlog::info("[Main] RGB-D VO processing completed successfully!");
        spdlog::info("[Main] Processed {} frames in total", result.processed_frames);
    } else {
        spdlog::error("[Main] RGB-D VO processing failed: {}", result.error_message);
        return -1;
    }

    spdlog::info("TUM RGB-D VO finished.");
    return 0;
}