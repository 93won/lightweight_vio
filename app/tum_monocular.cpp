/**
 * @file      tum_monocular.cpp
 * @brief     Main application entry point for the TUM Monocular pipeline (RGB only from RGB-D dataset)
 * @author    Seungwon Choi (csw3575@snu.ac.kr)
 * @date      2025-11-10
 * @copyright Copyright (c) 2025 Seungwon Choi. All rights reserved.
 *
 * @par License
 * This project is released under the MIT License.
 */

#include <iostream>
#include <memory>

#include <spdlog/spdlog.h>
#include <spdlog/sinks/stdout_color_sinks.h>
#include <glog/logging.h>

#include "player/tum_monocular_player.h"
#include "util/Config.h"

using namespace lightweight_vio;

void print_usage(const char* program_name) {
    std::cout << "Usage: " << program_name << " <config_file> <dataset_path> [options]\n";
    std::cout << "\nArguments:\n";
    std::cout << "  config_file     Path to monocular VO configuration file\n";
    std::cout << "  dataset_path    Path to TUM RGB-D dataset directory (uses RGB images only)\n";
    std::cout << "\nOptions:\n";
    std::cout << "  --no-viewer        Disable 3D visualization\n";
    std::cout << "  --save-trajectory  Save trajectory to file\n";
    std::cout << "\nExample:\n";
    std::cout << "  " << program_name << " config/euroc_vo_mono.yaml /path/to/rgbd_dataset_freiburg2_desk\n";
    std::cout << "  " << program_name << " config/tum_mono.yaml /path/to/dataset --no-viewer\n";
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
    bool enable_viewer = true;
    bool save_trajectory = false;
    
    // Parse optional arguments
    for (int i = 3; i < argc; ++i) {
        std::string arg = argv[i];
        
        if (arg == "--no-viewer") {
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
    spdlog::set_level(spdlog::level::debug);
    spdlog::set_pattern("[%H:%M:%S.%e] [%^%l%$] %v");

    spdlog::info("=== TUM Monocular Visual Odometry ===");
    spdlog::info("Config file: {}", config_file);
    spdlog::info("Dataset path: {}", dataset_path);
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
    spdlog::info("  camera_type: monocular (RGB only)");
    spdlog::info("  enable_viewer: {}", enable_viewer);
    spdlog::info("  viewer_width: {}", config.m_viewer_width);
    spdlog::info("  viewer_height: {}", config.m_viewer_height);

    // Setup TUM Monocular player configuration
    TUMMonocularPlayerConfig player_config;
    player_config.config_path = config_file;
    player_config.dataset_path = dataset_path;
    player_config.enable_viewer = enable_viewer;
    player_config.viewer_width = config.m_viewer_width;
    player_config.viewer_height = config.m_viewer_height;
    player_config.enable_statistics = true;
    player_config.enable_console_statistics = true;
    player_config.step_mode = false;

    // Create and run TUM Monocular player
    TUMMonocularPlayer player;
    auto result = player.run(player_config);

    if (result.success) {
        spdlog::info("[Main] Monocular VO processing completed successfully!");
        spdlog::info("[Main] Processed {} frames in total", result.processed_frames);
    } else {
        spdlog::error("[Main] Monocular VO processing failed: {}", result.error_message);
        return -1;
    }

    spdlog::info("TUM Monocular VO finished.");
    return 0;
}
