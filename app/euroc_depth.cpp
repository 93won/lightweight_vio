/**
 * @file      euroc_depth.cpp
 * @brief     Main application entry point for the RGBD pipeline (VO mode, VIO support planned).
 * @author    Seungwon Choi (csw3575@snu.ac.kr)
 * @date      2025-10-28
 * @copyright Copyright (c) 2025 Seungwon Choi. All rights reserved.
 *
 * @par License
 * This project is released under the MIT License.
 */

#include <spdlog/spdlog.h>
#include <spdlog/sinks/stdout_color_sinks.h>
#include "player/rgbd_player.h"
#include <util/Config.h>

using namespace lightweight_vio;

int main(int argc, char* argv[]) {
    // Initialize spdlog for immediate colored output
    spdlog::set_level(spdlog::level::debug);
    spdlog::set_pattern("[%Y-%m-%d %H:%M:%S.%e] [%^%l%$] %v");
    
    if (argc != 3) {
        spdlog::error("Usage: {} <config_file_path> <rgbd_dataset_path>", argv[0]);
        spdlog::error("Example: {} config/gemini.yaml /path/to/rgbd_dataset", argv[0]);
        spdlog::error("");
        spdlog::error("Supported dataset formats:");
        spdlog::error("");
        spdlog::error("1. HWASHIN format (recommended):");
        spdlog::error("  dataset/");
        spdlog::error("    timestamps.txt    # one timestamp per line (seconds)");
        spdlog::error("    color/            # RGB images (000000.png, 000001.png, ...)");
        spdlog::error("    depth/            # Depth images (000000.png, 000001.png, ...)");
        spdlog::error("");
        spdlog::error("2. TUM RGB-D format:");
        spdlog::error("  dataset/");
        spdlog::error("    rgb.txt           # timestamp filename");
        spdlog::error("    depth.txt         # timestamp filename");
        spdlog::error("    groundtruth.txt   # timestamp tx ty tz qx qy qz qw (optional)");
        spdlog::error("    rgb/              # RGB images folder");
        spdlog::error("    depth/            # Depth images folder (16-bit PNG)");
        return -1;
    }
    
    // Setup configuration
    RGBDPlayerConfig config;
    config.config_path = argv[1];
    config.dataset_path = argv[2];
    config.enable_statistics = true;          // File statistics
    config.enable_console_statistics = true;  // Console statistics
    config.step_mode = false;
    
    // Load config to get all settings from YAML
    Config::getInstance().load(argv[1]);
    
    // RGBD currently supports VO mode only (VIO support planned)
    config.use_vio_mode = false;
    
    config.enable_viewer = Config::getInstance().m_viewer_enable;
    config.viewer_width = Config::getInstance().m_viewer_width;
    config.viewer_height = Config::getInstance().m_viewer_height;
    
    spdlog::info("[Main] RGBD Player settings:");
    spdlog::info("  camera_type: RGBD");
    spdlog::info("  mode: VO (Visual Odometry)");
    spdlog::info("  enable_viewer: {}", config.enable_viewer);
    spdlog::info("  viewer_width: {}", config.viewer_width);
    spdlog::info("  viewer_height: {}", config.viewer_height);
    spdlog::info("  depth_scale: {}", Config::getInstance().m_rgbd_depth_scale);
    spdlog::info("  depth_range: [{}, {}] m", 
                 Config::getInstance().m_rgbd_min_depth,
                 Config::getInstance().m_rgbd_max_depth);
    
    // Create and run RGBD player
    RGBDPlayer player;
    auto result = player.run(config);
    
    if (result.success) {
        spdlog::info("[Main] RGBD VO processing completed successfully!");
        spdlog::info("[Main] Processed {} frames in {:.2f}ms average", 
                     result.processed_frames, result.average_processing_time_ms);
        return 0;
    } else {
        spdlog::error("[Main] RGBD VO processing failed: {}", result.error_message);
        return -1;
    }
}
