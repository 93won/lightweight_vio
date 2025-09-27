/**
 * @file      tum_stereo.cpp
 * @brief     Main application entry point for the TUM VI stereo pipeline (VO/VIO configurable via YAML).
 * @author    Seungwon Choi (csw3575@snu.ac.kr)
 * @date      2024-09-27
 * @copyright Copyright (c) 2025 Seungwon Choi. All rights reserved.
 *
 * @par License
 * This project is released under the MIT License.
 */

#include <spdlog/spdlog.h>
#include <spdlog/sinks/stdout_color_sinks.h>
#include "player/tum_player.h"
#include <util/Config.h>
#include <glog/logging.h>

using namespace lightweight_vio;

int main(int argc, char* argv[]) {
    // Suppress Google logging (Ceres) error messages
    google::InitGoogleLogging(argv[0]);
    FLAGS_logtostderr = false;
    FLAGS_minloglevel = 3;  // Only fatal messages (0=INFO, 1=WARNING, 2=ERROR, 3=FATAL)
    FLAGS_stderrthreshold = 3;
    
    // Initialize spdlog for immediate colored output
    spdlog::set_level(spdlog::level::debug);
    spdlog::set_pattern("[%Y-%m-%d %H:%M:%S.%e] [%^%l%$] %v");
    
    if (argc != 3) {
        spdlog::error("Usage: {} <config_file_path> <tum_dataset_path>", argv[0]);
        spdlog::error("Example: {} config/tum_vio.yaml /path/to/dataset-room1_512_16", argv[0]);
        spdlog::error("         {} config/tum_vo.yaml /path/to/dataset-room1_512_16", argv[0]);
        return -1;
    }
    
    // Setup configuration
    TUMPlayerConfig config;
    config.config_path = argv[1];
    config.dataset_path = argv[2];
    config.enable_statistics = true;          // File statistics
    config.enable_console_statistics = true;  // Console statistics
    config.step_mode = true;  // Enable step mode like euroc_player
    
    // Load config to get all settings from YAML
    Config::getInstance().load(argv[1]);
    config.use_vio_mode = (Config::getInstance().m_system_mode == "VIO");
    config.enable_viewer = Config::getInstance().m_viewer_enable;
    config.viewer_width = Config::getInstance().m_viewer_width;
    config.viewer_height = Config::getInstance().m_viewer_height;
    
    // Debug output to verify settings
    spdlog::info("[Main] System settings from YAML:");
    spdlog::info("  system_mode: {}", Config::getInstance().m_system_mode);
    spdlog::info("  use_vio_mode: {}", config.use_vio_mode);
    spdlog::info("  enable_viewer: {}", config.enable_viewer);
    spdlog::info("  viewer_width: {}", config.viewer_width);
    spdlog::info("  viewer_height: {}", config.viewer_height);
    
    // Create and run TUM VI player
    TUMPlayer player;
    auto result = player.run(config);
    
    if (result.success) {
        std::string mode_str = config.use_vio_mode ? "VIO" : "VO";
        spdlog::info("[Main] {} processing completed successfully!", mode_str);
        return 0;
    } else {
        std::string mode_str = config.use_vio_mode ? "VIO" : "VO";
        spdlog::error("[Main] {} processing failed: {}", mode_str, result.error_message);
        return -1;
    }
}
