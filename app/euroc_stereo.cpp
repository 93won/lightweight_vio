/**
 * @file      euroc_stereo.cpp
 * @brief     Main application entry point for the EuRoC stereo pipeline (VO/VIO configurable via YAML).
 * @author    Seungwon Choi (csw3575@snu.ac.kr)
 * @date      2025-09-16
 * @copyright Copyright (c) 2025 Seungwon Choi. All rights reserved.
 *
 * @par License
 * This project is released under the MIT License.
 */

#include <spdlog/spdlog.h>
#include <spdlog/sinks/stdout_color_sinks.h>
#include "player/euroc_player.h"
#include <util/Config.h>
#include <cstdlib>
#include <ctime>

using namespace lightweight_vio;

int main(int argc, char* argv[]) {
    // Set random seed for reproducibility
    srand(42);
    cv::setRNGSeed(42);
    cv::setNumThreads(1);
    
    // Initialize spdlog for immediate colored output
    spdlog::set_level(spdlog::level::debug);
    spdlog::set_pattern("[%Y-%m-%d %H:%M:%S.%e] [%^%l%$] %v");
    
    if (argc != 3) {
        spdlog::error("Usage: {} <config_file_path> <euroc_dataset_path>", argv[0]);
        spdlog::error("Example: {} config/euroc_vio.yaml /path/to/MH_01_easy", argv[0]);
        spdlog::error("         {} config/euroc_vo.yaml /path/to/MH_01_easy", argv[0]);
        return -1;
    }
    
    // Setup configuration
    EurocPlayerConfig config;
    config.config_path = argv[1];
    config.dataset_path = argv[2];
    config.enable_statistics = true;          // File statistics
    config.enable_console_statistics = true;  // Console statistics
    config.step_mode = false;
    
    // Load config to get all settings from YAML
    Config::getInstance().load(argv[1]);
    config.use_vio_mode = (Config::getInstance().m_system_mode == "VIO");
    config.enable_viewer = Config::getInstance().m_viewer_enable;
    config.viewer_width = Config::getInstance().m_viewer_width;
    config.viewer_height = Config::getInstance().m_viewer_height;
    
    // // Debug output to verify settings
    // spdlog::info("[Main] System settings from YAML:");
    // spdlog::info("  system_mode: {}", Config::getInstance().m_system_mode);
    // spdlog::info("  use_vio_mode: {}", config.use_vio_mode);
    // spdlog::info("  enable_viewer: {}", config.enable_viewer);
    // spdlog::info("  viewer_width: {}", config.viewer_width);
    // spdlog::info("  viewer_height: {}", config.viewer_height);
    
    // Create and run EuRoC player
    EurocPlayer player;
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
