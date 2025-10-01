/**
 * @file      PangolinViewer.h
 * @brief     Defines the Pangolin-based 3D viewer class for VIO.
 * @author    Seungwon Choi (csw3575@snu.ac.kr)
 * @date      2025-08-30
 * @copyright Copyright (c) 2025 Seungwon Choi. All rights reserved.
 *
 * @par License
 * This project is released under the MIT License.
 */

#pragma once

#include <pangolin/display/display.h>
#include <pangolin/display/view.h>
#include <pangolin/handler/handler.h>
#include <pangolin/gl/glinclude.h>
#include <pangolin/display/image_view.h>
#include <pangolin/gl/gldraw.h>
#include <pangolin/var/var.h>
#include <pangolin/display/widgets.h>
#include <pangolin/var/varextra.h>
#include <Eigen/Dense>
#include <vector>
#include <memory>
#include <mutex>
#include <opencv2/opencv.hpp>

// Forward declarations
namespace lightweight_vio {
class Feature;
class MapPoint;
class Frame;
}

namespace lightweight_vio {

class PangolinViewer {
public:
    PangolinViewer();
    ~PangolinViewer();

    // Initialization and shutdown
    bool initialize(int width = 1280, int height = 960);
    void shutdown();
    
    // Main loop
    bool should_close() const;
    bool is_ready() const;
    void render();
    
    // Camera control
    void reset_camera();
    
    // Data updates
    void update_points(const std::vector<Eigen::Vector3f>& points);
    void update_pose(const Eigen::Matrix4f& pose);
    void update_camera_pose(const Eigen::Matrix4f& T_wc);  // Update current frame camera pose
    void update_trajectory(const std::vector<Eigen::Vector3f>& trajectory);
    void update_trajectory_with_gt(const std::vector<Eigen::Matrix4f>& trajectory, const std::vector<Eigen::Matrix4f>& gt_trajectory);
    void update_keyframe_poses(const std::vector<Eigen::Matrix4f>& keyframe_poses);
    void add_ground_truth_pose(const Eigen::Matrix4f& gt_pose);
    
    // Map point updates with color differentiation
    void update_map_points(const std::vector<Eigen::Vector3f>& all_points, const std::vector<Eigen::Vector3f>& current_points);
    
    // Image updates
    void update_tracking_image(const cv::Mat& image);
    void update_tracking_image_with_map_points(const cv::Mat& image, 
                                              const std::vector<std::shared_ptr<Feature>>& features,
                                              const std::vector<std::shared_ptr<MapPoint>>& map_points);
    void update_stereo_image(const cv::Mat& image);
    
    // Frame and keyframe management (new sliding window approach)
    void add_frame(std::shared_ptr<Frame> frame);
    std::vector<std::shared_ptr<Frame>> get_all_frames() const;
    std::shared_ptr<Frame> get_first_frame() const;
    void update_keyframe_window(const std::vector<std::shared_ptr<Frame>>& keyframes);
    void set_last_keyframe(std::shared_ptr<Frame> last_keyframe);
        void update_relative_pose_from_last_keyframe(const Eigen::Matrix4f& relative_pose);
    
    // Map point management
    void update_all_map_points(const std::vector<std::shared_ptr<MapPoint>>& all_map_points);
    void update_window_map_points(const std::vector<std::shared_ptr<MapPoint>>& window_map_points);
    
    // Input processing
    void process_keyboard_input(bool& auto_play, bool& step_mode, bool& advance_frame);
    void sync_ui_state(bool& auto_play, bool& step_mode);  // Sync UI checkbox with mode state
    void set_space_pressed(bool pressed) { m_space_pressed = pressed; }
    void set_next_pressed(bool pressed) { m_next_pressed = pressed; }
    
    // Auto mode control
    bool is_auto_mode_enabled() const;
    bool is_follow_frame_enabled() const;
    bool is_finish_requested() const;  // New method to check finish button
    
    // Frame info updates
    void update_frame_info(int frame_id, int total_features, int tracked_features, int new_features);
    void update_tracking_stats(int frame_id, int total_features, int stereo_matches, int map_points, float success_rate, float position_error);

    // Gravity frame transformation
    void set_gravity_transformation(const Eigen::Matrix4f& Tgw);

private:
    // Pangolin components
    pangolin::OpenGlRenderState s_cam;
    pangolin::View d_cam;
    pangolin::View d_panel;
    pangolin::View d_img_left;
    pangolin::View d_img_right;
    
    // Data storage
    std::vector<Eigen::Vector3f> m_points;
    std::vector<Eigen::Vector3f> m_trajectory;
    std::vector<Eigen::Matrix4f> m_keyframe_poses;  // Store full poses for frustum drawing
    std::vector<Eigen::Matrix4f> m_gt_trajectory;
    Eigen::Matrix4f m_current_pose;
    Eigen::Matrix4f m_current_camera_pose;  // Store current frame camera pose (T_wc)
    
    // New sliding window data storage
    std::vector<std::shared_ptr<Frame>> m_all_frames;           // All frames for trajectory
    std::vector<std::shared_ptr<Frame>> m_keyframe_window;      // Sliding window of keyframes
    std::shared_ptr<Frame> m_last_keyframe;                     // Last keyframe for relative pose calculation
    Eigen::Matrix4f m_relative_pose_from_last_keyframe;         // Current relative pose from last keyframe
    
    // Map point storage
    std::vector<std::shared_ptr<MapPoint>> m_all_map_points_storage;     // All map points (new sliding window style)
    std::vector<std::shared_ptr<MapPoint>> m_window_map_points_storage;  // Window map points (new sliding window style)
    
    // Legacy map point vectors for drawing (derived from storage)
    std::vector<Eigen::Vector3f> m_all_map_points;      // White - accumulated map points (legacy style)
    std::vector<Eigen::Vector3f> m_current_map_points;  // Red - current frame tracking points (legacy style)
    
    // Thread safety
    mutable std::mutex m_data_mutex;
    
    // Image data
    pangolin::GlTexture m_tracking_image;
    pangolin::GlTexture m_stereo_image;
    bool m_has_tracking_image;
    bool m_has_stereo_image;
    
    // Control variables (simplified - no UI toggles)
    bool m_show_points;
    bool m_show_trajectory;
    bool m_show_keyframe_frustums;
    bool m_show_gt_trajectory;
    bool m_show_camera_frustum;
    bool m_show_grid;
    bool m_show_axis;
    bool m_follow_camera;
    float m_point_size;
    float m_trajectory_width;
    
    // Tracking debug information
    pangolin::Var<int> m_frame_id;
    pangolin::Var<int> m_successful_matches;
    
    // Control buttons
    pangolin::Var<bool> m_auto_mode_checkbox;
    pangolin::Var<bool> m_show_map_point_indices;
    pangolin::Var<bool> m_show_accumulated_map_points;
    pangolin::Var<bool> m_show_current_map_points;
    pangolin::Var<bool> m_show_estimated_trajectory;
    pangolin::Var<bool> m_show_ground_truth_trajectory;
    pangolin::Var<bool> m_show_sliding_window_keyframes;
    pangolin::Var<bool> m_follow_frame_checkbox;
    pangolin::Var<bool> m_step_forward_button;
    pangolin::Var<bool> m_finish_button;
    pangolin::Var<bool> m_show_dense_cloud;  // Control for dense color cloud
    mutable bool m_step_forward_pressed;
    mutable bool m_finish_pressed;
    
    // Gravity frame transformation
    Eigen::Matrix4f m_Tgw;                  // World-to-Gravity transformation matrix (SE(3))
    bool m_has_gravity_transformation;      // Flag to check if transformation is set
    
    // Input state
    bool m_space_pressed;
    bool m_next_pressed;
    
    // Initialization state
    bool m_initialized;
    
    // Window dimensions
    int m_window_width;
    int m_window_height;
    
    // Layout positions
    float m_tracking_image_bottom;
    float m_stereo_image_bottom;
    
    // Thread safety
    mutable std::mutex m_render_mutex;
    bool m_panels_created;
    
    // Drawing functions
    void draw_grid();
    void draw_axis();
    void draw_points();
    void draw_map_points();  // New function for colored map points
    void draw_dense_color_cloud();  // New function for RGB-D dense cloud
    void draw_trajectory();
    void draw_keyframe_frustums();
    void draw_gt_trajectory();
    void draw_pose();
    void draw_camera_frustum();
    void draw_feature_grid(cv::Mat& image);  // Grid overlay for feature distribution
    
    // Utility functions
    pangolin::GlTexture create_texture_from_cv_mat(const cv::Mat& mat);
    void setup_panels();
};

} // namespace lightweight_vio
