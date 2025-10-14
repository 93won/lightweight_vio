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
    void update_keyframe_poses(const std::vector<Eigen::Matrix4f>& keyframe_poses);
    
    // Map point updates with color differentiation
    void update_map_points(const std::vector<Eigen::Vector3f>& all_points, const std::vector<Eigen::Vector3f>& current_points);
    
    // Image updates
    void update_tracking_image(const cv::Mat& image);
    void update_tracking_image_with_map_points(const cv::Mat& image, 
                                              const std::vector<std::shared_ptr<Feature>>& features,
                                              const std::vector<std::shared_ptr<MapPoint>>& map_points);
    void update_tracking_image_with_uncertainty_debug(const cv::Mat& image, 
                                                      const std::vector<std::shared_ptr<Feature>>& features,
                                                      const std::vector<std::shared_ptr<MapPoint>>& map_points,
                                                      std::shared_ptr<Frame> current_frame);
    
    // Direct feature rendering (no OpenCV drawing - more efficient)
    void update_tracking_image_direct(const cv::Mat& raw_image,
                                     const std::vector<std::shared_ptr<Feature>>& features,
                                     const std::vector<std::shared_ptr<MapPoint>>& map_points);
    
    // Simple frame-based update (most efficient - just pass the frame)
    void update_tracking_with_frame(std::shared_ptr<Frame> current_frame);
    
    void update_uncertainty_debug_image(const cv::Mat& image);
    cv::Mat create_uncertainty_debug_image(const std::vector<std::shared_ptr<Feature>>& features,
                                           const std::vector<std::shared_ptr<MapPoint>>& map_points,
                                           std::shared_ptr<Frame> current_frame);
    
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
    
    // Uncertainty visualization
    void render_map_point_uncertainties(const std::vector<std::shared_ptr<MapPoint>>& map_points);
    void draw_uncertainty_ellipsoid(const Eigen::Vector3f& position, 
                                   const Eigen::Matrix3f& covariance,
                                   const Eigen::Vector3f& color = Eigen::Vector3f(1.0f, 0.0f, 0.0f),
                                   float alpha = 0.3f,
                                   float scale_factor = 2.0f);  // 2-sigma ellipsoid by default
    
    // Multi-view observation visualization
    void render_observation_point_clouds(const std::vector<std::shared_ptr<MapPoint>>& map_points);
    void draw_observation_connections(const Eigen::Vector3f& center_pos, 
                                    const std::vector<Eigen::Vector3f>& observation_positions,
                                    const Eigen::Vector3f& line_color = Eigen::Vector3f(0.0f, 0.5f, 1.0f));  // Blue lines
    Eigen::Matrix3f compute_observation_covariance(const std::vector<Eigen::Vector3f>& observation_positions,
                                                  const Eigen::Vector3f& center_position);
    
    // Uncertainty debugging - 2D projection visualization
    void debug_uncertainty_projection(cv::Mat& image, 
                                     std::shared_ptr<Frame> current_frame);
    void draw_uncertainty_ellipse_2d(cv::Mat& image,
                                    const cv::Point2f& center,
                                    const Eigen::Matrix2f& pixel_uncertainty);
    
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
    
    // Feature rendering data (OpenGL-based, more efficient than OpenCV)
    std::vector<Eigen::Vector2f> m_current_features;      // Normalized coordinates [-1, 1]
    std::vector<Eigen::Vector3f> m_current_feature_colors; // RGB colors for each feature
    
    // Current frame for direct access (simplest approach)
    std::shared_ptr<Frame> m_current_frame;
    
    // Thread safety
    mutable std::mutex m_data_mutex;
    
    // Image data
    pangolin::GlTexture m_tracking_image;
    pangolin::GlTexture m_uncertainty_debug_image;
    bool m_has_tracking_image;
    bool m_has_uncertainty_debug_image;
    
    // Control variables (simplified - no UI toggles)
    bool m_show_points;
    bool m_show_trajectory;
    bool m_show_keyframe_frustums;
    bool m_show_camera_frustum;
    bool m_show_grid;
    bool m_show_axis;
    bool m_follow_camera;
    float m_point_size;
    float m_trajectory_width;
    
    // Uncertainty visualization controls
    bool m_show_uncertainties;
    float m_uncertainty_scale;
    float m_min_uncertainty_size;
    
    // Tracking debug information
    pangolin::Var<int> m_frame_id;
    pangolin::Var<int> m_successful_matches;
    
    // Control buttons
    pangolin::Var<bool> m_auto_mode_checkbox;
    pangolin::Var<bool> m_show_map_point_indices;
    pangolin::Var<bool> m_show_accumulated_map_points;
    pangolin::Var<bool> m_show_current_map_points;
    pangolin::Var<bool> m_show_estimated_trajectory;
    pangolin::Var<bool> m_show_sliding_window_keyframes;
    pangolin::Var<bool> m_follow_frame_checkbox;
    pangolin::Var<bool> m_step_forward_button;
    pangolin::Var<bool> m_finish_button;
    pangolin::Var<bool> m_show_uncertainty_ellipsoids;  // New UI control
    pangolin::Var<bool> m_show_observation_point_clouds;  // Multi-view observation visualization
    mutable bool m_step_forward_pressed;
    mutable bool m_finish_pressed;
    
    // Follow frame state tracking
    mutable bool m_previous_follow_frame_state;
    
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
    float m_uncertainty_debug_image_bottom;
    
    // Thread safety
    mutable std::mutex m_render_mutex;
    bool m_panels_created;
    
    // Drawing functions
    void draw_grid();
    void draw_axis();
    void draw_points();
    void draw_map_points();  // New function for colored map points
    void draw_trajectory();
    void draw_keyframe_frustums();
    void draw_pose();
    void draw_camera_frustum();
    void draw_feature_grid(cv::Mat& image);  // Grid overlay for feature distribution
    
    // Uncertainty drawing helpers
    void draw_sphere(float radius, int slices = 16, int stacks = 16);
    void draw_wireframe_sphere(float radius, int slices = 16, int stacks = 16);  // New wireframe version
    void draw_wireframe_ellipsoid(float a, float b, float c, int slices = 16, int stacks = 16);  // True ellipsoid
    void generate_ellipsoid_vertices(const Eigen::Matrix3f& shape_matrix, 
                                   std::vector<Eigen::Vector3f>& vertices,
                                   std::vector<std::vector<int>>& faces,
                                   int slices = 16, int stacks = 16);
    Eigen::Vector3f uncertainty_to_color(float uncertainty_magnitude, float min_uncertainty, float max_uncertainty);
    
    // Utility functions
    pangolin::GlTexture create_texture_from_cv_mat(const cv::Mat& mat);
    void setup_panels();
};

} // namespace lightweight_vio
