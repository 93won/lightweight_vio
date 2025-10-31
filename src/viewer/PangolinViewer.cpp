/**
 * @file      PangolinViewer.cpp
 * @brief     Implements the Pangolin-based 3D viewer for VIO.
 * @author    Seungwon Choi (csw3575@snu.ac.kr)
 * @date      2025-08-30
 * @copyright Copyright (c) 2025 Seungwon Choi. All rights reserved.
 *
 * @par License
 * This project is released under the MIT License.
 */

#include "viewer/PangolinViewer.h"
#include "util/Config.h"
#include "database/Feature.h"
#include "database/MapPoint.h"
#include "database/Frame.h"

#include <iostream>
#include <cmath>
#include <cstdio>
#include <spdlog/spdlog.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

namespace lightweight_vio {

// Helper function to get max features from config
static int get_max_features_from_config() {
    try {
        const auto& config = Config::getInstance();
        return config.m_max_features;
    } catch (const std::exception& e) {
        // Fallback to default value
        return 150;
    }
}

PangolinViewer::PangolinViewer()
    : m_current_pose(Eigen::Matrix4f::Identity())
    , m_current_camera_pose(Eigen::Matrix4f::Identity())
    , m_relative_pose_from_last_keyframe(Eigen::Matrix4f::Identity())
    , m_has_tracking_image(false)
    , m_has_uncertainty_debug_image(false)
    , m_has_depth_image(false)
    , m_space_pressed(false)
    , m_next_pressed(false)
    , m_initialized(false)
    , m_window_width(1280)
    , m_window_height(960)
    , m_tracking_image_bottom(0.35f)
    , m_uncertainty_debug_image_bottom(0.0f)
    , m_depth_image_bottom(0.0f)
    , m_panels_created(false)
    , m_show_points(true)
    , m_show_trajectory(true)
    , m_show_keyframe_frustums(true)
    , m_show_camera_frustum(true)
    , m_show_grid(true)
    , m_show_axis(true)
    , m_follow_camera(true)
    , m_point_size(3.0f)
    , m_trajectory_width(2.0f)
    , m_show_uncertainties(true)
    , m_uncertainty_scale(1.0f)
    , m_min_uncertainty_size(0.01f)
    , m_frame_id("ui.Frame ID", 0)
    , m_successful_matches("ui.Num Tracked Map Points", 0, 0, get_max_features_from_config())
    , m_auto_mode_checkbox("ui.1. Auto Mode", false, true)
    , m_show_map_point_indices("ui.2. Show Map Point IDs", true, true)
    , m_show_accumulated_map_points("ui.3. Show Local Map Points", true, true)
    , m_show_current_map_points("ui.4. Show Current Map Points", true, true)
    , m_show_estimated_trajectory("ui.5. Show Estimated Trajectory", true, true)
    , m_show_sliding_window_keyframes("ui.6. Show Sliding Window Keyframes", true, true)
    , m_follow_frame_checkbox("ui.7. Follow Frame", true, true)
    , m_step_forward_button("ui.8. Step Forward", false, false)
    , m_finish_button("ui.9. Finish & Exit", false, false)
    , m_show_uncertainty_ellipsoids("ui.10. Show Uncertainty Ellipsoids", false, true)
    , m_show_observation_point_clouds("ui.11. Show Observation Point Clouds", true, true)
    , m_show_dense_point_cloud("ui.12. Show Dense Point Cloud (RGBD)", true, true)
    , m_toggle_dense_color_mode("ui.13. Toggle Dense Color (RGB/Depth)", false, false)
    , m_step_forward_pressed(false)
    , m_finish_pressed(false)
    , m_dense_color_mode_rgb(true)  // Start with RGB mode
    , m_previous_follow_frame_state(true)  // Initialize to true since follow frame starts enabled
    , m_Tgw(Eigen::Matrix4f::Identity())
    , m_has_gravity_transformation(false)
{
}

PangolinViewer::~PangolinViewer() {
    shutdown();
}

bool PangolinViewer::initialize(int width, int height) {
    // Store window dimensions
    m_window_width = width;
    m_window_height = height;
    
    // Create OpenGL window with Pangolin
    pangolin::CreateWindowAndBind("Statistical Uncertainty Learning for Robust Visual-Inertial State Estimation", width, height);
    glEnable(GL_DEPTH_TEST);
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

    // Set Pangolin UI text color to white for dark theme
    pangolin::RegisterKeyPressCallback(pangolin::PANGO_SPECIAL + pangolin::PANGO_KEY_F1, [](){});
    
    // Setup camera for 3D navigation with proper initial view
    // Use more reasonable focal length based on image dimensions
    float fx = width * 0.7f;  // Adjust focal length to be proportional to window size
    float fy = height * 0.7f;
    s_cam = pangolin::OpenGlRenderState(
        pangolin::ProjectionMatrix(width, height, fx, fy, width/2, height/2, 0.1, 1000),
        pangolin::ModelViewLookAt(-3, -3, 3, 0, 0, 0, pangolin::AxisZ)  // Changed to AxisZ for better orientation
    );

    // Setup display panels
    setup_panels();

    // Set clear color to dark navy background
    glClearColor(0.1f, 0.1f, 0.15f, 1.0f);  // Dark navy background

    m_initialized = true;
    return true;
}

void PangolinViewer::setup_panels() {
    // Set UI panel width to 1/4 of window width
    int ui_panel_width = m_window_width / 4;
    
    // Get actual image size and camera type from config
    float image_width, image_height;
    bool is_rgbd = false;
    try {
        const auto& config = Config::getInstance();
        image_width = static_cast<float>(config.m_image_width);
        image_height = static_cast<float>(config.m_image_height);
        is_rgbd = (config.m_camera_type == CameraType::RGBD);
    } catch (const std::exception& e) {
        // Fallback to default values (EuRoC standard)
        image_width = 752.0f;
        image_height = 480.0f;
        is_rgbd = false;
    }

    // 2. Feature tracking image (bottom position - always at bottom)
    float tracking_aspect = image_width / image_height;  // 752/480 = 1.567
    float display_width = static_cast<float>(m_window_width) * 0.25f;  // UI panel width (25% of window width)
    float tracking_height = display_width / tracking_aspect;
    float tracking_normalized_height = tracking_height / static_cast<float>(m_window_height);
    
    // 3. Depth heatmap image (above tracking image) - ONLY for RGBD
    float depth_aspect = image_width / image_height;  // Same aspect ratio
    float depth_height = display_width / depth_aspect;
    float depth_normalized_height = depth_height / static_cast<float>(m_window_height);
    
    // Layout from TOP to BOTTOM: UI panel -> [depth image (RGBD only)] -> tracking image
    // Tracking image at the very bottom (always)
    m_tracking_image_bottom = 0.0f;
    float tracking_image_top = m_tracking_image_bottom + tracking_normalized_height;
    
    // Depth image above tracking image (ONLY for RGBD)
    float depth_image_top;
    if (is_rgbd) {
        m_depth_image_bottom = tracking_image_top;
        depth_image_top = m_depth_image_bottom + depth_normalized_height;
    } else {
        // No depth image for stereo/monocular
        m_depth_image_bottom = 0.0f;
        depth_image_top = tracking_image_top;
    }
    
    // UI panel takes the rest of the space above depth image (or tracking image if no depth)
    float ui_panel_bottom = depth_image_top;
    float ui_panel_top = 1.0f;
    
    // Uncertainty image disabled
    m_uncertainty_debug_image_bottom = 0.0f;
    float uncertainty_image_top = 0.0f;
    
    if (!m_panels_created) {
        // Create panels only once
        
        // Calculate UI panel width as ratio (1/4 of window width)
        float ui_panel_ratio = 0.25f;  // 1/4
        
        // Create main 3D view (takes up most of the screen on the right side)
        d_cam = pangolin::CreateDisplay()
            .SetBounds(0.0, 1.0, pangolin::Attach::Frac(ui_panel_ratio), pangolin::Attach::Frac(1.0f))
            .SetHandler(new pangolin::Handler3D(s_cam));

        // 1. UI panel - above depth image (or tracking image if no depth)
        d_panel = pangolin::CreatePanel("ui")
            .SetBounds(ui_panel_bottom, ui_panel_top, 0.0, pangolin::Attach::Frac(ui_panel_ratio));
        
        // 2. Depth heatmap image in the middle (ONLY for RGBD)
        if (is_rgbd) {
            d_img_right = pangolin::CreateDisplay()
                .SetBounds(m_depth_image_bottom, depth_image_top, 0.0, pangolin::Attach::Frac(ui_panel_ratio), -depth_aspect);
        }
            
        // 3. Feature tracking image at the bottom (always)
        d_img_left = pangolin::CreateDisplay()
            .SetBounds(m_tracking_image_bottom, tracking_image_top, 0.0, pangolin::Attach::Frac(ui_panel_ratio), -tracking_aspect);
        
        m_panels_created = true;
    } else {
        // Update size of already created panels (based on ratio)
        float ui_panel_ratio = 0.25f;  // 1/4
        
        // Recalculate image sizes (reflecting window size changes)
        float new_display_width = static_cast<float>(m_window_width) * ui_panel_ratio;
        float new_tracking_height = new_display_width / tracking_aspect;
        float new_tracking_normalized_height = new_tracking_height / static_cast<float>(m_window_height);
        
        float new_depth_height = new_display_width / depth_aspect;
        float new_depth_normalized_height = new_depth_height / static_cast<float>(m_window_height);
        
        // Layout from TOP to BOTTOM: UI panel -> [depth image (RGBD only)] -> tracking image
        // Tracking at bottom
        m_tracking_image_bottom = 0.0f;
        float new_tracking_image_top = m_tracking_image_bottom + new_tracking_normalized_height;
        
        // Depth above tracking (ONLY for RGBD)
        float new_depth_image_top;
        if (is_rgbd) {
            m_depth_image_bottom = new_tracking_image_top;
            new_depth_image_top = m_depth_image_bottom + new_depth_normalized_height;
        } else {
            m_depth_image_bottom = 0.0f;
            new_depth_image_top = new_tracking_image_top;
        }
        
        // UI panel takes the rest of the space above depth image (or tracking image if no depth)
        float new_ui_panel_bottom = new_depth_image_top;
        float new_ui_panel_top = 1.0f;
        
        // Uncertainty image disabled
        float new_uncertainty_image_top = 0.0f;
        m_uncertainty_debug_image_bottom = 0.0f;
        
        d_cam.SetBounds(0.0, 1.0, pangolin::Attach::Frac(ui_panel_ratio), pangolin::Attach::Frac(1.0f));
        d_panel.SetBounds(new_ui_panel_bottom, new_ui_panel_top, 0.0, pangolin::Attach::Frac(ui_panel_ratio));
        
        // Update depth image bounds (middle) - ONLY for RGBD
        if (is_rgbd) {
            d_img_right.SetBounds(m_depth_image_bottom, new_depth_image_top, 0.0, pangolin::Attach::Frac(ui_panel_ratio), -depth_aspect);
        }
        
        // Update tracking image bounds (bottom)
        d_img_left.SetBounds(m_tracking_image_bottom, new_tracking_image_top, 0.0, pangolin::Attach::Frac(ui_panel_ratio), -tracking_aspect);
     
    }
}

void PangolinViewer::shutdown() {
    if (m_initialized) {
        pangolin::DestroyWindow("Statistical Uncertainty Learning for Robust Visual-Inertial State Estimation");
        m_initialized = false;
    }
}

bool PangolinViewer::should_close() const {
    return pangolin::ShouldQuit();
}

bool PangolinViewer::is_ready() const {
    return m_initialized;
}

void PangolinViewer::render() {
    if (!m_initialized) return;
    
    // 매 프레임마다 현재 창 크기를 확인하고 변경사항이 있으면 즉시 적용
    GLint viewport[4];
    glGetIntegerv(GL_VIEWPORT, viewport);
    int current_width = viewport[2];
    int current_height = viewport[3];
    
    // 창 크기가 조금이라도 변경되면 즉시 레이아웃 업데이트
    if (current_width != m_window_width || current_height != m_window_height) {
                  
        m_window_width = current_width;
        m_window_height = current_height;
        
        // 즉시 모든 패널 레이아웃 재설정
        setup_panels();
        
    }

    // Clear screen and activate view to render into
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    // Activate 3D view
    d_cam.Activate(s_cam);

    // Draw 3D content
    if (m_show_grid) {
        draw_grid();
    }

    if (m_show_axis) {
        draw_axis();
    }

    if (m_show_points && !m_points.empty()) {
        draw_points();
    }

    // Draw map points with color differentiation
    draw_map_points();

    // Draw uncertainty ellipsoids if enabled
    if (m_show_uncertainty_ellipsoids) {
        std::lock_guard<std::mutex> lock(m_data_mutex);
        if (!m_all_map_points_storage.empty()) {
            render_map_point_uncertainties(m_all_map_points_storage);
        }
    }

    // Draw observation point clouds if enabled
    if (m_show_observation_point_clouds) {
        std::lock_guard<std::mutex> lock(m_data_mutex);
        if (!m_all_map_points_storage.empty()) {
            render_observation_point_clouds(m_all_map_points_storage);
        }
    }

    // ⭐ Draw dense point cloud if enabled (RGBD only)
    if (m_show_dense_point_cloud) {
        draw_dense_point_cloud();
    }

    if (m_show_estimated_trajectory && m_show_trajectory && !m_trajectory.empty()) {
        draw_trajectory();
    }

    if (m_show_sliding_window_keyframes && m_show_keyframe_frustums && !m_keyframe_window.empty()) {
        draw_keyframe_frustums();
    }

    if (!m_current_pose.isZero()) {
        draw_pose();
        
        if (m_show_camera_frustum) {
            draw_camera_frustum();
        }
    }

    // Render tracking image at the bottom (always)
    if (m_has_tracking_image) {
        d_img_left.Activate();
        glColor3f(1.0, 1.0, 1.0);
        m_tracking_image.RenderToViewport();
    }

    // ⭐ Render depth heatmap image in the middle
    if (m_has_depth_image) {
        d_img_right.Activate();
        glColor3f(1.0, 1.0, 1.0);
        m_depth_image.RenderToViewport();
    }

    // Pangolin automatically renders the UI panel with tracking variables
    // No custom drawing needed - the pangolin::Var variables are displayed automatically

    // Check Step Forward button
    if (pangolin::Pushed(m_step_forward_button)) {
        m_step_forward_pressed = true;
    }

    // Check Finish button
    if (pangolin::Pushed(m_finish_button)) {
        m_finish_pressed = true;
    }
    
    // Check Toggle Dense Color button
    if (pangolin::Pushed(m_toggle_dense_color_mode)) {
        m_dense_color_mode_rgb = !m_dense_color_mode_rgb;
        spdlog::info("[PangolinViewer] Dense point cloud color mode: {}", 
                     m_dense_color_mode_rgb ? "RGB" : "Depth Heatmap");
    }

    // Process keyboard input - will be handled externally
    // Note: Space bar and 'n' key handling is done in the main application loop

    // ===== Follow Frame Mode (Camera Following) =====
    // This mode makes the viewer camera follow the current frame position
    // while still allowing user to zoom and rotate around it
    static bool was_follow_active = false;  // Track follow mode state across frames
    
    // Follow Frame mode - based on ORB-SLAM2 implementation
    static bool bFollow = false;       // Track if currently following
    static bool bFirstTime = true;     // Track if this is the first time ever
    
    if (m_follow_frame_checkbox && m_current_frame) {
        // Get camera-to-world transformation
        Eigen::Matrix4f T_wc = m_current_frame->get_Twc();
        
        // Convert Eigen matrix to Pangolin OpenGL matrix
        pangolin::OpenGlMatrix Twc_gl(T_wc);
        
        if (bFollow) {
            // Already following - just continue following
            s_cam.Follow(Twc_gl);
        } else 
        {
            // Just turned on
            if (bFirstTime) {
                // First time ever - set initial view based on current camera pose
                Eigen::Vector3f cam_pos = T_wc.block<3, 1>(0, 3);
                Eigen::Matrix3f R_wc = T_wc.block<3, 3>(0, 0);
                
                // Camera frame axes in world coordinates
                Eigen::Vector3f cam_z = R_wc.col(2);  // Forward direction
                Eigen::Vector3f cam_y = R_wc.col(1);  // Down direction
                
                // Place viewer behind and above the camera
                Eigen::Vector3f viewer_pos = cam_pos 
                    - cam_z * 4.0f   // 2m behind camera
                    - cam_y * 2.0f;  // 1m above camera
                
                pangolin::OpenGlMatrix initial_view = pangolin::ModelViewLookAt(
                    viewer_pos.x(), viewer_pos.y(), viewer_pos.z(),
                    cam_pos.x(), cam_pos.y(), cam_pos.z(),
                    -cam_y.x(), -cam_y.y(), -cam_y.z()
                );
                s_cam.SetModelViewMatrix(initial_view);
                bFirstTime = false;
            }
            s_cam.Follow(Twc_gl);
            bFollow = true;
        }
    } else if (bFollow) {
        // Just turned off - stop following (don't call Follow anymore)
        bFollow = false;
    }


    // Ensure UI text is rendered in white for dark theme
    glColor3f(1.0f, 1.0f, 1.0f);

    // Swap frames and Process Events
    pangolin::FinishFrame();
}

void PangolinViewer::reset_camera() {
    s_cam.SetModelViewMatrix(pangolin::ModelViewLookAt(-3, -3, 3, 0, 0, 0, pangolin::AxisZ));
}

void PangolinViewer::draw_grid() {
    const float grid_size = 10.0f;
    const float step = 1.0f;
    
    glLineWidth(1.0f);
    glBegin(GL_LINES);
    
    // Grid lines in light gray for dark background
    glColor3f(0.6f, 0.6f, 0.6f);
    for (float i = -grid_size; i <= grid_size; i += step) {
        // X direction lines
        glVertex3f(i, -grid_size, 0.0f);
        glVertex3f(i, grid_size, 0.0f);
        
        // Y direction lines
        glVertex3f(-grid_size, i, 0.0f);
        glVertex3f(grid_size, i, 0.0f);
    }
    
    // Axis lines in different colors
    glColor3f(1.0f, 0.0f, 0.0f); // X axis in red
    glVertex3f(-grid_size, 0.0f, 0.0f);
    glVertex3f(grid_size, 0.0f, 0.0f);
    
    glColor3f(0.0f, 1.0f, 0.0f); // Y axis in green
    glVertex3f(0.0f, -grid_size, 0.0f);
    glVertex3f(0.0f, grid_size, 0.0f);
    
    glEnd();
}

void PangolinViewer::draw_axis() {
    const float axis_length = 2.0f;
    
    glLineWidth(3.0f);
    glBegin(GL_LINES);
    
    // X axis - Red
    glColor3f(1.0f, 0.0f, 0.0f);
    glVertex3f(0.0f, 0.0f, 0.0f);
    glVertex3f(axis_length, 0.0f, 0.0f);
    
    // Y axis - Green
    glColor3f(0.0f, 1.0f, 0.0f);
    glVertex3f(0.0f, 0.0f, 0.0f);
    glVertex3f(0.0f, axis_length, 0.0f);
    
    // Z axis - Blue
    glColor3f(0.0f, 0.0f, 1.0f);
    glVertex3f(0.0f, 0.0f, 0.0f);
    glVertex3f(0.0f, 0.0f, axis_length);
    
    glEnd();
    glLineWidth(1.0f);
}

void PangolinViewer::draw_points() {
    glPointSize(m_point_size*5.0f); // Slightly larger for visibility
    glColor3f(1.0f, 0.0f, 0.0f); // Red points
    
    glBegin(GL_POINTS);
    for (const auto& point : m_points) {
        glVertex3f(point.x(), point.y(), point.z());
    }
    glEnd();
    
    glPointSize(1.0f);
}

void PangolinViewer::draw_map_points() {
    std::lock_guard<std::mutex> lock(m_data_mutex);
    
    // Draw all map points in light gray (background points)
    if (!m_all_map_points_storage.empty() && m_show_accumulated_map_points) {
        glPointSize(m_point_size);
        glColor3f(0.8f, 0.8f, 0.8f); // Light gray for all map points
        
        glBegin(GL_POINTS);
        for (const auto& point : m_all_map_points_storage) {
            if (point && !point->is_bad() && !point->is_multi_view_triangulated()) {
                Eigen::Vector3f position = point->get_position();
                glVertex3f(position.x(), position.y(), position.z());
            }
        }
        glEnd();
    }
    
    // Draw sliding window map points in white (more prominent) - excluding multi-view triangulated
    if (!m_window_map_points_storage.empty() && m_show_accumulated_map_points) {
        glPointSize(m_point_size * 1.0f); // Slightly larger
        glColor3f(1.0f, 1.0f, 1.0f); // White for window map points
        
        glBegin(GL_POINTS);
        for (const auto& point : m_window_map_points_storage) {
            if (point && !point->is_bad() && !point->is_multi_view_triangulated()) {
                Eigen::Vector3f position = point->get_position();
                glVertex3f(position.x(), position.y(), position.z());
            }
        }
        glEnd();
    }
   
    
    // Draw marginalized map points as green wireframe spheres (overlay on top of everything else)
    if (!m_all_map_points_storage.empty() && m_show_accumulated_map_points) {
        glEnable(GL_BLEND);
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
        glColor4f(0.0f, 1.0f, 0.0f, 0.1f); // Green for marginalized points with alpha 0.3
        glPolygonMode(GL_FRONT_AND_BACK, GL_LINE); // Enable wireframe mode
        glLineWidth(1.0f);
        
        for (const auto& point : m_all_map_points_storage) {
            if (point && !point->is_bad() && point->is_marginalized()) {
                Eigen::Vector3f position = point->get_position();
                
                // Draw wireframe sphere using OpenGL primitives
                glPushMatrix();
                glTranslatef(position.x(), position.y(), position.z());
                
                // Draw a wireframe sphere (radius 0.05 as requested)
                const float radius = 0.05f;
                const int slices = 12;
                const int stacks = 8;
                
                for (int i = 0; i < stacks; ++i) {
                    float lat0 = M_PI * (-0.5f + (float)i / stacks);
                    float z0 = radius * sin(lat0);
                    float zr0 = radius * cos(lat0);
                    
                    float lat1 = M_PI * (-0.5f + (float)(i + 1) / stacks);
                    float z1 = radius * sin(lat1);
                    float zr1 = radius * cos(lat1);
                    
                    glBegin(GL_LINE_STRIP);
                    for (int j = 0; j <= slices; ++j) {
                        float lng = 2 * M_PI * (float)j / slices;
                        float x = cos(lng);
                        float y = sin(lng);
                        
                        glVertex3f(x * zr0, y * zr0, z0);
                        glVertex3f(x * zr1, y * zr1, z1);
                    }
                    glEnd();
                }
                
                // Draw longitude lines
                for (int j = 0; j < slices; ++j) {
                    float lng = 2 * M_PI * (float)j / slices;
                    float x = cos(lng);
                    float y = sin(lng);
                    
                    glBegin(GL_LINE_STRIP);
                    for (int i = 0; i <= stacks; ++i) {
                        float lat = M_PI * (-0.5f + (float)i / stacks);
                        float z = radius * sin(lat);
                        float zr = radius * cos(lat);
                        
                        glVertex3f(x * zr, y * zr, z);
                    }
                    glEnd();
                }
                
                glPopMatrix();
            }
        }
        
        glPolygonMode(GL_FRONT_AND_BACK, GL_FILL); // Reset to fill mode
        glDisable(GL_BLEND); // Disable blending
    }
    
    // Also check window map points for marginalized ones
    if (!m_window_map_points_storage.empty() && m_show_accumulated_map_points) {
        glEnable(GL_BLEND);
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
        glColor4f(0.0f, 1.0f, 0.0f, 0.3f); // Green for marginalized points with alpha 0.3
        glPolygonMode(GL_FRONT_AND_BACK, GL_LINE); // Enable wireframe mode
        glLineWidth(1.0f);
        
        for (const auto& point : m_window_map_points_storage) {
            if (point && !point->is_bad() && point->is_marginalized()) {
                Eigen::Vector3f position = point->get_position();
                
                // Draw wireframe sphere using OpenGL primitives
                glPushMatrix();
                glTranslatef(position.x(), position.y(), position.z());
                
                // Draw a wireframe sphere (radius 0.05 as requested)
                const float radius = 0.05f;
                const int slices = 12;
                const int stacks = 8;
                
                for (int i = 0; i < stacks; ++i) {
                    float lat0 = M_PI * (-0.5f + (float)i / stacks);
                    float z0 = radius * sin(lat0);
                    float zr0 = radius * cos(lat0);
                    
                    float lat1 = M_PI * (-0.5f + (float)(i + 1) / stacks);
                    float z1 = radius * sin(lat1);
                    float zr1 = radius * cos(lat1);
                    
                    glBegin(GL_LINE_STRIP);
                    for (int j = 0; j <= slices; ++j) {
                        float lng = 2 * M_PI * (float)j / slices;
                        float x = cos(lng);
                        float y = sin(lng);
                        
                        glVertex3f(x * zr0, y * zr0, z0);
                        glVertex3f(x * zr1, y * zr1, z1);
                    }
                    glEnd();
                }
                
                // Draw longitude lines
                for (int j = 0; j < slices; ++j) {
                    float lng = 2 * M_PI * (float)j / slices;
                    float x = cos(lng);
                    float y = sin(lng);
                    
                    glBegin(GL_LINE_STRIP);
                    for (int i = 0; i <= stacks; ++i) {
                        float lat = M_PI * (-0.5f + (float)i / stacks);
                        float z = radius * sin(lat);
                        float zr = radius * cos(lat);
                        
                        glVertex3f(x * zr, y * zr, z);
                    }
                    glEnd();
                }
                
                glPopMatrix();
            }
        }
        
        glPolygonMode(GL_FRONT_AND_BACK, GL_FILL); // Reset to fill mode
        glDisable(GL_BLEND); // Disable blending
    }
   
    // Draw current frame tracking points in red (highest priority - overlay on top)
    if (!m_current_map_points.empty() && m_show_current_map_points) {
        glPointSize(m_point_size * 2.0f); // Largest for visibility
        glColor3f(1.0f, 0.0f, 0.0f); // Red for current frame tracking points
        
        glBegin(GL_POINTS);
        for (const auto& point : m_current_map_points) {
            glVertex3f(point.x(), point.y(), point.z());
        }
        glEnd();
    }
    
    glPointSize(1.0f); // Reset point size
}

void PangolinViewer::draw_trajectory() {
    // Use get_all_frames() for thread-safe access
    auto all_frames = get_all_frames();
    
    if (all_frames.size() < 2) return;
    
    // Enable blending for transparency
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
    
    glLineWidth(m_trajectory_width);
    glColor4f(1.0f, 1.0f, 0.0f, 0.3f); // Yellow trajectory with 0.3 alpha
    
    glBegin(GL_LINE_STRIP);
    for (const auto& frame : all_frames) {
        if (frame) {
            Eigen::Vector3f position = frame->get_Twb().block<3, 1>(0, 3);
            glVertex3f(position.x(), position.y(), position.z());
        }
    }
    glEnd();
    
    // ⭐ Draw ground truth trajectory (Green, thicker, semi-transparent)
    if (m_gt_trajectory.size() >= 2) {
        glLineWidth(m_trajectory_width * 1.5f);  // Slightly thicker
        glColor4f(0.0f, 1.0f, 0.0f, 0.5f); // Green with 0.5 alpha
        
        glBegin(GL_LINE_STRIP);
        for (const auto& pos : m_gt_trajectory) {
            glVertex3f(pos.x(), pos.y(), pos.z());
        }
        glEnd();
    }
    
    // Disable blending after drawing
    glDisable(GL_BLEND);
    
    glLineWidth(0.3f);
}

void PangolinViewer::draw_keyframe_frustums() {
    std::lock_guard<std::mutex> lock(m_data_mutex);
    
    // Use sliding window keyframes instead of old m_keyframe_poses
    if (m_keyframe_window.empty()) return;
    
    // Set sky blue color for keyframe frustums
    glColor3f(0.5f, 0.8f, 1.0f);
    
    for (const auto& keyframe : m_keyframe_window) {
        if (!keyframe) continue;
        
        // Get Twc from keyframe
        Eigen::Matrix4f Twc = keyframe->get_Twc();
        Eigen::Matrix4f T_wc = Twc;
        
        Eigen::Vector3f position = T_wc.block<3, 1>(0, 3);
        Eigen::Matrix3f rotation = T_wc.block<3, 3>(0, 0);
        
        // Draw smaller keyframe frustum
        float scale = 0.1f;  // Smaller scale for keyframes
        
        // Camera frustum vertices (in camera coordinate)
        std::vector<Eigen::Vector3f> frustum_points = {
            Eigen::Vector3f(0, 0, 0),                    // Camera center
            Eigen::Vector3f(-scale, -scale, scale * 2),  // Bottom-left
            Eigen::Vector3f(scale, -scale, scale * 2),   // Bottom-right  
            Eigen::Vector3f(scale, scale, scale * 2),    // Top-right
            Eigen::Vector3f(-scale, scale, scale * 2)    // Top-left
        };
        
        // Transform to world coordinates
        for (auto& point : frustum_points) {
            point = rotation * point + position;
        }
        
        // Draw frustum edges
        glBegin(GL_LINES);
        // Lines from camera center to corners
        for (int i = 1; i < 5; ++i) {
            glVertex3f(frustum_points[0].x(), frustum_points[0].y(), frustum_points[0].z());
            glVertex3f(frustum_points[i].x(), frustum_points[i].y(), frustum_points[i].z());
        }
        // Rectangle at far plane
        for (int i = 1; i < 5; ++i) {
            int next = (i % 4) + 1;
            glVertex3f(frustum_points[i].x(), frustum_points[i].y(), frustum_points[i].z());
            glVertex3f(frustum_points[next].x(), frustum_points[next].y(), frustum_points[next].z());
        }
        glEnd();
    }
}

void PangolinViewer::draw_pose() {
    Eigen::Vector3f position = m_current_pose.block<3, 1>(0, 3);
    Eigen::Matrix3f rotation = m_current_pose.block<3, 3>(0, 0);
    
    // Draw camera position
    glPointSize(8.0f);
    glColor3f(1.0f, 0.0f, 0.0f);
    glBegin(GL_POINTS);
    glVertex3f(position.x(), position.y(), position.z());
    glEnd();
    glPointSize(1.0f);
    
    // Draw body frame axes
    const float axis_length = 0.3f;
    glLineWidth(3.0f);
    glBegin(GL_LINES);
    
    // X-axis (Red)
    glColor3f(1.0f, 0.0f, 0.0f);
    Eigen::Vector3f x_axis = position + rotation.col(0) * axis_length;
    glVertex3f(position.x(), position.y(), position.z());
    glVertex3f(x_axis.x(), x_axis.y(), x_axis.z());
    
    // Y-axis (Green)
    glColor3f(0.0f, 1.0f, 0.0f);
    Eigen::Vector3f y_axis = position + rotation.col(1) * axis_length;
    glVertex3f(position.x(), position.y(), position.z());
    glVertex3f(y_axis.x(), y_axis.y(), y_axis.z());
    
    // Z-axis (Blue)
    glColor3f(0.0f, 0.0f, 1.0f);
    Eigen::Vector3f z_axis = position + rotation.col(2) * axis_length;
    glVertex3f(position.x(), position.y(), position.z());
    glVertex3f(z_axis.x(), z_axis.y(), z_axis.z());
    
    glEnd();
    glLineWidth(1.0f);
}

void PangolinViewer::draw_camera_frustum() {
    // Draw current frame frustum using T_wc (same as keyframes)
    // This ensures consistency with keyframe frustum rendering
    
    if (m_current_camera_pose.isZero()) {
        return; // No camera pose available
    }
    
    // Use the stored T_wc directly (same as keyframes)
    Eigen::Vector3f position = m_current_camera_pose.block<3, 1>(0, 3);
    Eigen::Matrix3f rotation = m_current_camera_pose.block<3, 3>(0, 0);
    
    // Draw current frame frustum - larger than keyframes
    float scale = 0.15f;  // Larger scale for current frame
    
    // Camera frustum vertices (in camera coordinate)
    std::vector<Eigen::Vector3f> frustum_points = {
        Eigen::Vector3f(0, 0, 0),                    // Camera center
        Eigen::Vector3f(-scale, -scale, scale * 2),  // Bottom-left
        Eigen::Vector3f(scale, -scale, scale * 2),   // Bottom-right  
        Eigen::Vector3f(scale, scale, scale * 2),    // Top-right
        Eigen::Vector3f(-scale, scale, scale * 2)    // Top-left
    };
    
    // Transform to world coordinates (same as keyframes)
    for (auto& point : frustum_points) {
        point = rotation * point + position;
    }
    
    glLineWidth(4.0f);  // Thicker lines for current frame
    glColor3f(1.0f, 0.4f, 0.7f); // Pink/Magenta frustum
    
    // Draw frustum edges
    glBegin(GL_LINES);
    // Lines from camera center to corners
    for (int i = 1; i < 5; ++i) {
        glVertex3f(frustum_points[0].x(), frustum_points[0].y(), frustum_points[0].z());
        glVertex3f(frustum_points[i].x(), frustum_points[i].y(), frustum_points[i].z());
    }
    // Rectangle at far plane
    for (int i = 1; i < 5; ++i) {
        int next = (i % 4) + 1;
        glVertex3f(frustum_points[i].x(), frustum_points[i].y(), frustum_points[i].z());
        glVertex3f(frustum_points[next].x(), frustum_points[next].y(), frustum_points[next].z());
    }
    glEnd();
    glLineWidth(1.0f);
}

pangolin::GlTexture PangolinViewer::create_texture_from_cv_mat(const cv::Mat& mat) {
    pangolin::GlTexture tex;
    
    // Convert BGR to RGB if needed and flip vertically to match OpenGL coordinate system
    cv::Mat rgb_mat;
    if (mat.channels() == 3) {
        cv::cvtColor(mat, rgb_mat, cv::COLOR_BGR2RGB);
    } else if (mat.channels() == 1) {
        cv::cvtColor(mat, rgb_mat, cv::COLOR_GRAY2RGB);
    } else {
        rgb_mat = mat;
    }
    
    // Flip vertically to match OpenGL coordinate system (OpenCV: top-left origin, OpenGL: bottom-left origin)
    cv::Mat flipped_mat;
    cv::flip(rgb_mat, flipped_mat, 0);
    
    tex.Reinitialise(flipped_mat.cols, flipped_mat.rows, GL_RGB, false, 0, GL_RGB, GL_UNSIGNED_BYTE);
    tex.Upload(flipped_mat.ptr(), GL_RGB, GL_UNSIGNED_BYTE);
    
    // Release temporary cv::Mat objects to free memory immediately
    rgb_mat.release();
    flipped_mat.release();
    
    return tex;
}

// Data update functions
void PangolinViewer::update_points(const std::vector<Eigen::Vector3f>& points) {
    m_points = points;
}

void PangolinViewer::update_pose(const Eigen::Matrix4f& pose) {
    m_current_pose = pose;
}

void PangolinViewer::update_camera_pose(const Eigen::Matrix4f& T_wc) {
    m_current_camera_pose = T_wc;
}

void PangolinViewer::update_trajectory(const std::vector<Eigen::Vector3f>& trajectory) {
    m_trajectory = trajectory;
}

void PangolinViewer::update_ground_truth_trajectory(const std::vector<Eigen::Vector3f>& gt_trajectory) {
    m_gt_trajectory = gt_trajectory;
}

// DEPRECATED: Use update_keyframe_window() instead
void PangolinViewer::update_keyframe_poses(const std::vector<Eigen::Matrix4f>& keyframe_poses) {
    // This function is deprecated but kept for backward compatibility
    // Convert to new sliding window format if needed
    spdlog::warn("update_keyframe_poses() is deprecated. Use update_keyframe_window() instead.");
    m_keyframe_poses = keyframe_poses;
}

void PangolinViewer::update_map_points(const std::vector<Eigen::Vector3f>& all_points, const std::vector<Eigen::Vector3f>& current_points) {
    m_all_map_points = all_points;
    m_current_map_points = current_points;
}

void PangolinViewer::update_tracking_image(const cv::Mat& image) {
    if (image.empty()) {
        spdlog::warn("[PangolinViewer] Received empty tracking image");
        return;
    }
    
    m_tracking_image = create_texture_from_cv_mat(image);
    m_has_tracking_image = true;
    
    // The bounds and aspect ratio are now handled exclusively by setup_panels().
    // This function is only responsible for updating the texture.
    // spdlog::debug("[PangolinViewer] Updated tracking image texture {}x{}", image.cols, image.rows);
}

void PangolinViewer::update_tracking_image_with_map_points(const cv::Mat& image, 
                                                          const std::vector<std::shared_ptr<Feature>>& features,
                                                          const std::vector<std::shared_ptr<MapPoint>>& map_points) {
    if (image.empty()) return;
    
    // Create a copy of the image to draw on
    cv::Mat image_with_grid = image.clone();
    
    // Draw grid overlay for feature distribution visualization
    draw_feature_grid(image_with_grid);
    
    // Draw map point IDs if enabled (NO colored circles - just text IDs)
    if (m_show_map_point_indices && !features.empty() && !map_points.empty()) {
        // Simple approach: assume features and map_points are aligned by index
        size_t min_size = std::min(features.size(), map_points.size());
        
        for (size_t i = 0; i < min_size; ++i) {
            if (!features[i] || !features[i]->is_valid()) continue;
            if (!map_points[i] || map_points[i]->is_bad()) continue;
            
            cv::Point2f pixel_coord = features[i]->get_pixel_coord();
            
            // Draw map point ID (white text for clean tracking view) - NO CIRCLES
            std::string id_text = std::to_string(map_points[i]->get_id());
            cv::Point2f text_pos(pixel_coord.x + 5, pixel_coord.y - 5);  // Offset text slightly
            
            // Use white color for text (BGR: 255,255,255) for clean tracking view
            cv::putText(image_with_grid, id_text, text_pos, 
                       cv::FONT_HERSHEY_SIMPLEX, 0.4, cv::Scalar(255, 255, 255), 1);  
        }
    }
    
    // Convert to texture
    m_tracking_image = create_texture_from_cv_mat(image_with_grid);
    m_has_tracking_image = true;
    
    // Release temporary cv::Mat to free memory immediately
    image_with_grid.release();
}

void PangolinViewer::update_tracking_with_frame(std::shared_ptr<Frame> frame) {
    if (!frame) return;

    m_current_frame = frame;
    
    // ⭐ For RGBD: use RGB image if available, otherwise use grayscale
    const cv::Mat& raw_image = frame->is_rgbd() && !frame->get_rgb_image().empty() 
                               ? frame->get_rgb_image() 
                               : frame->get_image();
    
    // Get features and map points from frame
    const auto& features = frame->get_features();
    const auto& map_points = frame->get_map_points();
    
    // Convert raw image to BGR if needed
    cv::Mat display_image;
    if (raw_image.channels() == 1) {
        cv::cvtColor(raw_image, display_image, cv::COLOR_GRAY2BGR);
    } else {
        display_image = raw_image.clone();
    }
    
    // Draw features directly on the image
    int drawn_count = 0;
    for (size_t i = 0; i < features.size(); ++i) {
        const auto& feature = features[i];
        if (feature && feature->is_valid()) {
            const cv::Point2f& pt = feature->get_pixel_coord();
            
            // Determine color based on map point validity
            cv::Scalar color;
            if (i < map_points.size() && map_points[i] && !map_points[i]->is_bad()) {
                color = cv::Scalar(0, 255, 0);  // Green (BGR) for valid map points
            } else {
                color = cv::Scalar(0, 0, 255);  // Red (BGR) for no map point
            }
            
            // Draw circle
            cv::circle(display_image, pt, 3, color, -1);
            drawn_count++;
        }
    }
    
    
    // Update texture with drawn image
    m_tracking_image = create_texture_from_cv_mat(display_image);
    m_has_tracking_image = true;
    
    // Release temporary cv::Mat to free memory immediately
    display_image.release();
    
    // Clear OpenGL feature storage (not used anymore)
    m_current_features.clear();
    m_current_feature_colors.clear();
}

void PangolinViewer::update_tracking_image_direct(const cv::Mat& raw_image,
                                                 const std::vector<std::shared_ptr<Feature>>& features,
                                                 const std::vector<std::shared_ptr<MapPoint>>& map_points) {
    // Convert raw image to BGR if needed (no feature drawing here)
    cv::Mat display_image;
    if (raw_image.channels() == 1) {
        cv::cvtColor(raw_image, display_image, cv::COLOR_GRAY2BGR);
    } else {
        display_image = raw_image.clone();
    }
    
    // Just update the texture with raw image - features will be rendered by OpenGL
    m_tracking_image = create_texture_from_cv_mat(display_image);
    m_has_tracking_image = true;
    
    // Release temporary cv::Mat to free memory immediately
    display_image.release();
    
    // Debug: Print feature count
    spdlog::info("Updated tracking image: {}x{}, {} features", 
                 raw_image.cols, raw_image.rows, features.size());
    
    // Store features data for OpenGL rendering - just draw map point features as circles
    m_current_features.clear();
    m_current_feature_colors.clear();
    
    for (size_t i = 0; i < features.size(); ++i) {
        const auto& feature = features[i];
        if (feature && feature->is_valid()) {
            const cv::Point2f& pt = feature->get_pixel_coord();
            
            // Convert pixel coordinates to normalized coordinates [-1, 1]
            float norm_x = (2.0f * pt.x / raw_image.cols) - 1.0f;
            float norm_y = 1.0f - (2.0f * pt.y / raw_image.rows);  // Flip Y
            
            m_current_features.push_back(Eigen::Vector2f(norm_x, norm_y));
            
            // Determine color based on map point validity
            if (i < map_points.size() && map_points[i] && !map_points[i]->is_bad()) {
                m_current_feature_colors.push_back(Eigen::Vector3f(0.0f, 1.0f, 0.0f)); // Green for valid map points
            } else {
                m_current_feature_colors.push_back(Eigen::Vector3f(1.0f, 0.0f, 0.0f)); // Red for no map point
            }
        }
    }
    
    spdlog::info("Prepared {} features for OpenGL rendering", m_current_features.size());
}

void PangolinViewer::update_tracking_image_with_uncertainty_debug(const cv::Mat& image, 
                                                                      const std::vector<std::shared_ptr<Feature>>& features,
                                                                      const std::vector<std::shared_ptr<MapPoint>>& map_points,
                                                                      std::shared_ptr<Frame> current_frame) {
    if (image.empty()) return;
    
    // Create a copy of the image to draw on
    cv::Mat image_with_debug = image.clone();
    
    // Draw grid overlay for feature distribution visualization
    draw_feature_grid(image_with_debug);
    
    // Draw uncertainty ellipses for all map points (debugging)
    debug_uncertainty_projection(image_with_debug, current_frame);
    
    // Draw map point indices if enabled
    if (m_show_map_point_indices && !features.empty() && !map_points.empty()) {
        // Simple approach: assume features and map_points are aligned by index
        size_t min_size = std::min(features.size(), map_points.size());
        
        for (size_t i = 0; i < min_size; ++i) {
            if (!features[i] || !features[i]->is_valid()) continue;
            if (!map_points[i] || map_points[i]->is_bad()) continue;
            
            cv::Point2f pixel_coord = features[i]->get_pixel_coord();
            
            // Draw map point ID (yellow text for better visibility over uncertainty ellipses)
            std::string id_text = std::to_string(map_points[i]->get_id());
            cv::Point2f text_pos(pixel_coord.x + 5, pixel_coord.y - 5);  // Offset text slightly
            
            // Use yellow color for text (BGR: 0,255,255) and font size 0.4
            cv::putText(image_with_debug, id_text, text_pos, 
                       cv::FONT_HERSHEY_SIMPLEX, 0.4, cv::Scalar(0, 255, 255), 1);  
        }
    }
    
    // Convert to texture
    m_tracking_image = create_texture_from_cv_mat(image_with_debug);
    m_has_tracking_image = true;
    
    // Release temporary cv::Mat to free memory immediately
    image_with_debug.release();
}

void PangolinViewer::update_uncertainty_debug_image(const cv::Mat& image) {
    if (image.empty()) {
        spdlog::warn("[PangolinViewer] Received empty uncertainty debug image");
        return;
    }
    
    m_uncertainty_debug_image = create_texture_from_cv_mat(image);
    m_has_uncertainty_debug_image = true;

    // The bounds and aspect ratio are now handled exclusively by setup_panels().
    // This function is only responsible for updating the texture.
    // spdlog::debug("[PangolinViewer] Updated uncertainty debug image texture {}x{}", image.cols, image.rows);
}

void PangolinViewer::process_keyboard_input(bool& auto_play, bool& step_mode, bool& advance_frame) {
    // Handle space key press - toggle between auto and step mode
    if (m_space_pressed) {
        if (auto_play) {
            auto_play = false;
            step_mode = true;
            m_auto_mode_checkbox = false;  // Update UI checkbox
        } else {
            auto_play = true;
            step_mode = false;
            m_auto_mode_checkbox = true;   // Update UI checkbox
        }
        m_space_pressed = false;
    }
    
    // Handle next frame key press - advance one frame in step mode
    if (m_next_pressed) {
        if (step_mode) {
            advance_frame = true;
        }
        m_next_pressed = false;
    }
    
    // Handle Step Forward button press - advance one frame in step mode
    if (m_step_forward_pressed) {
        if (step_mode) {
            advance_frame = true;
        }
        m_step_forward_pressed = false;
    }
}

void PangolinViewer::sync_ui_state(bool& auto_play, bool& step_mode) {
    // Check if UI checkbox state has changed and update mode accordingly
    bool ui_auto_mode = m_auto_mode_checkbox;
    
    if (ui_auto_mode && !auto_play) {
        // UI checkbox enabled but currently in step mode - switch to auto
        auto_play = true;
        step_mode = false;
    } else if (!ui_auto_mode && auto_play) {
        // UI checkbox disabled but currently in auto mode - switch to step
        auto_play = false;
        step_mode = true;
    }
}

void PangolinViewer::update_frame_info(int frame_id, int total_features, int tracked_features, int new_features) {
    // Frame info is no longer stored in UI variables - just ignore the call
    // This function is kept for compatibility but does nothing
}

void PangolinViewer::update_tracking_stats(int frame_id, int total_features, int stereo_matches, int map_points, float success_rate, float position_error) {
    m_frame_id = frame_id;
    m_successful_matches = stereo_matches;  // Show successful stereo matches
}

bool PangolinViewer::is_auto_mode_enabled() const {
    return m_auto_mode_checkbox;
}

bool PangolinViewer::is_follow_frame_enabled() const {
    return m_follow_frame_checkbox;
}

bool PangolinViewer::is_finish_requested() const {
    return m_finish_pressed;
}

void PangolinViewer::draw_feature_grid(cv::Mat& image) {
    const auto& config = Config::getInstance();
    
    const int grid_cols = config.m_grid_cols;  // From config
    const int grid_rows = config.m_grid_rows;  // From config
    const cv::Scalar grid_color(100, 100, 100);  // Gray color for grid lines
    const int thickness = 1;
    
    // Draw vertical grid lines (20 divisions) - use more precise calculation
    for (int i = 1; i < grid_cols; i++) {
        // Use integer arithmetic to avoid floating-point precision issues
        int x = (i * image.cols) / grid_cols;
        cv::line(image, cv::Point(x, 0), cv::Point(x, image.rows), grid_color, thickness);
    }
    
    // Draw horizontal grid lines (10 divisions) - use more precise calculation
    for (int i = 1; i < grid_rows; i++) {
        // Use integer arithmetic to avoid floating-point precision issues
        int y = (i * image.rows) / grid_rows;
        cv::line(image, cv::Point(0, y), cv::Point(image.cols, y), grid_color, thickness);
    }
}

// New sliding window keyframe management functions
void PangolinViewer::add_frame(std::shared_ptr<Frame> frame) {
    std::lock_guard<std::mutex> lock(m_data_mutex);
    m_all_frames.push_back(frame);
    
    // Automatically extract current frame map points (red dots)
    m_current_map_points.clear();
    if (frame) {
        const auto& frame_map_points = frame->get_map_points();
        for (const auto& mp : frame_map_points) {
            if (mp && !mp->is_bad()) {
                Eigen::Vector3f position = mp->get_position();
                m_current_map_points.push_back(position);
            }
        }
    }
}

std::vector<std::shared_ptr<Frame>> PangolinViewer::get_all_frames() const {
    std::lock_guard<std::mutex> lock(m_data_mutex);
    return m_all_frames;
}

std::shared_ptr<Frame> PangolinViewer::get_first_frame() const {
    std::lock_guard<std::mutex> lock(m_data_mutex);
    if (!m_all_frames.empty()) {
        return m_all_frames.front();
    }
    return nullptr;
}

void PangolinViewer::update_keyframe_window(const std::vector<std::shared_ptr<Frame>>& keyframes) {
    std::lock_guard<std::mutex> lock(m_data_mutex);
    m_keyframe_window = keyframes;
}

void PangolinViewer::set_last_keyframe(std::shared_ptr<Frame> last_keyframe) {
    std::lock_guard<std::mutex> lock(m_data_mutex);
    m_last_keyframe = last_keyframe;
}

void PangolinViewer::update_relative_pose_from_last_keyframe(const Eigen::Matrix4f& relative_pose) {
    std::lock_guard<std::mutex> lock(m_data_mutex);
    m_relative_pose_from_last_keyframe = relative_pose;
}

void PangolinViewer::update_all_map_points(const std::vector<std::shared_ptr<MapPoint>>& map_points) {
    std::lock_guard<std::mutex> lock(m_data_mutex);
    m_all_map_points_storage = map_points;
}

void PangolinViewer::update_window_map_points(const std::vector<std::shared_ptr<MapPoint>>& window_map_points) {
    std::lock_guard<std::mutex> lock(m_data_mutex);
    m_window_map_points_storage = window_map_points;
}

void PangolinViewer::set_gravity_transformation(const Eigen::Matrix4f& Tgw) {
    std::lock_guard<std::mutex> lock(m_data_mutex);
    m_Tgw = Tgw;
    m_has_gravity_transformation = true;
    spdlog::info("[PangolinViewer] Set gravity transformation matrix Tgw");
    spdlog::info("  [{:.6f}, {:.6f}, {:.6f}, {:.6f}]", Tgw(0,0), Tgw(0,1), Tgw(0,2), Tgw(0,3));
    spdlog::info("  [{:.6f}, {:.6f}, {:.6f}, {:.6f}]", Tgw(1,0), Tgw(1,1), Tgw(1,2), Tgw(1,3));
    spdlog::info("  [{:.6f}, {:.6f}, {:.6f}, {:.6f}]", Tgw(2,0), Tgw(2,1), Tgw(2,2), Tgw(2,3));
    spdlog::info("  [{:.6f}, {:.6f}, {:.6f}, {:.6f}]", Tgw(3,0), Tgw(3,1), Tgw(3,2), Tgw(3,3));
}

// Uncertainty visualization implementation
void PangolinViewer::render_map_point_uncertainties(const std::vector<std::shared_ptr<MapPoint>>& map_points) {
    if (map_points.empty()) return;
    
    // First pass: find min and max uncertainty sizes for normalization
    float min_size = std::numeric_limits<float>::max();
    float max_size = std::numeric_limits<float>::min();
    
    for (const auto& mp : map_points) {
        if (!mp || !mp->has_uncertainty()) continue;
        
        Eigen::Matrix3f combined_uncertainty = mp->get_world_uncertainty();
        
        // Calculate uncertainty size as the trace (sum of eigenvalues) of covariance matrix
        Eigen::SelfAdjointEigenSolver<Eigen::Matrix3f> solver(combined_uncertainty);
        if (solver.info() == Eigen::Success) {
            Eigen::Vector3f eigenvalues = solver.eigenvalues();
            float uncertainty_size = eigenvalues.sum(); // Total uncertainty volume
            min_size = std::min(min_size, uncertainty_size);
            max_size = std::max(max_size, uncertainty_size);
        }
    }
    
    // Render each map point's COMBINED uncertainty with size-based coloring
    for (const auto& mp : map_points) {
        if (!mp || !mp->has_uncertainty()) continue;
        
        Eigen::Vector3f position = mp->get_position();
        Eigen::Matrix3f combined_uncertainty = mp->get_world_uncertainty();
        
        // Calculate uncertainty size and normalize it
        Eigen::SelfAdjointEigenSolver<Eigen::Matrix3f> solver(combined_uncertainty);
        Eigen::Vector3f color(0.3f, 0.9f, 0.7f); // Mint color for all 3D ellipsoids
        
        if (solver.info() == Eigen::Success) {
            // Keep mint color regardless of uncertainty size
            // Mint color: light green-blue combination
            color.x() = 0.3f; // Red component (low for mint)
            color.y() = 0.9f; // Green component (high for mint)
            color.z() = 0.7f; // Blue component (medium-high for mint)
        }
        
        float exaggeration_factor = 1.96f;
        
        // Draw combined uncertainty ellipsoid with size-based color
        draw_uncertainty_ellipsoid(position, combined_uncertainty, color, 0.3f, m_uncertainty_scale * exaggeration_factor);
    }
}

void PangolinViewer::draw_uncertainty_ellipsoid(const Eigen::Vector3f& position, 
                                               const Eigen::Matrix3f& covariance,
                                               const Eigen::Vector3f& color,
                                               float alpha,
                                               float scale_factor) {
    // Check if covariance is valid
    if (covariance.determinant() <= 0) {
        return;  // Skip invalid covariance matrices
    }



    Eigen::SelfAdjointEigenSolver<Eigen::Matrix3f> orig_solver(covariance);
    if (orig_solver.info() == Eigen::Success) {
        Eigen::Vector3f orig_eigenvalues = orig_solver.eigenvalues();
    }

    Eigen::Vector3f eigenvalues = orig_solver.eigenvalues();
    Eigen::Matrix3f eigenvectors = orig_solver.eigenvectors();
  
    // Save current OpenGL state
    glPushMatrix();
    
    // Translate to position
    glTranslatef(position.x(), position.y(), position.z());
    
    // Apply rotation (eigenvectors as rotation matrix)
    Eigen::Matrix4f transform = Eigen::Matrix4f::Identity();
    transform.block<3,3>(0,0) = eigenvectors;
    
    // Convert to column-major for OpenGL
    float gl_matrix[16];
    for (int i = 0; i < 4; ++i) {
        for (int j = 0; j < 4; ++j) {
            gl_matrix[i*4 + j] = transform(j, i);
        }
    }
    glMultMatrixf(gl_matrix);
    
    // Set color with transparency (더 투명하게)
    glColor4f(color.x(), color.y(), color.z(), alpha * 0.8f);
    
    // Enable blending for transparency
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
    
    // Draw wireframe ellipsoid using lines (no need for glScalef since we pass radii directly)
    draw_wireframe_ellipsoid(eigenvalues.x(), eigenvalues.y(), eigenvalues.z(), 16, 12);
    
    // Restore OpenGL state
    glDisable(GL_BLEND);
    glPopMatrix();
}

void PangolinViewer::draw_wireframe_ellipsoid(float a, float b, float c, int slices, int stacks) {
    // Generate ellipsoid vertices with different radii for each axis
    std::vector<Eigen::Vector3f> vertices;
    
    // Generate vertices for ellipsoid
    for (int i = 0; i <= stacks; ++i) {
        float phi = M_PI * float(i) / float(stacks);  // Vertical angle (0 to π)
        float sin_phi = sin(phi);
        float cos_phi = cos(phi);
        
        for (int j = 0; j <= slices; ++j) {
            float theta = 2.0f * M_PI * float(j) / float(slices);  // Horizontal angle (0 to 2π)
            float sin_theta = sin(theta);
            float cos_theta = cos(theta);
            
            // Parametric ellipsoid equations:
            // x = a * sin(phi) * cos(theta)
            // y = b * sin(phi) * sin(theta)  
            // z = c * cos(phi)
            float x = a * sin_phi * cos_theta;
            float y = b * sin_phi * sin_theta;
            float z = c * cos_phi;
            
            vertices.push_back(Eigen::Vector3f(x, y, z));
        }
    }
    
    // Draw horizontal circles (latitude lines)
    glBegin(GL_LINES);
    for (int i = 0; i < stacks; ++i) {
        for (int j = 0; j < slices; ++j) {
            int current = i * (slices + 1) + j;
            int next = i * (slices + 1) + ((j + 1) % (slices + 1));
            
            // Skip the last slice to avoid duplication
            if (j < slices) {
                glVertex3f(vertices[current].x(), vertices[current].y(), vertices[current].z());
                glVertex3f(vertices[next].x(), vertices[next].y(), vertices[next].z());
            }
        }
    }
    
    // Draw vertical circles (longitude lines)
    for (int j = 0; j < slices; j += 2) {  // Draw every 2nd longitude line to avoid clutter
        for (int i = 0; i < stacks; ++i) {
            int current = i * (slices + 1) + j;
            int below = (i + 1) * (slices + 1) + j;
            
            glVertex3f(vertices[current].x(), vertices[current].y(), vertices[current].z());
            glVertex3f(vertices[below].x(), vertices[below].y(), vertices[below].z());
        }
    }
    glEnd();
}

void PangolinViewer::draw_wireframe_sphere(float radius, int slices, int stacks) {
    // This is just a special case of ellipsoid where a = b = c = radius
    draw_wireframe_ellipsoid(radius, radius, radius, slices, stacks);
}
Eigen::Vector3f PangolinViewer::uncertainty_to_color(float uncertainty_magnitude, 
                                                   float min_uncertainty, 
                                                   float max_uncertainty) {
    if (max_uncertainty <= min_uncertainty) {
        return Eigen::Vector3f(0.0f, 0.0f, 1.0f);  // Blue for uniform uncertainty
    }
    
    // Normalize uncertainty to [0, 1]
    float normalized = (uncertainty_magnitude - min_uncertainty) / (max_uncertainty - min_uncertainty);
    normalized = std::max(0.0f, std::min(1.0f, normalized));
    
    // HSV color mapping: Blue (low) -> Green -> Yellow -> Red (high)
    float hue = (1.0f - normalized) * 240.0f;  // 240° = blue, 0° = red
    float saturation = 1.0f;
    float value = 1.0f;
    
    // Convert HSV to RGB
    float c = value * saturation;
    float x = c * (1.0f - std::abs(std::fmod(hue / 60.0f, 2.0f) - 1.0f));
    float m = value - c;
    
    float r, g, b;
    if (hue >= 0 && hue < 60) {
        r = c; g = x; b = 0;
    } else if (hue >= 60 && hue < 120) {
        r = x; g = c; b = 0;
    } else if (hue >= 120 && hue < 180) {
        r = 0; g = c; b = x;
    } else if (hue >= 180 && hue < 240) {
        r = 0; g = x; b = c;
    } else if (hue >= 240 && hue < 300) {
        r = x; g = 0; b = c;
    } else {
        r = c; g = 0; b = x;
    }
    
    return Eigen::Vector3f(r + m, g + m, b + m);
}

// Uncertainty debugging implementation
void PangolinViewer::debug_uncertainty_projection(cv::Mat& image, 
                                                  std::shared_ptr<Frame> current_frame) {
    if (!current_frame) return;
    
    // Get features from the current frame
    const auto& features = current_frame->get_features();
    const auto& map_points = current_frame->get_map_points();
    
    // Ensure features and map_points arrays are aligned
    size_t min_size = std::min(features.size(), map_points.size());
    
    // Process each feature-mappoint pair
    for (size_t i = 0; i < min_size; ++i) {
        const auto& feature = features[i];
        const auto& mp = map_points[i];
        
        if (!feature || !feature->is_valid()) continue;
        if (!mp || mp->is_bad() || !mp->has_uncertainty()) continue;
        
        // Use feature's pixel coordinates directly (no reprojection needed)
        cv::Point2f pixel_center = feature->get_pixel_coord();
        
        // Check if pixel is within image bounds
        if (pixel_center.x < 0 || pixel_center.x >= image.cols || 
            pixel_center.y < 0 || pixel_center.y >= image.rows) continue;
        
        // Get world uncertainty and transform to pixel
        Eigen::Matrix3f world_uncertainty = mp->get_world_uncertainty();
        Eigen::Matrix2f pixel_uncertainty = mp->transform_uncertainty_world_to_pixel(world_uncertainty, current_frame);
        
        // Draw uncertainty ellipse (color is determined inside based on axis product)
        draw_uncertainty_ellipse_2d(image, pixel_center, pixel_uncertainty);
    }
}

void PangolinViewer::draw_uncertainty_ellipse_2d(cv::Mat& image,
                                                const cv::Point2f& center,
                                                const Eigen::Matrix2f& pixel_uncertainty) {
    // Check if uncertainty matrix is valid
    if (pixel_uncertainty.determinant() <= 0) return;
    
    // Eigenvalue decomposition for ellipse shape
    Eigen::SelfAdjointEigenSolver<Eigen::Matrix2f> solver(pixel_uncertainty);
    if (solver.info() != Eigen::Success) return;
    
    Eigen::Vector2f eigenvalues = solver.eigenvalues();
    Eigen::Matrix2f eigenvectors = solver.eigenvectors();
    
    // Ensure positive eigenvalues
    for (int i = 0; i < 2; ++i) {
        if (eigenvalues(i) <= 0) eigenvalues(i) = 0.1f;
    }
    
    // Convert to ellipse parameters (2-sigma ellipse)
    float scale_factor = 2.0f; // 2-sigma
    float major_axis = scale_factor * std::sqrt(eigenvalues(1)); // Larger eigenvalue
    float minor_axis = scale_factor * std::sqrt(eigenvalues(0)); // Smaller eigenvalue
    
    // Ensure minimum axis lengths for visibility
    float min_axis_length = 10.0f;
    float max_axis_length = 100.0f;
    major_axis = std::max(major_axis, min_axis_length);
    minor_axis = std::max(minor_axis, min_axis_length);
    major_axis = std::min(major_axis, max_axis_length);
    minor_axis = std::min(minor_axis, max_axis_length);
    
    // Calculate axis product for color mapping
    float axis_product = major_axis * minor_axis;
    
    // Color mapping based on axis product:
    // <= 100 -> Red
    // >= 1600 -> Blue (40x40 max size)
    // Between -> Gradient
    float min_product = min_axis_length*min_axis_length;
    float max_product = max_axis_length*max_axis_length;
    
    float normalized_size;
    if (axis_product <= min_product) {
        normalized_size = 0.0f;  // Red
    } else if (axis_product >= max_product) {
        normalized_size = 1.0f;  // Blue
    } else {
        normalized_size = (axis_product - min_product) / (max_product - min_product);
    }
    
    // BGR color interpolation: Red (0,0,255) -> Blue (255,0,0)
    cv::Scalar ellipse_color;
    ellipse_color[0] = normalized_size * 255;           // Blue component
    ellipse_color[1] = 0;                              // Green component  
    ellipse_color[2] = (1.0f - normalized_size) * 255; // Red component
    
    // Calculate rotation angle (in degrees)
    Eigen::Vector2f major_eigenvector = eigenvectors.col(1); // Eigenvector of larger eigenvalue
    float angle = std::atan2(major_eigenvector.y(), major_eigenvector.x()) * 180.0f / M_PI;
    
    // Define ellipse parameters for OpenCV
    cv::Size2f axes(major_axis, minor_axis);
    
    // Draw filled ellipse with transparency
    cv::Mat overlay = image.clone();
    cv::ellipse(overlay, center, axes, angle, 0, 360, ellipse_color, -1); // Filled ellipse
    cv::addWeighted(image, 0.9, overlay, 0.1, 0, image); // Blend with transparency
    
    // Draw ellipse border
    cv::ellipse(image, center, axes, angle, 0, 360, ellipse_color, 2); // Border
    
    // // Draw center point
    // cv::circle(image, center, 3, ellipse_color, -1);
}

cv::Mat PangolinViewer::create_uncertainty_debug_image(const std::vector<std::shared_ptr<Feature>>& features,
                                                       const std::vector<std::shared_ptr<MapPoint>>& map_points,
                                                       std::shared_ptr<Frame> current_frame) {
    if (!current_frame) {
        return cv::Mat::zeros(480, 752, CV_8UC3);  // Default EuRoC size
    }
    
    // Get the actual camera image from current frame
    cv::Mat debug_image;
    cv::Mat frame_image = current_frame->get_left_image(); // Get actual camera image
    
    if (!frame_image.empty()) {
        // Use the real camera image as background
        if (frame_image.channels() == 1) {
            cv::cvtColor(frame_image, debug_image, cv::COLOR_GRAY2BGR);
        } else {
            debug_image = frame_image.clone();
        }
    } else {
        // Fallback to black background if no image available
        debug_image = cv::Mat::zeros(480, 752, CV_8UC3);
    }
    
    // Draw uncertainty projections on top of the real image
    debug_uncertainty_projection(debug_image, current_frame);


    return debug_image;
}

// Multi-view observation visualization implementation
void PangolinViewer::render_observation_point_clouds(const std::vector<std::shared_ptr<MapPoint>>& map_points) {
    if (map_points.empty()) return;
    
    // Blue color for observation points and connections
    Eigen::Vector3f observation_color(0.3f, 0.6f, 1.0f);  // Light blue
    Eigen::Vector3f connection_color(0.0f, 0.5f, 1.0f);   // Deep blue
    
    for (const auto& mp : map_points) {
        if (!mp || mp->is_bad() || mp->get_observation_count() < 2) {
            continue;  // Skip points with insufficient observations
        }
        
        // // Get the optimized map point position (center)
        // Eigen::Vector3f center_position = mp->get_position();
        
        // // Use cached observation positions (updated after optimization)
        // if (!mp->has_valid_observation_positions()) {
        //     continue;  // Skip if no valid cached positions
        // }
        
        const std::vector<Eigen::Vector3f>& observation_positions = mp->get_observation_positions();
        
        // if (observation_positions.size() < 2) {
        //     continue;  // Need at least 2 observations for meaningful visualization
        // }
        
        // // Draw observation points as small spheres
        // glPointSize(6.0f);
        // glColor3f(observation_color.x(), observation_color.y(), observation_color.z());
        
        // glBegin(GL_POINTS);
        // for (const auto& obs_pos : observation_positions) {
        //     glVertex3f(obs_pos.x(), obs_pos.y(), obs_pos.z());
        // }
        // glEnd();
        
        // // Draw center point (optimized position) in brighter color
        // glPointSize(8.0f);
        // glColor3f(1.0f, 1.0f, 0.0f);  // Yellow for center
        
        // glBegin(GL_POINTS);
        // glVertex3f(center_position.x(), center_position.y(), center_position.z());
        // glEnd();
        
        // // Draw connections from center to all observation points
        // draw_observation_connections(center_position, observation_positions, connection_color);
        

        
        // Fit ellipsoid to observation distribution if we have enough points
        if (observation_positions.size() >= 3 && mp->has_world_uncertainty() && !mp->is_bad()) {  
            // Use current MapPoint position as ellipsoid center (optimized position)
            Eigen::Vector3f mappoint_position = mp->get_position();
            
            // Use world uncertainty matrix (updated from observation covariance)
            Eigen::Matrix3f world_uncertainty = mp->get_world_uncertainty();
            
            // Draw fitted ellipsoid at current MapPoint position
            Eigen::Vector3f ellipsoid_color(0.0f, 1.0f, 1.0f);  // Cyan for observation distribution
            draw_uncertainty_ellipsoid(mappoint_position, world_uncertainty, ellipsoid_color, 0.2f, 1.0f);  // 1-sigma for conservative visualization
        }
    }
    
    glPointSize(1.0f);  // Reset point size
}

void PangolinViewer::draw_observation_connections(const Eigen::Vector3f& center_pos, 
                                                const std::vector<Eigen::Vector3f>& observation_positions,
                                                const Eigen::Vector3f& line_color) {
    if (observation_positions.empty()) return;
    
    glLineWidth(1.5f);
    glColor3f(line_color.x(), line_color.y(), line_color.z());
    
    glBegin(GL_LINES);
    
    // Draw lines from center to each observation point
    for (const auto& obs_pos : observation_positions) {
        glVertex3f(center_pos.x(), center_pos.y(), center_pos.z());
        glVertex3f(obs_pos.x(), obs_pos.y(), obs_pos.z());
    }
    
    // Optional: Draw connections between observation points (creates a web-like structure)
    // for (size_t i = 0; i < observation_positions.size(); ++i) {
    //     for (size_t j = i + 1; j < observation_positions.size(); ++j) {
    //         glVertex3f(observation_positions[i].x(), observation_positions[i].y(), observation_positions[i].z());
    //         glVertex3f(observation_positions[j].x(), observation_positions[j].y(), observation_positions[j].z());
    //     }
    // }
    
    glEnd();
    glLineWidth(1.0f);  // Reset line width
}

Eigen::Matrix3f PangolinViewer::compute_observation_covariance(const std::vector<Eigen::Vector3f>& observation_positions,
                                                              const Eigen::Vector3f& mean_position) {
    if (observation_positions.size() < 3) {
        // Return identity matrix for insufficient data
        return Eigen::Matrix3f::Identity() * 0.01f;  // Small default covariance
    }
    
    // Use provided mean_position (already calculated)
    Eigen::Vector3f mean_pos = mean_position;
    
    // Compute covariance matrix
    Eigen::Matrix3f covariance = Eigen::Matrix3f::Zero();
    for (const auto& pos : observation_positions) {
        Eigen::Vector3f diff = pos - mean_pos;
        covariance += diff * diff.transpose();
    }
    
    // Normalize by (n-1) for sample covariance
    covariance /= static_cast<float>(observation_positions.size() - 1);
    
    // Add small regularization to ensure positive definiteness
    covariance += Eigen::Matrix3f::Identity() * 1e-6f;
    
    return covariance;
}

// ⭐ RGBD Dense Point Cloud Visualization
void PangolinViewer::update_dense_point_cloud(std::shared_ptr<Frame> frame) {
    if (!frame || !frame->is_rgbd()) {
        return;
    }
    
    const Config& config = Config::getInstance();
    
    // Check if dense cloud is enabled
    if (!config.m_rgbd_enable_dense_cloud) {
        std::lock_guard<std::mutex> lock(m_data_mutex);
        // Just clear without shrinking - keep capacity for reuse
        m_dense_point_cloud.clear();
        m_dense_point_colors_rgb.clear();
        m_dense_point_colors_depth.clear();
        m_dense_point_depths.clear();
        return;
    }
    
    // Get parameters from config
    int stride = config.m_rgbd_dense_cloud_stride;
    float min_depth = config.m_min_depth;
    float max_depth = config.m_max_depth;
    float vis_min_depth = config.m_rgbd_vis_min_depth;
    float vis_max_depth = config.m_rgbd_vis_max_depth;
    
    // Update storage (thread-safe)
    std::lock_guard<std::mutex> lock(m_data_mutex);
    
    // Clear previous data but keep capacity to avoid reallocation
    m_dense_point_cloud.clear();
    m_dense_point_colors_rgb.clear();
    m_dense_point_colors_depth.clear();
    m_dense_point_depths.clear();
    
    // Generate colored point cloud with RGB colors (color_mode=1)
    // Do this INSIDE the lock to minimize the lifetime of colored_points
    auto colored_points = frame->generate_colored_point_cloud(stride, min_depth, max_depth, 1);
    
    // Reserve space if needed (only grows, never shrinks)
    if (m_dense_point_cloud.capacity() < colored_points.size()) {
        m_dense_point_cloud.reserve(colored_points.size());
        m_dense_point_colors_rgb.reserve(colored_points.size());
        m_dense_point_colors_depth.reserve(colored_points.size());
        m_dense_point_depths.reserve(colored_points.size());
    }
    
    // Move data efficiently and compute depth heatmap colors
    for (auto& cp : colored_points) {
        m_dense_point_cloud.push_back(std::move(cp.position));
        m_dense_point_colors_rgb.push_back(std::move(cp.color));
        m_dense_point_depths.push_back(cp.depth);
        
        // Compute depth heatmap color
        float depth = cp.depth;
        Eigen::Vector3f depth_color;
        
        if (depth <= 0.0f) {
            depth_color = Eigen::Vector3f(0.0f, 0.0f, 0.0f);  // Black for invalid
        } else {
            float clamped_depth = std::max(vis_min_depth, std::min(vis_max_depth, depth));
            float normalized_depth = (clamped_depth - vis_min_depth) / (vis_max_depth - vis_min_depth);
            normalized_depth = std::max(0.0f, std::min(1.0f, normalized_depth));
            
            // Heatmap: Red (near) -> Yellow -> Green -> Cyan -> Blue (far)
            float r, g, b;
            if (normalized_depth < 0.25f) {
                float t = normalized_depth / 0.25f;
                r = 1.0f; g = t; b = 0.0f;
            } else if (normalized_depth < 0.5f) {
                float t = (normalized_depth - 0.25f) / 0.25f;
                r = 1.0f - t; g = 1.0f; b = 0.0f;
            } else if (normalized_depth < 0.75f) {
                float t = (normalized_depth - 0.5f) / 0.25f;
                r = 0.0f; g = 1.0f; b = t;
            } else {
                float t = (normalized_depth - 0.75f) / 0.25f;
                r = 0.0f; g = 1.0f - t; b = 1.0f;
            }
            depth_color = Eigen::Vector3f(r, g, b);
        }
        
        m_dense_point_colors_depth.push_back(depth_color);
    }
    
    // Explicitly release colored_points memory immediately
    colored_points.clear();
    colored_points.shrink_to_fit();
}

void PangolinViewer::draw_dense_point_cloud() {
    std::lock_guard<std::mutex> lock(m_data_mutex);
    
    if (m_dense_point_cloud.empty()) {
        return;
    }
    
    // Use small point size for dense cloud
    glPointSize(1.0f);
    
    glBegin(GL_POINTS);
    
    // Choose color based on current mode
    if (m_dense_color_mode_rgb) {
        // RGB mode - use RGB colors from image
        if (!m_dense_point_colors_rgb.empty()) {
            for (size_t i = 0; i < m_dense_point_cloud.size(); ++i) {
                const auto& pt = m_dense_point_cloud[i];
                const auto& color = m_dense_point_colors_rgb[i];
                glColor3f(color.x(), color.y(), color.z());
                glVertex3f(pt.x(), pt.y(), pt.z());
            }
        } else {
            // Fallback to cyan if no RGB data
            glColor3f(0.0f, 1.0f, 1.0f);
            for (const auto& pt : m_dense_point_cloud) {
                glVertex3f(pt.x(), pt.y(), pt.z());
            }
        }
    } else {
        // Depth heatmap mode - use depth colors
        if (!m_dense_point_colors_depth.empty()) {
            for (size_t i = 0; i < m_dense_point_cloud.size(); ++i) {
                const auto& pt = m_dense_point_cloud[i];
                const auto& color = m_dense_point_colors_depth[i];
                glColor3f(color.x(), color.y(), color.z());
                glVertex3f(pt.x(), pt.y(), pt.z());
            }
        } else {
            // Fallback to magenta if no depth data
            glColor3f(1.0f, 0.0f, 1.0f);
            for (const auto& pt : m_dense_point_cloud) {
                glVertex3f(pt.x(), pt.y(), pt.z());
            }
        }
    }
    
    glEnd();
    
    // Reset point size
    glPointSize(1.0f);
}

// ⭐ RGBD Depth Image Visualization
void PangolinViewer::update_depth_image(std::shared_ptr<Frame> frame) {
    if (!frame || !frame->is_rgbd()) {
        return;
    }
    
    const cv::Mat& depth_map = frame->get_depth_map();
    if (depth_map.empty()) {
        return;
    }
    
    const Config& config = Config::getInstance();
    float min_depth = config.m_rgbd_vis_min_depth;  // Use vis_min_depth from config
    float max_depth = config.m_rgbd_vis_max_depth;  // Use vis_max_depth from config
    
    // Create depth heatmap
    cv::Mat depth_heatmap = create_depth_heatmap(depth_map, min_depth, max_depth);
    
    // Update texture (thread-safe)
    std::lock_guard<std::mutex> lock(m_data_mutex);
    m_depth_image = create_texture_from_cv_mat(depth_heatmap);
    m_has_depth_image = true;
    
    // Release temporary Mat to free memory immediately
    depth_heatmap.release();
}

cv::Mat PangolinViewer::create_depth_heatmap(const cv::Mat& depth_map, float min_depth, float max_depth) {
    if (depth_map.empty()) {
        return cv::Mat();
    }
    
    // Create RGB heatmap image
    cv::Mat heatmap(depth_map.rows, depth_map.cols, CV_8UC3);
    
    for (int v = 0; v < depth_map.rows; ++v) {
        for (int u = 0; u < depth_map.cols; ++u) {
            float depth = depth_map.at<float>(v, u);
            
            cv::Vec3b color;
            
            // ⭐ Depth = 0 → Black (no depth data)
            if (depth <= 0.0f) {
                color = cv::Vec3b(0, 0, 0);  // Black for invalid/missing depth
                heatmap.at<cv::Vec3b>(v, u) = color;
                continue;
            }
            
            // Clamp depth to [min_depth, max_depth] range
            // Below min_depth → Red, Above max_depth → Blue
            float clamped_depth = std::max(min_depth, std::min(max_depth, depth));
            
            // Normalize depth to [0, 1]
            float normalized_depth = (clamped_depth - min_depth) / (max_depth - min_depth);
            normalized_depth = std::max(0.0f, std::min(1.0f, normalized_depth));
            
            // Heatmap: Red (near/below min) -> Yellow -> Green -> Cyan -> Blue (far/above max)
            float r, g, b;
                if (normalized_depth < 0.25f) {
                    // Red to Yellow
                    float t = normalized_depth / 0.25f;
                    r = 1.0f;
                    g = t;
                    b = 0.0f;
                } else if (normalized_depth < 0.5f) {
                    // Yellow to Green
                    float t = (normalized_depth - 0.25f) / 0.25f;
                    r = 1.0f - t;
                    g = 1.0f;
                    b = 0.0f;
                } else if (normalized_depth < 0.75f) {
                    // Green to Cyan
                    float t = (normalized_depth - 0.5f) / 0.25f;
                    r = 0.0f;
                    g = 1.0f;
                    b = t;
                } else {
                    // Cyan to Blue
                    float t = (normalized_depth - 0.75f) / 0.25f;
                    r = 0.0f;
                    g = 1.0f - t;
                    b = 1.0f;
                }
                
                // Convert to BGR for OpenCV (note: reversed order!)
                color = cv::Vec3b(
                    static_cast<uchar>(b * 255),
                    static_cast<uchar>(g * 255),
                    static_cast<uchar>(r * 255)
                );
            
            heatmap.at<cv::Vec3b>(v, u) = color;
        }
    }
    
    return heatmap;
}

}