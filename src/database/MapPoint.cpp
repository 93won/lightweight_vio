/**
 * @file      MapPoint.cpp
 * @brief     Implements the MapPoint class.
 * @author    Seungwon Choi (csw3575@snu.ac.kr)
 * @date      2025-08-18
 * @copyright Copyright (c) 2025 Seungwon Choi. All rights reserved.
 *
 * @par License
 * This project is released under the MIT License.
 */

#include "database/MapPoint.h"
#include "database/Frame.h"
#include "database/Feature.h"
#include "util/Config.h"
#include <algorithm>
#include <iostream>
#include <cstdlib>
#include <spdlog/spdlog.h>

namespace lightweight_vio {

int MapPoint::s_next_id = 0;

MapPoint::MapPoint() 
    : m_id(s_next_id++)
    , m_position(0.0f, 0.0f, 0.0f)
    , m_is_bad(false)
    , m_is_multi_view_triangulated(false)
    , m_is_marginalized(false)
    , m_world_uncertainty(Eigen::Matrix3f::Identity() * Config::getInstance().m_uncertainty_max_eigenvalue)
    , m_has_uncertainty(false)
    , m_observation_positions_valid(false)
{
}

MapPoint::MapPoint(const Eigen::Vector3f& position)
    : m_id(s_next_id++)
    , m_position(position)
    , m_is_bad(false)
    , m_is_multi_view_triangulated(false)
    , m_is_marginalized(false)
    , m_world_uncertainty(Eigen::Matrix3f::Identity())
    , m_has_uncertainty(false)
    , m_observation_positions_valid(false)
{
}

MapPoint::~MapPoint() {
}

void MapPoint::set_position(const Eigen::Vector3f& position) {
    std::lock_guard<std::mutex> lock(m_position_mutex);
    m_position = position;
}

const Eigen::Vector3f& MapPoint::get_position() const {
    std::lock_guard<std::mutex> lock(m_position_mutex);
    return m_position;
}

void MapPoint::add_observation(std::shared_ptr<Frame> frame, int feature_index) {
    std::lock_guard<std::mutex> lock(m_data_mutex);
    if (!frame) return;
    
    // Check if this frame is already observing this map point
    for (auto& obs : m_observations) {
        if (auto existing_frame = obs.frame.lock()) {
            if (existing_frame == frame) {
                // Update feature index if frame already exists
                obs.feature_index = feature_index;
                return;
            }
        }
    }
    
    // Add new observation
    m_observations.emplace_back(frame, feature_index);
    
    // Increment accumulated observation count for the feature
    auto feature = frame->get_feature(feature_index);
    if (feature) {
        feature->increment_observations_accumulated();
    }
}

void MapPoint::remove_observation(std::shared_ptr<Frame> frame) {
    if (!frame) return;
    
    m_observations.erase(
        std::remove_if(m_observations.begin(), m_observations.end(),
                      [frame](const Observation& obs) {
                          if (auto obs_frame = obs.frame.lock()) {
                              return obs_frame == frame;
                          }
                          return true; // Remove expired weak_ptr
                      }),
        m_observations.end()
    );
    
    // Mark as bad if no observations left
    if (m_observations.empty()) {
        set_bad();
    }
}

const std::vector<Observation>& MapPoint::get_observations() const {
    return m_observations;
}

int MapPoint::get_observation_count() const {
    return static_cast<int>(m_observations.size());
}

bool MapPoint::is_observed_by_frame(std::shared_ptr<Frame> frame) const {
    if (!frame) return false;
    
    for (const auto& obs : m_observations) {
        if (auto obs_frame = obs.frame.lock()) {
            if (obs_frame == frame) {
                return true;
            }
        }
    }
    return false;
}

int MapPoint::get_feature_index_in_frame(std::shared_ptr<Frame> frame) const {
    if (!frame) return -1;
    
    for (const auto& obs : m_observations) {
        if (auto obs_frame = obs.frame.lock()) {
            if (obs_frame == frame) {
                return obs.feature_index;
            }
        }
    }
    return -1;
}

void MapPoint::set_id(int id) {
    m_id = id;
}

int MapPoint::get_id() const {
    return m_id;
}

void MapPoint::set_bad() {
    std::lock_guard<std::mutex> lock(m_data_mutex);
    m_is_bad = true;
}

bool MapPoint::is_bad() const {
    std::lock_guard<std::mutex> lock(m_data_mutex);
    return m_is_bad;
}

void MapPoint::set_multi_view_triangulated(bool flag) {
    std::lock_guard<std::mutex> lock(m_data_mutex);
    m_is_multi_view_triangulated = flag;
}

bool MapPoint::is_multi_view_triangulated() const {
    std::lock_guard<std::mutex> lock(m_data_mutex);
    return m_is_multi_view_triangulated;
}

void MapPoint::set_marginalized(bool flag) {
    std::lock_guard<std::mutex> lock(m_data_mutex);
    m_is_marginalized = flag;
}

bool MapPoint::is_marginalized() const {
    std::lock_guard<std::mutex> lock(m_data_mutex);
    return m_is_marginalized;
}

double MapPoint::compute_reprojection_error() const {
    if (m_observations.empty()) {
        return 0.0;
    }
    
    double total_error = 0.0;
    int valid_observations = 0;
    
    for (const auto& obs : m_observations) {
        if (auto frame = obs.frame.lock()) {
            // TODO: Implement reprojection error calculation
            // This would require camera intrinsics and pose information
            valid_observations++;
        }
    }
    
    return valid_observations > 0 ? total_error / valid_observations : 0.0;
}




Eigen::Matrix2f MapPoint::compute_uncertainty(std::shared_ptr<Frame> frame) const
{
    // Transform world uncertainty to pixel uncertainty
    return transform_uncertainty_world_to_pixel(m_world_uncertainty, frame);
}

Eigen::Matrix3f MapPoint::initial_unproject_pixel_uncertainty_to_world(const Eigen::Matrix2f& pixel_uncertainty, std::shared_ptr<Frame> frame)
{
    // Get camera intrinsics
    double fx, fy, cx, cy;
    fx = frame->get_fx(); fy = frame->get_fy(); cx = frame->get_cx(); cy = frame->get_cy();
    
    // Get 3D point in camera coordinates
    Eigen::Vector3f world_pos = get_position();
    Eigen::Matrix4f T_cw = frame->get_Twc().inverse();
    Eigen::Vector3f camera_pos = (T_cw * world_pos.homogeneous()).head<3>();
    
    // Avoid division by zero
    if (std::abs(camera_pos.z()) < 1e-6) {
        return Eigen::Matrix3f::Identity();
    }
    
    // Standard inverse projection Jacobian: J_inv = [Z/fx   0    X/fx]
    //                                               [0     Z/fy  Y/fy]
    //                                               [0      0     1  ]
    float X = camera_pos.x();
    float Y = camera_pos.y();
    float Z = camera_pos.z();
    float fx_inv = 1.0f / fx;
    float fy_inv = 1.0f / fy;
    
    Eigen::Matrix<float, 3, 2> J_inv_proj;
    J_inv_proj << Z * fx_inv, 0,
                  0,          Z * fy_inv,
                  X * fx_inv, Y * fy_inv;

    // Depth uncertainty based on stereo geometry
    // float pixel_error = std::sqrt(pixel_uncertainty(0,0));
    // float depth_uncertainty = Z*0.01f; // ToDo (Heuristic)

    // Create camera uncertainty with proper depth component
    Eigen::Matrix3f camera_uncertainty = J_inv_proj * pixel_uncertainty * J_inv_proj.transpose();
    // camera_uncertainty(2, 2) += depth_uncertainty * depth_uncertainty;

    // Get camera pose (T_wc = T_wb * T_bc)
    Eigen::Matrix4f T_wc = frame->get_Twc();
    
    // Get viewing direction information
    Eigen::Vector3f camera_pos_world = T_wc.block<3, 1>(0, 3);
    Eigen::Vector3f viewing_direction = (world_pos - camera_pos_world).normalized();
    float distance = (world_pos - camera_pos_world).norm();
    
    // Create viewing direction-based coordinate system
    // Z-axis: along viewing direction (from camera to point)
    Eigen::Vector3f z_view = viewing_direction;
    
    // X-axis: perpendicular to viewing direction and camera up vector
    Eigen::Vector3f camera_up = T_wc.block<3, 3>(0, 0).col(1); // camera Y-axis (down in camera frame)
    Eigen::Vector3f x_view = z_view.cross(-camera_up).normalized(); // Use -camera_up for proper orientation
    
    // Y-axis: complete the right-handed system
    Eigen::Vector3f y_view = z_view.cross(x_view).normalized();
    
    // Build viewing direction-based rotation matrix
    Eigen::Matrix3f R_view_to_world;
    R_view_to_world.col(0) = x_view;
    R_view_to_world.col(1) = y_view;
    R_view_to_world.col(2) = z_view;


    // std::cout<<"R_view_to_world:\n" << R_view_to_world << std::endl;
    
    // Transform uncertainty from viewing direction frame to world frame
    Eigen::Matrix3f base_uncertainty = R_view_to_world * camera_uncertainty * R_view_to_world.transpose();

    // Add viewing direction-dependent uncertainty
    // Points further from camera center have more uncertainty in viewing direction
    float directional_uncertainty = distance * 0.01f; // Small factor for viewing direction uncertainty
    Eigen::Matrix3f directional_component = directional_uncertainty * directional_uncertainty * 
                                           (viewing_direction * viewing_direction.transpose());


    // std::cout<<"Base uncertainty (world frame):\n" << base_uncertainty << std::endl;
    // std::cout<<"Directional component:\n" << directional_component << std::endl;
    
    return base_uncertainty + directional_component;

}


// Uncertainty transformation functions
Eigen::Matrix3f MapPoint::transform_uncertainty_world_to_camera(const Eigen::Matrix3f& world_uncertainty, 
                                                                 std::shared_ptr<Frame> frame) const {
    if (!frame) {
        return Eigen::Matrix3f::Identity();
    }
    
    // Get camera pose (T_cw = T_cb * T_bw)
    Eigen::Matrix4f T_wc = frame->get_Twc();
    Eigen::Matrix4f T_cw = T_wc.inverse();
    Eigen::Matrix3f R_cw = T_cw.block<3, 3>(0, 0);
    
    // Transform uncertainty: Σ_camera = R_cw * Σ_world * R_cw^T
    return R_cw * world_uncertainty * R_cw.transpose();
}

Eigen::Matrix2f MapPoint::transform_uncertainty_world_to_pixel(const Eigen::Matrix3f& world_uncertainty, 
                                                               std::shared_ptr<Frame> frame) const {
    // First transform to camera coordinates
    Eigen::Matrix3f camera_uncertainty = transform_uncertainty_world_to_camera(world_uncertainty, frame);
    
    // Then transform to pixel coordinates
    return transform_uncertainty_camera_to_pixel(camera_uncertainty, frame);
}



Eigen::Matrix2f MapPoint::transform_uncertainty_camera_to_pixel(const Eigen::Matrix3f& camera_uncertainty, 
                                                                std::shared_ptr<Frame> frame) const {
    if (!frame) {
        return Eigen::Matrix2f::Identity();
    }
    
    // Get camera intrinsics
    float fx = frame->get_fx();
    float fy = frame->get_fy();
    
    // Get 3D point in camera coordinates
    Eigen::Vector3f world_pos = get_position();
    Eigen::Matrix4f T_cw = frame->get_Twc().inverse();
    Eigen::Vector3f camera_pos = (T_cw * world_pos.homogeneous()).head<3>();
    
    // Avoid division by zero
    if (std::abs(camera_pos.z()) < 1e-6) {
        return Eigen::Matrix2f::Identity();
    }
    
    // Projection Jacobian: J = [fx/Z  0  -fx*X/Z^2]
    //                          [0   fy/Z -fy*Y/Z^2]
    float X = camera_pos.x();
    float Y = camera_pos.y();
    float Z = camera_pos.z();
    float Z_inv = 1.0f / Z;
    float Z_inv_sq = Z_inv * Z_inv;
    
    Eigen::Matrix<float, 2, 3> J_proj;
    J_proj << fx * Z_inv,      0,        -fx * X * Z_inv_sq,
              0,               fy * Z_inv, -fy * Y * Z_inv_sq;
    
    // Transform uncertainty: Σ_pixel = J * Σ_camera * J^T
    return J_proj * camera_uncertainty * J_proj.transpose();
}

Eigen::Matrix3f MapPoint::transform_uncertainty_pixel_to_camera(const Eigen::Matrix2f& pixel_uncertainty, 
                                                                 std::shared_ptr<Frame> frame) const {
    if (!frame) {
        return Eigen::Matrix3f::Identity();
    }
    
    // Get camera intrinsics
    double fx, fy, cx, cy;
    fx = frame->get_fx(); fy = frame->get_fy(); cx = frame->get_cx(); cy = frame->get_cy();
    
    // Get 3D point in camera coordinates
    Eigen::Vector3f world_pos = get_position();
    Eigen::Matrix4f T_cw = frame->get_Twc().inverse();
    Eigen::Vector3f camera_pos = (T_cw * world_pos.homogeneous()).head<3>();
    
    // Avoid division by zero
    if (std::abs(camera_pos.z()) < 1e-6) {
        return Eigen::Matrix3f::Identity();
    }
    
    // Standard inverse projection Jacobian: J_inv = [Z/fx   0    X/fx]
    //                                               [0     Z/fy  Y/fy]
    //                                               [0      0     1  ]
    float X = camera_pos.x();
    float Y = camera_pos.y();
    float Z = camera_pos.z();
    float fx_inv = 1.0f / fx;
    float fy_inv = 1.0f / fy;
    
    Eigen::Matrix<float, 3, 2> J_inv_proj;
    J_inv_proj << Z * fx_inv, 0,
                  0,          Z * fy_inv,
                  X * fx_inv, Y * fy_inv;
    
    // Depth uncertainty based on stereo geometry
    float pixel_error = std::sqrt(pixel_uncertainty(0,0));
    float depth_uncertainty = Z*0.1f; // stereo baseline ~0.11m
    
    // Create camera uncertainty with proper depth component
    Eigen::Matrix3f camera_uncertainty = J_inv_proj * pixel_uncertainty * J_inv_proj.transpose();
    camera_uncertainty(2, 2) += depth_uncertainty * depth_uncertainty;
    
    return camera_uncertainty;
}

Eigen::Matrix3f MapPoint::transform_uncertainty_camera_to_world(const Eigen::Matrix3f& camera_uncertainty, 
                                                                 std::shared_ptr<Frame> frame) const {
    if (!frame) {
        return Eigen::Matrix3f::Identity();
    }
    
    // Get camera pose (T_wc = T_wb * T_bc)
    Eigen::Matrix4f T_wc = frame->get_Twc();
    
    Eigen::Matrix3f R_wc = T_wc.block<3, 3>(0, 0);

    Eigen::Matrix3f world_uncertainty = R_wc * camera_uncertainty * R_wc.transpose();

    
    return world_uncertainty;
}

Eigen::Matrix3f MapPoint::transform_uncertainty_pixel_to_world(const Eigen::Matrix2f& pixel_uncertainty, 
                                                                std::shared_ptr<Frame> frame) const {
    // First transform to camera coordinates
    Eigen::Matrix3f camera_uncertainty = transform_uncertainty_pixel_to_camera(pixel_uncertainty, frame);
    
    // Then transform to world coordinates
    return transform_uncertainty_camera_to_world(camera_uncertainty, frame);
}

// Uncertainty management functions
void MapPoint::set_world_uncertainty(const Eigen::Matrix3f& uncertainty) {
    std::lock_guard<std::mutex> lock(m_data_mutex);
    
    // Check eigenvalues before setting
    Eigen::SelfAdjointEigenSolver<Eigen::Matrix3f> solver(uncertainty);
    if (solver.info() == Eigen::Success) {
        float max_eigenvalue = solver.eigenvalues().maxCoeff();
    }
    
    m_world_uncertainty = uncertainty;
    m_has_uncertainty = true;
}

void MapPoint::set_min_world_uncertainty(const Eigen::Matrix3f& uncertainty) {
    std::lock_guard<std::mutex> lock(m_data_mutex);
    m_world_uncertainty_min = uncertainty;
    m_has_uncertainty = true;
}

const Eigen::Matrix3f& MapPoint::get_world_uncertainty() const {
    std::lock_guard<std::mutex> lock(m_data_mutex);
    
    // Periodically check eigenvalues when accessed
    static int access_count = 0;
    access_count++;
    if (access_count % 50 == 0) { // Print every 50th access
        Eigen::SelfAdjointEigenSolver<Eigen::Matrix3f> solver(m_world_uncertainty);
        if (solver.info() == Eigen::Success) {
            float max_eigenvalue = solver.eigenvalues().maxCoeff();
        }
    }
    
    return m_world_uncertainty;
}

const Eigen::Matrix3f& MapPoint::get_min_world_uncertainty() const {
    std::lock_guard<std::mutex> lock(m_data_mutex);
    return m_world_uncertainty_min;
}

bool MapPoint::has_uncertainty() const {
    std::lock_guard<std::mutex> lock(m_data_mutex);
    return m_has_uncertainty;
}

bool MapPoint::has_world_uncertainty() const {
    std::lock_guard<std::mutex> lock(m_data_mutex);
    return m_has_uncertainty;
}

// void MapPoint::update_world_uncertainty_with_observations()
// {

//     auto observations = get_observations();

//     clear_all_world_uncertainties();

//     // Information Matrix 방식으로 결합 
//     Eigen::Matrix3f combined_information = Eigen::Matrix3f::Zero();
//     Eigen::Matrix3f combined_information_min = Eigen::Matrix3f::Zero();
//     float count = 0.0f;

//     for (const auto& obs : observations) {
//         if (auto frame = obs.frame.lock()) {
//             // 각 observation에서 world uncertainty 계산

//             // reprojection error 계산
//             Eigen::Vector3f world_pos = get_position();
//             Eigen::Matrix4f T_cw = frame->get_Twc().inverse();
//             Eigen::Vector3f camera_pos = (T_cw * world_pos.homogeneous()).head<3>();
//             float u_proj = frame->get_fx() * (camera_pos.x() / camera_pos.z()) + frame->get_cx();
//             float v_proj = frame->get_fy() * (camera_pos.y() / camera_pos.z()) + frame->get_cy();
//             int feature_index = obs.feature_index;
//             auto feature = frame->get_feature(feature_index);
                
//             if (!feature) continue;
            
//             cv::Point2f undistorted_coord = feature->get_undistorted_coord();
//             float u_obs = undistorted_coord.x;
//             float v_obs = undistorted_coord.y;

//             float reproj_error = std::sqrt((u_proj - u_obs)*(u_proj - u_obs) + (v_proj - v_obs)*(v_proj - v_obs));

//             float min_reproj_error = Config::getInstance().m_min_reprojection_error;

//             reproj_error = std::max(reproj_error, min_reproj_error);


//             Eigen::Matrix2f pixel_uncertainty = Eigen::Matrix2f::Identity() * (reproj_error * reproj_error);
//             Eigen::Matrix2f pixel_uncertainty_min = Eigen::Matrix2f::Identity() * (min_reproj_error * min_reproj_error);


//             // spdlog::info("Reprojection error x y : {} {}", abs(u_proj - u_obs), abs(v_proj - v_obs));
//             // spdlog::info("Pixel uncertainty: {} {}", pixel_uncertainty(0,0), pixel_uncertainty(1,1));


//             Eigen::Matrix3f world_uncertainty_i = initial_unproject_pixel_uncertainty_to_world(pixel_uncertainty, frame);
//             Eigen::Matrix3f world_uncertainty_i_min = initial_unproject_pixel_uncertainty_to_world(pixel_uncertainty_min, frame);

           
//             // Covariance → Information Matrix (Σ^(-1))
//             Eigen::Matrix3f information_i = world_uncertainty_i.inverse();
//             Eigen::Matrix3f information_i_min = world_uncertainty_i_min.inverse();
            
//             // Information Matrix들을 더함
//             combined_information += information_i;
//             combined_information_min += information_i_min;
//         }
//     }

//     Eigen::Matrix3f world_uncertainty_combined = combined_information.inverse();
//     Eigen::Matrix3f world_uncertainty_combined_min = combined_information_min.inverse();
//     set_world_uncertainty(world_uncertainty_combined);
//     set_min_world_uncertainty(world_uncertainty_combined_min);

// }

std::vector<Eigen::Vector3f> MapPoint::compute_multi_view_positions() const {
    std::vector<Eigen::Vector3f> positions;
    std::lock_guard<std::mutex> lock(m_data_mutex);
    
    // For each stereo observation, use the computed depth from triangulation
    for (const auto& obs : m_observations) {
        auto frame = obs.frame.lock();
        if (!frame) continue;
        
        // Get feature from the observation
        const auto& features = frame->get_features();
        if (obs.feature_index >= 0 && obs.feature_index < features.size()) {
            auto feature = features[obs.feature_index];
            if (!feature || !feature->is_valid()) {
                continue;
            }
            
            // Use the pre-computed depth from frame
            if (!frame->has_depth(obs.feature_index)) {
                continue;
            }
            
            double depth = frame->get_depth(obs.feature_index);
            
            // Use config depth range for validation
            const auto& config = Config::getInstance();
            if (depth <= config.m_min_depth || depth >= config.m_max_depth) {
                continue;
            }
            
            // Get pixel coordinates
            cv::Point2f left_pixel = feature->get_undistorted_coord();
            
            // Get camera parameters using individual getters
            float fx = frame->get_fx();
            float fy = frame->get_fy();
            float cx = frame->get_cx();
            float cy = frame->get_cy();
            
            // Convert to 3D point in camera frame using pre-computed depth
            Eigen::Vector3f camera_point;
            camera_point.x() = (left_pixel.x - cx) * depth / fx;
            camera_point.y() = (left_pixel.y - cy) * depth / fy;
            camera_point.z() = depth;
            
            // Transform to world frame
            Eigen::Matrix4f Twc = frame->get_Twc();
            Eigen::Vector3f world_point = (Twc * camera_point.homogeneous()).head<3>();
            
            positions.push_back(world_point);
        }
    }
    
    return positions;
}

void MapPoint::update_uncertainty() {

    m_observation_positions = compute_multi_view_positions();
    m_observation_positions_valid = true;

    
    // Compute covariance matrix using current MapPoint position as center and set as world uncertainty
    if (m_observation_positions.size() >= 3) {
        // Use the current optimized MapPoint position as the center
        Eigen::Vector3f current_position;
        {
            std::lock_guard<std::mutex> pos_lock(m_position_mutex);
            current_position = m_position;
        }
        
        // Compute covariance matrix and directly set as world uncertainty
        Eigen::Matrix3f observation_covariance = compute_covariance_from_positions(m_observation_positions, current_position);
        
        // Scale covariance matrix to limit maximum eigenvalue properly
        Eigen::SelfAdjointEigenSolver<Eigen::Matrix3f> solver(observation_covariance);
        if (solver.info() == Eigen::Success) {
            Eigen::Vector3f eigenvalues = solver.eigenvalues();
            Eigen::Matrix3f eigenvectors = solver.eigenvectors();
            
            // float max_eigenvalue = eigenvalues.maxCoeff();
            float max_allowed = Config::getInstance().m_uncertainty_max_eigenvalue;
            float min_allowed = Config::getInstance().m_uncertainty_min_eigenvalue;
            
            bool modified = false;


            for(int i=0; i<3; i++) {
                if (eigenvalues(i) < min_allowed) {
                    eigenvalues(i) = min_allowed; // Small positive value to ensure positive definiteness
                    modified = true;
                }
                else if (eigenvalues(i) > max_allowed) {
                    eigenvalues(i) = max_allowed; // Cap to maximum allowed
                    modified = true;
                }
            }


            
            // // Apply maximum eigenvalue constraint
            // if (max_eigenvalue > max_allowed) {
            //     float scale = max_allowed / max_eigenvalue;
            //     eigenvalues *= scale;  // Scale eigenvalues
            //     modified = true;
            // }
            
            // // Apply minimum eigenvalue constraint
            // float min_eigenvalue = eigenvalues.minCoeff();
            // if (min_eigenvalue < min_allowed) {
            //     float scale = min_allowed / min_eigenvalue;
            //     eigenvalues *= scale;  // Scale eigenvalues up
            //     modified = true;
            // }
            
            // Reconstruct covariance matrix if modified
            if (modified) {


                // spdlog::info("Changing eigenvalues for MapPoint {}: [{:.6f}, {:.6f}, {:.6f}] -> [{:.6f}, {:.6f}, {:.6f}]",
                //              m_id,
                //              solver.eigenvalues()(0), solver.eigenvalues()(1), solver.eigenvalues()(2),
                //              eigenvalues(0), eigenvalues(1), eigenvalues(2));
                observation_covariance = eigenvectors * eigenvalues.asDiagonal() * eigenvectors.transpose();
                
                // Verify scaling worked
                Eigen::SelfAdjointEigenSolver<Eigen::Matrix3f> solver_after(observation_covariance);
                if (solver_after.info() == Eigen::Success) {
                    float new_max_eigenvalue = solver_after.eigenvalues().maxCoeff();
                    float new_min_eigenvalue = solver_after.eigenvalues().minCoeff();
                }
            }
        }
        
        set_world_uncertainty(observation_covariance);
    } 
    else
    {
        // For insufficient data, use small default covariance
        Eigen::Matrix3f default_covariance = Eigen::Matrix3f::Identity() * Config::getInstance().m_uncertainty_max_eigenvalue;
        set_world_uncertainty(default_covariance);
    }
    
    // else if (m_observation_positions.size() >= 1) {
    //     // For insufficient data, use small default covariance
    //     // Eigen::Matrix3f default_covariance = Eigen::Matrix3f::Identity() * 0.01f;
    //     // set_world_uncertainty(default_covariance);
    // } 
    // No need to handle else case since world uncertainty already exists
}

const std::vector<Eigen::Vector3f>& MapPoint::get_observation_positions() const {
    std::lock_guard<std::mutex> lock(m_data_mutex);
    return m_observation_positions;
}

bool MapPoint::has_valid_observation_positions() const {
    std::lock_guard<std::mutex> lock(m_data_mutex);
    return m_observation_positions_valid && !m_observation_positions.empty();
}

Eigen::Matrix3f MapPoint::compute_covariance_from_positions(const std::vector<Eigen::Vector3f>& positions, 
                                                           const Eigen::Vector3f& mean_position) const {
    if (positions.size() < 3) {
        // Return identity matrix for insufficient data
        return Eigen::Matrix3f::Identity() * 0.01f;  // Small default covariance
    }
    
    // Compute covariance matrix - calculate variance for each axis separately
    float var_x = 0, var_y = 0, var_z = 0;
    for (const auto& pos : positions) {
        Eigen::Vector3f diff = pos - mean_position;
        var_x += diff.x() * diff.x();
        var_y += diff.y() * diff.y();  
        var_z += diff.z() * diff.z();
    }
    
    // Normalize by (n-1) for sample variance
    float n_minus_1 = static_cast<float>(positions.size() - 1);
    var_x /= n_minus_1;
    var_y /= n_minus_1;
    var_z /= n_minus_1;
    
    // Create diagonal covariance matrix
    Eigen::Matrix3f covariance = Eigen::Matrix3f::Zero();
    covariance(0,0) = var_x;
    covariance(1,1) = var_y;
    covariance(2,2) = var_z;
    
    // Add small regularization to ensure positive definiteness
    covariance += Eigen::Matrix3f::Identity() * 1e-6f;
    
    return covariance;
}

} // namespace lightweight_vio
