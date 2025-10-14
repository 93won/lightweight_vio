/**
 * @file      MapPoint.h
 * @brief     Defines the MapPoint class, representing a 3D point in the map.
 * @author    Seungwon Choi (csw3575@snu.ac.kr)
 * @date      2025-08-18
 * @copyright Copyright (c) 2025 Seungwon Choi. All rights reserved.
 *
 * @par License
 * This project is released under the MIT License.
 */

#pragma once

#include <vector>
#include <memory>
#include <Eigen/Dense>
#include <mutex>

namespace lightweight_vio {

class Frame;
class Feature;

struct Observation {
    std::weak_ptr<Frame> frame;
    int feature_index;
    
    Observation(std::shared_ptr<Frame> f, int idx) 
        : frame(f), feature_index(idx) {}
};

class MapPoint {
public:
    MapPoint();
    MapPoint(const Eigen::Vector3f& position);
    ~MapPoint();

    // Position management
    void set_position(const Eigen::Vector3f& position);
    const Eigen::Vector3f& get_position() const;
    
    // Observation management
    void add_observation(std::shared_ptr<Frame> frame, int feature_index);
    void remove_observation(std::shared_ptr<Frame> frame);
    const std::vector<Observation>& get_observations() const;
    int get_observation_count() const;
    
    // Utility functions
    bool is_observed_by_frame(std::shared_ptr<Frame> frame) const;
    int get_feature_index_in_frame(std::shared_ptr<Frame> frame) const;
    
    // MapPoint management
    void set_id(int id);
    int get_id() const;
    
    void set_bad();
    bool is_bad() const;
    
    // Multi-view triangulation flag
    void set_multi_view_triangulated(bool flag);
    bool is_multi_view_triangulated() const;
    
    // Marginalization flag
    void set_marginalized(bool flag);
    bool is_marginalized() const;
    
    // Triangulation and refinement
    double compute_reprojection_error() const;

    Eigen::Matrix3f initial_unproject_pixel_uncertainty_to_world(const Eigen::Matrix2f& pixel_uncertainty, std::shared_ptr<Frame> frame);

    void update_world_uncertainty_with_observations();

    // Uncertainty transformation functions
    Eigen::Matrix3f transform_uncertainty_world_to_camera(const Eigen::Matrix3f& world_uncertainty, 
                                                           std::shared_ptr<Frame> frame) const;
    Eigen::Matrix2f transform_uncertainty_world_to_pixel(const Eigen::Matrix3f& world_uncertainty, 
                                                          std::shared_ptr<Frame> frame) const;
    Eigen::Matrix2f transform_uncertainty_camera_to_pixel(const Eigen::Matrix3f& camera_uncertainty, 
                                                           std::shared_ptr<Frame> frame) const;
    Eigen::Matrix3f transform_uncertainty_pixel_to_camera(const Eigen::Matrix2f& pixel_uncertainty, 
                                                           std::shared_ptr<Frame> frame) const;
    Eigen::Matrix3f transform_uncertainty_camera_to_world(const Eigen::Matrix3f& camera_uncertainty, 
                                                           std::shared_ptr<Frame> frame) const;
    Eigen::Matrix3f transform_uncertainty_pixel_to_world(const Eigen::Matrix2f& pixel_uncertainty, 
                                                          std::shared_ptr<Frame> frame) const;

    Eigen::Matrix3f get_uncertainty_world_from_min_reproj_err() const;

    Eigen::Matrix2f compute_uncertainty(std::shared_ptr<Frame> frame) const;
    
    // Uncertainty management
    void set_world_uncertainty(const Eigen::Matrix3f& uncertainty);
    void set_min_world_uncertainty(const Eigen::Matrix3f& uncertainty);
    const Eigen::Matrix3f& get_world_uncertainty() const;
    const Eigen::Matrix3f& get_min_world_uncertainty() const;
    bool has_uncertainty() const;
    bool has_world_uncertainty() const;
    
    // Multi-view position computation
    std::vector<Eigen::Vector3f> compute_multi_view_positions() const;
    void update_uncertainty();  // Initialize uncertainty from observation positions (once only)
    const std::vector<Eigen::Vector3f>& get_observation_positions() const;  // Get cached positions
    bool has_valid_observation_positions() const;  // Check if cache is valid
    
   
    std::vector<Eigen::Matrix3f> get_all_world_uncertainties() const
    {
        std::lock_guard<std::mutex> lock(m_data_mutex);
        return m_all_world_uncertainties;
    }
    void clear_all_world_uncertainties()
    {
        std::lock_guard<std::mutex> lock(m_data_mutex);
        m_all_world_uncertainties.clear();
    }
    // void insert_world_uncertainty(const Eigen::Matrix3f& uncertainty)
    // {
    //     std::lock_guard<std::mutex> lock(m_data_mutex);
    //     m_all_world_uncertainties.push_back(uncertainty);
    // }

private:
    int m_id;
    Eigen::Vector3f m_position;
    std::vector<Observation> m_observations;
    bool m_is_bad;
    bool m_is_multi_view_triangulated;
    bool m_is_marginalized;
    
    // Uncertainty data
    Eigen::Matrix3f m_world_uncertainty;
    Eigen::Matrix3f m_world_uncertainty_min;
    bool m_has_uncertainty;

    std::vector<Eigen::Matrix3f> m_all_world_uncertainties;
    
    // Multi-view observation data (cached for performance)
    std::vector<Eigen::Vector3f> m_observation_positions;
    bool m_observation_positions_valid;
    
    // Helper functions
    Eigen::Matrix3f compute_covariance_from_positions(const std::vector<Eigen::Vector3f>& positions, 
                                                      const Eigen::Vector3f& mean_position) const;
    
    // Thread safety
    mutable std::mutex m_position_mutex; // Mutex for position operations
    mutable std::mutex m_data_mutex;     // Mutex for other data operations
    
    static int s_next_id;
};

} // namespace lightweight_vio
