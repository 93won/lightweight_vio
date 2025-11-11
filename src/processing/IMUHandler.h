/**
 * @file      IMUHandler.h
 * @brief     Handles IMU data preintegration and bias management for VIO
 * @author    Seungwon Choi (csw3575@snu.ac.kr)
 * @date      2025-09-08
 * @copyright Copyright (c) 2025 Seungwon Choi. All rights reserved.
 *
 * @par License
 * This project is released under the MIT License.
 *
 * @par Mathematical Foundation
 * The IMU preintegration theory and mathematical formulations implemented in this file
 * are based on the work presented in:
 * "IMU Preintegration on Manifold for Efficient Visual-Inertial Maximum-a-Posteriori Estimation"
 * by C. Forster, L. Carlone, F. Dellaert, and D. Scaramuzza (RSS 2015)
 */

#pragma once

#include <memory>
#include <vector>
#include <Eigen/Dense>
#include <Eigen/Core>
#include <Eigen/Geometry>
#include <sophus/se3.hpp>

namespace lightweight_vio {

class Config;
class MapPoint;  // Forward declaration
class Frame;
struct IMUData;

// IMU preintegration measurement structure
struct IMUPreintegration {
    // Preintegrated measurements (from keyframe i to j)
    Eigen::Matrix3f delta_R;        // Rotation increment
    Eigen::Vector3f delta_V;        // Velocity increment  
    Eigen::Vector3f delta_P;        // Position increment
    
    // Jacobians w.r.t. bias (for efficient updates)
    Eigen::Matrix3f J_Rg;            // ∂δR/∂bg (gyro bias)
    Eigen::Matrix3f J_Vg, J_Va;       // ∂δV/∂bg, ∂δV/∂ba
    Eigen::Matrix3f J_Pg, J_Pa;       // ∂δP/∂bg, ∂δP/∂ba
    
    // Covariance matrix (15x15: δR,δV,δP,δbg,δba)
    Eigen::Matrix<float, 15, 15> covariance;
    
    // Bias used for preintegration
    Eigen::Vector3f gyro_bias;
    Eigen::Vector3f accel_bias;
    
    // Time span
    double dt_total;
    
    IMUPreintegration();
    void reset();
    bool is_valid() const;
};

/**
 * @brief Handles IMU data processing and preintegration for VIO
 */
class IMUHandler {
public:
    IMUHandler();
    ~IMUHandler() = default;
    
    /**
     * @brief Reset IMU handler state
     */
    void reset();
    
    /**
     * @brief Set current IMU bias estimates
     * @param gyro_bias Current gyroscope bias
     * @param accel_bias Current accelerometer bias
     */
    void set_bias(const Eigen::Vector3f& gyro_bias, const Eigen::Vector3f& accel_bias);
    
    /**
     * @brief Get current IMU bias estimates
     * @param gyro_bias Output gyroscope bias
     * @param accel_bias Output accelerometer bias
     */
    void get_bias(Eigen::Vector3f& gyro_bias, Eigen::Vector3f& accel_bias) const;
    
    /**
     * @brief Get current gyroscope bias estimate
     * @return Current gyroscope bias
     */
    Eigen::Vector3f get_gyro_bias() const { return m_gyro_bias; }
    
    /**
     * @brief Get current accelerometer bias estimate
     * @return Current accelerometer bias
     */
    Eigen::Vector3f get_accel_bias() const { return m_accel_bias; }
    
    /**
     * @brief Preintegrate IMU measurements between two timestamps
     * @param imu_measurements IMU measurements to integrate
     * @param start_time Start timestamp  
     * @param end_time End timestamp
     * @return Raw preintegrated IMU measurement (rotation compensation handled later)
     */
    std::shared_ptr<IMUPreintegration> preintegrate(
        const std::vector<IMUData>& imu_measurements,
        double start_time,
        double end_time
    );
    
    /**
     * @brief Update preintegration result when bias changes (using Jacobians)
     * @param preint Preintegration to update
     * @param new_gyro_bias New gyro bias values
     * @param new_accel_bias New accel bias values
     */
    void update_preintegration_with_new_bias(
        std::shared_ptr<IMUPreintegration> preint,
        const Eigen::Vector3f& new_gyro_bias,
        const Eigen::Vector3f& new_accel_bias
    );
    
    /**
     * @brief Estimate initial IMU bias from static measurements
     * @param imu_measurements Static IMU measurements
     * @param gravity_magnitude Expected gravity magnitude (default: 9.81)
     */
    void estimate_initial_bias(
        const std::vector<IMUData>& imu_measurements,
        float gravity_magnitude = 9.81f
    );
    
    /**
     * @brief Test workflow for gravity estimation (demonstration)
     * @param frames Vector of frames with pose estimates
     * @param all_imu_data All available IMU measurements
     * @return Success flag
     */
    bool test_gravity_estimation_workflow(
        const std::vector<Frame*>& frames,
        const std::vector<IMUData>& all_imu_data
    );
    
    /**
     * @brief Preintegrate IMU measurements with gravity compensation
     * @param imu_measurements IMU measurements to integrate
     * @param start_time Start timestamp
     * @param end_time End timestamp
     * @param R_wb_initial Initial world-to-body rotation matrix
     * @return Gravity-compensated preintegrated IMU measurement
     */
    std::shared_ptr<IMUPreintegration> preintegrate_with_gravity_compensation(
        const std::vector<IMUData>& imu_measurements,
        double start_time,
        double end_time,
        const Eigen::Matrix3f& R_wb_initial
    );
    
    /**
     * @brief Recompute all preintegrations with gravity compensation after gravity estimation
     * @param frames Vector of frames with pose estimates
     * @param all_imu_data All available IMU measurements
     * @param gravity_compensated_preints Output vector of gravity-compensated preintegrations
     * @return Success flag
     */
    bool recompute_preintegrations_with_gravity_compensation(
        const std::vector<Frame*>& frames,
        const std::vector<IMUData>& all_imu_data,
        std::vector<std::shared_ptr<IMUPreintegration>>& gravity_compensated_preints
    );
    

    
    /**
     * @brief Estimate gravity vector using stereo visual constraints
     * @param frames Vector of frames with pose estimates
     * @param all_imu_data All available IMU measurements  
     * @param gravity_magnitude Expected gravity magnitude (default: 9.81)
     * @param[out] initial_cost Optional output for initial cost before optimization
     * @param[out] final_cost Optional output for final cost after optimization
     * @return Success flag
     */
    bool estimate_gravity_with_stereo_constraints(
        const std::vector<Frame*>& frames,
        const std::vector<IMUData>& all_imu_data,
        float gravity_magnitude = 9.81f,
        double* initial_cost = nullptr,
        double* final_cost = nullptr
    );
    

    
    /**
     * @brief Check if IMU has been initialized
     * @return True if initialized
     */
    bool is_initialized() const { return m_initialized; }

    void set_initialized(bool initialized) { m_initialized = initialized; }
    
    /**
     * @brief Get gravity vector in world frame
     * @return Gravity vector
     */
    Eigen::Vector3f get_gravity() const { return m_gravity; }
    
    /**
     * @brief Set gravity vector and compute Rgw transformation
     * @param gravity Gravity vector in world frame
     */
    void set_gravity(const Eigen::Vector3f& gravity) { 
        m_gravity = gravity; 
        compute_rgw_transformation();
    }

    /**
     * @brief Compute World-to-Gravity transformation matrix from current gravity vector
     */
    void compute_rgw_transformation();

    /**
     * @brief Set coordinate system to gravity-aligned (Rgw = Identity, gravity = [0,0,-9.81])
     */
    void set_gravity_aligned_coordinate_system();

    /**
     * @brief Debug velocity comparison between stereo VO and gravity-compensated IMU
     * @param frames Vector of frames with pose estimates
     * @param all_imu_data All available IMU measurements
     * @return Success flag (true if velocity errors are reasonable)
     */
    bool debug_velocity_comparison(
        const std::vector<Frame*>& frames,
        const std::vector<IMUData>& all_imu_data
    );

    /**
     * @brief Check if gravity alignment has been computed
     * @return True if Rgw transformation has been calculated
     */
    bool is_gravity_aligned() const { return m_gravity_aligned; }

    /**
     * @brief Get the World-to-Gravity transformation matrix
     * @return Rgw matrix (returns Identity if not yet computed)
     */
    Eigen::Matrix3f get_Rgw() const { 
        return m_gravity_aligned ? m_Rgw : Eigen::Matrix3f::Identity(); 
    }

    /**
     * @brief Get the World-to-Gravity SE(3) transformation matrix
     * @return Tgw matrix (returns Identity if not yet computed)
     */
    Eigen::Matrix4f get_Tgw() const { 
        Eigen::Matrix4f Tgw = Eigen::Matrix4f::Identity();
        if (m_gravity_aligned) {
            Tgw.block<3,3>(0,0) = m_Rgw;
        }
        return Tgw;
    }

    /**
     * @brief Transform keyframe poses and map points to gravity-aligned coordinate frame
     * @param keyframes Vector of keyframes to transform
     * @param map_points Vector of map points to transform (positions will be modified)
     * @param T_gw Output transformation matrix
     * @return Success flag
     */
    bool transform_to_gravity_frame(
        const std::vector<std::shared_ptr<Frame>>& keyframes,
        const std::vector<std::shared_ptr<MapPoint>>& map_points,
        Eigen::Matrix4f& T_gw
    );

    /**
     * @brief Verify gravity alignment by checking raw IMU accelerometer data
     * @param keyframes Vector of keyframes with IMU data to verify
     * @return True if gravity is properly aligned (accelerations average to [0,0,-9.81])
     */
    bool verify_gravity_alignment_with_imu_data(
        const std::vector<std::shared_ptr<Frame>>& keyframes
    );

  
    /**
     * @brief Apply optimized bias and velocity results to keyframes
     * @param optimized_keyframes Vector of keyframes with optimization results
     * @param optimized_biases Vector of optimized IMU biases [gyro_bias, accel_bias] pairs
     * @param optimized_velocities Vector of optimized velocities
     * @return Success flag
     */
    bool apply_optimization_results(
        const std::vector<Frame*>& optimized_keyframes,
        const std::vector<std::pair<Eigen::Vector3f, Eigen::Vector3f>>& optimized_biases,
        const std::vector<Eigen::Vector3f>& optimized_velocities
    );

    /**
     * @brief Initialize first keyframe from second keyframe (same bias and velocity)
     * @param first_keyframe First keyframe to initialize
     * @param second_keyframe Second keyframe to copy from
     * @return Success flag
     */
    bool initialize_first_keyframe_from_second(
        Frame* first_keyframe,
        const Frame* second_keyframe
    );

    /**
     * @brief Inherit bias from the last keyframe when creating new frames
     * @param new_frame New frame to set bias for
     * @param reference_keyframe Reference keyframe to inherit bias from
     * @return Success flag
     */
    bool inherit_bias_from_keyframe(
        Frame* new_frame,
        const Frame* reference_keyframe
    );



    /**
     * @brief Calculate gravity-compensated velocity (real motion velocity without gravity effects)
     * @param frame Frame to calculate velocity for
     * @param dt Time interval
     * @return Gravity-compensated velocity vector
     */
    Eigen::Vector3f get_gravity_compensated_velocity(
        const Frame* frame,
        float dt
    ) const;

    /**
     * @brief Update preintegration data with new optimized bias
     * @param frames Vector of frames with preintegration data
     * @param optimized_biases Vector of optimized bias pairs [gyro_bias, accel_bias]
     * @return Success flag
     */
    bool update_preintegrations_with_optimized_bias(
        const std::vector<Frame*>& frames,
        const std::vector<std::pair<Eigen::Vector3f, Eigen::Vector3f>>& optimized_biases
    );

    /**
     * @brief Rodrigues formula for rotation
     * @param omega Rotation vector
     * @return Rotation matrix
     */
    Eigen::Matrix3f rodrigues(const Eigen::Vector3f& omega) const;

private:
    // Current bias estimates
    Eigen::Vector3f m_gyro_bias;
    Eigen::Vector3f m_accel_bias;
    
    // Gravity vector (world frame)
    Eigen::Vector3f m_gravity;
    
    // Gravity alignment transformation
    Eigen::Matrix3f m_Rgw;          // World-to-Gravity rotation matrix
    bool m_gravity_aligned;         // Flag indicating if gravity alignment has been computed
    
    // IMU noise parameters
    float m_gyro_noise;         // Gyroscope noise density
    float m_accel_noise;        // Accelerometer noise density
    float m_gyro_bias_noise;    // Gyroscope bias random walk
    float m_accel_bias_noise;   // Accelerometer bias random walk
    
    // State
    bool m_initialized;
    
    /**
     * @brief Integrate single IMU measurement
     * @param preint Preintegration object to update
     * @param imu IMU measurement
     * @param dt Time step
     */
    void integrate_measurement(
        std::shared_ptr<IMUPreintegration> preint,
        const IMUData& imu,
        float dt
    );
    
    /**
     * @brief Integrate single IMU measurement with gravity compensation
     * @param preint Preintegration object to update
     * @param imu IMU measurement
     * @param dt Time step
     * @param R_wb World-to-body rotation matrix for gravity compensation
     */
    void integrate_measurement_with_gravity(
        std::shared_ptr<IMUPreintegration> preint,
        const IMUData& imu,
        float dt,
        const Eigen::Matrix3f& R_wb
    );
    
    /**
     * @brief Update covariance during integration
     * @param preint Preintegration object
     * @param dt Time step
     */
    void update_covariance(
        std::shared_ptr<IMUPreintegration> preint,
        float dt
    );
    
    /**
     * @brief Skew-symmetric matrix
     * @param v Input vector
     * @return Skew-symmetric matrix
     */
    Eigen::Matrix3f skew_symmetric(const Eigen::Vector3f& v) const;
    
    /**
     * @brief Right Jacobian for SO(3)
     * @param omega Rotation vector
     * @return Right Jacobian matrix
     */
    Eigen::Matrix3f right_jacobian(const Eigen::Vector3f& omega) const;
    
    /**
     * @brief Calculate body frame velocity between two frames
     * @param frame1 First frame
     * @param frame2 Second frame
     * @return Body frame linear velocity
     */
    Eigen::Vector3f calculate_body_velocity(Frame* frame1, Frame* frame2) const;
};

} // namespace lightweight_vio
