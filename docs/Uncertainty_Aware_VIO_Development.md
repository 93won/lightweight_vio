# Uncertainty-Aware VIO Development Documentation

## Overview

이 문서는 Lightweight VIO 시스템에 **불확실성 기반 가중치 최적화**를 도입하는 개발 과정과 향후 작업 방향을 정리합니다.

---

## 1. 현재 구현 상태 (Phase 1: Multi-View Consensus Uncertainty)

### 1.1 핵심 개념

각 MapPoint의 3D 위치 불확실성을 **다중 뷰에서 삼각측량된 위치들의 일관성(consensus)**으로 추정합니다.

```
                    ┌─────────────────────────────────────────────┐
                    │           Multi-View Triangulation          │
                    │                                             │
                    │   View 1 ──► P₁ = (x₁, y₁, z₁)             │
                    │   View 2 ──► P₂ = (x₂, y₂, z₂)             │
                    │   View 3 ──► P₃ = (x₃, y₃, z₃)             │
                    │        ...                                  │
                    │   View N ──► Pₙ = (xₙ, yₙ, zₙ)             │
                    │                                             │
                    │   Covariance Matrix Σ = Cov(P₁...Pₙ)       │
                    │                                             │
                    │   Low variance → High weight (reliable)     │
                    │   High variance → Low weight (uncertain)    │
                    └─────────────────────────────────────────────┘
```

### 1.2 구현 위치

| 파일 | 함수 | 설명 |
|------|------|------|
| `src/database/MapPoint.cpp` | `compute_multi_view_positions()` | 각 관측에서 삼각측량된 3D 위치 계산 |
| `src/database/MapPoint.cpp` | `update_uncertainty()` | 위치들의 공분산 행렬 계산 및 저장 |
| `src/database/MapPoint.cpp` | `transform_uncertainty_world_to_pixel()` | 3D 불확실성을 2D 픽셀 불확실성으로 변환 |
| `src/processing/Optimizer.cpp` | `create_information_from_uncertainty_propagation()` | 불확실성 → Information Matrix 생성 |

### 1.3 데이터 흐름

```
1. 각 프레임에서 stereo 매칭으로 depth 계산
   └─► Frame::set_depth(feature_idx, depth)

2. MapPoint가 관측될 때마다 다중 뷰 위치 수집
   └─► MapPoint::compute_multi_view_positions()
       - 각 관측 프레임의 depth + camera pose로 world 좌표 계산
       - 결과: vector<Eigen::Vector3f> positions

3. 위치들로부터 공분산 행렬 계산
   └─► MapPoint::update_uncertainty()
       - Σ = (1/N) Σᵢ (Pᵢ - P̄)(Pᵢ - P̄)ᵀ
       - Eigenvalue clipping: [min_eigenvalue, max_eigenvalue]
       - 결과: 3x3 world uncertainty matrix

4. 최적화 시 pixel-space information matrix로 변환
   └─► create_information_from_uncertainty_propagation()
       - World uncertainty → Pixel uncertainty via Jacobian
       - Ω = Σ_pixel⁻¹ + I (numerical stability)
```

### 1.4 설정 파라미터 (config/euroc_vio.yaml)

```yaml
uncertainty:
  enable: 1                  # 불확실성 가중치 활성화
  max_eigenvalue: 1e-1       # 공분산 행렬 최대 고유값 (너무 불확실한 점 제한)
  min_eigenvalue: 1e-5       # 공분산 행렬 최소 고유값 (수치 안정성)
```

### 1.5 성능 결과 (EuRoC Dataset - 11 Sequences)

**Baseline 측정일: 2025년 12월 11일**

#### Baseline 결과 (Identity Weight, uncertainty 비활성화)

| Sequence | Rotation RMSE (°) | Translation RMSE (m) |
|----------|-------------------|----------------------|
| MH_01_easy | 0.0521 | 0.002113 |
| MH_02_easy | 0.0555 | 0.002083 |
| MH_03_medium | 0.0684 | 0.004463 |
| MH_04_difficult | 0.0709 | 0.005984 |
| MH_05_difficult | 0.0680 | 0.004589 |
| V1_01_easy | 0.0653 | 0.003041 |
| V1_02_medium | 0.1118 | 0.005514 |
| V1_03_difficult | 0.1187 | 0.005815 |
| V2_01_easy | 0.1064 | 0.002222 |
| V2_02_medium | 0.0949 | 0.004052 |
| V2_03_difficult | 0.3697 | 0.021868 |
| **Average** | **0.1074** | **0.005613** |

#### Phase 1 결과 (Multi-view Consensus Uncertainty)

| Sequence | Rotation RMSE (°) | Translation RMSE (m) | Rot 개선율 | Trans 개선율 |
|----------|-------------------|----------------------|------------|--------------|
| MH_01_easy | 0.0310 | 0.001512 | 40.5% | 28.4% |
| MH_02_easy | 0.0355 | 0.001372 | 36.0% | 34.1% |
| MH_03_medium | 0.0465 | 0.003728 | 32.0% | 16.5% |
| MH_04_difficult | 0.0440 | 0.004847 | 37.9% | 19.0% |
| MH_05_difficult | 0.0405 | 0.003537 | 40.4% | 22.9% |
| V1_01_easy | 0.0501 | 0.002782 | 23.3% | 8.5% |
| V1_02_medium | 0.1113 | 0.005744 | 0.4% | -4.2% |
| V1_03_difficult | 0.1076 | 0.005252 | 9.4% | 9.7% |
| V2_01_easy | 0.0605 | 0.001681 | 43.1% | 24.3% |
| V2_02_medium | 0.1008 | 0.004235 | -6.2% | -4.5% |
| V2_03_difficult | 0.3342 | 0.023216 | 9.6% | -6.2% |
| **Average** | **0.0875** | **0.005264** | **18.5%** | **6.2%** |

**개선율 요약:**
- Rotation RMSE: 평균 **18.5%** 개선 (0.1074° → 0.0875°)
- Translation RMSE: 평균 **6.2%** 개선 (0.005613m → 0.005264m)
- **MH 시퀀스에서 특히 효과적** (30~40% 개선)
- V1_02, V2_02, V2_03는 개선 미미 또는 악화 → 추가 튜닝 필요

---

## 2. 향후 개발 필요 사항 (Phase 2: Marginalization)

> ⚠️ **중요:** Marginalization은 현재 **구현되어 있지 않습니다.** 처음부터 구현이 필요합니다.

### 2.1 Marginalization이란?

Sliding window 최적화에서 오래된 프레임을 제거할 때, 해당 프레임이 가지고 있던 **정보를 prior constraint로 보존**하는 기법입니다.

```
                    ┌─────────────────────────────────────────────┐
                    │              Sliding Window                 │
                    │                                             │
                    │   [KF₀] ─── [KF₁] ─── [KF₂] ─── [KF₃]      │
                    │     │         │         │         │         │
                    │    MP₁      MP₂       MP₃       MP₄        │
                    │                                             │
                    │   When KF₀ is removed:                      │
                    │   ┌───────────────────────────────────────┐ │
                    │   │ Marginalization creates PRIOR on:     │ │
                    │   │ - KF₁ pose (from IMU factor KF₀→KF₁)  │ │
                    │   │ - Shared MPs (from visual factors)    │ │
                    │   └───────────────────────────────────────┘ │
                    └─────────────────────────────────────────────┘
```

### 2.2 왜 필요한가?

1. **정보 손실 방지**: 오래된 프레임 제거 시 관측 정보 유실 → drift 증가
2. **일관성 유지**: 이전 최적화 결과를 현재 최적화에 반영
3. **특히 VIO에서 중요**: IMU preintegration이 프레임 간 연결하므로 pose prior 필수

### 2.3 구현 필요 사항 (처음부터 구현 필요)

#### Step 1: 기본 클래스 구조 생성

새로 만들어야 할 파일: `src/optimization/Marginalization.h`, `src/optimization/Marginalization.cpp`

```cpp
// src/optimization/Marginalization.h

// ResidualBlockInfo: 하나의 residual block 정보를 저장
struct ResidualBlockInfo {
    ceres::CostFunction* cost_function;
    ceres::LossFunction* loss_function;
    std::vector<double*> parameter_blocks;
    std::vector<int> drop_set;  // marginalize할 parameter block 인덱스
};

// MarginalizationInfo: 전체 marginalization 정보 관리
class MarginalizationInfo {
public:
    void addResidualBlockInfo(ResidualBlockInfo* residual_block_info);
    void preMarginalize();   // Jacobian 및 residual 계산
    void marginalize();      // Schur complement 수행
    bool isValid() const;
    
    // Linearized prior 정보
    Eigen::MatrixXd getLinearizedJacobian() const;
    Eigen::VectorXd getLinearizedResidual() const;
    std::vector<double*> getKeepBlockData() const;
    std::vector<int> getKeepBlockSizes() const;
    
private:
    std::vector<ResidualBlockInfo*> m_factors;
    Eigen::MatrixXd m_linearized_jacobians;  // sqrt(H) after marginalization
    Eigen::VectorXd m_linearized_residuals;  // b vector
    std::vector<double*> m_keep_block_data;
    std::vector<int> m_keep_block_sizes;
    bool m_valid = false;
};

// MarginalizationFactor: Ceres에 추가할 prior factor
class MarginalizationFactor : public ceres::CostFunction {
public:
    MarginalizationFactor(MarginalizationInfo* marg_info);
    virtual bool Evaluate(double const* const* parameters,
                         double* residuals,
                         double** jacobians) const override;
private:
    MarginalizationInfo* m_marginalization_info;
};
```

#### Step 2: Schur Complement 구현

```cpp
// marginalize() 함수 핵심 로직

// 1. 모든 factor에서 Jacobian과 residual 수집
//    H = J^T * J,  b = J^T * r

// 2. Hessian을 marginalize/keep 블록으로 분할
//    H = | H_mm  H_mk |    b = | b_m |
//        | H_km  H_kk |        | b_k |

// 3. Schur complement 계산
//    H_prior = H_kk - H_km * H_mm^(-1) * H_mk
//    b_prior = b_k - H_km * H_mm^(-1) * b_m

// 4. H_prior = J_prior^T * J_prior (Cholesky decomposition)
//    b_prior = J_prior^T * r_prior
```

#### Step 3: Visual Factor Marginalization

```cpp
// src/processing/Optimizer.cpp에 추가

void SlidingWindowOptimizer::perform_marginalization(
    const std::vector<std::shared_ptr<Frame>>& keyframes,
    const std::vector<std::shared_ptr<MapPoint>>& map_points,
    /* ... */) 
{
    if (keyframes.empty()) return;
    
    auto* marg_info = new MarginalizationInfo();
    
    // 가장 오래된 KF (인덱스 0)를 marginalize
    int marg_kf_idx = 0;
    
    // 1. 오래된 KF에서만 관측되는 점 (exclusive) vs 다른 KF에서도 관측되는 점 (shared) 분류
    std::set<int> exclusive_mp_indices;
    std::set<int> shared_mp_indices;
    // ... 분류 로직 ...
    
    // 2. 오래된 KF의 관측 factor들을 marginalization에 추가
    for (const auto& obs : observations) {
        if (obs.keyframe_index != marg_kf_idx) continue;
        
        std::vector<double*> param_blocks = {pose_ptr, point_ptr};
        std::vector<int> drop_set = {0};  // KF pose는 항상 marginalize
        
        if (exclusive_mp_indices.count(obs.mappoint_index) > 0) {
            drop_set.push_back(1);  // Exclusive point도 marginalize
        }
        
        marg_info->addResidualBlockInfo(new ResidualBlockInfo(
            obs.cost_function, nullptr, param_blocks, drop_set));
    }
    
    // 3. Marginalization 수행
    marg_info->preMarginalize();
    marg_info->marginalize();
    
    // 4. 다음 최적화에서 사용할 수 있도록 저장
    m_last_marginalization_info = marg_info;
}
```

#### Step 4: IMU Factor Marginalization

```cpp
// VIO에서 필수: IMU factor도 marginalize해야 pose prior 생성됨

if (imu_enabled && keyframes.size() >= 2) {
    auto preint = keyframes[1]->get_imu_preintegration_from_last_keyframe();
    
    // IMU factor: KF0 -> KF1
    auto* imu_cost = new IMUFactor(preint, /* ... */);
    
    std::vector<double*> param_blocks = {
        pose_kf0, velocity_kf0, bias_kf0,    // Marginalize these
        pose_kf1, velocity_kf1, bias_kf1     // Keep these (get prior)
    };
    std::vector<int> drop_set = {0, 1, 2};  // KF0 states
    
    marg_info->addResidualBlockInfo(new ResidualBlockInfo(
        imu_cost, nullptr, param_blocks, drop_set));
}
```

#### Step 5: Prior를 다음 최적화에 추가

```cpp
// optimize_sliding_window() 함수에서

if (m_last_marginalization_info && m_last_marginalization_info->isValid()) {
    // 이전 marginalization 결과를 prior factor로 추가
    auto* marg_factor = new MarginalizationFactor(m_last_marginalization_info);
    
    // Parameter block 포인터 remapping 필요 (ID 기반)
    std::vector<double*> remapped_blocks = remap_parameter_blocks(
        m_last_marginalization_info->getKeepBlockData());
    
    problem.AddResidualBlock(marg_factor, 
                            new ceres::HuberLoss(sqrt(5.991)), 
                            remapped_blocks);
}
```

### 2.4 구현 시 주의사항

1. **Huber Loss Delta**: `sqrt(5.991)` ≈ 2.448 (chi-squared 95% threshold for 2 DOF)
   ```cpp
   ceres::LossFunction* marg_loss = new ceres::HuberLoss(m_huber_delta);
   ```

2. **Block Size Matching**: Parameter block 개수와 prior dimension 일치 확인
   ```cpp
   if (remapped_blocks.size() != expected_blocks) {
       // Prior cannot be added - blocks missing
   }
   ```

3. **ID-based Remapping**: 포인터가 변경되므로 Frame ID / MapPoint ID로 추적
   ```cpp
   std::unordered_map<double*, int> ptr_to_frame_id;
   std::unordered_map<double*, int> ptr_to_mp_id;
   ```

4. **Uncertainty Integration**: Phase 1의 불확실성을 marginalization에 반영
   ```cpp
   // In Marginalization.cpp::marginalize()
   // Scale Jacobian by uncertainty: Ji *= (1.0 / sqrt(scale))
   // Higher uncertainty → Lower weight in prior
   ```

### 2.5 테스트 전략

1. **단위 테스트**: 
   - Marginalization이 실행되는지 로그 확인
   - Prior가 추가되는지 확인
   - Prior dimension이 올바른지 확인

2. **벤치마크**: EuRoC 11개 시퀀스
   ```bash
   # Expected improvement: ~5-10% over Phase 1
   ./build/euroc_stereo config/euroc_vio.yaml /path/to/MH_01_easy
   ```

3. **비교 실험**:
   - Phase 1 only (current best)
   - Phase 1 + Visual Marginalization
   - Phase 1 + Visual + IMU Marginalization (목표)

---

## 3. 코드 구조 요약

```
lightweight_vio/
├── config/
│   └── euroc_vio.yaml          # uncertainty 파라미터
├── src/
│   ├── database/
│   │   ├── MapPoint.h/.cpp     # ✅ Multi-view uncertainty 계산 (구현됨)
│   │   └── Frame.h/.cpp        # ✅ Depth 저장/관리 (구현됨)
│   ├── processing/
│   │   ├── Optimizer.h/.cpp    # ✅ Sliding window 최적화 (구현됨)
│   │   │                       # ❌ Marginalization 호출 (미구현)
│   │   └── IMUHandler.h/.cpp   # ✅ IMU preintegration (구현됨)
│   └── optimization/
│       ├── Marginalization.h/.cpp   # ❌ 파일 없음 - 새로 생성 필요!
│       ├── Factors.h/.cpp           # ✅ Ceres cost functions (구현됨)
│       └── Parameters.h/.cpp        # ✅ Lie group parameterization (구현됨)
└── docs/
    └── Uncertainty_Aware_VIO_Development.md  # 이 문서
```

### 구현 상태 요약

| 컴포넌트 | 상태 | 설명 |
|----------|------|------|
| Multi-view Uncertainty | ✅ 완료 | MapPoint에서 다중 뷰 위치 공분산 계산 |
| Uncertainty → Weight | ✅ 완료 | 불확실성을 Information Matrix로 변환 |
| Marginalization 클래스 | ❌ 없음 | `Marginalization.h/.cpp` 새로 생성 필요 |
| Visual Factor Marg | ❌ 없음 | Optimizer에서 호출 로직 구현 필요 |
| IMU Factor Marg | ❌ 없음 | IMU factor marginalization 구현 필요 |
| Prior Factor 추가 | ❌ 없음 | 최적화에 prior 추가 로직 구현 필요 |

---

## 4. 참고 자료

1. **VINS-Mono**: Marginalization 구현 참고
   - https://github.com/HKUST-Aerial-Robotics/VINS-Mono

2. **Schur Complement**:
   - Gauss-Newton Hessian에서 marginalize할 변수 제거
   - H_keep = H_kk - H_km * H_mm^(-1) * H_mk

3. **Chi-squared Threshold**:
   - 2 DOF (pixel observation): 5.991 at 95% confidence
   - Huber delta = sqrt(5.991) ≈ 2.448

---

*Last Updated: December 17, 2025*
