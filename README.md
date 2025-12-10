# 🦎 Origaker: Adaptive Metamorphic Legged Robot Locomotion

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyBullet](https://img.shields.io/badge/PyBullet-3.2.5-green.svg)](https://pybullet.org/)
[![Stable-Baselines3](https://img.shields.io/badge/SB3-2.0+-orange.svg)](https://stable-baselines3.readthedocs.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Paper](https://img.shields.io/badge/Paper-MSc_Thesis-red.svg)](link-to-paper)
[![King's College London](https://img.shields.io/badge/Institution-King's_College_London-blue.svg)](https://www.kcl.ac.uk/)

> **Enhancing Metamorphic Legged Robot Locomotion Using Machine Learning and Nature-Inspired Design**

*MSc Robotics Individual Project | King's College London | August 2025*

<p align="center">
  <img src="assets/origaker_hero.gif" alt="Origaker in action" width="800"/>
</p>

---

## 📋 Table of Contents

- [Project Motivation](#-project-motivation)
- [Key Features](#-key-features)
- [System Architecture](#-system-architecture)
- [Simulation Environment](#-simulation-environment)
- [Hybrid CPG Architecture](#-hybrid-cpg-architecture)
- [Reinforcement Learning Framework](#-reinforcement-learning-framework)
- [SLAM & Planning Pipeline](#-slam--planning-pipeline)
- [Morphology Reconfiguration](#-morphology-reconfiguration)
- [Results](#-results)
- [Demonstrations](#-demonstrations)
- [Installation](#-installation)
- [Usage](#-usage)
- [Project Structure](#-project-structure)
- [Citation](#-citation)
- [Future Work](#-future-work)
- [Acknowledgements](#-acknowledgements)
- [License](#-license)

---

## 🎯 Project Motivation

### The Problem

Metamorphic robots promise superior adaptability through physical reconfiguration, yet current systems face critical limitations:

<p align="center">
  <img src="assets/figures/origaker_robot.png" alt="Origaker Robot" width="600"/>
  <br>
  <em>Figure 1: The Origaker metamorphic quadruped robot platform</em>
</p>

#### **Current Limitations:**

1. **Fixed Gaits**: Pre-scripted locomotion patterns cannot adapt to dynamic terrain variations
2. **No Perception**: Lack of real-time environmental awareness and mapping capabilities
3. **No Morphological Autonomy**: Manual transitions between body configurations
4. **Dynamic Terrain Failures**: High failure rates on unstructured surfaces
5. **Limited Real-World Deployment**: Poor generalization beyond training conditions

#### **Real-World Impact:**

<table>
<tr>
<td width="50%">
<img src="assets/figures/fukushima_failure.jpg" alt="Fukushima Robot" width="100%"/>
<p align="center"><em><b>2011 Fukushima Disaster</b></em><br>Ground robots immobilized by debris due to morphology rigidity [Murphy et al., 2016]</p>
</td>
<td width="50%">
<img src="assets/figures/exomars_rover.png" alt="ExoMars Rover" width="100%"/>
<p align="center"><em><b>ExoMars Mission</b></em><br>Multiple design revisions after prototypes became stuck in soft Martian regolith [ESA, 2025]</p>
</td>
</tr>
</table>

#### **Market Need:**

According to the UN Office for Disaster Risk Reduction (2020):
- **300+ natural disasters annually** affect 200M+ people
- **Limited robotic assistance** due to terrain-accessibility issues
- **Critical need** for autonomous, adaptive ground robots in:
  - 🚨 Search & rescue operations
  - 🌍 Planetary exploration
  - 🏭 Industrial inspection
  - ⚠️ Hazardous environment navigation

### Our Solution

This project presents a **unified simulation-based framework** enabling autonomous navigation and real-time morphological adaptation through:

✅ **Bio-inspired rhythmic control** (Hybrid CPG networks)  
✅ **Adaptive learning** (PPO-based reinforcement learning)  
✅ **Environmental perception** (SLAM-based mapping)  
✅ **Intelligent planning** (A* global + DWA local)  
✅ **Dynamic reconfiguration** (Terrain-aware morphology switching)  
✅ **Robust generalization** (Domain randomization)

---

## ⚡ Key Features

### 🧬 **Hybrid CPG-RL Control**
- Combines Matsuoka + Hopf oscillators for biologically plausible gaits
- PPO agent modulates CPG parameters for terrain adaptation
- **30% faster convergence** vs. naive reward approaches

### 🗺️ **Perception-Driven Navigation**
- Real-time SLAM with depth sensor and IMU fusion
- A* global path planning + DWA local trajectory control
- **84.3% mapping accuracy** in complex environments

### 🦎 **Autonomous Morphology Adaptation**
- 4 discrete modes: Crawler, Walker, Spreader, High-Step
- Terrain-aware switching based on obstacle height, corridor width, roughness
- **22% reduction in pose variance** (stability improvement)

### 🎯 **Performance Metrics**
| Metric | Improvement |
|--------|-------------|
| **Task Success Rate** | 92% (vs 68% baseline) |
| **Cost of Transport** | ↓ 15% |
| **Pose Stability** | ↓ 22% variance |
| **Path Efficiency** | ↑ 9-17% |

### 🔄 **Robust Generalization**
- Annealed domain randomization schedule
- ±10% friction, ±5% restitution, ±15% compliance variation
- **25% improvement** in terrain traversal under perturbations

---

## 🏗️ System Architecture

<p align="center">
  <img src="assets/figures/integrated_framework.png" alt="System Architecture" width="900"/>
  <br>
  <em>Figure 4: Integrated simulation-based framework for autonomous morphological adaptation</em>
</p>

### Module Overview
```
┌─────────────────────────────────────────────────────────────┐
│                    AUTONOMY PIPELINE                         │
├─────────────────────────────────────────────────────────────┤
│                                                               │
│  ┌──────────┐    ┌──────────┐    ┌──────────────┐          │
│  │ Sensors  │───▶│   SLAM   │───▶│   Planning   │          │
│  │ (Depth,  │    │ (Point   │    │  • A* Global │          │
│  │  IMU)    │    │  Cloud,  │    │  • DWA Local │          │
│  └──────────┘    │   Grid)  │    └──────┬───────┘          │
│                  └──────────┘           │                   │
│                                          │                   │
│                       ┌──────────────────▼─────────┐        │
│                       │  Morphology Planner        │        │
│                       │  • Terrain Classification  │        │
│                       │  • Mode Selection Logic    │        │
│                       └──────────┬─────────────────┘        │
│                                  │                           │
│  ┌───────────────────────────────▼──────────────────────┐  │
│  │            Hybrid CPG-RL Controller                   │  │
│  │  ┌─────────────┐          ┌──────────────┐          │  │
│  │  │ CPG Network │◀────────▶│  PPO Agent   │          │  │
│  │  │ (Matsuoka + │          │ (Modulation) │          │  │
│  │  │    Hopf)    │          └──────────────┘          │  │
│  │  └─────────────┘                                     │  │
│  └──────────────────────────┬───────────────────────────┘  │
│                              │                               │
│                    ┌─────────▼──────────┐                   │
│                    │  PyBullet Sim      │                   │
│                    │  (Torque Control)  │                   │
│                    └────────────────────┘                   │
└─────────────────────────────────────────────────────────────┘
```

### Key Components

#### 1️⃣ **CPG Subsystem**
- **Matsuoka oscillators**: Neuron-inspired adaptation dynamics
- **Hopf oscillators**: Stable limit-cycle generation
- **Hybrid coupling**: Hopf modulates Matsuoka tonic input
- **Output**: Phase-coordinated joint trajectories

#### 2️⃣ **RL Subsystem**
- **Algorithm**: Proximal Policy Optimization (PPO)
- **Observations**: Joint states, body pose, oscillator phases
- **Actions**: CPG parameter modulation (scale, offset)
- **Reward**: Multi-objective (forward progress, energy, jerk)

#### 3️⃣ **SLAM Module**
- **Inputs**: Depth camera (640×480), IMU (100Hz)
- **Processing**: Point cloud → RANSAC ground removal → Voxel filter
- **Output**: 2D occupancy grid (0.05m resolution)
- **Update Rate**: 10Hz

#### 4️⃣ **Planning Layer**
- **Global**: A* with Euclidean heuristic + obstacle inflation
- **Local**: Dynamic Window Approach (DWA) with clearance scoring
- **Integration**: Real-time waypoint tracking

#### 5️⃣ **Morphology Planner**
- **Inputs**: Terrain features (elevation σ, corridor width, obstacle height)
- **Logic**: Rule-based classifier → mode selection
- **Execution**: Joint-space interpolation (0.5s transition time)

<p align="center">
  <img src="assets/figures/autonomy_loop.png" alt="Perception-Action Loop" width="700"/>
  <br>
  <em>Figure 10: Integrated perception-action control loop</em>
</p>

---

## 🌍 Simulation Environment

### PyBullet Configuration

<table>
<tr>
<td width="50%">
<img src="assets/figures/urdf_model_pybullet.png" alt="URDF in PyBullet" width="100%"/>
<p align="center"><em><b>Figure 11:</b> Origaker URDF model in PyBullet</em></p>
</td>
<td width="50%">

**Simulation Parameters:**
- **Physics Engine**: PyBullet 3.2.5
- **Time Step**: 1ms (1000 Hz)
- **Gravity**: -9.81 m/s²
- **Control Mode**: Torque-based
- **Solver**: Featherstone algorithm
- **Contact Model**: Soft constraints

**Model Specifications:**
- **DOF**: 12 (3 per leg)
- **Total Mass**: 8.2 kg
- **Base Dimensions**: 350×250×120 mm
- **Leg Length**: 280 mm
</td>
</tr>
</table>

### Dynamics Validation

<p align="center">
  <img src="assets/figures/dynamics_sanity_check.png" alt="Dynamics Validation" width="900"/>
  <br>
  <em>Figure 12: URDF model validation - Link mass and inertia tensor comparison against CAD reference</em>
</p>

**Validation Process:**
1. ✅ Extract mass/inertia from `getDynamicsInfo()`
2. ✅ Compare with CAD specifications
3. ✅ Enforce <10% deviation threshold
4. ✅ Correct URDF `<inertial>` tags if needed

### Domain Randomization Schedule

The annealed randomization schedule ensures robust policy generalization:
```python
r_t = r_init * (1 - t/T) + r_final * (t/T)
```

Where:
- `r_t`: Randomized parameter at step t
- `r_init`: Initial perturbation range (wide)
- `r_final`: Final range (nominal)
- `T`: Total training steps (1M)

**Randomized Parameters:**
| Parameter | Initial Range | Final Range |
|-----------|---------------|-------------|
| Friction | ±10% | ±2% |
| Restitution | ±5% | ±1% |
| Link Mass | ±8% | ±2% |
| Terrain Slope | ±15° | ±5° |
| Sensor Latency | 0-50ms | 0-10ms |

---

## 🧬 Hybrid CPG Architecture

### Mathematical Foundation

<p align="center">
  <img src="assets/figures/oscillator_equations.png" alt="CPG Equations" width="700"/>
</p>

#### **Matsuoka Oscillator**

Six coupled first-order ODEs representing mutual inhibition and adaptation:
```
ẋᵢ = -xᵢ - wᵢⱼyⱼ - βvᵢ + uᵢ    (membrane potential)
v̇ᵢ = -vᵢ + yᵢ                  (adaptation state)
yᵢ = max(0, xᵢ)                (firing rate)
```

**Parameters:**
- `wᵢⱼ`: Inhibitory connection weight
- `β`: Adaptation gain
- `uᵢ`: External tonic input ← **Hopf modulates this**

#### **Hopf Oscillator**

Two-dimensional system with stable limit cycle:
```
ẋ = (μ - x² - y²)x - ωy    (polar dynamics)
ẏ = (μ - x² - y²)y + ωx
```

**Parameters:**
- `μ`: Amplitude control
- `ω`: Angular frequency

### Phase Portrait Analysis

<p align="center">
  <img src="assets/figures/oscillator_comparison.png" alt="Oscillator Phase Portraits" width="900"/>
  <br>
  <em>Figure 6: Comparative phase portraits - Hopf (circular limit cycle), Matsuoka (convergent), and hybrid α-interpolations</em>
</p>

**Key Observations:**
- **Hopf**: Perfect circular limit cycle → stable rhythms
- **Matsuoka**: Fixed-point attractor → adaptive bursting
- **Hybrid α=0.3**: Slight spiral convergence (more Hopf-like)
- **Hybrid α=0.7**: Straight trajectories (more Matsuoka-like)

### Coupling Mechanism
```
┌─────────────┐       modulation      ┌──────────────┐
│    Hopf     │─────────────────────▶│   Matsuoka   │
│  Oscillator │      (tonic input)    │  Oscillator  │
│   (μ, ω)    │                       │  (w, β, u)   │
└─────────────┘                       └──────────────┘
       │                                      │
       │                                      │
       └──────────────────┬───────────────────┘
                          │
                   Phase-coordinated
                   joint trajectories
```

### Parameter Optimization

**Grid Search Strategy:**
- **Search Space**: 1000+ parameter combinations
- **Biological Seeding**: Based on quadruped gait data [Alexander, 2003]
- **Objective**: Pareto-optimal (energy, stability)
- **Storage**: JSON gait library for runtime retrieval

**Optimized Parameter Ranges:**
| Parameter | Range | Selected |
|-----------|-------|----------|
| Matsuoka β | 0.5-2.5 | 1.2 |
| Matsuoka wᵢⱼ | 1.0-5.0 | 2.8 |
| Hopf μ | 0.1-1.0 | 0.5 |
| Hopf ω | 1.0-10.0 | 4.2 |
| Coupling α | 0.0-1.0 | 0.6 |

---

## 🤖 Reinforcement Learning Framework

### PPO Architecture

<p align="center">
  <img src="assets/figures/rl_training_loop.png" alt="RL Training Loop" width="800"/>
  <br>
  <em>Figure 7: Adaptive hybrid RL-CPG control architecture</em>
</p>

**Network Structure:**
```
Observations (36-dim)
      │
      ├─ Joint positions (12)
      ├─ Joint velocities (12)
      ├─ Base pose (6: x,y,z,roll,pitch,yaw)
      ├─ CPG phases (4: one per leg)
      └─ Terrain features (2: slope, roughness)
      │
      ▼
┌─────────────────┐
│  Actor Network  │  256→256 (ReLU)
│  (Policy π)     │  ─────────────▶ Actions (8-dim)
└─────────────────┘                 - CPG scale (4)
                                    - CPG offset (4)
┌─────────────────┐
│ Critic Network  │  256→256 (ReLU)
│  (Value V)      │  ─────────────▶ State Value (1-dim)
└─────────────────┘
```

### Reward Function Design

Multi-objective reward shaping balances speed, efficiency, and smoothness:
```python
R = w₁·Δx - w₂·∑(τᵢ·q̇ᵢ) - w₃·‖q̈‖₂
    ↑         ↑            ↑
  Progress  Energy      Jerk
            Cost      Penalty
```

<p align="center">
  <img src="assets/figures/reward_decomposition.png" alt="Reward Components" width="800"/>
  <br>
  <em>Figure 14: Reward component analysis over full gait cycle</em>
</p>

**Component Analysis:**
| Term | Weight | Purpose | Impact |
|------|--------|---------|--------|
| Forward Progress (Δx) | w₁=1.0 | Encourage locomotion | Primary drive |
| Energy Cost (τ·q̇) | w₂=0.01 | Minimize power | 15% COT reduction |
| Jerk Penalty (‖q̈‖₂) | w₃=0.005 | Smooth motion | 22% stability ↑ |

### Training Configuration

**Hyperparameters:**
```yaml
Algorithm: PPO
Total Timesteps: 1,000,000
Learning Rate: 3e-4 (linear decay)
Batch Size: 64
n_epochs: 10
Clip Range: 0.3 → 0.1 (annealed)
GAE Lambda: 0.95
Discount (γ): 0.99
Value Coef: 0.5
Entropy Coef: 0.01
Max Grad Norm: 0.5
```

**Hardware:**
- Platform: Windows 11, Intel i7, 16GB RAM
- Training Time: ~18 hours
- Checkpoint Interval: Every 20k steps

### Learning Curves

<p align="center">
  <img src="assets/figures/training_curves.png" alt="Training Progress" width="900"/>
</p>

**Key Milestones:**
- **100k steps**: Basic forward locomotion acquired
- **300k steps**: Energy-efficient gait emerges
- **500k steps**: Stable morphology transitions
- **1M steps**: Convergence with 30% improvement vs. baseline

---

## 🗺️ SLAM & Planning Pipeline

### Perception Architecture

<p align="center">
  <img src="assets/figures/slam_pipeline.png" alt="SLAM Architecture" width="900"/>
  <br>
  <em>Figure 8: SLAM system - Front-end and back-end processing</em>
</p>

#### **Data Flow:**
```
Depth Camera (640×480, 30Hz)
         │
         ▼
    Point Cloud
         │
         ▼
   RANSAC Ground Removal
         │
         ▼
    Voxel Downsampling
         │
         ▼
   2D Occupancy Grid (10Hz)
         │
         ├───▶ Global Planner (A*)
         │
         └───▶ Local Planner (DWA)
```

### SLAM Visualization

<table>
<tr>
<td width="50%">
<img src="assets/figures/slam_3d_pointcloud.png" alt="3D Point Cloud" width="100%"/>
<p align="center"><em><b>(a)</b> 3D Point Cloud Reconstruction</em></p>
</td>
<td width="50%">
<img src="assets/figures/slam_2d_occupancy.png" alt="2D Occupancy Grid" width="100%"/>
<p align="center"><em><b>(b)</b> 2D Occupancy Grid Map</em></p>
</td>
</tr>
</table>

<p align="center">
  <em>Figure 15: Simulated SLAM system with multi-modal camera input</em>
</p>

### Global Path Planning (A*)

<p align="center">
  <img src="assets/figures/astar_planning.png" alt="A* Planning" width="900"/>
  <br>
  <em>Figure 16: A* global path planning in (a) simple maze and (b) corridor maze environments</em>
</p>

**Algorithm Configuration:**
- **Heuristic**: Euclidean distance
- **Obstacle Inflation**: 0.15m radius
- **Cost Function**: g(n) + h(n)
- **Resolution**: 0.05m grid cells

### Local Trajectory Control (DWA)

**Dynamic Window Approach Parameters:**
```yaml
Velocity Search Space:
  - Linear: [-0.5, 1.0] m/s
  - Angular: [-π/2, π/2] rad/s
  
Sampling:
  - dt: 0.1s
  - prediction_horizon: 1.5s
  - num_samples: 50

Scoring Weights:
  - heading: 0.4
  - clearance: 0.3
  - velocity: 0.3
```

<p align="center">
  <img src="assets/figures/dwa_trajectories.png" alt="DWA Candidates" width="700"/>
  <br>
  <em>Sampled DWA trajectories with clearance-based scoring</em>
</p>

---

## 🦎 Morphology Reconfiguration

### Discrete Locomotion Modes

<p align="center">
  <img src="assets/figures/morphology_modes.png" alt="4 Morphology Modes" width="900"/>
  <br>
  <em>Figure 17: Discrete morphological modes - (a) Crawler, (b) Walker, (c) Spreader, (d) High-Step</em>
</p>

### Mode Specifications

| Mode | Use Case | Joint Config | Energy | Stability |
|------|----------|--------------|--------|-----------|
| **Crawler** | Narrow spaces, low clearance | Legs tucked (30° from body) | Low | High |
| **Walker** | Normal terrain, standard gait | Balanced stance (60° spread) | Medium | High |
| **Spreader** | Wide obstacles, lateral stability | Wide stance (90° spread) | Medium | Very High |
| **High-Step** | Tall obstacles, rough terrain | Extended legs (45° elevation) | High | Medium |

### Terrain Classification Logic

**Decision Tree:**
```
Input: Local terrain features
  ├─ Obstacle Height > 0.12m?
  │    └─ YES → High-Step Mode
  │
  ├─ Corridor Width < 0.4m?
  │    └─ YES → Crawler Mode
  │
  ├─ Surface Roughness σ > 0.08?
  │    └─ YES → Spreader Mode
  │
  └─ ELSE → Walker Mode (default)
```

**Feature Extraction:**
```python
# From SLAM occupancy grid
elevation_variance = np.std(heightmap[local_window])
corridor_width = detect_lateral_clearance(occupancy_grid)
forward_obstacle = max_height_in_path(occupancy_grid, lookahead=1.0m)
```

### Mode Switching Timeline

<p align="center">
  <img src="assets/figures/morphology_timeline.png" alt="Mode Timeline" width="900"/>
  <br>
  <em>Figure 18: Origaker morphology timeline over 40s navigation sequence</em>
</p>

**Transition Statistics:**
- **Total Transitions**: 8 over 40s (0.2 trans/s)
- **Most Frequent**: Walker ↔ Spreader (stable terrain)
- **Strategic**: High-Step used in 2 short bursts (energy-intensive)
- **Smooth**: Zero failed transitions (kinematic continuity maintained)

### Transition Implementation

**Joint-Space Interpolation:**
```python
def interpolate_morphology(current_config, target_config, duration=0.5):
    """
    Smooth transition between morphologies using cubic interpolation
    """
    t = np.linspace(0, duration, num_steps)
    interpolated_angles = []
    
    for joint_idx in range(12):
        q_start = current_config[joint_idx]
        q_end = target_config[joint_idx]
        
        # Cubic polynomial ensures smooth velocity profile
        q_t = cubic_interpolate(q_start, q_end, t)
        interpolated_angles.append(q_t)
    
    return interpolated_angles
```

**Safety Constraints:**
- **Transition Time**: 0.5s (prevents dynamic instability)
- **Max Angular Velocity**: 2.0 rad/s
- **Kinematic Limits**: Joint angles within [−π, π]

---

## 📊 Results

### Performance Metrics Summary

<p align="center">
  <img src="assets/figures/kpi_comparison_table.png" alt="KPI Table" width="700"/>
  <br>
  <em>Table 4: Controller performance comparison across key metrics</em>
</p>

#### **Quantitative Improvements:**

| Metric | Scripted CPG | PPO-Only | **Hybrid PPO-CPG** | Improvement |
|--------|--------------|----------|-------------------|-------------|
| **Cost of Transport ↓** | 2.1 | 1.8 | **1.6** | **24% ↓** |
| **Jerk Index ↓** | 1.03 | 0.71 | **0.45** | **56% ↓** |
| **Slip Ratio ↓** | 0.21 | 0.13 | **0.09** | **57% ↓** |
| **Tracking Error ↓** | 0.12 m | 0.08 m | **0.05 m** | **58% ↓** |
| **Recovery Time ↓** | 1.8 s | 1.2 s | **0.8 s** | **56% ↓** |

### Success Rate Analysis

<p align="center">
  <img src="assets/figures/success_rate_chart.png" alt="Success Rates" width="600"/>
</p>
```
Full System (Hybrid + SLAM + Morphing):  ████████████████████ 92%
Fixed-Mode CPG Baseline:                 ████████████▌        68%
No SLAM (Oracle Map):                    ██████████████       75%
No Domain Randomization:                 ███████████████      81%
```

**Key Finding**: Integrated system achieves **36% relative improvement** over baseline.

### Energy Efficiency (COT)

<p align="center">
  <img src="assets/figures/cot_bar_chart.png" alt="Cost of Transport" width="700"/>
</p>

**Per-Mode Energy Profile:**
| Mode | Avg. Power (W) | Duration (s) | COT |
|------|----------------|--------------|-----|
| Crawler | 8.2 | 12.5 | 1.42 |
| Walker | 10.5 | 18.0 | 1.55 |
| Spreader | 11.8 | 6.5 | 1.68 |
| High-Step | **15.3** | 3.0 | **2.12** |

**Insight**: Strategic mode selection minimizes High-Step usage (high energy) to critical moments.

### Stability Analysis

<p align="center">
  <img src="assets/figures/stability_plot.png" alt="Pose Stability" width="900"/>
</p>

**Pose Variance (Roll/Pitch):**
- **Full System**: σ = 0.08 rad
- **Fixed-Mode**: σ = 0.14 rad
- **Improvement**: **43% reduction** in pose instability

### Ablation Study Heatmap

<p align="center">
  <img src="assets/figures/ablation_heatmap.png" alt="Ablation Study" width="800"/>
  <br>
  <em>Component contribution analysis across 5 terrain types</em>
</p>

**Key Contributions:**
| Component Removed | Success Rate ↓ | COT ↑ | Explanation |
|-------------------|---------------|-------|-------------|
| SLAM | -17% | +12% | Blind navigation fails obstacle avoidance |
| Morphology Switching | -14% | +8% | Fixed configuration limits versatility |
| Domain Randomization | -11% | +6% | Overfitting to training conditions |
| Hybrid CPG | -9% | +15% | Pure RL lacks rhythmic stability |

### Trajectory Following Performance

<p align="center">
  <img src="assets/figures/trajectory_following.png" alt="Path Tracking" width="900"/>
</p>

**Metrics:**
- **Path Deviation**: Mean = 0.05m, Max = 0.12m
- **Goal Reach Accuracy**: 0.03m (within tolerance)
- **Completion Time**: 38.2s (vs. 45.1s baseline)

### Integrated Dashboard

<p align="center">
  <img src="assets/figures/autonomous_dashboard.png" alt="System Dashboard" width="1000"/>
  <br>
  <em>Figure 19: Real-time autonomous navigation system visualization</em>
</p>

**Dashboard Components:**
1. **SLAM Mapping**: 84.3% coverage, real-time point cloud
2. **Terrain Classification**: Confidence levels per region
3. **Morphology Distribution**: Mode usage histogram
4. **Navigation Trajectory**: Planned vs. executed path
5. **PPO Action Selection**: Policy output distribution
6. **Performance Metrics**: Live KPI monitoring

---

## 🎬 Demonstrations

### 1. Hybrid CPG-RL Locomotion

<p align="center">
  <img src="demos/gait_locomotion.gif" alt="Gait Demo" width="600"/>
  <br>
  <em><b>Smooth, energy-efficient trot gait</b> generated by hybrid CPG-RL controller</em>
</p>

[📹 **Full Video (MP4)**](demos/gait_locomotion.mp4) | Duration: 0:30

---

### 2. Autonomous Morphology Switching

<p align="center">
  <img src="demos/morphology_switching.gif" alt="Morphology Demo" width="600"/>
  <br>
  <em><b>Real-time adaptation:</b> Walker → High-Step (obstacle) → Crawler (narrow passage)</em>
</p>

[📹 **Full Video (MP4)**](demos/morphology_switching.mp4) | Duration: 0:45

---

### 3. SLAM Reconstruction

<p align="center">
  <img src="demos/slam_reconstruction.gif" alt="SLAM Demo" width="600"/>
  <br>
  <em><b>Live mapping:</b> Depth sensor → Point cloud → Occupancy grid</em>
</p>

[📹 **Full Video (MP4)**](demos/slam_reconstruction.mp4) | Duration: 0:40

---

### 4. Maze Navigation (Full Pipeline)

<p align="center">
  <img src="demos/maze_navigation.gif" alt="Maze Demo" width="600"/>
  <br>
  <em><b>Complete autonomy:</b> SLAM → A* planning → DWA control → Goal reach</em>
</p>

[📹 **Full Video (MP4)**](demos/maze_navigation.mp4) | Duration: 1:20

---

### 5. Domain Randomization Robustness

<p align="center">
  <img src="demos/domain_randomization.gif" alt="Robustness Demo" width="600"/>
  <br>
  <em><b>Generalization test:</b> Varying friction, slopes, masses - zero retraining</em>
</p>

[📹 **Full Video (MP4)**](demos/domain_randomization.mp4) | Duration: 1:00

---

## 🚀 Installation

### Prerequisites
```bash
Python >= 3.8
CUDA 11.7+ (optional, for GPU-accelerated training)
```

### Step 1: Clone Repository
```bash
git clone https://github.com/Degas01/origaker_sources.git
cd origaker_sources
```

### Step 2: Create Virtual Environment
```bash
# Using venv
python -m venv origaker_env
source origaker_env/bin/activate  # Linux/Mac
origaker_env\Scripts\activate     # Windows

# Or using conda
conda create -n origaker python=3.8
conda activate origaker
```

### Step 3: Install Dependencies
```bash
pip install -r requirements.txt
```

**Key Dependencies:**
```txt
pybullet==3.2.5
stable-baselines3==2.0.0
torch==2.0.1
numpy==1.24.3
scipy==1.10.1
matplotlib==3.7.1
opencv-python==4.7.0
open3d==0.17.0
```

### Step 4: Verify Installation
```bash
python scripts/smoke_test.py
```

Expected output:
```
✓ PyBullet initialized
✓ Origaker URDF loaded (12 joints)
✓ Torque control enabled
✓ Smoke test passed: Simulation stable
```

---

## 💻 Usage

### Quick Start: Pre-trained Model Demo
```bash
python demo.py --mode full --terrain maze --gui
```

**Arguments:**
- `--mode`: `full` | `cpg_only` | `rl_only` | `fixed`
- `--terrain`: `flat` | `maze` | `slopes` | `obstacles` | `mixed`
- `--gui`: Launch PyBullet GUI (default: headless)

### Training from Scratch

#### 1. Train PPO Agent
```bash
python train.py \
  --total-timesteps 1000000 \
  --save-freq 20000 \
  --log-dir logs/ \
  --model-save-path models/ppo_origaker \
  --domain-randomization
```

**Monitor Training:**
```bash
tensorboard --logdir=logs/
```

#### 2. Evaluate Trained Policy
```bash
python evaluate.py \
  --model models/ppo_origaker_best.zip \
  --num-episodes 50 \
  --render
```

### Custom Terrain Generation
```bash
python scripts/generate_terrain.py \
  --type maze \
  --complexity 0.7 \
  --size 10x10 \
  --obstacles 15 \
  --save-path terrains/custom_maze.urdf
```

### SLAM Visualization
```bash
python scripts/visualize_slam.py \
  --replay-log logs/slam_episode_042.pkl \
  --show-pointcloud \
  --export-video
```

### Morphology Mode Testing
```bash
python scripts/test_morphology.py \
  --modes crawler walker spreader high_step \
  --transitions-only \
  --save-metrics results/morphology_test.csv
```

---

## 📁 Project Structure
```
origaker_sources/
│
├── 📄 README.md                    ← You are here
├── 📄 requirements.txt             ← Python dependencies
├── 📄 LICENSE                      ← MIT License
│
├── 📂 assets/                      ← Media assets
│   ├── figures/                    ← Paper figures
│   ├── demos/                      ← GIFs and videos
│   └── models/                     ← 3D models (URDF, meshes)
│
├── 📂 origaker_urdf/               ← Robot model files
│   ├── origaker.urdf               ← Main URDF description
│   ├── meshes/                     ← STL collision/visual meshes
│   └── config/                     ← Joint limits, calibration
│
├── 📂 src/                         ← Source code
│   ├── controllers/
│   │   ├── cpg_network.py          ← Hybrid Matsuoka+Hopf CPGs
│   │   ├── rl_agent.py             ← PPO policy wrapper
│   │   └── torque_controller.py    ← Low-level joint control
│   │
│   ├── perception/
│   │   ├── slam.py                 ← Point cloud SLAM
│   │   └── terrain_classifier.py   ← Feature extraction
│   │
│   ├── planning/
│   │   ├── astar_planner.py        ← Global path planning
│   │   └── dwa_controller.py       ← Local trajectory control
│   │
│   ├── morphology/
│   │   ├── mode_selector.py        ← Terrain-aware switching
│   │   └── interpolator.py         ← Smooth joint transitions
│   │
│   └── simulation/
│       ├── environment.py          ← PyBullet Gym env
│       ├── domain_randomizer.py    ← Parameter perturbations
│       └── terrain_generator.py    ← Procedural terrains
│
├── 📂 scripts/                     ← Utility scripts
│   ├── train.py                    ← PPO training pipeline
│   ├── evaluate.py                 ← Model evaluation
│   ├── demo.py                     ← Interactive demo
│   ├── smoke_test.py               ← Basic sanity checks
│   ├── visualize_slam.py           ← SLAM replay tool
│   └── generate_terrain.py         ← Custom terrain creator
│
├── 📂 configs/                     ← Configuration files
│   ├── training_config.yaml        ← PPO hyperparameters
│   ├── cpg_params.json             ← Optimized CPG library
│   └── morphology_modes.json       ← Joint configurations
│
├── 📂 logs/                        ← Training logs (TensorBoard)
├── 📂 models/                      ← Saved model checkpoints
├── 📂 results/                     ← Evaluation metrics (CSV)
├── 📂 tests/                       ← Unit tests
│
└── 📂 docs/                        ← Documentation
    ├── PAPER.pdf                   ← Full MSc thesis
    ├── ARCHITECTURE.md             ← System design details
    ├── API_REFERENCE.md            ← Code documentation
    └── TUTORIAL.ipynb              ← Jupyter tutorial notebook
```

---

## 📖 Citation

If you use this work in your research, please cite:
```bibtex
@mastersthesis{masone2025origaker,
  title={Enhancing Metamorphic Legged Robot Locomotion Using Machine Learning and Nature-Inspired Design},
  author={Masone, Giacomo Demetrio},
  year={2025},
  school={King's College London},
  type={MSc Thesis},
  department={Engineering Department},
  supervisor={Spyrakos-Papastavridis, Emmanouil}
}
```

**Related Publications:**
```bibtex
@article{tang2022origaker,
  title={Origaker: A Novel Multi-Mimicry Quadruped Robot Based on a Metamorphic Mechanism},
  author={Tang, Z. and Wang, K. and Spyrakos-Papastavridis, E. and Dai, J.S.},
  journal={Journal of Mechanisms and Robotics},
  volume={14},
  number={6},
  year={2022}
}
```

---

## 🔮 Future Work

### Short-Term Extensions

#### 1. **Sim-to-Real Transfer**
- [ ] System identification on physical Origaker platform
- [ ] Adaptive domain randomization refinement
- [ ] Real-time sensor noise characterization
- [ ] Contact dynamics calibration
- [ ] Power consumption validation

#### 2. **Vision-Based SLAM**
- [ ] RGB-D integration (currently depth-only)
- [ ] ORB feature tracking for loop closure
- [ ] Semantic segmentation for terrain classification
- [ ] Multi-modal sensor fusion (LiDAR + camera)

#### 3. **Continuous Morphology Optimization**
- [ ] Replace discrete modes with continuous joint-space optimization
- [ ] Online trajectory optimization (e.g., iLQR, DDP)
- [ ] Learned mode selection via RL (meta-learning)
- [ ] Energy-optimal configuration search

### Mid-Term Goals

#### 4. **RL-Based Morphology Switching**
- [ ] Train hierarchical policy: meta-controller selects modes
- [ ] Multi-task learning across terrain types
- [ ] Transfer learning from simulation clusters
- [ ] Curriculum learning for progressively harder terrains

#### 5. **Multi-Terrain Generalization**
- [ ] Expand test suite: sand, mud, ice, gravel, vegetation
- [ ] Deformable terrain simulation (e.g., Taichi-MPM)
- [ ] Dynamic obstacles and moving platforms
- [ ] Outdoor field trials (unstructured environments)

#### 6. **Robustness Enhancements**
- [ ] Failure recovery strategies (e.g., self-righting)
- [ ] Fault-tolerant control (leg damage scenarios)
- [ ] Battery-aware planning (energy-constrained missions)
- [ ] Communication loss resilience

### Long-Term Vision

#### 7. **Multi-Agent Collaboration**
- [ ] Fleet coordination for search & rescue
- [ ] Distributed SLAM and map merging
- [ ] Task allocation and role specialization
- [ ] Swarm behavior emergence

#### 8. **Real-World Deployment**
- [ ] King's College campus autonomous navigation trials
- [ ] Industrial inspection applications (nuclear, offshore)
- [ ] Disaster response scenario testing (UK Fire Service collaboration)
- [ ] Planetary analog missions (ESA partnership)

#### 9. **Open-Source Community**
- [ ] ROS2 integration for broader compatibility
- [ ] Web-based simulation interface (JavaScript/WebAssembly)
- [ ] Benchmarking suite for locomotion research
- [ ] Educational modules for university courses

---

## 🙏 Acknowledgements

This research was conducted at **King's College London** as part of the MSc Robotics program.

### Supervision & Mentorship
- **Prof./Dr. Emmanouil Spyrakos-Papastavridis** – Primary Supervisor  
  *For invaluable guidance, expertise, and unwavering support throughout this project*

- **Dr. Taisir Elgorashi** – Degree Committee Member  
  *For insightful feedback and scholarly input that enriched this work*

### Academic Community
- **MSc Robotics Cohort 2024-2025** – Course Colleagues  
  *For collaborative discussions, moral support, and friendship*

- **King's College London Engineering Department**  
  *For providing world-class resources, facilities, and academic environment*

### Technical Foundations
This project builds upon foundational work:
- **Origaker Platform** – Tang et al. (2022)
- **Stable-Baselines3** – Raffin et al.
- **PyBullet** – Erwin Coumans & team

### Personal Support
- **My Parents** – *Driving force behind every achievement*  
  *For their unconditional love, sacrifice, and belief in my potential*

---

## 📜 License

This project is licensed under the **MIT License** - see the [LICENSE](LICENSE) file for details.

### Third-Party Licenses
- PyBullet: Zlib License
- Stable-Baselines3: MIT License
- Open3D: MIT License

---

## 📬 Contact

**Giacomo Demetrio Masone**  
MSc Robotics, King's College London  

📧 Email: [your.email@kcl.ac.uk](mailto:your.email@kcl.ac.uk)  
🔗 LinkedIn: [linkedin.com/in/your-profile](https://linkedin.com/in/your-profile)  
🐙 GitHub: [@Degas01](https://github.com/Degas01)  
🎓 Google Scholar: [Your Scholar Profile](https://scholar.google.com)

---

<p align="center">
  <img src="assets/figures/kcl_logo.png" alt="King's College London" height="60"/>
  &nbsp;&nbsp;&nbsp;&nbsp;
  <img src="assets/figures/robotics_lab_logo.png" alt="Robotics Lab" height="60"/>
</p>

<p align="center">
  <sub>Made with ❤️ and a lot of ☕ in London, 2025</sub>
</p>

---

## 📊 Repository Statistics

![GitHub stars](https://img.shields.io/github/stars/Degas01/origaker_sources?style=social)
![GitHub forks](https://img.shields.io/github/forks/Degas01/origaker_sources?style=social)
![GitHub watchers](https://img.shields.io/github/watchers/Degas01/origaker_sources?style=social)
![GitHub commit activity](https://img.shields.io/github/commit-activity/m/Degas01/origaker_sources)
![GitHub last commit](https://img.shields.io/github/last-commit/Degas01/origaker_sources)
![GitHub repo size](https://img.shields.io/github/repo-size/Degas01/origaker_sources)

<p align="center">
  <strong>⭐ Star this repository if you found it helpful!</strong>
</p>

---

**[⬆ Back to Top](#-origaker-adaptive-metamorphic-legged-robot-locomotion)**










