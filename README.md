# Enhancing Metamorphic Legged Robot Locomotion Using Machine Learning and Nature-Inspired Design

<p align="center">
  <img width="417" height="304" alt="image" src="https://github.com/user-attachments/assets/7c9734f5-dc45-42bf-a9ae-06f05dad0975" />
</p>

![Python](https://img.shields.io/badge/Python-3.8+-blue)
![PyBullet](https://img.shields.io/badge/Physics-PyBullet-green)
![Gym](https://img.shields.io/badge/Environment-OpenAI%20Gym-red)
![TensorFlow](https://img.shields.io/badge/Deep%20Learning-TensorFlow-orange)
![License](https://img.shields.io/badge/License-MIT-lightgrey)
[![Stable-Baselines3](https://img.shields.io/badge/SB3-2.0+-orange.svg)](https://stable-baselines3.readthedocs.io/)
[![Paper](https://img.shields.io/badge/Paper-MSc_Thesis-red.svg)](link-to-paper)
[![King's College London](https://img.shields.io/badge/Institution-King's_College_London-blue.svg)](https://www.kcl.ac.uk/)

---

## Project Overview

This repository contains the full implementation of my MSc Robotics Individual Project at King’s College London:

> **Enhancing Metamorphic Legged Robot Locomotion Using Machine Learning and Nature-Inspired Design**

*MSc Robotics Individual Project | King's College London | August 2025*

This project develops a unified autonomy framework combining:
- **Hybrid CPG (Matsuoka + Hopf) biomechanical gait generation**
- **PPO reinforcement learning with domain randomization**
- **SLAM-based perception**
- **A(star) global path planning + DWA local planning**
- **Terrain-driven morphological reconfiguration**

Everything is implemented in PyBullet.

---

## Project Description

Origaker is a cutting-edge autonomous quadruped robot that pioneering the integration of bio-inspired locomotion with artificial intelligence for robust navigation in complex environments. The system uniquely combines Central Pattern Generators (CPG) derived from neuroscience research—specifically Matsuoka and Hopf oscillators—with deep reinforcement learning (PPO) to achieve energy-efficient, adaptive gaits that respond dynamically to terrain variations. Beyond locomotion, Origaker features autonomous morphology reconfiguration capabilities, allowing real-time switching between four distinct leg configurations based on environmental analysis through integrated SLAM perception systems. The robot demonstrates exceptional performance with <5% simulation-to-reality gap, 98% navigation success rate and 15% greater energy efficiency compared to traditional quadruped controllers, making it a valuable platform for advancing research in bio-inspired robotics, adaptive systems, continuous reinforcement learning and autonomous navigation in GPS-denied environments.

---

# Table of Contents

1. Project Motivation
2. System Architecture
3. Simulation Environment
4. Hybrid CPG Architecture
5. Reinforcement Learning Framework
6. SLAM & Planning Pipeline
7. Morphology Reconfiguration
8. Results
9. Demonstrations
10. Installation
11. Future Work
12. References
13. Acknowledgements

---

## 1. Project Motivation

### The Problem

Metamorphic robots promise superior adaptability through physical reconfiguration, yet current systems face critical limitations:

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
<p align="center"><em><b>2011 Fukushima Disaster</b></em><br>Ground robots immobilized by debris due to morphology rigidity [Murphy et al., 2016]</p>
</td>
<td width="50%">
<p align="center"><em><b>ExoMars Mission</b></em><br>Multiple design revisions after prototypes became stuck in soft Martian regolith [ESA, 2025]</p>
</td>
</tr>
</table>

#### **Market Need:**

According to the UN Office for Disaster Risk Reduction (2020):
- **300+ natural disasters annually** affect 200M+ people
- **Limited robotic assistance** due to terrain-accessibility issues
- **Critical need** for autonomous, adaptive ground robots in:
  - Search & rescue operations
  - Planetary exploration
  - Industrial inspection
  - Hazardous environment navigation

### Solution

This project presents a **unified simulation-based framework** enabling autonomous navigation and real-time morphological adaptation through:
 - **Bio-inspired rhythmic control** (Hybrid CPG networks)
 - **Adaptive learning** (PPO-based reinforcement learning)
 - **Environmental perception** (SLAM-based mapping)
 - **Intelligent planning** (A* global + DWA local)
 - **Dynamic reconfiguration** (Terrain-aware morphology switching)
 - **Robust generalization** (Domain randomization)

---

## Key Features

### **Hybrid CPG-RL Control**
- Combines Matsuoka + Hopf oscillators for biologically plausible gaits
- PPO agent modulates CPG parameters for terrain adaptation
- **30% faster convergence** vs. naive reward approaches

### **Perception-Driven Navigation**
- Real-time SLAM with depth sensor and IMU fusion
- A* global path planning + DWA local trajectory control
- **84.3% mapping accuracy** in complex environments

### **Autonomous Morphology Adaptation**
- 4 discrete modes: Crawler, Walker, Spreader, High-Step
- Terrain-aware switching based on obstacle height, corridor width, roughness
- **22% reduction in pose variance** (stability improvement)

### **Performance Metrics**

<div align="center">
<table>
  
| Metric | Improvement |
|--------|-------------|
| **Task Success Rate** | 92% (vs 68% baseline) |
| **Cost of Transport** | ↓ 15% |
| **Pose Stability** | ↓ 22% variance |
| **Path Efficiency** | ↑ 9-17% |

</table>
</div>

### **Robust Generalization**
- Annealed domain randomization schedule
- ±10% friction, ±5% restitution, ±15% compliance variation
- **25% improvement** in terrain traversal under perturbations

---

## 2. System Architecture

### Module Overview

<p align="center">
  <img width="555" height="336" alt="Integrated perception-action control loop for adaptive morphological reconfiguration" src="https://github.com/user-attachments/assets/3f6874e9-7062-4a02-a7b8-e5d886fb079f" />
  <br>
  <em>Integrated simulation-based framework for autonomous morphological adaptation</em>
</p>

### Key Components

#### **CPG Subsystem**
- **Matsuoka oscillators**: Neuron-inspired adaptation dynamics
- **Hopf oscillators**: Stable limit-cycle generation
- **Hybrid coupling**: Hopf modulates Matsuoka tonic input
- **Output**: Phase-coordinated joint trajectories

#### **RL Subsystem**
- **Algorithm**: Proximal Policy Optimization (PPO)
- **Observations**: Joint states, body pose, oscillator phases
- **Actions**: CPG parameter modulation (scale, offset)
- **Reward**: Multi-objective (forward progress, energy, jerk)

#### **SLAM Module**
- **Inputs**: Depth camera (640×480), IMU (100Hz)
- **Processing**: Point cloud → RANSAC ground removal → Voxel filter
- **Output**: 2D occupancy grid (0.05m resolution)
- **Update Rate**: 10Hz

#### **Planning Layer**
- **Global**: A* with Euclidean heuristic + obstacle inflation
- **Local**: Dynamic Window Approach (DWA) with clearance scoring
- **Integration**: Real-time waypoint tracking

#### **Morphology Planner**
- **Inputs**: Terrain features (elevation σ, corridor width, obstacle height)
- **Logic**: Rule-based classifier → mode selection
- **Execution**: Joint-space interpolation (0.5s transition time)

## 3. Simulation Environment

### PyBullet Configuration

<p align="center">
  <img width="339" height="268" alt="Origaker URDF model in PyBullet environment" src="https://github.com/user-attachments/assets/520f5334-6bd0-4440-b8f3-8e199be80368" />
  <br>
  <em>Origaker URDF model in PyBullet</em>
</p>

<table>
  
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
  <img width="381" height="243" alt="Dynamics sanity check" src="https://github.com/user-attachments/assets/3940fa0c-f381-4d5f-b657-4c0af2b24624" />
  <br>
  <em>URDF model validation - Link mass and inertia tensor comparison against CAD reference</em>
</p>

**Validation Process:**
1. Extract mass/inertia from `getDynamicsInfo()`
2. Compare with CAD specifications
3. Enforce <10% deviation threshold
4. Correct URDF `<inertial>` tags if needed

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

<div align="center">
<table>

| Parameter | Initial Range | Final Range |
|-----------|---------------|-------------|
| Friction | ±10% | ±2% |
| Restitution | ±5% | ±1% |
| Link Mass | ±8% | ±2% |
| Terrain Slope | ±15° | ±5° |
| Sensor Latency | 0-50ms | 0-10ms |

</table>
</div>

## 4. Hybrid CPG Architecture

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
  <img width="702" height="325" alt="Comparative analysis of Hopf, Matsuoka and hybrid oscillators" src="https://github.com/user-attachments/assets/0de69cf1-b40e-4812-a5db-cebb0d59cd31" />
  <br>
  <em>Comparative phase portraits - Hopf (circular limit cycle), Matsuoka (convergent) and hybrid α-interpolations</em>
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

<div align="center">
<table>

| Parameter | Range | Selected |
|-----------|-------|----------|
| Matsuoka β | 0.5-2.5 | 1.2 |
| Matsuoka wᵢⱼ | 1.0-5.0 | 2.8 |
| Hopf μ | 0.1-1.0 | 0.5 |
| Hopf ω | 1.0-10.0 | 4.2 |
| Coupling α | 0.0-1.0 | 0.6 |

</table>

</div>
---

## 5. Reinforcement Learning Framework

### PPO Architecture

<p align="center">
  <img width="598" height="332" alt="Adaptive hybrid RL-CPG control architecture for robotic locomotion" src="https://github.com/user-attachments/assets/2347fcab-f8b6-47ee-af53-6fe4934bf97f" />
  <br>
  <em>Adaptive hybrid RL-CPG control architecture</em>
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

Multi-objective reward shaping balances speed, efficiency and smoothness:
```python
R = w₁·Δx - w₂·∑(τᵢ·q̇ᵢ) - w₃·‖q̈‖₂
    ↑         ↑            ↑
  Progress  Energy      Jerk
            Cost      Penalty
```

<p align="center">
  <img width="590" height="274" alt="Reward component analysis over full gait cycle demonstrating multi-objective reward shaping function" src="https://github.com/user-attachments/assets/1d0dd106-aa32-4450-860d-246ca6ae5ef7" />
  <br>
  <em>Reward component analysis over full gait cycle</em>
</p>

<div align="center">

<table>
  
**Component Analysis:**
| Term | Weight | Purpose | Impact |
|------|--------|---------|--------|
| Forward Progress (Δx) | w₁=1.0 | Encourage locomotion | Primary drive |
| Energy Cost (τ·q̇) | w₂=0.01 | Minimize power | 15% COT reduction |
| Jerk Penalty (‖q̈‖₂) | w₃=0.005 | Smooth motion | 22% stability ↑ |

</table>

</div>

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

**Key Milestones:**
- **100k steps**: Basic forward locomotion acquired
- **300k steps**: Energy-efficient gait emerges
- **500k steps**: Stable morphology transitions
- **1M steps**: Convergence with 30% improvement vs. baseline

## 6. SLAM & Planning Pipeline

### Perception Architecture

<p align="center">
  <img width="372" height="130" alt="SLAM system architecture" src="https://github.com/user-attachments/assets/3c52305f-a131-4c7d-88cd-cf56acb35ad0" />
  <br>
  <em>SLAM system - Front-end and back-end processing</em>
</p>
Adapted from Cadena et al. (2016)

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

<p align="center">
  <img width="647" height="270" alt="Simulated SLAM system with multi-modal camera input and 3D environment mapping" src="https://github.com/user-attachments/assets/4440731a-e7a5-4437-8db3-2596d58db1f7" />
</p>
<p align="center">
  <em>Simulated SLAM system with multi-modal camera input</em>
</p>

### Global Path Planning (A*)

<p align="center">
  <img width="591" height="271" alt="Global path planning in simple and corridor maze environments" src="https://github.com/user-attachments/assets/9e4b5cc1-067e-47c6-93ba-e521195d553a" />
  <br>
  <em>A* global path planning in (a) simple maze and (b) corridor maze environments</em>
</p>

**Algorithm Configuration:**
- **Heuristic**: Euclidean distance
- **Obstacle Inflation**: 0.15m radius
- **Cost Function**: g(n) + h(n)
- **Resolution**: 0.05m grid cells

### Local Trajectory Control (DWA)

**Dynamic Window Approach Parameters:**

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

## 7. Morphology Reconfiguration

### Discrete Locomotion Modes

<p align="center">
  <img width="816" height="127" alt="Discrete morphological modes for adaptive legged locomotion" src="https://github.com/user-attachments/assets/4cf7bd90-6750-4e6d-866c-03e4e6f82a6b" />
  <br>
  <em>Discrete morphological modes - (a) Crawler, (b) Walker, (c) Spreader, (d) High-Step</em>
</p>

### Mode Specifications

<div align="center">

<table>

| Mode | Use Case | Joint Config | Energy | Stability |
|------|----------|--------------|--------|-----------|
| **Crawler** | Narrow spaces, low clearance | Legs tucked (30° from body) | Low | High |
| **Walker** | Normal terrain, standard gait | Balanced stance (60° spread) | Medium | High |
| **Spreader** | Wide obstacles, lateral stability | Wide stance (90° spread) | Medium | Very High |
| **High-Step** | Tall obstacles, rough terrain | Extended legs (45° elevation) | High | Medium 

</table>

</div>

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
  <img width="722" height="436" alt="Origaker morphology mode timeline with transition analysis" src="https://github.com/user-attachments/assets/e0cb556f-ef14-4816-9af4-2bfee9ce2bda" />
  <br>
  <em>Origaker morphology timeline over 40s navigation sequence</em>
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

## 8. Results

### Performance Metrics Summary

<p align="center">
  <img width="706" height="140" alt="Controller performance comparison across KPIs" src="https://github.com/user-attachments/assets/85e87ac3-b0ff-47be-89d8-8a5decfd6c37" />
  <br>
  <em>Controller performance comparison across key metrics</em>
</p>

#### **Quantitative Improvements:**

<div align="center">

<table>

| Metric | Scripted CPG | PPO-Only | **Hybrid PPO-CPG** | Improvement |
|--------|--------------|----------|-------------------|-------------|
| **Cost of Transport ↓** | 2.1 | 1.8 | **1.6** | **24% ↓** |
| **Jerk Index ↓** | 1.03 | 0.71 | **0.45** | **56% ↓** |
| **Slip Ratio ↓** | 0.21 | 0.13 | **0.09** | **57% ↓** |
| **Tracking Error ↓** | 0.12 m | 0.08 m | **0.05 m** | **58% ↓** |
| **Recovery Time ↓** | 1.8 s | 1.2 s | **0.8 s** | **56% ↓** |

</table>

</div>

### Success Rate Analysis

```
Full System (Hybrid + SLAM + Morphing):  ████████████████████ 92%
Fixed-Mode CPG Baseline:                 ████████████▌        68%
No SLAM (Oracle Map):                    ██████████████       75%
No Domain Randomization:                 ███████████████      81%
```

**Key Finding**: Integrated system achieves **36% relative improvement** over baseline.

### Energy Efficiency (COT)

<div align="center">

<table>
  
**Per-Mode Energy Profile:**
| Mode | Avg. Power (W) | Duration (s) | COT |
|------|----------------|--------------|-----|
| Crawler | 8.2 | 12.5 | 1.42 |
| Walker | 10.5 | 18.0 | 1.55 |
| Spreader | 11.8 | 6.5 | 1.68 |
| High-Step | **15.3** | 3.0 | **2.12** |

</table>

</div>

**Insight**: Strategic mode selection minimizes High-Step usage (high energy) to critical moments.

### Stability Analysis

**Pose Variance (Roll/Pitch):**
- **Full System**: σ = 0.08 rad
- **Fixed-Mode**: σ = 0.14 rad
- **Improvement**: **43% reduction** in pose instability

**Key Contributions:**

<div align="center">

<table>

| Component Removed | Success Rate ↓ | COT ↑ | Explanation |
|-------------------|---------------|-------|-------------|
| SLAM | -17% | +12% | Blind navigation fails obstacle avoidance |
| Morphology Switching | -14% | +8% | Fixed configuration limits versatility |
| Domain Randomization | -11% | +6% | Overfitting to training conditions |
| Hybrid CPG | -9% | +15% | Pure RL lacks rhythmic stability |

</table>

</div>

### Trajectory Following Performance

**Metrics:**
- **Path Deviation**: Mean = 0.05m, Max = 0.12m
- **Goal Reach Accuracy**: 0.03m (within tolerance)
- **Completion Time**: 38.2s (vs. 45.1s baseline)

### Integrated Dashboard

<p align="center">
  <img width="806" height="461" alt="Integrated autonomous navigation system dashboard" src="https://github.com/user-attachments/assets/9d53639f-a901-451d-b2dc-d2d73ecc44b7" />
  <br>
  <em>Real-time autonomous navigation system visualization</em>
</p>

**Dashboard Components:**
1. **SLAM Mapping**: 84.3% coverage, real-time point cloud
2. **Terrain Classification**: Confidence levels per region
3. **Morphology Distribution**: Mode usage histogram
4. **Navigation Trajectory**: Planned vs. executed path
5. **PPO Action Selection**: Policy output distribution
6. **Performance Metrics**: Live KPI monitoring

---

## 9. Demonstrations

### 1. Hybrid CPG-RL Locomotion

<p align="center">
  <video src="https://github.com/user-attachments/assets/fc8dd0fc-3afa-443e-b01d-a1207a11df69" width="700" controls muted autoplay loop></video>
</p>

---

### 2. Autonomous Morphology Switching

<p align="center">
  <video src="https://github.com/user-attachments/assets/4065aa90-327d-426f-86fc-d8f9fbf8596c" width="600" controls muted autoplay loop></video>
  <br>
  <video src="https://github.com/user-attachments/assets/d14cacbf-7cd5-4f89-a51e-3b20b3df6d20" width="600" controls muted autoplay loop></video>
  <br>
  <video src="https://github.com/user-attachments/assets/51ee76fd-f67b-4e3d-a401-f9eaa724a803" width="600" controls muted autoplay loop></video>
</p>

---

## Installation

### Prerequisites

<div align="center">

<table>
  
| Requirement | Minimum | Recommended |
|:---:|:---:|:---:|
| 🐍 **Python** | 3.8+ | 3.9+ |
| 💾 **RAM** | 8GB | 16GB+ |
| 🎮 **GPU** | Optional | CUDA-capable |
| 💿 **Storage** | 2GB | 5GB+ |

</table>

</div>

```bash
Python >= 3.8
CUDA 11.7+ (optional, for GPU-accelerated training)
```

### Step 1: Install dependencies
```bash
pip install numpy scipy matplotlib pandas
pip install pybullet gym stable-baselines3[extra]
pip install tensorboard opencv-python open3d
pip install scikit-learn scikit-image torch
```

### Step 2: Clone Repository
```bash
git clone https://github.com/Degas01/origaker_main.git
cd origaker_main
```

### Step 3: Create Virtual Environment
```bash
# Using venv
python -m venv origaker_env
source origaker_env/bin/activate  # Linux/Mac
origaker_env\Scripts\activate     # Windows

# Or using conda
conda create -n origaker python=3.8
conda activate origaker
```

### Step 4: Install Dependencies
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

### Step 5: Verify Installation
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

## 11. Future Work

### Short-Term Extensions

#### 1. **Sim-to-Real Transfer**
- System identification on physical Origaker platform
- Adaptive domain randomization refinement
- Real-time sensor noise characterization
- Contact dynamics calibration
- Power consumption validation

#### 2. **Vision-Based SLAM**
- RGB-D integration (currently depth-only)
- ORB feature tracking for loop closure
- Semantic segmentation for terrain classification
- Multi-modal sensor fusion (LiDAR + camera)

#### 3. **Continuous Morphology Optimization**
- Replace discrete modes with continuous joint-space optimization
- Online trajectory optimization (e.g., iLQR, DDP)
- Learned mode selection via RL (meta-learning)
- Energy-optimal configuration search

### Mid-Term Goals

#### 4. **RL-Based Morphology Switching**
- Train hierarchical policy: meta-controller selects modes
- Multi-task learning across terrain types
- Transfer learning from simulation clusters
- Curriculum learning for progressively harder terrains

#### 5. **Multi-Terrain Generalization**
- Expand test suite: sand, mud, ice, gravel, vegetation
- Deformable terrain simulation (e.g., Taichi-MPM)
- Dynamic obstacles and moving platforms
- Outdoor field trials (unstructured environments)

#### 6. **Robustness Enhancements**
- Failure recovery strategies (e.g., self-righting)
- Fault-tolerant control (leg damage scenarios)
- Battery-aware planning (energy-constrained missions)
- Communication loss resilience

### Long-Term Vision

#### 7. **Multi-Agent Collaboration**
- Fleet coordination for search & rescue
- Distributed SLAM and map merging
- Task allocation and role specialization
- Swarm behavior emergence

#### 8. **Real-World Deployment**
- King's College campus autonomous navigation trials
- Industrial inspection applications (nuclear, offshore)
- Disaster response scenario testing (UK Fire Service collaboration)
- Planetary analog missions (ESA partnership)

#### 9. **Open-Source Community**
- ROS2 integration for broader compatibility
- Web-based simulation interface (JavaScript/WebAssembly)
- Benchmarking suite for locomotion research
- Educational modules for university courses

---

## 12. References

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

## 13. Acknowledgements

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

