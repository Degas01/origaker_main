# 🦾 Origaker: Autonomous Locomotion and Reconfiguration for a Metamorphic Robot

![Origaker Banner](docs/banner.png)

![License](https://img.shields.io/github/license/your-org/origaker)
![Python](https://img.shields.io/badge/Python-3.10+-blue)
![MuJoCo](https://img.shields.io/badge/MuJoCo-2.3.7-green)
![Stable-Baselines3](https://img.shields.io/badge/Stable--Baselines3-PPO-orange)
![ROS2](https://img.shields.io/badge/ROS2-Foxy-purple)

---

### 🤖 Description
**Origaker** is a reconfigurable quadruped robot developed for autonomous gait adaptation and dynamic morphology control across unstructured terrains. This repository contains the **full sim-to-real pipeline** for:

- Hybrid CPG design
- PPO-based reinforcement learning
- Calibrated MuJoCo simulation
- Serial-based Arduino control interface
- Physical validation with onboard logging

The project is structured across 12 methodical stages, covering both software and hardware implementation.

---

### 📸 Demo

![Origaker Gait Demo](docs/origaker_gait.gif)

---

### ✨ Key Features

- Hybrid **Hopf–Matsuoka CPG Oscillators** for multi-modal gait generation
- Physics-grounded **reward shaping** for energy, jerk, and stability
- **Domain randomization** for terrain, friction, mass and dynamics
- **Real-time gait perturbation recovery** & phase analysis
- Fully parametric MJCF simulation & ".urdf" → ".xml" converter
- **Live serial control** from PPO policy via ROS2–Arduino bridge
- Plotting tools: **TensorBoard**, gait overlays, contact maps, perturbation tests

---

### 🛠 Installation

#### 🐍 Python Environment
```bash
conda create -n origaker_env python=3.10
conda activate origaker_env
pip install -r requirements.txt
```

#### 💻 MuJoCo Setup
1. Download [MuJoCo 2.3.x](https://mujoco.org/)
2. Set environment variable:
```bash
export MUJOCO_PY_MUJOCO_PATH="/path/to/mujoco"
```

#### 🤖 Firmware (for physical testing)
```bash
cd origaker_firmware/
platformio run --target upload
```

---

### 🚀 Usage

#### 🔬 Train PPO Policy
```bash
python origaker_control/train_ppo.py --env Origaker-v0 --tensorboard-log logs/
```

#### 📊 Visualize Results
```bash
tensorboard --logdir=logs/
```

#### 🧪 Evaluate Gait Perturbations
```bash
python origaker_analysis/evaluate_perturbation_response.py
```

#### 🎮 Drive Real Robot (via Serial)
```bash
ros2 run origaker_serial_bridge controller_node.py
```

---

### 🧱 Project Structure

```
├── origaker_sim/             # MJCF, URDF, domain randomization
├── origaker_control/        # CPGs, PPO, reward shaping, training
├── origaker_firmware/       # Teensy/Arduino microcontroller firmware
├── origaker_analysis/       # Evaluation tools and visualization scripts
├── origaker_serial_bridge/  # ROS2 serial communication (Stage 9)
├── docs/                    # Project report, diagrams, banner, GIFs
└── README.md                # This file
```

---

### 📈 Logging & Monitoring
- TensorBoard
- `results/logs.csv` for reward trends, perturbation, energy
- `compare_gaits.py` for gait overlays
- `plot_foot_contacts.py` to visualize foot-ground impacts

---

### 🤝 Contributing
We welcome contributions! Feel free to submit issues, PRs, or questions.
Please follow our [CONTRIBUTING.md](docs/CONTRIBUTING.md) guidelines.

---

### 📜 License
This project is licensed under the [MIT License](LICENSE).

---

### 📚 References
- Tang et al., "Origaker: A Reconfigurable Metamorphic Robot for Dynamic Locomotion", *IEEE RA-L*, 2022
- Schulman et al., "Proximal Policy Optimization Algorithms", *OpenAI*, 2017
- Micro-ROS & ROS 2 Docs: [https://micro.ros.org/](https://micro.ros.org/)

---

### 🔗 Citation
> G. Masone, “Autonomous Reconfiguration and Navigation for a Metamorphic Quadruped Robot,” MSc Dissertation, King’s College London, 2025.

---

### 🌍 Contact
For academic queries, contact: `your_email@kcl.ac.uk`

---

🧠 Powered by MuJoCo + ROS 2 + Stable-Baselines3
