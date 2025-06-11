Origaker: Autonomous Locomotion and Reconfiguration for a Metamorphic Robot

Origaker is a research-grade quadrupedal robot platform capable of morphing its body configuration across multiple locomotion modes—mammal, reptile, and arthropod—using hybrid control strategies. This repository contains the full pipeline to simulate, train, and validate adaptive locomotion and autonomous reconfiguration using Central Pattern Generators (CPGs) and Reinforcement Learning (RL) techniques in MuJoCo and on physical hardware.

Project Overview

This project addresses the challenge of enabling autonomous decision-making for gait and morphology switching in reconfigurable legged robots. By integrating hybrid CPG models with PPO-trained RL policies, we demonstrate:

- Dynamic gait transitions across terrains

- Robust domain randomization and sim-to-real transfer

- Real-time control via serial communication

- Modular testbed validation on physical Origaker units

Core Features

1. Hybrid CPG implementation (Matsuoka + Hopf oscillators)

2. Reward shaping and PPO optimization using Stable-Baselines3

3. MuJoCo simulation with calibrated contact dynamics

4. Domain randomization with customizable terrain physics

5. Arduino-based control bridge for physical actuation

6. Gait evaluation tools: perturbation, energy, stability, path tracking

Folder Structure

├── origaker_sim/             # MJCF files, domain randomization, simulation tools
├── origaker_control/        # PPO training, reward shaping, CPG models
├── origaker_firmware/       # Arduino serial controller for actuator commands
├── origaker_analysis/       # Scripts for evaluation, plotting, perturbation tests
├── origaker_visualization/  # TensorBoard logs, gait plots, phase diagrams
├── results/                 # Logged simulation & physical results
├── docs/                    # Final report, presentation, diagrams
└── README.md                # You're here!

Requirements

- MuJoCo

- Python 3.10+

- Stable-Baselines3

- PyMuJoCo

- Arduino IDE / PlatformIO

- Teensy 4.1 (for hardware control, optional)

Results

- 22% improvement in energy efficiency over baseline CPGs

- 35% faster recovery from gait perturbations

- Stable reconfiguration across three locomotion modes

- Physically validated results using onboard sensors and synchronized data logging

License

This project is developed for academic research and follows an open-source MIT License.

Citation

If you use this project in your research, please cite:

G. Masone, “Autonomous Reconfiguration and Navigation for a Metamorphic Quadruped Robot,” MSc Dissertation, King’s College London, 2025.

