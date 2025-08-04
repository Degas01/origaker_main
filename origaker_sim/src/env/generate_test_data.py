#!/usr/bin/env python3
"""
Quick script to generate test randomization data for plotting demonstration
"""

import os
import csv
import numpy as np


def generate_test_randomization_data(output_file: str = "logs/randomization_log.csv", 
                                   total_steps: int = 5000):
    """Generate synthetic randomization data for testing plots"""
    
    print(f"🎲 Generating test randomization data")
    print(f"   Output file: {output_file}")
    print(f"   Total steps: {total_steps:,}")
    
    # Create logs directory
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    # Nominal values (as specified in your task)
    nominal_friction = 0.7
    nominal_restitution = 0.1
    nominal_compliance = 50000.0
    
    # Randomization ranges (as specified)
    friction_range = 0.10    # ±10%
    restitution_range = 0.05 # ±5%
    compliance_range = 0.15  # ±15%
    
    # Generate data
    data = []
    
    for step in range(0, total_steps, 10):  # Every 10 steps for reasonable file size
        # Calculate alpha (linear annealing)
        alpha = max(0.0, 1.0 - step / total_steps)
        
        # Calculate ranges for each parameter
        # Friction (μ)
        delta_mu = friction_range * nominal_friction
        mu_min = nominal_friction - alpha * delta_mu
        mu_max = nominal_friction + alpha * delta_mu
        mu_sampled = nominal_friction + alpha * np.random.uniform(-delta_mu, delta_mu)
        
        # Restitution (e)
        delta_e = restitution_range * nominal_restitution
        e_min = nominal_restitution - alpha * delta_e
        e_max = nominal_restitution + alpha * delta_e
        e_sampled = nominal_restitution + alpha * np.random.uniform(-delta_e, delta_e)
        
        # Compliance (k)
        delta_k = compliance_range * nominal_compliance
        k_min = nominal_compliance - alpha * delta_k
        k_max = nominal_compliance + alpha * delta_k
        k_sampled = nominal_compliance + alpha * np.random.uniform(-delta_k, delta_k)
        
        # Create log entry
        entry = {
            'step': step,
            'alpha': alpha,
            'mu_min': mu_min,
            'mu_max': mu_max,
            'mu_sampled': mu_sampled,
            'e_min': e_min,
            'e_max': e_max,
            'e_sampled': e_sampled,
            'k_min': k_min,
            'k_max': k_max,
            'k_sampled': k_sampled
        }
        
        data.append(entry)
    
    # Write to CSV
    with open(output_file, 'w', newline='') as f:
        fieldnames = ['step', 'alpha', 'mu_min', 'mu_max', 'mu_sampled',
                     'e_min', 'e_max', 'e_sampled', 'k_min', 'k_max', 'k_sampled']
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(data)
    
    print(f"✓ Generated {len(data):,} randomization entries")
    print(f"✓ Saved to: {output_file}")
    print(f"✓ Alpha range: {data[0]['alpha']:.3f} → {data[-1]['alpha']:.3f}")
    
    return output_file, len(data)


if __name__ == "__main__":
    print("🎯 Test Data Generator for Domain Randomization Plots")
    
    # Generate test data
    log_file, num_entries = generate_test_randomization_data()
    
    print(f"\n📊 Now you can generate plots with:")
    print(f"   python plot_randomization.py --log_file {log_file}")
    print(f"   python plot_randomization.py --log_file {log_file} --output_dir plots/test")
    