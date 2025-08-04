"""
Python Phase Portrait Visualization for CPG Oscillators

This script generates high-quality phase portrait visualizations using matplotlib
instead of Blender. Perfect for scientific presentations and publications.

Usage:
    python visualize_phase_portraits.py
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import sys
from matplotlib import cm
from matplotlib.colors import LinearSegmentedColormap
import seaborn as sns

# Set style for publication-quality plots
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

# Add the src directory to Python path for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.join(current_dir, 'src')
sys.path.insert(0, src_dir)

# Import oscillators
try:
    from cpg.matsuoka import MatsuokaOscillator
    from cpg.hopf import HopfOscillator
    from cpg.hybrid import HybridCPG
except ImportError:
    try:
        sys.path.insert(0, os.path.join(current_dir, 'src', 'cpg'))
        from matsuoka import MatsuokaOscillator
        from hopf import HopfOscillator
        from hybrid import HybridCPG
    except ImportError:
        print("Could not import CPG modules. Generating sample data instead.")


def generate_and_export_data():
    """Generate phase portrait data and export to CSV files."""
    print("Generating phase portrait data...")
    
    # Create output directory
    output_dir = "phase_portraits"
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Matsuoka Oscillator Data
    print("  - Generating Matsuoka oscillator data...")
    try:
        osc = MatsuokaOscillator(tau=0.3, tau_r=1.5, beta=2.0, w=2.5, u=1.2)
        osc.reset()
        osc.x1 = 0.1  # Initial perturbation
        
        dt = 0.01
        duration = 10.0
        steps = int(duration / dt)
        
        matsuoka_data = []
        for i in range(steps):
            y1, y2 = osc.step(dt)
            state = osc.get_state()
            matsuoka_data.append({
                'time': i * dt,
                'y1': y1,
                'y2': y2,
                'x1': state['x1'],
                'x2': state['x2'],
                'v1': state['v1'],
                'v2': state['v2']
            })
        
        # Save to CSV
        df = pd.DataFrame(matsuoka_data)
        df.to_csv(os.path.join(output_dir, "matsuoka_phase_portrait.csv"), index=False)
        print(f"    Saved {len(matsuoka_data)} Matsuoka data points")
        
    except Exception as e:
        print(f"    Error generating Matsuoka data: {e}")
    
    # 2. Hopf Oscillator Data
    print("  - Generating Hopf oscillator data...")
    try:
        hopf_osc = HopfOscillator(mu=1.0, omega=2*np.pi)
        hopf_osc.reset()
        
        hopf_data = []
        for i in range(steps):
            x, y = hopf_osc.step(dt)
            radius = np.sqrt(x*x + y*y)
            phase = np.arctan2(y, x)
            hopf_data.append({
                'time': i * dt,
                'x': x,
                'y': y,
                'radius': radius,
                'phase': phase
            })
        
        # Save to CSV
        df = pd.DataFrame(hopf_data)
        df.to_csv(os.path.join(output_dir, "hopf_phase_portrait.csv"), index=False)
        print(f"    Saved {len(hopf_data)} Hopf data points")
        
    except Exception as e:
        print(f"    Error generating Hopf data: {e}")
    
    # 3. Hybrid CPG Data
    print("  - Generating Hybrid CPG data...")
    try:
        config = {
            "matsuoka_params": {"tau": 0.3, "tau_r": 1.0, "beta": 2.5, "w": 2.0, "u": 1.0},
            "hopf_params": {"mu": 1.0, "omega": 2 * np.pi * 0.8},
            "num_matsuoka": 1,
            "num_hopf": 1,
            "coupling": {
                "hopf_to_matsuoka": [[0.5]],
                "matsuoka_to_matsuoka": [[0.0]],
                "hopf_to_hopf": [[0.0]],
                "matsuoka_to_hopf": [[0.1]]
            },
            "output_mapping": {
                "joint_weights": [
                    [1.0, 0.0, 0.2, 0.0],
                    [0.0, 1.0, 0.0, 0.2]
                ]
            }
        }
        
        cpg = HybridCPG(config)
        cpg.reset()
        cpg.matsuoka_oscs[0].x1 = 0.1
        
        hybrid_data = []
        for i in range(steps):
            joint_outputs = cpg.step(dt)
            states = cpg.get_oscillator_states()
            
            mat_state = states["matsuoka"][0]
            hopf_state = states["hopf"][0]
            
            hybrid_data.append({
                'time': i * dt,
                'matsuoka_y1': mat_state['y1'],
                'matsuoka_y2': mat_state['y2'],
                'hopf_x': hopf_state['x'],
                'hopf_y': hopf_state['y'],
                'joint_0': joint_outputs[0],
                'joint_1': joint_outputs[1],
                'hopf_radius': hopf_state['radius']
            })
        
        # Save to CSV
        df = pd.DataFrame(hybrid_data)
        df.to_csv(os.path.join(output_dir, "hybrid_cpg_phase_portrait.csv"), index=False)
        print(f"    Saved {len(hybrid_data)} Hybrid CPG data points")
        
    except Exception as e:
        print(f"    Error generating Hybrid data: {e}")
    
    # 4. Parameter Comparison Data
    print("  - Generating parameter comparison data...")
    try:
        param_sets = [
            (0.25, "hopf_small_amplitude.csv"),
            (1.0, "hopf_medium_amplitude.csv"),
            (4.0, "hopf_large_amplitude.csv"),
            (1.0, "hopf_slow_frequency.csv", 1*np.pi),
            (1.0, "hopf_fast_frequency.csv", 4*np.pi)
        ]
        
        for params in param_sets:
            if len(params) == 3:
                mu, filename, omega = params
            else:
                mu, filename = params
                omega = 2*np.pi
            
            param_osc = HopfOscillator(mu=mu, omega=omega)
            param_osc.reset()
            
            param_data = []
            for i in range(int(8.0 / dt)):  # 8 seconds
                x, y = param_osc.step(dt)
                radius = np.sqrt(x*x + y*y)
                param_data.append({
                    'time': i * dt,
                    'x': x,
                    'y': y,
                    'radius': radius,
                    'mu': mu,
                    'omega': omega
                })
            
            df = pd.DataFrame(param_data)
            df.to_csv(os.path.join(output_dir, filename), index=False)
        
        print(f"    Saved parameter comparison data")
        
    except Exception as e:
        print(f"    Error generating parameter data: {e}")
    
    return output_dir


def create_publication_figure(data, x_col, y_col, title, filename, 
                            figsize=(10, 8), color_by_time=True, 
                            reference_circle=None, labels=None):
    """Create a publication-quality phase portrait figure."""
    
    fig, ax = plt.subplots(figsize=figsize, dpi=300)
    
    if color_by_time and 'time' in data.columns:
        # Color trajectory by time progression
        points = ax.scatter(data[x_col], data[y_col], 
                          c=data['time'], cmap='viridis', 
                          s=1, alpha=0.7, linewidths=0)
        
        # Add colorbar
        cbar = plt.colorbar(points, ax=ax)
        cbar.set_label('Time (s)', fontsize=12)
        
        # Also plot the trajectory as a line
        ax.plot(data[x_col], data[y_col], 'k-', alpha=0.3, linewidth=0.5)
        
    else:
        # Simple line plot
        ax.plot(data[x_col], data[y_col], 'b-', linewidth=2, alpha=0.8)
    
    # Add reference circle if specified
    if reference_circle:
        circle = plt.Circle((0, 0), reference_circle, 
                          fill=False, color='red', linestyle='--', 
                          linewidth=2, alpha=0.7, label=f'Target radius = {reference_circle:.1f}')
        ax.add_patch(circle)
        ax.legend()
    
    # Mark start and end points
    if len(data) > 1:
        ax.plot(data[x_col].iloc[0], data[y_col].iloc[0], 
               'go', markersize=8, label='Start', zorder=5)
        ax.plot(data[x_col].iloc[-1], data[y_col].iloc[-1], 
               'ro', markersize=8, label='End', zorder=5)
        ax.legend()
    
    # Formatting
    ax.set_xlabel(labels[0] if labels else x_col, fontsize=14, fontweight='bold')
    ax.set_ylabel(labels[1] if labels else y_col, fontsize=14, fontweight='bold')
    ax.set_title(title, fontsize=16, fontweight='bold', pad=20)
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal', adjustable='box')
    
    # Add subtle background
    ax.set_facecolor('#f8f9fa')
    
    # Tight layout
    plt.tight_layout()
    
    # Save with high quality
    plt.savefig(filename, dpi=300, bbox_inches='tight', 
                facecolor='white', edgecolor='none')
    plt.close()
    
    print(f"  ✅ Saved: {filename}")


def visualize_matsuoka_phase_portrait(data_dir, output_dir):
    """Generate Matsuoka phase portrait visualization."""
    print("Creating Matsuoka phase portrait...")
    
    try:
        csv_path = os.path.join(data_dir, "matsuoka_phase_portrait.csv")
        data = pd.read_csv(csv_path)
        
        # Skip initial transient (first 2 seconds)
        burn_in = int(2.0 / 0.01)
        data_stable = data.iloc[burn_in:].reset_index(drop=True)
        
        # Create the visualization
        create_publication_figure(
            data_stable, 'y1', 'y2',
            'Matsuoka Oscillator Phase Portrait\nMutual Inhibition Dynamics',
            os.path.join(output_dir, 'matsuoka_phase_portrait.png'),
            labels=['Y₁ (Neuron 1 Output)', 'Y₂ (Neuron 2 Output)']
        )
        
        # Also create a time series plot
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), dpi=300)
        
        time_data = data.iloc[burn_in:]
        ax1.plot(time_data['time'], time_data['y1'], 'b-', linewidth=2, label='Y₁')
        ax1.plot(time_data['time'], time_data['y2'], 'r-', linewidth=2, label='Y₂')
        ax1.set_ylabel('Neural Output', fontweight='bold')
        ax1.set_title('Matsuoka Oscillator Time Series', fontweight='bold', fontsize=14)
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        ax2.plot(time_data['time'], time_data['x1'], 'g-', linewidth=2, label='X₁ (Activity)')
        ax2.plot(time_data['time'], time_data['x2'], 'm-', linewidth=2, label='X₂ (Activity)')
        ax2.set_xlabel('Time (s)', fontweight='bold')
        ax2.set_ylabel('Neural Activity', fontweight='bold')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'matsuoka_time_series.png'), 
                   dpi=300, bbox_inches='tight')
        plt.close()
        print("  ✅ Saved: matsuoka_time_series.png")
        
    except Exception as e:
        print(f"  ❌ Error creating Matsuoka visualization: {e}")


def visualize_hopf_phase_portrait(data_dir, output_dir):
    """Generate Hopf phase portrait visualization."""
    print("Creating Hopf phase portrait...")
    
    try:
        csv_path = os.path.join(data_dir, "hopf_phase_portrait.csv")
        data = pd.read_csv(csv_path)
        
        # Skip initial transient
        burn_in = int(2.0 / 0.01)
        data_stable = data.iloc[burn_in:].reset_index(drop=True)
        
        # Create the visualization with reference circle
        mu = 1.0  # Known parameter
        expected_radius = np.sqrt(mu)
        
        create_publication_figure(
            data_stable, 'x', 'y',
            f'Hopf Oscillator Phase Portrait\nLimit Cycle Dynamics (μ = {mu})',
            os.path.join(output_dir, 'hopf_phase_portrait.png'),
            reference_circle=expected_radius,
            labels=['X', 'Y']
        )
        
        # Create radius convergence plot
        fig, ax = plt.subplots(figsize=(10, 6), dpi=300)
        
        ax.plot(data['time'], data['radius'], 'b-', linewidth=2, label='Actual Radius')
        ax.axhline(y=expected_radius, color='r', linestyle='--', linewidth=2, 
                  label=f'Expected Radius = √μ = {expected_radius:.2f}')
        
        ax.set_xlabel('Time (s)', fontweight='bold')
        ax.set_ylabel('Radius', fontweight='bold')
        ax.set_title('Hopf Oscillator Radius Convergence', fontweight='bold', fontsize=14)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'hopf_radius_convergence.png'), 
                   dpi=300, bbox_inches='tight')
        plt.close()
        print("  ✅ Saved: hopf_radius_convergence.png")
        
    except Exception as e:
        print(f"  ❌ Error creating Hopf visualization: {e}")


def visualize_hybrid_cpg(data_dir, output_dir):
    """Generate Hybrid CPG visualization."""
    print("Creating Hybrid CPG visualization...")
    
    try:
        csv_path = os.path.join(data_dir, "hybrid_cpg_phase_portrait.csv")
        data = pd.read_csv(csv_path)
        
        # Skip initial transient
        burn_in = int(2.0 / 0.01)
        data_stable = data.iloc[burn_in:].reset_index(drop=True)
        
        # Create side-by-side phase portraits
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7), dpi=300)
        
        # Matsuoka trajectory
        points1 = ax1.scatter(data_stable['matsuoka_y1'], data_stable['matsuoka_y2'], 
                            c=data_stable['time'], cmap='Reds', s=2, alpha=0.7)
        ax1.plot(data_stable['matsuoka_y1'], data_stable['matsuoka_y2'], 
               'k-', alpha=0.3, linewidth=0.5)
        ax1.set_xlabel('Y₁ (Neuron 1)', fontweight='bold')
        ax1.set_ylabel('Y₂ (Neuron 2)', fontweight='bold')
        ax1.set_title('Matsuoka Component', fontweight='bold')
        ax1.grid(True, alpha=0.3)
        ax1.set_aspect('equal')
        
        # Hopf trajectory
        points2 = ax2.scatter(data_stable['hopf_x'], data_stable['hopf_y'], 
                            c=data_stable['time'], cmap='Blues', s=2, alpha=0.7)
        ax2.plot(data_stable['hopf_x'], data_stable['hopf_y'], 
               'k-', alpha=0.3, linewidth=0.5)
        ax2.set_xlabel('X', fontweight='bold')
        ax2.set_ylabel('Y', fontweight='bold')
        ax2.set_title('Hopf Component', fontweight='bold')
        ax2.grid(True, alpha=0.3)
        ax2.set_aspect('equal')
        
        # Add reference circle to Hopf
        circle = plt.Circle((0, 0), 1.0, fill=False, color='red', 
                          linestyle='--', linewidth=2, alpha=0.7)
        ax2.add_patch(circle)
        
        plt.suptitle('Hybrid CPG Network - Coupled Oscillators', 
                    fontsize=16, fontweight='bold', y=1.02)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'hybrid_cpg_phase_portrait.png'), 
                   dpi=300, bbox_inches='tight')
        plt.close()
        print("  ✅ Saved: hybrid_cpg_phase_portrait.png")
        
        # Create joint output plot
        fig, ax = plt.subplots(figsize=(12, 6), dpi=300)
        
        time_data = data.iloc[burn_in:]
        ax.plot(time_data['time'], time_data['joint_0'], 'b-', linewidth=2, label='Joint 0')
        ax.plot(time_data['time'], time_data['joint_1'], 'r-', linewidth=2, label='Joint 1')
        
        ax.set_xlabel('Time (s)', fontweight='bold')
        ax.set_ylabel('Joint Command', fontweight='bold')
        ax.set_title('Hybrid CPG Joint Outputs', fontweight='bold', fontsize=14)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'hybrid_cpg_joint_outputs.png'), 
                   dpi=300, bbox_inches='tight')
        plt.close()
        print("  ✅ Saved: hybrid_cpg_joint_outputs.png")
        
    except Exception as e:
        print(f"  ❌ Error creating Hybrid CPG visualization: {e}")


def visualize_parameter_comparison(data_dir, output_dir):
    """Generate parameter comparison visualization."""
    print("Creating parameter comparison visualization...")
    
    try:
        # Load parameter comparison data
        param_files = [
            ("hopf_small_amplitude.csv", "μ = 0.25", '#e74c3c'),
            ("hopf_medium_amplitude.csv", "μ = 1.0", '#3498db'),
            ("hopf_large_amplitude.csv", "μ = 4.0", '#2ecc71')
        ]
        
        fig, ax = plt.subplots(figsize=(12, 10), dpi=300)
        
        for filename, label, color in param_files:
            try:
                csv_path = os.path.join(data_dir, filename)
                data = pd.read_csv(csv_path)
                
                # Skip transient
                burn_in = int(2.0 / 0.01)
                data_stable = data.iloc[burn_in:].reset_index(drop=True)
                
                ax.plot(data_stable['x'], data_stable['y'], 
                       color=color, linewidth=2.5, alpha=0.8, label=label)
                
                # Add expected radius circle
                if 'mu' in data.columns:
                    mu = data['mu'].iloc[0]
                    expected_r = np.sqrt(mu)
                    circle = plt.Circle((0, 0), expected_r, fill=False, 
                                      color=color, linestyle='--', linewidth=1.5, alpha=0.6)
                    ax.add_patch(circle)
                
            except Exception as e:
                print(f"    Warning: Could not load {filename}: {e}")
        
        ax.set_xlabel('X', fontweight='bold', fontsize=14)
        ax.set_ylabel('Y', fontweight='bold', fontsize=14)
        ax.set_title('Hopf Oscillator Parameter Comparison\nEffect of μ on Limit Cycle Radius', 
                    fontweight='bold', fontsize=16, pad=20)
        ax.legend(fontsize=12, framealpha=0.9)
        ax.grid(True, alpha=0.3)
        ax.set_aspect('equal')
        
        # Set reasonable axis limits
        ax.set_xlim(-2.5, 2.5)
        ax.set_ylim(-2.5, 2.5)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'hopf_parameter_comparison.png'), 
                   dpi=300, bbox_inches='tight')
        plt.close()
        print("  ✅ Saved: hopf_parameter_comparison.png")
        
    except Exception as e:
        print(f"  ❌ Error creating parameter comparison: {e}")


def main():
    """Main function to generate all visualizations."""
    print("=== CPG Phase Portrait Visualization (Python) ===")
    
    # Step 1: Generate data
    data_dir = generate_and_export_data()
    
    # Step 2: Create output directory for images
    output_dir = "phase_portrait_images"
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"\nGenerating high-quality visualizations...")
    print(f"Data source: {os.path.abspath(data_dir)}")
    print(f"Output directory: {os.path.abspath(output_dir)}")
    
    # Step 3: Generate all visualizations
    visualize_matsuoka_phase_portrait(data_dir, output_dir)
    visualize_hopf_phase_portrait(data_dir, output_dir)
    visualize_hybrid_cpg(data_dir, output_dir)
    visualize_parameter_comparison(data_dir, output_dir)
    
    print(f"\n🎨 Visualization Complete!")
    print(f"📁 Images saved to: {os.path.abspath(output_dir)}")
    print(f"📊 Generated files:")
    
    # List all generated files
    if os.path.exists(output_dir):
        for file in sorted(os.listdir(output_dir)):
            if file.endswith('.png'):
                print(f"   - {file}")
    
    print(f"\n✨ Ready for presentation slides!")


if __name__ == "__main__":
    main()