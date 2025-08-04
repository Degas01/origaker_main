"""
Task 6.3: Reward Terms Time Series Plotting
Plots progress, energy penalty, and jerk penalty over time.

Save as: origaker_sim/src/analysis/plot_reward_terms.py
Run from: origaker_main directory
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
from pathlib import Path

def setup_plotting():
    """Setup matplotlib for better plots."""
    plt.style.use('default')
    plt.rcParams['figure.figsize'] = (12, 8)
    plt.rcParams['font.size'] = 10
    plt.rcParams['lines.linewidth'] = 1.5
    plt.rcParams['grid.alpha'] = 0.3

def plot_reward_terms():
    """Load validation data and create time series plots."""
    
    print("📊 Task 6.3: Plotting Reward Terms Time Series")
    print("=" * 50)
    
    # Setup plotting
    setup_plotting()
    
    # Load data
    data_file = Path("data/reward_terms_validation.csv")
    
    if not data_file.exists():
        print(f"❌ Data file not found: {data_file}")
        print("   Run validation script first:")
        print("   python origaker_sim/src/analysis/validate_reward_terms.py")
        return False
    
    print(f"✅ Loading data from: {data_file}")
    df = pd.read_csv(data_file)
    
    print(f"   Data shape: {df.shape}")
    print(f"   Time range: {df['time'].min():.2f} - {df['time'].max():.2f} seconds")
    
    # Create output directory
    output_dir = Path("data/analysis")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create main time series plot
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Reward Terms Time Series Analysis', fontsize=16, fontweight='bold')
    
    # Plot 1: All reward components together
    ax1 = axes[0, 0]
    ax1.plot(df['time'], df['progress'], label='Progress (d_x)', color='green', alpha=0.8)
    ax1.plot(df['time'], -df['energy'], label='Energy Penalty', color='red', alpha=0.8)
    ax1.plot(df['time'], -df['jerk'], label='Jerk Penalty', color='blue', alpha=0.8)
    ax1.set_xlabel('Time (s)')
    ax1.set_ylabel('Term Value')
    ax1.set_title('All Reward Components')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Total reward
    ax2 = axes[0, 1]
    ax2.plot(df['time'], df['total_reward'], label='Total Reward', color='purple', alpha=0.8)
    ax2.set_xlabel('Time (s)')
    ax2.set_ylabel('Total Reward')
    ax2.set_title('Total Reward Over Time')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Individual components (separate scales)
    ax3 = axes[1, 0]
    
    # Use separate y-axes for different scales
    ax3_twin1 = ax3.twinx()
    ax3_twin2 = ax3.twinx()
    
    # Offset the third y-axis
    ax3_twin2.spines['right'].set_position(('outward', 60))
    
    p1 = ax3.plot(df['time'], df['progress'], label='Progress', color='green', linewidth=2)
    p2 = ax3_twin1.plot(df['time'], df['energy'], label='Energy Cost', color='red', linewidth=2)
    p3 = ax3_twin2.plot(df['time'], df['jerk'], label='Jerk Penalty', color='blue', linewidth=2)
    
    ax3.set_xlabel('Time (s)')
    ax3.set_ylabel('Progress', color='green')
    ax3_twin1.set_ylabel('Energy Cost', color='red')
    ax3_twin2.set_ylabel('Jerk Penalty', color='blue')
    ax3.set_title('Individual Components (Separate Scales)')
    
    # Color the y-axis labels
    ax3.tick_params(axis='y', labelcolor='green')
    ax3_twin1.tick_params(axis='y', labelcolor='red')
    ax3_twin2.tick_params(axis='y', labelcolor='blue')
    
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Action magnitude reference
    ax4 = axes[1, 1]
    if 'action_norm' in df.columns:
        ax4.plot(df['time'], df['action_norm'], label='Action Magnitude', color='orange', alpha=0.8)
        ax4.set_xlabel('Time (s)')
        ax4.set_ylabel('Action Norm')
        ax4.set_title('Action Magnitude (Reference)')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
    else:
        ax4.text(0.5, 0.5, 'Action data\nnot available', ha='center', va='center', 
                transform=ax4.transAxes, fontsize=12)
        ax4.set_title('Action Magnitude (Not Available)')
    
    plt.tight_layout()
    
    # Save the main plot
    main_plot_file = output_dir / "reward_terms_timeseries.png"
    plt.savefig(main_plot_file, dpi=300, bbox_inches='tight')
    print(f"✅ Main plot saved: {main_plot_file}")
    
    # Create additional detailed plots
    plt.figure(figsize=(12, 6))
    
    # Subplot 1: Progress component
    plt.subplot(1, 3, 1)
    plt.plot(df['time'], df['progress'], color='green', linewidth=2)
    plt.xlabel('Time (s)')
    plt.ylabel('Progress (d_x)')
    plt.title('Progress Component')
    plt.grid(True, alpha=0.3)
    
    # Subplot 2: Energy cost
    plt.subplot(1, 3, 2)
    plt.plot(df['time'], df['energy'], color='red', linewidth=2)
    plt.xlabel('Time (s)')
    plt.ylabel('Energy Cost')
    plt.title('Energy Cost Component')
    plt.grid(True, alpha=0.3)
    
    # Subplot 3: Jerk penalty
    plt.subplot(1, 3, 3)
    plt.plot(df['time'], df['jerk'], color='blue', linewidth=2)
    plt.xlabel('Time (s)')
    plt.ylabel('Jerk Penalty')
    plt.title('Jerk Penalty Component')
    plt.grid(True, alpha=0.3)
    
    plt.suptitle('Individual Reward Components (Detailed View)', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    # Save detailed plot
    detailed_plot_file = output_dir / "reward_components_detailed.png"
    plt.savefig(detailed_plot_file, dpi=300, bbox_inches='tight')
    print(f"✅ Detailed plot saved: {detailed_plot_file}")
    
    # Statistical analysis
    print(f"\n📈 Statistical Analysis:")
    print(f"   Progress: mean={df['progress'].mean():.4f}, std={df['progress'].std():.4f}")
    print(f"   Energy:   mean={df['energy'].mean():.4f}, std={df['energy'].std():.4f}")
    print(f"   Jerk:     mean={df['jerk'].mean():.4f}, std={df['jerk'].std():.4f}")
    print(f"   Total:    mean={df['total_reward'].mean():.4f}, std={df['total_reward'].std():.4f}")
    
    # Variation analysis
    print(f"\n🔍 Non-Zero Contribution Analysis:")
    
    progress_range = df['progress'].max() - df['progress'].min()
    energy_range = df['energy'].max() - df['energy'].min()
    jerk_range = df['jerk'].max() - df['jerk'].min()
    
    print(f"   Progress range: {progress_range:.6f}")
    print(f"   Energy range:   {energy_range:.6f}")
    print(f"   Jerk range:     {jerk_range:.6f}")
    
    # Check for non-trivial contributions
    checks = []
    
    if progress_range > 1e-4:
        checks.append("✅ Progress shows non-trivial variation")
    else:
        checks.append("⚠️  Progress shows minimal variation")
    
    if energy_range > 1e-4:
        checks.append("✅ Energy shows non-trivial variation")  
    else:
        checks.append("⚠️  Energy shows minimal variation")
    
    if jerk_range > 0.1:
        checks.append("✅ Jerk shows non-trivial variation")
    else:
        checks.append("⚠️  Jerk shows minimal variation")
    
    print(f"\n📋 Validation Results:")
    for check in checks:
        print(f"   {check}")
    
    # Create summary table
    summary_data = {
        'Component': ['Progress', 'Energy', 'Jerk', 'Total'],
        'Mean': [df['progress'].mean(), df['energy'].mean(), df['jerk'].mean(), df['total_reward'].mean()],
        'Std': [df['progress'].std(), df['energy'].std(), df['jerk'].std(), df['total_reward'].std()],
        'Min': [df['progress'].min(), df['energy'].min(), df['jerk'].min(), df['total_reward'].min()],
        'Max': [df['progress'].max(), df['energy'].max(), df['jerk'].max(), df['total_reward'].max()],
        'Range': [progress_range, energy_range, jerk_range, 
                 df['total_reward'].max() - df['total_reward'].min()]
    }
    
    summary_df = pd.DataFrame(summary_data)
    summary_file = output_dir / "reward_terms_summary.csv"
    summary_df.to_csv(summary_file, index=False)
    print(f"✅ Summary statistics saved: {summary_file}")
    
    print(f"\n🎯 Task 6.3 Complete!")
    print(f"   Main plot: {main_plot_file}")
    print(f"   Detailed plot: {detailed_plot_file}")
    print(f"   Summary stats: {summary_file}")
    
    return True

if __name__ == "__main__":
    success = plot_reward_terms()
    if success:
        print(f"\n🎉 Plotting completed successfully!")
        print(f"   Check the data/analysis/ directory for output files.")
    else:
        print(f"\n❌ Plotting failed.")
        print(f"   Make sure validation data exists.")