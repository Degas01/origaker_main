#!/usr/bin/env python3
"""
Domain Randomization Visualization Script

This script reads the randomization log CSV file and creates plots showing:
- Parameter bounds (min/max) vs training steps
- Sampled values over time
- Annealing progression for all three parameters

"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import argparse
from pathlib import Path


def load_randomization_data(csv_path: str) -> pd.DataFrame:
    """Load randomization data from CSV file"""
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Log file not found: {csv_path}")
    
    df = pd.read_csv(csv_path)
    print(f"✓ Loaded {len(df)} randomization entries from {csv_path}")
    
    # Verify required columns
    required_cols = ['step', 'alpha', 'mu_min', 'mu_max', 'mu_sampled', 
                     'e_min', 'e_max', 'e_sampled', 'k_min', 'k_max', 'k_sampled']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")
    
    return df


def plot_parameter_annealing(df: pd.DataFrame, param_name: str, param_symbol: str, 
                           param_unit: str, output_dir: str):
    """Plot annealing for a single parameter"""
    
    # Column names for this parameter
    min_col = f"{param_symbol}_min"
    max_col = f"{param_symbol}_max" 
    sampled_col = f"{param_symbol}_sampled"
    
    # Create figure
    plt.figure(figsize=(12, 8))
    
    # Plot bounds (min/max envelope)
    plt.fill_between(df['step'], df[min_col], df[max_col], 
                     alpha=0.3, color='lightblue', label=f'{param_name} Range')
    plt.plot(df['step'], df[min_col], 'b--', linewidth=2, label='Min Bound')
    plt.plot(df['step'], df[max_col], 'b--', linewidth=2, label='Max Bound')
    
    # Plot sampled values (with alpha blending for density)
    plt.scatter(df['step'], df[sampled_col], alpha=0.6, s=10, color='red', 
                label='Sampled Values', zorder=5)
    
    # Plot nominal value line
    nominal_value = df[sampled_col].iloc[0] if len(df) > 0 else 0
    if param_symbol == 'mu':
        nominal_value = 0.7  # Friction nominal
    elif param_symbol == 'e':
        nominal_value = 0.1  # Restitution nominal
    elif param_symbol == 'k':
        nominal_value = 50000.0  # Compliance nominal
        
    plt.axhline(y=nominal_value, color='green', linestyle='-', linewidth=2, 
                alpha=0.8, label=f'Nominal Value ({nominal_value})')
    
    # Formatting
    plt.xlabel('Training Step', fontsize=14)
    plt.ylabel(f'{param_name} {param_unit}', fontsize=14)
    plt.title(f'Domain Randomization: {param_name} Annealing\n'
              f'Linear annealing over {df["step"].max():,} training steps', fontsize=16)
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    
    # Format x-axis with thousands separator
    ax = plt.gca()
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{int(x):,}'))
    
    # Add alpha progression annotation
    if len(df) > 1:
        alpha_start = df['alpha'].iloc[0]
        alpha_end = df['alpha'].iloc[-1]
        plt.text(0.02, 0.98, f'α: {alpha_start:.3f} → {alpha_end:.3f}', 
                transform=ax.transAxes, fontsize=12, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # Save plot
    filename = f'randomization_{param_symbol}_{param_name.lower().replace(" ", "_")}.png'
    filepath = os.path.join(output_dir, filename)
    plt.tight_layout()
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    print(f"✓ Saved {param_name} plot: {filepath}")
    plt.close()


def plot_alpha_progression(df: pd.DataFrame, output_dir: str):
    """Plot alpha (annealing factor) progression"""
    plt.figure(figsize=(12, 6))
    
    plt.plot(df['step'], df['alpha'], 'purple', linewidth=3, label='Annealing Factor (α)')
    plt.fill_between(df['step'], 0, df['alpha'], alpha=0.3, color='purple')
    
    plt.xlabel('Training Step', fontsize=14)
    plt.ylabel('Alpha (α)', fontsize=14)
    plt.title('Domain Randomization Annealing Factor\n'
              'Linear decrease from 1.0 to 0.0 over training', fontsize=16)
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.ylim(0, 1.1)
    
    # Format x-axis
    ax = plt.gca()
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{int(x):,}'))
    
    # Add annotations
    plt.text(0.02, 0.98, 'α = 1.0: Full randomization', 
             transform=ax.transAxes, fontsize=12, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8))
    plt.text(0.98, 0.02, 'α = 0.0: No randomization', 
             transform=ax.transAxes, fontsize=12, verticalalignment='bottom',
             horizontalalignment='right',
             bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.8))
    
    # Save plot
    filepath = os.path.join(output_dir, 'randomization_alpha_progression.png')
    plt.tight_layout()
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    print(f"✓ Saved alpha progression plot: {filepath}")
    plt.close()


def plot_combined_overview(df: pd.DataFrame, output_dir: str):
    """Create combined overview plot with all parameters"""
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    
    # Alpha progression
    ax1.plot(df['step'], df['alpha'], 'purple', linewidth=2)
    ax1.fill_between(df['step'], 0, df['alpha'], alpha=0.3, color='purple')
    ax1.set_title('Annealing Factor (α)', fontsize=14)
    ax1.set_xlabel('Training Step')
    ax1.set_ylabel('Alpha')
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(0, 1.1)
    
    # Friction
    ax2.fill_between(df['step'], df['mu_min'], df['mu_max'], alpha=0.3, color='blue')
    ax2.plot(df['step'], df['mu_min'], 'b--', linewidth=1)
    ax2.plot(df['step'], df['mu_max'], 'b--', linewidth=1)
    ax2.scatter(df['step'], df['mu_sampled'], alpha=0.4, s=5, color='red')
    ax2.axhline(y=0.7, color='green', linestyle='-', alpha=0.8)
    ax2.set_title('Friction (μ)', fontsize=14)
    ax2.set_xlabel('Training Step')
    ax2.set_ylabel('Friction Coefficient')
    ax2.grid(True, alpha=0.3)
    
    # Restitution
    ax3.fill_between(df['step'], df['e_min'], df['e_max'], alpha=0.3, color='orange')
    ax3.plot(df['step'], df['e_min'], 'orange', linestyle='--', linewidth=1)
    ax3.plot(df['step'], df['e_max'], 'orange', linestyle='--', linewidth=1)
    ax3.scatter(df['step'], df['e_sampled'], alpha=0.4, s=5, color='red')
    ax3.axhline(y=0.1, color='green', linestyle='-', alpha=0.8)
    ax3.set_title('Restitution (e)', fontsize=14)
    ax3.set_xlabel('Training Step')
    ax3.set_ylabel('Restitution Coefficient')
    ax3.grid(True, alpha=0.3)
    
    # Compliance
    ax4.fill_between(df['step'], df['k_min'], df['k_max'], alpha=0.3, color='brown')
    ax4.plot(df['step'], df['k_min'], 'brown', linestyle='--', linewidth=1)
    ax4.plot(df['step'], df['k_max'], 'brown', linestyle='--', linewidth=1)
    ax4.scatter(df['step'], df['k_sampled'], alpha=0.4, s=5, color='red')
    ax4.axhline(y=50000, color='green', linestyle='-', alpha=0.8)
    ax4.set_title('Ground Compliance (k)', fontsize=14)
    ax4.set_xlabel('Training Step')
    ax4.set_ylabel('Stiffness (N/m)')
    ax4.grid(True, alpha=0.3)
    
    # Format all x-axes
    for ax in [ax1, ax2, ax3, ax4]:
        ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{int(x/1000):,}k' if x >= 1000 else f'{int(x)}'))
    
    plt.suptitle('CPG-RL Domain Randomization Overview\n'
                 'Annealed parameter sampling over training', fontsize=18)
    plt.tight_layout()
    
    # Save combined plot
    filepath = os.path.join(output_dir, 'randomization_combined_overview.png')
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    print(f"✓ Saved combined overview plot: {filepath}")
    plt.close()


def generate_summary_stats(df: pd.DataFrame, output_dir: str):
    """Generate and save summary statistics"""
    stats = {
        'Total Steps': len(df),
        'Step Range': f"{df['step'].min():,} - {df['step'].max():,}",
        'Alpha Range': f"{df['alpha'].max():.3f} - {df['alpha'].min():.3f}",
        'Friction Range': f"{df['mu_sampled'].min():.3f} - {df['mu_sampled'].max():.3f}",
        'Restitution Range': f"{df['e_sampled'].min():.3f} - {df['e_sampled'].max():.3f}",
        'Compliance Range': f"{df['k_sampled'].min():.1f} - {df['k_sampled'].max():.1f}",
        'Final Alpha': f"{df['alpha'].iloc[-1]:.3f}",
        'Annealing Complete': df['alpha'].iloc[-1] < 0.01 if len(df) > 0 else False
    }
    
    # Save stats to text file
    stats_file = os.path.join(output_dir, 'randomization_summary.txt')
    with open(stats_file, 'w') as f:
        f.write("CPG-RL Domain Randomization Summary\n")
        f.write("="*40 + "\n\n")
        for key, value in stats.items():
            f.write(f"{key}: {value}\n")
    
    print(f"✓ Saved summary statistics: {stats_file}")
    return stats


def main():
    """Main plotting function"""
    parser = argparse.ArgumentParser(description='Plot domain randomization annealing results')
    parser.add_argument('--log_file', '-l', type=str, 
                       default='logs/randomization_log.csv',
                       help='Path to randomization log CSV file')
    parser.add_argument('--output_dir', '-o', type=str,
                       default='plots/randomization',
                       help='Output directory for plots')
    parser.add_argument('--show_plots', '-s', action='store_true',
                       help='Show plots interactively')
    
    args = parser.parse_args()
    
    print(f"🎨 CPG-RL Domain Randomization Plotting Tool")
    print(f"   Log file: {args.log_file}")
    print(f"   Output directory: {args.output_dir}")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    try:
        # Load data
        df = load_randomization_data(args.log_file)
        
        if len(df) == 0:
            print("❌ No data found in log file")
            return
        
        # Generate plots
        print(f"\n📊 Generating plots...")
        
        # Individual parameter plots
        plot_parameter_annealing(df, 'Friction', 'mu', '(coefficient)', args.output_dir)
        plot_parameter_annealing(df, 'Restitution', 'e', '(coefficient)', args.output_dir)
        plot_parameter_annealing(df, 'Ground Compliance', 'k', '(N/m)', args.output_dir)
        
        # Alpha progression
        plot_alpha_progression(df, args.output_dir)
        
        # Combined overview
        plot_combined_overview(df, args.output_dir)
        
        # Summary statistics
        stats = generate_summary_stats(df, args.output_dir)
        
        print(f"\n✅ All plots generated successfully!")
        print(f"   Output directory: {args.output_dir}")
        print(f"   Total plots: 5 PNG files + 1 summary")
        print(f"   Steps analyzed: {stats['Total Steps']:,}")
        print(f"   Annealing complete: {stats['Annealing Complete']}")
        
        if args.show_plots:
            print(f"\n🖼️  Opening plots directory...")
            if os.name == 'nt':  # Windows
                os.startfile(args.output_dir)
            elif os.name == 'posix':  # macOS/Linux
                os.system(f'open "{args.output_dir}"')
    
    except Exception as e:
        print(f"❌ Error: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())