#!/usr/bin/env python3
"""
Grid Search Results Analysis and Visualization

This script processes grid search results from CPG parameter sweeps,
generates heatmaps, identifies sweet-spots, and exports visualizations
for use in dynamic demos.
"""

import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from pathlib import Path
import os

class GridSearchAnalyzer:
    """Analyzes and visualizes CPG parameter grid search results."""
    
    def __init__(self, results_file="data/gaits/grid_search_results.json"):
        """
        Initialize analyzer with results file path.
        
        Args:
            results_file: Path to grid search results JSON file
        """
        self.results_file = results_file
        self.results = None
        self.df = None
        self.output_dir = Path("data/analysis")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
    def load_results(self):
        """Load grid search results from JSON file."""
        try:
            with open(self.results_file, 'r') as f:
                self.results = json.load(f)
            
            # Convert to DataFrame for easier manipulation
            self.df = pd.DataFrame(self.results)
            print(f"Loaded {len(self.df)} grid search results")
            print(f"Columns: {list(self.df.columns)}")
            
            # Display basic statistics
            print("\nBasic Statistics:")
            print(f"Frequency range: {self.df['frequency'].min():.2f} - {self.df['frequency'].max():.2f} Hz")
            print(f"Amplitude range: {self.df['amplitude'].min():.2f} - {self.df['amplitude'].max():.2f} rad")
            print(f"Energy range: {self.df['energy'].min():.4f} - {self.df['energy'].max():.4f}")
            
            if 'stability' in self.df.columns:
                print(f"Stability range: {self.df['stability'].min():.4f} - {self.df['stability'].max():.4f}")
            
        except FileNotFoundError:
            print(f"Error: Results file '{self.results_file}' not found.")
            print("Please run grid search first to generate results.")
            return False
        except json.JSONDecodeError:
            print(f"Error: Invalid JSON in results file '{self.results_file}'")
            return False
        except Exception as e:
            print(f"Error loading results: {e}")
            return False
        
        return True
    
    def generate_heatmap(self, metric='energy', save_path=None):
        """
        Generate heatmap for specified metric.
        
        Args:
            metric: Metric to visualize ('energy', 'stability', etc.)
            save_path: Custom save path, defaults to data/analysis/
        """
        if self.df is None:
            print("No data loaded. Call load_results() first.")
            return
        
        if metric not in self.df.columns:
            print(f"Error: Metric '{metric}' not found in results.")
            print(f"Available metrics: {list(self.df.columns)}")
            return
        
        # Create pivot table for heatmap
        pivot = self.df.pivot(index="amplitude", columns="frequency", values=metric)
        
        # Create figure with proper sizing
        plt.figure(figsize=(12, 8))
        
        # Generate heatmap
        im = plt.imshow(pivot, origin="lower", aspect="auto", cmap='viridis')
        
        # Add colorbar
        cbar = plt.colorbar(im, label=metric.capitalize())
        
        # Set labels and title
        plt.xlabel("Frequency (Hz)", fontsize=12)
        plt.ylabel("Amplitude (rad)", fontsize=12)
        plt.title(f"{metric.capitalize()} Heatmap - CPG Parameter Grid Search", fontsize=14)
        
        # Improve tick labels
        freq_ticks = np.linspace(0, len(pivot.columns)-1, 6)
        amp_ticks = np.linspace(0, len(pivot.index)-1, 6)
        
        plt.xticks(freq_ticks, [f"{pivot.columns[int(i)]:.1f}" for i in freq_ticks])
        plt.yticks(amp_ticks, [f"{pivot.index[int(i)]:.2f}" for i in amp_ticks])
        
        # Grid for better readability
        plt.grid(True, alpha=0.3)
        
        # Save figure
        if save_path is None:
            save_path = self.output_dir / f"{metric}_heatmap.png"
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Heatmap saved to: {save_path}")
        
        return pivot
    
    def identify_sweet_spots(self, energy_percentile=10, stability_threshold=0.1):
        """
        Identify sweet-spot parameter combinations.
        
        Args:
            energy_percentile: Bottom percentile for energy (lower is better)
            stability_threshold: Maximum stability value (lower is better)
        
        Returns:
            DataFrame with sweet-spot parameters
        """
        if self.df is None:
            print("No data loaded. Call load_results() first.")
            return None
        
        # Calculate energy threshold (bottom percentile)
        energy_threshold = np.percentile(self.df['energy'], energy_percentile)
        
        # Filter for sweet spots
        sweet_spots = self.df[
            (self.df['energy'] <= energy_threshold) &
            (self.df.get('stability', 0) <= stability_threshold)
        ].copy()
        
        print(f"\nSweet-Spot Analysis:")
        print(f"Energy threshold (bottom {energy_percentile}%): {energy_threshold:.4f}")
        print(f"Stability threshold: {stability_threshold:.4f}")
        print(f"Found {len(sweet_spots)} sweet-spot parameter combinations")
        
        if len(sweet_spots) > 0:
            print(f"\nTop 5 Sweet-Spots (lowest energy):")
            top_spots = sweet_spots.nsmallest(5, 'energy')
            for i, (_, row) in enumerate(top_spots.iterrows(), 1):
                print(f"{i}. Freq: {row['frequency']:.2f} Hz, "
                      f"Amp: {row['amplitude']:.3f} rad, "
                      f"Energy: {row['energy']:.4f}")
        
        return sweet_spots
    
    def generate_sweet_spot_heatmap(self, sweet_spots, metric='energy', save_path=None):
        """
        Generate heatmap with sweet-spots overlaid.
        
        Args:
            sweet_spots: DataFrame with sweet-spot parameters
            metric: Base metric for heatmap
            save_path: Custom save path
        """
        if self.df is None or sweet_spots is None:
            print("No data or sweet-spots available.")
            return
        
        # Generate base heatmap
        pivot = self.generate_heatmap(metric, save_path=None)
        
        # Overlay sweet-spots
        if len(sweet_spots) > 0:
            # Convert sweet-spot coordinates to heatmap indices
            freq_values = pivot.columns.values
            amp_values = pivot.index.values
            
            for _, spot in sweet_spots.iterrows():
                # Find closest indices
                freq_idx = np.argmin(np.abs(freq_values - spot['frequency']))
                amp_idx = np.argmin(np.abs(amp_values - spot['amplitude']))
                
                # Add circle marker
                circle = patches.Circle((freq_idx, amp_idx), 0.3, 
                                      facecolor='red', edgecolor='white', 
                                      linewidth=2, alpha=0.8)
                plt.gca().add_patch(circle)
        
        # Add legend
        red_patch = patches.Patch(color='red', label='Sweet-Spots')
        plt.legend(handles=[red_patch], loc='upper right')
        
        plt.title(f"{metric.capitalize()} Heatmap with Sweet-Spots", fontsize=14)
        
        # Save figure
        if save_path is None:
            save_path = self.output_dir / f"{metric}_heatmap_sweet_spots.png"
        
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Sweet-spot heatmap saved to: {save_path}")
        
        return pivot
    
    def export_for_demos(self, pivot, metric='energy'):
        """
        Export heatmap data for use in dynamic demos.
        
        Args:
            pivot: Pivot table with heatmap data
            metric: Metric name for file naming
        """
        # Save heatmap as high-resolution PNG for texture mapping
        texture_path = self.output_dir / f"{metric}_texture.png"
        plt.figure(figsize=(10, 10))
        plt.imshow(pivot, origin="lower", aspect="auto", cmap='viridis')
        plt.axis('off')  # Remove axes for clean texture
        plt.tight_layout()
        plt.savefig(texture_path, dpi=300, bbox_inches='tight', pad_inches=0)
        plt.close()
        
        print(f"Texture saved to: {texture_path}")
        
        # Create simple OBJ plane for texture mapping
        obj_path = self.output_dir / "heatmap_plane.obj"
        self.create_textured_plane(obj_path)
        
        # Export data as CSV for other applications
        csv_path = self.output_dir / f"{metric}_heatmap_data.csv"
        pivot.to_csv(csv_path)
        print(f"Heatmap data exported to: {csv_path}")
        
        # Export metadata
        metadata = {
            "metric": metric,
            "frequency_range": [float(pivot.columns.min()), float(pivot.columns.max())],
            "amplitude_range": [float(pivot.index.min()), float(pivot.index.max())],
            "value_range": [float(pivot.min().min()), float(pivot.max().max())],
            "shape": list(pivot.shape),
            "texture_file": str(texture_path.name),
            "obj_file": "heatmap_plane.obj"
        }
        
        metadata_path = self.output_dir / f"{metric}_metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"Metadata exported to: {metadata_path}")
    
    def create_textured_plane(self, obj_path):
        """Create a simple OBJ plane for texture mapping."""
        obj_content = """# Simple plane for heatmap texture
# Vertices
v -1.0 -1.0 0.0
v  1.0 -1.0 0.0
v  1.0  1.0 0.0
v -1.0  1.0 0.0

# Texture coordinates
vt 0.0 0.0
vt 1.0 0.0
vt 1.0 1.0
vt 0.0 1.0

# Normals
vn 0.0 0.0 1.0

# Faces (vertex/texture/normal)
f 1/1/1 2/2/1 3/3/1
f 1/1/1 3/3/1 4/4/1
"""
        
        with open(obj_path, 'w') as f:
            f.write(obj_content)
        
        print(f"OBJ plane created: {obj_path}")
    
    def generate_summary_report(self, sweet_spots):
        """Generate a comprehensive analysis report."""
        if self.df is None:
            return
        
        report_path = self.output_dir / "grid_search_report.txt"
        
        with open(report_path, 'w') as f:
            f.write("CPG Parameter Grid Search Analysis Report\n")
            f.write("=" * 45 + "\n\n")
            
            f.write(f"Total parameter combinations tested: {len(self.df)}\n")
            f.write(f"Frequency range: {self.df['frequency'].min():.2f} - {self.df['frequency'].max():.2f} Hz\n")
            f.write(f"Amplitude range: {self.df['amplitude'].min():.2f} - {self.df['amplitude'].max():.2f} rad\n\n")
            
            f.write("Energy Statistics:\n")
            f.write(f"  Mean: {self.df['energy'].mean():.4f}\n")
            f.write(f"  Std:  {self.df['energy'].std():.4f}\n")
            f.write(f"  Min:  {self.df['energy'].min():.4f}\n")
            f.write(f"  Max:  {self.df['energy'].max():.4f}\n\n")
            
            if 'stability' in self.df.columns:
                f.write("Stability Statistics:\n")
                f.write(f"  Mean: {self.df['stability'].mean():.4f}\n")
                f.write(f"  Std:  {self.df['stability'].std():.4f}\n")
                f.write(f"  Min:  {self.df['stability'].min():.4f}\n")
                f.write(f"  Max:  {self.df['stability'].max():.4f}\n\n")
            
            if sweet_spots is not None and len(sweet_spots) > 0:
                f.write(f"Sweet-Spots Found: {len(sweet_spots)}\n\n")
                f.write("Top 10 Sweet-Spots (lowest energy):\n")
                top_spots = sweet_spots.nsmallest(10, 'energy')
                for i, (_, row) in enumerate(top_spots.iterrows(), 1):
                    f.write(f"{i:2d}. Freq: {row['frequency']:5.2f} Hz, "
                           f"Amp: {row['amplitude']:6.3f} rad, "
                           f"Energy: {row['energy']:8.4f}")
                    if 'stability' in row:
                        f.write(f", Stability: {row['stability']:8.4f}")
                    f.write("\n")
        
        print(f"Analysis report saved to: {report_path}")

def main():
    """Main execution function."""
    # Initialize analyzer
    analyzer = GridSearchAnalyzer()
    
    # Load results
    if not analyzer.load_results():
        return
    
    # Generate basic energy heatmap
    print("\n" + "="*50)
    print("Generating Energy Heatmap...")
    energy_pivot = analyzer.generate_heatmap('energy')
    
    # Generate stability heatmap if available
    if 'stability' in analyzer.df.columns:
        print("\nGenerating Stability Heatmap...")
        stability_pivot = analyzer.generate_heatmap('stability')
    
    # Identify sweet-spots
    print("\n" + "="*50)
    print("Identifying Sweet-Spots...")
    sweet_spots = analyzer.identify_sweet_spots(energy_percentile=10, stability_threshold=0.1)
    
    # Generate sweet-spot overlay heatmap
    if sweet_spots is not None and len(sweet_spots) > 0:
        print("\nGenerating Sweet-Spot Heatmap...")
        analyzer.generate_sweet_spot_heatmap(sweet_spots, 'energy')
    
    # Export for dynamic demos
    print("\n" + "="*50)
    print("Exporting for Dynamic Demos...")
    analyzer.export_for_demos(energy_pivot, 'energy')
    
    # Generate summary report
    print("\nGenerating Summary Report...")
    analyzer.generate_summary_report(sweet_spots)
    
    print("\n" + "="*50)
    print("Grid Search Analysis Complete!")
    print("Check 'data/analysis/' directory for all outputs.")

if __name__ == "__main__":
    main()