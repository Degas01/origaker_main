#!/usr/bin/env python3
"""
Advanced Analyses & Plots for Validation Results

"""

import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import pandas as pd
from typing import Dict, List, Tuple, Any
import warnings
warnings.filterwarnings('ignore')

# Set style for professional plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

class ValidationPlotter:
    """Advanced plotting class for validation analysis"""
    
    def __init__(self, summary_path: str, output_dir: str = "data/analysis"):
        """
        Initialize the plotter
        
        Args:
            summary_path: Path to evaluation_summary.json
            output_dir: Directory to save plots
        """
        self.summary_path = Path(summary_path)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Load data
        self.data = self.load_summary()
        self.metrics_df = self.prepare_dataframe()
        
        # Plot settings
        self.dpi = 300  # High resolution
        self.figsize_large = (12, 8)
        self.figsize_medium = (10, 6)
        self.figsize_square = (8, 8)
        
    def load_summary(self) -> Dict[str, Any]:
        """Load and parse the evaluation summary JSON"""
        try:
            with open(self.summary_path, 'r') as f:
                data = json.load(f)
            print(f"✓ Loaded summary from {self.summary_path}")
            
            # Debug: Print the JSON structure
            print(f"\n📊 JSON STRUCTURE DEBUG:")
            print(f"Root keys: {list(data.keys()) if isinstance(data, dict) else 'Not a dict'}")
            print(f"Data type: {type(data)}")
            
            if isinstance(data, dict):
                for key, value in data.items():
                    print(f"  '{key}': {type(value)} - {len(value) if hasattr(value, '__len__') and not isinstance(value, str) else 'N/A'}")
                    if isinstance(value, dict) and len(value) <= 10:
                        print(f"    Sample: {dict(list(value.items())[:3])}")
                    elif isinstance(value, list) and len(value) <= 10:
                        print(f"    Sample: {value[:3]}")
            
            return data
        except FileNotFoundError:
            raise FileNotFoundError(f"Summary file not found: {self.summary_path}")
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON in summary file: {e}")
    
    def prepare_dataframe(self) -> pd.DataFrame:
        """Convert summary data to structured DataFrame"""
        rows = []
        
        print(f"\n🔍 EXTRACTING DATA FROM JSON:")
        
        # Strategy 1: Look for 'results' section
        if 'results' in self.data:
            print("Found 'results' section")
            for terrain, results in self.data['results'].items():
                if isinstance(results, dict):
                    row = {'terrain': terrain}
                    
                    if 'metrics' in results:
                        # Extract mean values from nested metrics
                        metrics = results['metrics']
                        print(f"  {terrain} metrics keys: {list(metrics.keys())}")
                        
                        for metric_name, metric_data in metrics.items():
                            if isinstance(metric_data, dict):
                                # Extract mean value for primary metric
                                if 'mean' in metric_data:
                                    row[metric_name] = metric_data['mean']
                                    print(f"    {metric_name}: {metric_data['mean']}")
                                    
                                    # Also extract std if available for error bars
                                    if 'std' in metric_data:
                                        row[f"{metric_name}_std"] = metric_data['std']
                                
                                # Handle special cases like success rate
                                elif 'rate' in metric_data:
                                    row[metric_name] = metric_data['rate']
                                    print(f"    {metric_name}: {metric_data['rate']} (rate)")
                                
                                # Handle count/total metrics
                                elif 'count' in metric_data:
                                    row[metric_name] = metric_data['count']
                                    print(f"    {metric_name}: {metric_data['count']} (count)")
                            else:
                                # Direct numeric value
                                row[metric_name] = metric_data
                                print(f"    {metric_name}: {metric_data} (direct)")
                    else:
                        # Direct structure without nested 'metrics'
                        row.update(results)
                        print(f"  {terrain}: {results}")
                    
                    rows.append(row)
        
        # Strategy 2: Look for 'terrains' section
        elif 'terrains' in self.data:
            print("Found 'terrains' section")
            terrains_data = self.data['terrains']
            if isinstance(terrains_data, list):
                for i, terrain_data in enumerate(terrains_data):
                    if isinstance(terrain_data, dict):
                        row = terrain_data.copy()
                        if 'terrain' not in row:
                            row['terrain'] = f'terrain_{i}'
                        
                        # Extract mean values from any nested dictionaries
                        for key, value in list(row.items()):
                            if isinstance(value, dict) and 'mean' in value:
                                row[key] = value['mean']
                                if 'std' in value:
                                    row[f"{key}_std"] = value['std']
                            elif isinstance(value, dict) and 'rate' in value:
                                row[key] = value['rate']
                        
                        print(f"  terrain_{i}: {terrain_data}")
                        rows.append(row)
            elif isinstance(terrains_data, dict):
                for terrain, data in terrains_data.items():
                    row = {'terrain': terrain}
                    if isinstance(data, dict):
                        for key, value in data.items():
                            if isinstance(value, dict) and 'mean' in value:
                                row[key] = value['mean']
                                if 'std' in value:
                                    row[f"{key}_std"] = value['std']
                            elif isinstance(value, dict) and 'rate' in value:
                                row[key] = value['rate']
                            else:
                                row[key] = value
                    print(f"  {terrain}: {data}")
                    rows.append(row)
        
        # Strategy 3: Look for direct metrics structure
        elif 'metrics' in self.data:
            print("Found 'metrics' section")
            for terrain, metrics in self.data['metrics'].items():
                row = {'terrain': terrain}
                if isinstance(metrics, dict):
                    for metric_name, metric_data in metrics.items():
                        if isinstance(metric_data, dict) and 'mean' in metric_data:
                            row[metric_name] = metric_data['mean']
                            if 'std' in metric_data:
                                row[f"{metric_name}_std"] = metric_data['std']
                        elif isinstance(metric_data, dict) and 'rate' in metric_data:
                            row[metric_name] = metric_data['rate']
                        else:
                            row[metric_name] = metric_data
                    print(f"  {terrain}: {metrics}")
                rows.append(row)
        
        # Strategy 4: Look for terrain-named keys at root level
        elif any(key.startswith('terrain') for key in self.data.keys() if isinstance(self.data, dict)):
            print("Found terrain keys at root level")
            for key, value in self.data.items():
                if key.startswith('terrain') or key.startswith('Terrain'):
                    row = {'terrain': key}
                    if isinstance(value, dict):
                        row.update(value)
                        print(f"  {key}: {value}")
                    rows.append(row)
        
        # Strategy 5: Direct list structure
        elif isinstance(self.data, list):
            print("Found list structure")
            for i, item in enumerate(self.data):
                if isinstance(item, dict):
                    row = item.copy()
                    if 'terrain' not in row:
                        row['terrain'] = f'terrain_{i}'
                    print(f"  terrain_{i}: {item}")
                    rows.append(row)
        
        # Strategy 6: Try any nested dictionary structure
        else:
            print("Searching nested structures...")
            for main_key, main_value in self.data.items():
                print(f"Checking key: {main_key}")
                if isinstance(main_value, dict):
                    # Check if this looks like terrain data
                    if any(key.startswith('terrain') for key in main_value.keys()):
                        print(f"  Found terrain data in '{main_key}'")
                        for terrain, metrics in main_value.items():
                            row = {'terrain': terrain}
                            if isinstance(metrics, dict):
                                row.update(metrics)
                                print(f"    {terrain}: {metrics}")
                            rows.append(row)
                        break
                elif isinstance(main_value, list):
                    # Check if this is a list of terrain data
                    if main_value and isinstance(main_value[0], dict):
                        print(f"  Found list of terrain data in '{main_key}'")
                        for i, item in enumerate(main_value):
                            row = item.copy()
                            if 'terrain' not in row:
                                row['terrain'] = f'terrain_{i}'
                            print(f"    terrain_{i}: {item}")
                            rows.append(row)
                        break
        
        print(f"\n📋 EXTRACTED ROWS: {len(rows)}")
        for row in rows:
            print(f"  {row}")
        
        if not rows:
            raise ValueError("No data found in summary - check JSON structure")
        
        df = pd.DataFrame(rows)
        
        # Convert numeric columns properly
        numeric_columns = []
        for col in df.columns:
            if col.lower() not in ['terrain', 'terrain_type', 'name']:
                try:
                    # Check if column contains non-null, non-string values
                    non_null_values = df[col].dropna()
                    if len(non_null_values) > 0:
                        # Check if values are already numeric
                        if pd.api.types.is_numeric_dtype(df[col]):
                            numeric_columns.append(col)
                            print(f"✓ '{col}' is already numeric: {df[col].dropna().tolist()}")
                        else:
                            # Try to convert to numeric
                            converted = pd.to_numeric(df[col], errors='coerce')
                            # Check if conversion was successful (not all NaN)
                            if not converted.isna().all():
                                df[col] = converted
                                numeric_columns.append(col)
                                print(f"✓ Converted '{col}' to numeric: {df[col].dropna().tolist()}")
                            else:
                                print(f"⚠️ Could not convert '{col}' to numeric - all values became NaN")
                                print(f"    Sample values: {df[col].head().tolist()}")
                    else:
                        print(f"⚠️ Column '{col}' has no non-null values")
                except Exception as e:
                    print(f"⚠️ Error processing '{col}': {e}")
                    print(f"    Sample values: {df[col].head().tolist()}")
        
        print(f"\n✓ Prepared DataFrame with {len(df)} terrain records")
        print(f"  Columns: {list(df.columns)}")
        print(f"  Numeric columns: {numeric_columns}")
        print(f"  Data types: {dict(df.dtypes)}")
        
        # Show sample data for debugging
        print(f"\n📊 FINAL DATA PREVIEW:")
        print(df.head())
        
        return df
    
    def get_numeric_columns(self) -> List[str]:
        """Get list of numeric column names, excluding terrain identifiers"""
        # First try pandas numeric detection
        numeric_cols = self.metrics_df.select_dtypes(include=[np.number]).columns.tolist()
        
        # Filter out columns that are all NaN
        valid_numeric_cols = []
        for col in numeric_cols:
            if not self.metrics_df[col].isna().all():
                valid_numeric_cols.append(col)
        
        # Remove terrain column if somehow included
        terrain_identifiers = ['terrain', 'terrain_type', 'name']
        valid_numeric_cols = [col for col in valid_numeric_cols if col.lower() not in terrain_identifiers]
        
        return valid_numeric_cols
    
    def get_terrain_column(self) -> str:
        """Get the terrain identifier column name"""
        possible_names = ['terrain', 'terrain_type', 'name']
        for name in possible_names:
            if name in self.metrics_df.columns:
                return name
        
        # Return first non-numeric column as fallback
        numeric_cols = self.get_numeric_columns()
        for col in self.metrics_df.columns:
            if col not in numeric_cols:
                return col
        
        # Final fallback
        return self.metrics_df.columns[0]
    
    def generate_bar_charts(self):
        """Generate bar charts with error bars for each metric across terrains"""
        # Get numeric columns using helper method
        metric_cols = self.get_numeric_columns()
        terrain_col = self.get_terrain_column()
        
        if not metric_cols:
            print("Warning: No numeric metrics found for bar charts")
            print(f"Available columns: {list(self.metrics_df.columns)}")
            print(f"Data types: {dict(self.metrics_df.dtypes)}")
            return
        
        print(f"Generating bar charts for metrics: {metric_cols}")
        
        # Create subplots
        n_metrics = len(metric_cols)
        cols = min(3, n_metrics)
        rows = (n_metrics + cols - 1) // cols
        
        fig, axes = plt.subplots(rows, cols, figsize=(cols * 4, rows * 3))
        if n_metrics == 1:
            axes = [axes]
        elif rows == 1:
            axes = axes.flatten() if cols > 1 else [axes]
        else:
            axes = axes.flatten()
        
        for idx, metric in enumerate(metric_cols):
            ax = axes[idx]
            
            # Get data for this metric - ensure it's numeric
            terrain_names = self.metrics_df[terrain_col].astype(str)
            values = pd.to_numeric(self.metrics_df[metric], errors='coerce')
            
            # Skip if all values are NaN
            if values.isna().all():
                ax.text(0.5, 0.5, f'No valid data for {metric}', ha='center', va='center', transform=ax.transAxes)
                continue
            
            # Calculate error bars (use std if available, otherwise use 10% of mean)
            std_col = f"{metric}_std"
            if std_col in self.metrics_df.columns:
                errors = pd.to_numeric(self.metrics_df[std_col], errors='coerce')
                errors = errors.fillna(values * 0.1)  # Fallback for NaN std values
            else:
                errors = values * 0.1  # 10% error approximation
            
            # Create bar chart
            valid_indices = ~values.isna()
            valid_terrains = terrain_names[valid_indices]
            valid_values = values[valid_indices]
            valid_errors = errors[valid_indices]
            
            if len(valid_values) == 0:
                ax.text(0.5, 0.5, f'No valid data for {metric}', ha='center', va='center', transform=ax.transAxes)
                continue
            
            bars = ax.bar(range(len(valid_terrains)), valid_values, yerr=valid_errors, 
                         capsize=5, alpha=0.8, edgecolor='black', linewidth=0.5)
            
            # Customize
            ax.set_xlabel('Terrain Type')
            ax.set_ylabel(metric.replace('_', ' ').title())
            ax.set_title(f'{metric.replace("_", " ").title()} by Terrain')
            ax.set_xticks(range(len(valid_terrains)))
            ax.set_xticklabels(valid_terrains, rotation=45, ha='right')
            ax.grid(True, alpha=0.3)
            
            # Add value labels on bars
            for i, (bar, val, err) in enumerate(zip(bars, valid_values, valid_errors)):
                height = bar.get_height()
                label_height = height + err if not pd.isna(err) else height
                ax.text(bar.get_x() + bar.get_width()/2., label_height,
                       f'{val:.3f}', ha='center', va='bottom', fontsize=8)
        
        # Hide unused subplots
        for idx in range(n_metrics, len(axes)):
            axes[idx].axis('off')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'metrics_bar_charts.png', dpi=self.dpi, bbox_inches='tight')
        plt.close()
        print(f"✓ Saved bar charts: {self.output_dir / 'metrics_bar_charts.png'}")
    
    def generate_pareto_heatmaps(self):
        """Generate Pareto heatmaps showing trade-offs between metrics"""
        metric_cols = self.get_numeric_columns()
        terrain_col = self.get_terrain_column()
        
        if len(metric_cols) < 2:
            print(f"Warning: Need at least 2 metrics for Pareto analysis. Found: {metric_cols}")
            return
        
        print(f"Generating Pareto heatmaps for metrics: {metric_cols}")
        
        # Create pairwise Pareto heatmaps
        important_pairs = [
            ('COT', 'path_deviation'),
            ('cost_of_transport', 'path_deviation'),
            ('energy_consumed', 'distance_traveled'),
            ('stability_index', 'success'),
            ('success', 'cost_of_transport')
        ]
        
        # Find actual column names that match our important pairs
        actual_pairs = []
        for pair in important_pairs:
            col1_matches = [col for col in metric_cols if pair[0].lower() in col.lower()]
            col2_matches = [col for col in metric_cols if pair[1].lower() in col.lower()]
            
            if col1_matches and col2_matches:
                actual_pairs.append((col1_matches[0], col2_matches[0]))
        
        # If no matches, use first available metrics
        if not actual_pairs and len(metric_cols) >= 2:
            # Create pairs from available metrics
            for i in range(0, len(metric_cols)-1, 2):
                if i+1 < len(metric_cols):
                    actual_pairs.append((metric_cols[i], metric_cols[i+1]))
        
        if not actual_pairs:
            print("Warning: Could not create metric pairs for Pareto analysis")
            return
        
        for i, (metric1, metric2) in enumerate(actual_pairs):
            fig, ax = plt.subplots(figsize=self.figsize_square)
            
            # Create scatter plot
            terrains = self.metrics_df[terrain_col].astype(str)
            x_vals = pd.to_numeric(self.metrics_df[metric1], errors='coerce')
            y_vals = pd.to_numeric(self.metrics_df[metric2], errors='coerce')
            
            # Remove NaN values
            valid_mask = ~(x_vals.isna() | y_vals.isna())
            if not valid_mask.any():
                print(f"Warning: No valid data for {metric1} vs {metric2}")
                plt.close()
                continue
            
            x_vals = x_vals[valid_mask]
            y_vals = y_vals[valid_mask]
            terrains_valid = terrains[valid_mask]
            
            # Create 2D histogram/heatmap
            try:
                if len(x_vals) > 1 and x_vals.std() > 0 and y_vals.std() > 0:
                    # Create grid for heatmap
                    x_range = np.linspace(x_vals.min(), x_vals.max(), 20)
                    y_range = np.linspace(y_vals.min(), y_vals.max(), 20)
                    
                    # Create efficiency matrix (example: closer to origin is better)
                    X, Y = np.meshgrid(x_range, y_range)
                    
                    # Efficiency score (normalize and invert - lower values = better efficiency)
                    norm_x = (X - x_vals.min()) / (x_vals.max() - x_vals.min() + 1e-8)
                    norm_y = (Y - y_vals.min()) / (y_vals.max() - y_vals.min() + 1e-8)
                    efficiency = 1 / (1 + norm_x + norm_y)  # Higher is better
                    
                    # Plot heatmap
                    im = ax.imshow(efficiency, extent=[x_vals.min(), x_vals.max(), 
                                                     y_vals.min(), y_vals.max()],
                                  origin='lower', cmap='RdYlGn', alpha=0.7)
                    
                    # Add colorbar
                    cbar = plt.colorbar(im, ax=ax)
                    cbar.set_label('Efficiency Score', rotation=270, labelpad=15)
                else:
                    print(f"Insufficient variation in data for heatmap: {metric1} vs {metric2}")
                
                # Overlay data points
                scatter = ax.scatter(x_vals, y_vals, c='blue', s=100, alpha=0.8, 
                               edgecolors='white', linewidth=2, zorder=5)
                
                # Add terrain labels
                for x, y, terrain in zip(x_vals, y_vals, terrains_valid):
                    ax.annotate(terrain, (x, y), xytext=(5, 5), 
                              textcoords='offset points', fontsize=8,
                              bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
                
            except Exception as e:
                print(f"Warning: Could not create heatmap for {metric1} vs {metric2}: {e}")
                # Fallback to simple scatter plot
                ax.scatter(x_vals, y_vals, alpha=0.7, s=100)
                for x, y, terrain in zip(x_vals, y_vals, terrains_valid):
                    ax.annotate(terrain, (x, y), xytext=(5, 5), textcoords='offset points')
            
            ax.set_xlabel(metric1.replace('_', ' ').title())
            ax.set_ylabel(metric2.replace('_', ' ').title())
            ax.set_title(f'Pareto Analysis: {metric1.replace("_", " ").title()} vs {metric2.replace("_", " ").title()}')
            ax.grid(True, alpha=0.3)
            
            plt.tight_layout()
            filename = f'pareto_heatmap_{metric1}_vs_{metric2}.png'
            plt.savefig(self.output_dir / filename, dpi=self.dpi, bbox_inches='tight')
            plt.close()
            print(f"✓ Saved Pareto heatmap: {self.output_dir / filename}")
    
    def generate_confidence_maps(self):
        """Generate confidence maps overlaying mean±std on terrain 2D layout"""
        # Create a simulated 2D terrain layout
        terrain_col = self.get_terrain_column()
        metric_cols = self.get_numeric_columns()
        
        if not metric_cols:
            print("Warning: No numeric metrics available for confidence maps")
            return
        
        terrains = self.metrics_df[terrain_col].astype(str).unique()
        n_terrains = len(terrains)
        
        # Create 2D grid layout for terrains
        grid_size = int(np.ceil(np.sqrt(n_terrains)))
        terrain_positions = {}
        
        idx = 0
        for i in range(grid_size):
            for j in range(grid_size):
                if idx < n_terrains:
                    terrain_positions[terrains[idx]] = (i, j)
                    idx += 1
        
        for metric in metric_cols[:4]:  # Limit to first 4 metrics
            fig, ax = plt.subplots(figsize=self.figsize_square)
            
            # Create confidence map data
            confidence_grid = np.zeros((grid_size, grid_size))
            mean_grid = np.zeros((grid_size, grid_size))
            std_grid = np.zeros((grid_size, grid_size))
            
            for _, row in self.metrics_df.iterrows():
                terrain = str(row[terrain_col])
                if terrain in terrain_positions:
                    i, j = terrain_positions[terrain]
                    
                    # Convert to numeric
                    mean_val = pd.to_numeric(row[metric], errors='coerce')
                    if pd.isna(mean_val):
                        continue
                    
                    # Get std if available
                    std_col = f"{metric}_std"
                    if std_col in self.metrics_df.columns:
                        std_val = pd.to_numeric(row[std_col], errors='coerce')
                        if pd.isna(std_val):
                            std_val = mean_val * 0.1  # 10% approximation
                    else:
                        std_val = mean_val * 0.1  # 10% approximation
                    
                    mean_grid[i, j] = mean_val
                    std_grid[i, j] = std_val
                    # Confidence = 1 / (1 + coefficient_of_variation)
                    confidence_grid[i, j] = 1 / (1 + abs(std_val) / (abs(mean_val) + 1e-8))
            
            # Only plot if we have valid data
            if np.any(confidence_grid > 0):
                # Plot confidence as heatmap
                im1 = ax.imshow(confidence_grid, cmap='RdYlGn', alpha=0.8, vmin=0, vmax=1)
                
                # Add mean values as text annotations
                for i in range(grid_size):
                    for j in range(grid_size):
                        if confidence_grid[i, j] > 0:  # Only for valid positions
                            # Find terrain name for this position
                            terrain_name = None
                            for terrain, pos in terrain_positions.items():
                                if pos == (i, j):
                                    terrain_name = terrain
                                    break
                            
                            if terrain_name:
                                mean_val = mean_grid[i, j]
                                std_val = std_grid[i, j]
                                
                                # Add terrain name
                                ax.text(j, i-0.2, terrain_name, ha='center', va='center', 
                                      fontsize=8, fontweight='bold')
                                
                                # Add mean ± std
                                ax.text(j, i, f'{mean_val:.3f}', ha='center', va='center', 
                                      fontsize=10, color='black', 
                                      bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.8))
                                
                                ax.text(j, i+0.2, f'±{std_val:.3f}', ha='center', va='center', 
                                      fontsize=8, color='gray')
                
                # Customize plot
                ax.set_title(f'Confidence Map: {metric.replace("_", " ").title()}\n(Green=High Confidence, Red=Low Confidence)')
                ax.set_xticks([])
                ax.set_yticks([])
                
                # Add colorbar
                cbar = plt.colorbar(im1, ax=ax, fraction=0.046, pad=0.04)
                cbar.set_label('Confidence Score', rotation=270, labelpad=15)
                
                plt.tight_layout()
                filename = f'confidence_map_{metric}.png'
                plt.savefig(self.output_dir / filename, dpi=self.dpi, bbox_inches='tight')
                plt.close()
                print(f"✓ Saved confidence map: {self.output_dir / filename}")
            else:
                plt.close()
                print(f"Warning: No valid data for confidence map of {metric}")
    
    def generate_correlation_matrix(self):
        """Generate correlation matrix heatmap for all metrics"""
        metric_cols = self.get_numeric_columns()
        
        if len(metric_cols) < 2:
            print(f"Warning: Need at least 2 numeric columns for correlation matrix. Found: {metric_cols}")
            return
        
        # Get numeric data and convert to ensure it's actually numeric
        numeric_data = pd.DataFrame()
        for col in metric_cols:
            numeric_data[col] = pd.to_numeric(self.metrics_df[col], errors='coerce')
        
        # Remove columns that are all NaN
        numeric_data = numeric_data.dropna(axis=1, how='all')
        
        if numeric_data.shape[1] < 2:
            print("Warning: Not enough valid numeric data for correlation matrix")
            return
        
        # Calculate correlation matrix
        corr_matrix = numeric_data.corr()
        
        # Create heatmap
        fig, ax = plt.subplots(figsize=self.figsize_square)
        
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool))  # Mask upper triangle
        
        sns.heatmap(corr_matrix, mask=mask, annot=True, cmap='RdBu_r', center=0,
                   square=True, ax=ax, cbar_kws={"shrink": .8})
        
        ax.set_title('Metrics Correlation Matrix')
        plt.tight_layout()
        plt.savefig(self.output_dir / 'correlation_matrix.png', dpi=self.dpi, bbox_inches='tight')
        plt.close()
        print(f"✓ Saved correlation matrix: {self.output_dir / 'correlation_matrix.png'}")
    
    def generate_summary_report(self):
        """Generate a summary statistics report"""
        metric_cols = self.get_numeric_columns()
        terrain_col = self.get_terrain_column()
        
        if not metric_cols:
            print("Warning: No numeric data available for summary report")
            return
        
        # Get numeric data and ensure it's actually numeric
        numeric_data = pd.DataFrame()
        for col in metric_cols:
            numeric_data[col] = pd.to_numeric(self.metrics_df[col], errors='coerce')
        
        # Remove columns that are all NaN
        numeric_data = numeric_data.dropna(axis=1, how='all')
        
        if numeric_data.empty:
            print("Warning: No valid numeric data for summary report")
            return
        
        # Filter out std columns for main analysis (but keep them for error bars)
        main_metrics = [col for col in numeric_data.columns if not col.endswith('_std')]
        if main_metrics:
            numeric_data_main = numeric_data[main_metrics]
        else:
            numeric_data_main = numeric_data
        
        # Calculate summary statistics
        summary_stats = numeric_data_main.describe()
        
        # Create summary plot with better sizing and spacing
        fig = plt.figure(figsize=(16, 12))  # Larger figure
        gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)  # Better spacing
        
        ax1 = fig.add_subplot(gs[0, 0])
        ax2 = fig.add_subplot(gs[0, 1], projection='polar')
        ax3 = fig.add_subplot(gs[1, 0])
        ax4 = fig.add_subplot(gs[1, 1])
        
        # 1. Box plots for main metrics (excluding std columns)
        try:
            if len(numeric_data_main.columns) > 0:
                # Create box plot with better formatting
                box_data = []
                labels = []
                for col in numeric_data_main.columns:
                    if not numeric_data_main[col].isna().all():
                        box_data.append(numeric_data_main[col].dropna())
                        # Clean up column names for display
                        clean_name = col.replace('_', ' ').title()
                        if len(clean_name) > 15:  # Truncate long names
                            clean_name = clean_name[:12] + '...'
                        labels.append(clean_name)
                
                if box_data:
                    bp = ax1.boxplot(box_data, labels=labels, patch_artist=True)
                    
                    # Color the boxes
                    colors = plt.cm.Set3(np.linspace(0, 1, len(box_data)))
                    for patch, color in zip(bp['boxes'], colors):
                        patch.set_facecolor(color)
                        patch.set_alpha(0.7)
                    
                    ax1.set_title('Distribution of Main Metrics', fontsize=14, fontweight='bold')
                    ax1.tick_params(axis='x', labelsize=10, rotation=45)
                    ax1.grid(True, alpha=0.3)
                else:
                    ax1.text(0.5, 0.5, 'No valid data for box plots', ha='center', va='center', 
                            transform=ax1.transAxes, fontsize=12)
            else:
                ax1.text(0.5, 0.5, 'No numeric data available', ha='center', va='center', 
                        transform=ax1.transAxes, fontsize=12)
        except Exception as e:
            ax1.text(0.5, 0.5, f'Error generating box plots:\n{str(e)[:50]}...', 
                    ha='center', va='center', transform=ax1.transAxes, fontsize=10)
        
        # 2. Improved terrain comparison radar chart
        if terrain_col and len(self.metrics_df[terrain_col].unique()) > 1 and len(numeric_data_main.columns) >= 3:
            try:
                # Normalize data for radar chart
                normalized_data = (numeric_data_main - numeric_data_main.min()) / (numeric_data_main.max() - numeric_data_main.min() + 1e-8)
                
                # Limit to 6 metrics max for readability
                display_metrics = list(normalized_data.columns)[:6]
                normalized_display = normalized_data[display_metrics]
                
                angles = np.linspace(0, 2*np.pi, len(display_metrics), endpoint=False)
                angles = np.concatenate((angles, [angles[0]]))  # Complete the circle
                
                # Color palette for terrains
                colors = plt.cm.tab10(np.linspace(0, 1, len(self.metrics_df)))
                
                for idx, (_, row) in enumerate(self.metrics_df.iterrows()):
                    terrain_name = str(row[terrain_col]) if terrain_col else f'Sample {idx}'
                    values = normalized_display.iloc[idx].values
                    
                    # Skip if all values are NaN
                    if not np.isnan(values).all():
                        values = np.concatenate((values, [values[0]]))  # Complete the circle
                        
                        ax2.plot(angles, values, 'o-', linewidth=2, label=terrain_name, 
                               color=colors[idx], markersize=4)
                        ax2.fill(angles, values, alpha=0.15, color=colors[idx])
                
                # Improve label positioning and readability
                ax2.set_xticks(angles[:-1])
                
                # Create clean, readable labels
                clean_labels = []
                for col in display_metrics:
                    clean_label = col.replace('_', '\n').title()
                    if len(clean_label) > 20:
                        clean_label = clean_label[:17] + '...'
                    clean_labels.append(clean_label)
                
                ax2.set_xticklabels(clean_labels, fontsize=9)
                ax2.set_ylim(0, 1)
                ax2.set_title('Terrain Performance Comparison\n(Normalized)', 
                             fontsize=14, fontweight='bold', pad=20)
                
                # Improve legend positioning
                ax2.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0), fontsize=10)
                ax2.grid(True, alpha=0.3)
                
            except Exception as e:
                ax2.text(0.5, 0.5, f'Error in radar chart:\n{str(e)[:50]}...', 
                        ha='center', va='center', transform=ax2.transAxes, fontsize=10)
        else:
            ax2.text(0.5, 0.5, 'Insufficient data for\nradar chart\n(need multiple terrains\nand metrics)', 
                    ha='center', va='center', transform=ax2.transAxes, fontsize=12)
            ax2.set_title('Terrain Comparison', fontsize=14, fontweight='bold')
        
        # 3. Improved performance ranking
        if len(numeric_data_main.columns) > 0:
            try:
                # Calculate overall performance score (normalized sum of main metrics)
                normalized_scores = (numeric_data_main - numeric_data_main.min()) / (numeric_data_main.max() - numeric_data_main.min() + 1e-8)
                overall_scores = normalized_scores.mean(axis=1)
                
                terrain_names = self.metrics_df[terrain_col].astype(str) if terrain_col else [f'Sample {i}' for i in range(len(self.metrics_df))]
                ranking_df = pd.DataFrame({'Terrain': terrain_names, 'Score': overall_scores}).sort_values('Score', ascending=True)  # Ascending for horizontal bars
                
                # Remove NaN scores
                ranking_df = ranking_df.dropna()
                
                if not ranking_df.empty:
                    # Create horizontal bar chart
                    colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(ranking_df)))
                    bars = ax3.barh(range(len(ranking_df)), ranking_df['Score'], 
                                   color=colors, alpha=0.8, edgecolor='white', linewidth=1)
                    
                    ax3.set_yticks(range(len(ranking_df)))
                    ax3.set_yticklabels(ranking_df['Terrain'], fontsize=11)
                    ax3.set_xlabel('Overall Performance Score', fontsize=12)
                    ax3.set_title('Terrain Performance Ranking', fontsize=14, fontweight='bold')
                    ax3.grid(True, alpha=0.3, axis='x')
                    
                    # Add score labels with better positioning
                    for i, (bar, score) in enumerate(zip(bars, ranking_df['Score'])):
                        label_x = bar.get_width() + max(ranking_df['Score']) * 0.02
                        ax3.text(label_x, bar.get_y() + bar.get_height()/2, 
                                f'{score:.3f}', va='center', fontsize=10, fontweight='bold')
                    
                    # Set x-axis limits to accommodate labels
                    ax3.set_xlim(0, max(ranking_df['Score']) * 1.15)
                else:
                    ax3.text(0.5, 0.5, 'No valid scores\nfor ranking', ha='center', va='center', 
                            transform=ax3.transAxes, fontsize=12)
            except Exception as e:
                ax3.text(0.5, 0.5, f'Error in ranking:\n{str(e)[:50]}...', 
                        ha='center', va='center', transform=ax3.transAxes, fontsize=10)
        else:
            ax3.text(0.5, 0.5, 'No data available\nfor ranking', ha='center', va='center', 
                    transform=ax3.transAxes, fontsize=12)
        
        # 4. Improved data quality assessment
        try:
            total_points = len(self.metrics_df)
            total_metrics = len(metric_cols)
            valid_metrics = len(numeric_data.columns)
            complete_records = len(numeric_data.dropna())
            
            quality_data = {
                'Total\nSamples': total_points,
                'Total\nMetrics': total_metrics,
                'Valid\nMetrics': valid_metrics,
                'Complete\nRecords': complete_records
            }
            
            # Create bar chart with better colors and styling
            x_positions = range(len(quality_data))
            values = list(quality_data.values())
            colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4']
            
            bars = ax4.bar(x_positions, values, color=colors, alpha=0.8, 
                          edgecolor='white', linewidth=2)
            
            ax4.set_xticks(x_positions)
            ax4.set_xticklabels(list(quality_data.keys()), fontsize=11)
            ax4.set_ylabel('Count', fontsize=12)
            ax4.set_title('Data Quality Summary', fontsize=14, fontweight='bold')
            ax4.grid(True, alpha=0.3, axis='y')
            
            # Add value labels on top of bars
            for i, (bar, value) in enumerate(zip(bars, values)):
                ax4.text(bar.get_x() + bar.get_width()/2, 
                        bar.get_height() + max(values) * 0.02, 
                        str(value), ha='center', va='bottom', 
                        fontsize=12, fontweight='bold')
            
            # Set y-axis limits to accommodate labels
            ax4.set_ylim(0, max(values) * 1.15)
            
        except Exception as e:
            ax4.text(0.5, 0.5, f'Error in quality summary:\n{str(e)[:50]}...', 
                    ha='center', va='center', transform=ax4.transAxes, fontsize=10)
        
        # Final layout adjustments
        plt.suptitle('Validation Analysis Summary Report', fontsize=18, fontweight='bold', y=0.98)
        plt.tight_layout()
        plt.subplots_adjust(top=0.93)  # Make room for main title
        
        # Save with higher DPI for better quality
        plt.savefig(self.output_dir / 'summary_report.png', dpi=self.dpi, bbox_inches='tight', 
                   facecolor='white', edgecolor='none')
        plt.close()
        print(f"✓ Saved improved summary report: {self.output_dir / 'summary_report.png'}")
    
    def run_all_analyses(self):
        """Run all analysis and plotting functions"""
        print(f"\n{'='*50}")
        print("ADVANCED VALIDATION ANALYSIS - TASK 8.3")
        print(f"{'='*50}")
        print(f"Data source: {self.summary_path}")
        print(f"Output directory: {self.output_dir}")
        print(f"Available data shape: {self.metrics_df.shape}")
        print(f"{'='*50}\n")
        
        try:
            # Generate all plots
            self.generate_bar_charts()
            self.generate_pareto_heatmaps()
            self.generate_confidence_maps()
            self.generate_correlation_matrix()
            self.generate_summary_report()
            
            print(f"\n{'='*50}")
            print("✅ ALL ANALYSES COMPLETED SUCCESSFULLY")
            print(f"✅ All plots saved to: {self.output_dir}")
            print(f"{'='*50}")
            
        except Exception as e:
            print(f"\n❌ Error during analysis: {e}")
            raise


def main():
    """Main function to run the validation plotting"""
    # Update this path to match your actual file location
    summary_path = r"C:\Users\Giacomo\Desktop\MSc Robotics\7CCEMPRJ MSc Individual Project\origaker_main\data\validation\evaluation_summary.json"
    output_dir = "data/analysis"
    
    # Create and run plotter
    plotter = ValidationPlotter(summary_path, output_dir)
    plotter.run_all_analyses()


if __name__ == "__main__":
    main()