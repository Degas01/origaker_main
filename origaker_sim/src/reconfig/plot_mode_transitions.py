"""
Origaker Mode Transition Visualization and Analysis
File: src/analysis/plot_mode_transitions.py

This module provides comprehensive visualization and analysis tools for Origaker
morphology mode transitions, including timeline plots, performance analysis,
and terrain correlation visualizations.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.dates as mdates
from matplotlib.animation import FuncAnimation
from matplotlib.collections import LineCollection
import seaborn as sns
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
from datetime import datetime, timedelta
import json
from pathlib import Path

# Import Origaker components
try:
    from .reconfig import OrigakerPoseMode
    from .graph import OrigakerTransitionGraph
except ImportError:
    try:
        from reconfig import OrigakerPoseMode
        from graph import OrigakerTransitionGraph
    except ImportError:
        # Define minimal enum for standalone operation
        class OrigakerPoseMode:
            SPREADER = 1
            HIGH_STEP = 2
            CRAWLER = 3
            ROLLING = 4

# Set visualization style
plt.style.use('default')
sns.set_palette("husl")


class OrigakerTransitionPlotter:
    """
    Comprehensive visualization system for Origaker morphology mode transitions.
    Provides timeline analysis, performance metrics, and terrain correlation plots.
    """
    
    def __init__(self):
        """Initialize the transition plotter with Origaker-specific styling."""
        
        # Origaker-specific color scheme
        self.mode_colors = {
            'SPREADER': '#4ECDC4',      # Teal - stable, spread stance
            'HIGH_STEP': '#45B7D1',     # Blue - elevated, obstacle-clearing
            'CRAWLER': '#FF6B6B',       # Red - compact, low-profile
            'ROLLING': '#96CEB4',       # Green - spherical, mobile
            'UNKNOWN': '#808080'        # Gray - undefined/error state
        }
        
        # Terrain type colors
        self.terrain_colors = {
            'flat_surface': '#00B894',      # Green - easy terrain
            'rough_terrain': '#74B9FF',     # Light blue - challenging
            'obstacles': '#D63031',         # Red - difficult obstacles
            'narrow_corridor': '#FF9F43'    # Orange - confined spaces
        }
        
        # Transition quality indicators
        self.quality_colors = {
            'excellent': '#00B894',
            'good': '#74B9FF', 
            'fair': '#FDCB6E',
            'poor': '#E17055',
            'failed': '#D63031'
        }
        
        # Figure style settings
        self.figure_style = {
            'figsize': (15, 10),
            'dpi': 300,
            'facecolor': 'white',
            'edgecolor': 'none'
        }
        
        print("✅ Origaker Transition Plotter initialized")
    
    def plot_transition_timeline(self,
                                transition_history: List[Dict],
                                episode_duration: float,
                                robot_trajectory: Optional[List[Tuple[float, float]]] = None,
                                terrain_events: Optional[List[Dict]] = None,
                                save_path: Optional[str] = None,
                                show_plot: bool = True) -> str:
        """
        Create comprehensive timeline plot of mode transitions.
        
        Args:
            transition_history: List of transition records
            episode_duration: Total episode duration in seconds
            robot_trajectory: Optional robot trajectory data
            terrain_events: Optional terrain change events
            save_path: Path to save the plot
            show_plot: Whether to display the plot
            
        Returns:
            Path to saved plot file
        """
        
        try:
            # Create figure with subplots
            fig = plt.figure(figsize=self.figure_style['figsize'])
            
            if terrain_events or robot_trajectory:
                gs = fig.add_gridspec(4, 1, height_ratios=[3, 1, 1, 1], hspace=0.3)
                ax_main = fig.add_subplot(gs[0])
                ax_freq = fig.add_subplot(gs[1])
                ax_terrain = fig.add_subplot(gs[2]) if terrain_events else None
                ax_traj = fig.add_subplot(gs[3]) if robot_trajectory else None
            else:
                gs = fig.add_gridspec(2, 1, height_ratios=[3, 1], hspace=0.3)
                ax_main = fig.add_subplot(gs[0])
                ax_freq = fig.add_subplot(gs[1])
                ax_terrain = None
                ax_traj = None
            
            # Prepare timeline data
            times = [0.0]
            modes = ['SPREADER']  # Default starting mode
            
            for transition in transition_history:
                times.append(transition['timestamp'])
                if hasattr(transition['to_pose'], 'name'):
                    modes.append(transition['to_pose'].name)
                elif hasattr(transition['to_pose'], 'value'):
                    modes.append(f"POSE_MODEL_{transition['to_pose'].value}")
                else:
                    modes.append(str(transition['to_pose']).split('.')[-1])
            
            times.append(episode_duration)
            modes.append(modes[-1])
            
            # Main timeline plot
            for i in range(len(times) - 1):
                mode = modes[i]
                color = self.mode_colors.get(mode, self.mode_colors['UNKNOWN'])
                duration = times[i+1] - times[i]
                
                # Draw mode segment
                rect = patches.Rectangle((times[i], -0.4), duration, 0.8,
                                       facecolor=color, alpha=0.8, 
                                       edgecolor='black', linewidth=1)
                ax_main.add_patch(rect)
                
                # Add mode label if segment is wide enough
                if duration > episode_duration * 0.08:
                    label_text = mode.replace('_', '\n') if '_' in mode else mode
                    ax_main.text(times[i] + duration/2, 0, label_text,
                               ha='center', va='center', fontweight='bold',
                               fontsize=10, color='white' if mode != 'UNKNOWN' else 'black')
            
            # Add transition markers and annotations
            for i, transition in enumerate(transition_history):
                timestamp = transition['timestamp']
                
                # Transition marker
                ax_main.axvline(timestamp, color='red', linestyle='--', 
                              alpha=0.8, linewidth=2, zorder=10)
                
                # Transition arrow and info
                ax_main.annotate('🔄', xy=(timestamp, 0.6), 
                               ha='center', va='center', fontsize=14, zorder=15)
                
                # Success/failure indicator
                success = transition.get('success', True)
                marker_color = '#00B894' if success else '#D63031'
                marker_symbol = '✓' if success else '✗'
                
                ax_main.annotate(marker_symbol, xy=(timestamp, -0.6),
                               ha='center', va='center', fontsize=12,
                               color=marker_color, fontweight='bold', zorder=15)
                
                # Transition cost indicator
                cost = transition.get('transition_cost', 0)
                if cost > 0:
                    cost_height = min(0.3, cost / 10)  # Scale cost to height
                    ax_main.bar(timestamp, cost_height, width=episode_duration*0.01,
                              bottom=0.7, color='orange', alpha=0.7, zorder=5)
            
            # Main plot formatting
            ax_main.set_xlim(0, episode_duration)
            ax_main.set_ylim(-1, 1.2)
            ax_main.set_ylabel('Morphology Mode', fontweight='bold')
            ax_main.set_title('Origaker Morphology Mode Timeline with Transition Analysis', 
                            fontsize=16, fontweight='bold', pad=20)
            ax_main.set_yticks([])
            ax_main.grid(True, alpha=0.3, axis='x')
            
            # Add mode legend
            legend_elements = []
            for mode, color in self.mode_colors.items():
                if mode != 'UNKNOWN':
                    legend_elements.append(
                        patches.Patch(color=color, label=mode.replace('_', ' ').title())
                    )
            
            ax_main.legend(handles=legend_elements, loc='upper left', 
                         bbox_to_anchor=(1.02, 1), fontsize=10)
            
            # Transition frequency plot
            if transition_history:
                transition_times = [t['timestamp'] for t in transition_history]
                bins = np.linspace(0, episode_duration, 20)
                counts, bin_edges = np.histogram(transition_times, bins=bins)
                bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
                
                bars = ax_freq.bar(bin_centers, counts, width=bin_edges[1]-bin_edges[0]*0.8,
                                 alpha=0.7, color='steelblue', edgecolor='black')
                
                # Add value labels on bars
                for bar, count in zip(bars, counts):
                    if count > 0:
                        ax_freq.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                                   str(int(count)), ha='center', va='bottom', fontsize=9)
                
                ax_freq.set_ylabel('Transitions', fontweight='bold')
                ax_freq.set_title('Transition Frequency Distribution', fontweight='bold')
                ax_freq.grid(True, alpha=0.3)
                
                # Add average line
                avg_transitions = len(transition_history) / 20
                ax_freq.axhline(avg_transitions, color='red', linestyle='--', 
                              alpha=0.7, label=f'Average: {avg_transitions:.1f}')
                ax_freq.legend()
            
            # Terrain events plot
            if ax_terrain and terrain_events:
                for event in terrain_events:
                    start_time = event.get('start_time', 0)
                    end_time = event.get('end_time', episode_duration)
                    terrain_type = event.get('terrain_type', 'flat_surface')
                    color = self.terrain_colors.get(terrain_type, '#808080')
                    
                    rect = patches.Rectangle((start_time, -0.4), end_time - start_time, 0.8,
                                           facecolor=color, alpha=0.7, edgecolor='black')
                    ax_terrain.add_patch(rect)
                    
                    # Add terrain label
                    if end_time - start_time > episode_duration * 0.08:
                        label = terrain_type.replace('_', '\n')
                        ax_terrain.text(start_time + (end_time - start_time)/2, 0, label,
                                      ha='center', va='center', fontweight='bold', fontsize=9)
                
                ax_terrain.set_xlim(0, episode_duration)
                ax_terrain.set_ylim(-0.5, 0.5)
                ax_terrain.set_ylabel('Terrain', fontweight='bold')
                ax_terrain.set_title('Terrain Types Encountered', fontweight='bold')
                ax_terrain.set_yticks([])
                ax_terrain.grid(True, alpha=0.3, axis='x')
            
            # Trajectory plot
            if ax_traj and robot_trajectory:
                # Extract x and y coordinates
                x_coords = [pos[0] for pos in robot_trajectory]
                y_coords = [pos[1] for pos in robot_trajectory]
                
                # Create time array
                traj_times = np.linspace(0, episode_duration, len(robot_trajectory))
                
                # Plot x and y separately
                ax_traj.plot(traj_times, x_coords, 'b-', alpha=0.7, linewidth=2, label='X Position')
                ax_traj_twin = ax_traj.twinx()
                ax_traj_twin.plot(traj_times, y_coords, 'r-', alpha=0.7, linewidth=2, label='Y Position')
                
                # Mark transition points on trajectory
                for transition in transition_history:
                    timestamp = transition['timestamp']
                    ax_traj.axvline(timestamp, color='orange', linestyle=':', alpha=0.5)
                
                ax_traj.set_ylabel('X Position (m)', color='blue', fontweight='bold')
                ax_traj_twin.set_ylabel('Y Position (m)', color='red', fontweight='bold')
                ax_traj.set_title('Robot Trajectory During Mode Transitions', fontweight='bold')
                ax_traj.grid(True, alpha=0.3)
                
                # Combined legend
                lines1, labels1 = ax_traj.get_legend_handles_labels()
                lines2, labels2 = ax_traj_twin.get_legend_handles_labels()
                ax_traj.legend(lines1 + lines2, labels1 + labels2, loc='upper right')
            
            # Final formatting
            if ax_traj:
                ax_traj.set_xlabel('Time (seconds)', fontweight='bold')
            elif ax_terrain:
                ax_terrain.set_xlabel('Time (seconds)', fontweight='bold')
            else:
                ax_freq.set_xlabel('Time (seconds)', fontweight='bold')
            
            plt.tight_layout()
            
            # Save plot
            if save_path is None:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                save_path = f"origaker_transition_timeline_{timestamp}.png"
            
            plt.savefig(save_path, dpi=self.figure_style['dpi'], bbox_inches='tight')
            print(f"✅ Transition timeline saved to {save_path}")
            
            if show_plot:
                plt.show()
            else:
                plt.close()
            
            return save_path
            
        except Exception as e:
            print(f"⚠️ Timeline plot failed: {e}")
            return ""
    
    def plot_transition_performance_matrix(self,
                                         transition_history: List[Dict],
                                         save_path: Optional[str] = None,
                                         show_plot: bool = True) -> str:
        """
        Create performance matrix showing success rates between all pose pairs.
        
        Args:
            transition_history: List of transition records
            save_path: Path to save the plot
            show_plot: Whether to display the plot
            
        Returns:
            Path to saved plot file
        """
        
        try:
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
            
            # Extract transition data
            from_poses = []
            to_poses = []
            successes = []
            costs = []
            
            for transition in transition_history:
                # Handle different pose representation formats
                if hasattr(transition['from_pose'], 'name'):
                    from_pose = transition['from_pose'].name
                else:
                    from_pose = str(transition['from_pose']).split('.')[-1]
                
                if hasattr(transition['to_pose'], 'name'):
                    to_pose = transition['to_pose'].name
                else:
                    to_pose = str(transition['to_pose']).split('.')[-1]
                
                from_poses.append(from_pose)
                to_poses.append(to_pose)
                successes.append(transition.get('success', True))
                costs.append(transition.get('transition_cost', 2.0))
            
            if not transition_history:
                # Handle empty transition history
                for ax in [ax1, ax2, ax3, ax4]:
                    ax.text(0.5, 0.5, 'No transition data available', 
                           ha='center', va='center', transform=ax.transAxes,
                           fontsize=14, style='italic')
                plt.tight_layout()
                if save_path:
                    plt.savefig(save_path, dpi=300, bbox_inches='tight')
                if show_plot:
                    plt.show()
                else:
                    plt.close()
                return save_path or "empty_matrix.png"
            
            # Get all unique poses
            all_poses = sorted(list(set(from_poses + to_poses)))
            n_poses = len(all_poses)
            
            # Initialize matrices
            success_matrix = np.zeros((n_poses, n_poses))
            attempt_matrix = np.zeros((n_poses, n_poses))
            cost_matrix = np.zeros((n_poses, n_poses))
            
            # Fill matrices
            for from_pose, to_pose, success, cost in zip(from_poses, to_poses, successes, costs):
                i, j = all_poses.index(from_pose), all_poses.index(to_pose)
                attempt_matrix[i, j] += 1
                if success:
                    success_matrix[i, j] += 1
                cost_matrix[i, j] += cost
            
            # Calculate success rates
            success_rate_matrix = np.divide(success_matrix, attempt_matrix,
                                          out=np.zeros_like(success_matrix),
                                          where=attempt_matrix!=0) * 100
            
            # Calculate average costs
            avg_cost_matrix = np.divide(cost_matrix, attempt_matrix,
                                      out=np.zeros_like(cost_matrix),
                                      where=attempt_matrix!=0)
            
            # Plot 1: Success Rate Matrix
            im1 = ax1.imshow(success_rate_matrix, cmap='RdYlGn', vmin=0, vmax=100, aspect='equal')
            ax1.set_title('Transition Success Rates (%)', fontweight='bold', fontsize=14)
            ax1.set_xticks(range(n_poses))
            ax1.set_yticks(range(n_poses))
            ax1.set_xticklabels([pose.replace('_', '\n') for pose in all_poses], rotation=45)
            ax1.set_yticklabels([pose.replace('_', '\n') for pose in all_poses])
            ax1.set_xlabel('To Pose', fontweight='bold')
            ax1.set_ylabel('From Pose', fontweight='bold')
            
            # Add text annotations
            for i in range(n_poses):
                for j in range(n_poses):
                    if attempt_matrix[i, j] > 0:
                        text = f'{success_rate_matrix[i, j]:.0f}%\n({int(attempt_matrix[i, j])})'
                        ax1.text(j, i, text, ha='center', va='center',
                               color='white' if success_rate_matrix[i, j] < 50 else 'black',
                               fontsize=9, fontweight='bold')
            
            plt.colorbar(im1, ax=ax1, label='Success Rate (%)')
            
            # Plot 2: Attempt Count Matrix
            im2 = ax2.imshow(attempt_matrix, cmap='Blues', aspect='equal')
            ax2.set_title('Transition Attempt Counts', fontweight='bold', fontsize=14)
            ax2.set_xticks(range(n_poses))
            ax2.set_yticks(range(n_poses))
            ax2.set_xticklabels([pose.replace('_', '\n') for pose in all_poses], rotation=45)
            ax2.set_yticklabels([pose.replace('_', '\n') for pose in all_poses])
            ax2.set_xlabel('To Pose', fontweight='bold')
            ax2.set_ylabel('From Pose', fontweight='bold')
            
            # Add text annotations
            for i in range(n_poses):
                for j in range(n_poses):
                    if attempt_matrix[i, j] > 0:
                        ax2.text(j, i, f'{int(attempt_matrix[i, j])}',
                               ha='center', va='center', fontweight='bold',
                               color='white' if attempt_matrix[i, j] > np.max(attempt_matrix)/2 else 'black')
            
            plt.colorbar(im2, ax=ax2, label='Attempt Count')
            
            # Plot 3: Average Cost Matrix
            # Only show costs where attempts > 0
            masked_cost_matrix = np.ma.masked_where(attempt_matrix == 0, avg_cost_matrix)
            
            im3 = ax3.imshow(masked_cost_matrix, cmap='plasma', aspect='equal')
            ax3.set_title('Average Transition Costs', fontweight='bold', fontsize=14)
            ax3.set_xticks(range(n_poses))
            ax3.set_yticks(range(n_poses))
            ax3.set_xticklabels([pose.replace('_', '\n') for pose in all_poses], rotation=45)
            ax3.set_yticklabels([pose.replace('_', '\n') for pose in all_poses])
            ax3.set_xlabel('To Pose', fontweight='bold')
            ax3.set_ylabel('From Pose', fontweight='bold')
            
            # Add text annotations
            for i in range(n_poses):
                for j in range(n_poses):
                    if attempt_matrix[i, j] > 0:
                        ax3.text(j, i, f'{avg_cost_matrix[i, j]:.1f}',
                               ha='center', va='center', fontweight='bold', color='white')
            
            plt.colorbar(im3, ax=ax3, label='Average Cost')
            
            # Plot 4: Performance Summary Statistics
            # Create bar plots for overall statistics
            pose_stats = {}
            for pose in all_poses:
                idx = all_poses.index(pose)
                
                # Outgoing transitions (from this pose)
                outgoing_attempts = np.sum(attempt_matrix[idx, :])
                outgoing_successes = np.sum(success_matrix[idx, :])
                outgoing_success_rate = (outgoing_successes / outgoing_attempts * 100) if outgoing_attempts > 0 else 0
                
                # Incoming transitions (to this pose)
                incoming_attempts = np.sum(attempt_matrix[:, idx])
                incoming_successes = np.sum(success_matrix[:, idx])
                incoming_success_rate = (incoming_successes / incoming_attempts * 100) if incoming_attempts > 0 else 0
                
                pose_stats[pose] = {
                    'outgoing_rate': outgoing_success_rate,
                    'incoming_rate': incoming_success_rate,
                    'total_attempts': outgoing_attempts + incoming_attempts
                }
            
            # Plot statistics
            poses_short = [pose.replace('_', '\n') for pose in all_poses]
            outgoing_rates = [pose_stats[pose]['outgoing_rate'] for pose in all_poses]
            incoming_rates = [pose_stats[pose]['incoming_rate'] for pose in all_poses]
            
            x = np.arange(len(all_poses))
            width = 0.35
            
            bars1 = ax4.bar(x - width/2, outgoing_rates, width, label='Outgoing Success Rate',
                          color='skyblue', edgecolor='black', alpha=0.8)
            bars2 = ax4.bar(x + width/2, incoming_rates, width, label='Incoming Success Rate',
                          color='lightcoral', edgecolor='black', alpha=0.8)
            
            ax4.set_xlabel('Pose Mode', fontweight='bold')
            ax4.set_ylabel('Success Rate (%)', fontweight='bold')
            ax4.set_title('Pose Transition Performance Summary', fontweight='bold', fontsize=14)
            ax4.set_xticks(x)
            ax4.set_xticklabels(poses_short)
            ax4.legend()
            ax4.grid(True, alpha=0.3, axis='y')
            ax4.set_ylim(0, 105)
            
            # Add value labels on bars
            for bars in [bars1, bars2]:
                for bar in bars:
                    height = bar.get_height()
                    if height > 0:
                        ax4.text(bar.get_x() + bar.get_width()/2., height + 2,
                               f'{height:.0f}%', ha='center', va='bottom', fontsize=9)
            
            plt.tight_layout()
            
            # Save plot
            if save_path is None:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                save_path = f"origaker_transition_matrix_{timestamp}.png"
            
            plt.savefig(save_path, dpi=self.figure_style['dpi'], bbox_inches='tight')
            print(f"✅ Performance matrix saved to {save_path}")
            
            if show_plot:
                plt.show()
            else:
                plt.close()
            
            return save_path
            
        except Exception as e:
            print(f"⚠️ Performance matrix plot failed: {e}")
            return ""
    
    def plot_terrain_correlation_analysis(self,
                                        transition_history: List[Dict],
                                        terrain_analysis_history: List[Dict],
                                        save_path: Optional[str] = None,
                                        show_plot: bool = True) -> str:
        """
        Analyze and plot correlations between terrain features and mode transitions.
        
        Args:
            transition_history: List of transition records
            terrain_analysis_history: List of terrain analysis results
            save_path: Path to save the plot
            show_plot: Whether to display the plot
            
        Returns:
            Path to saved plot file
        """
        
        try:
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
            
            if not terrain_analysis_history:
                # Handle empty terrain history
                for ax in [ax1, ax2, ax3, ax4]:
                    ax.text(0.5, 0.5, 'No terrain analysis data available', 
                           ha='center', va='center', transform=ax.transAxes,
                           fontsize=14, style='italic')
                plt.tight_layout()
                if save_path:
                    plt.savefig(save_path, dpi=300, bbox_inches='tight')
                if show_plot:
                    plt.show()
                else:
                    plt.close()
                return save_path or "empty_terrain_correlation.png"
            
            # Create DataFrame from terrain analysis
            terrain_df = pd.DataFrame(terrain_analysis_history)
            
            # Extract key terrain metrics
            terrain_metrics = ['max_elevation_ahead', 'corridor_width', 'terrain_roughness', 'obstacle_density']
            available_metrics = [metric for metric in terrain_metrics if metric in terrain_df.columns]
            
            if len(available_metrics) < 2:
                print("⚠️ Insufficient terrain metrics for correlation analysis")
                return ""
            
            # Plot 1: Terrain Metrics Over Time
            time_steps = range(len(terrain_df))
            
            for i, metric in enumerate(available_metrics[:4]):  # Max 4 metrics
                color = plt.cm.tab10(i)
                ax1.plot(time_steps, terrain_df[metric], label=metric.replace('_', ' ').title(),
                        color=color, linewidth=2, alpha=0.8)
            
            # Mark transition points
            for transition in transition_history:
                # Approximate time step (assuming regular intervals)
                time_step = int(transition['timestamp'] / 0.1)  # Assume 0.1s per analysis
                if time_step < len(terrain_df):
                    ax1.axvline(time_step, color='red', linestyle='--', alpha=0.6)
            
            ax1.set_xlabel('Analysis Time Step', fontweight='bold')
            ax1.set_ylabel('Metric Value', fontweight='bold')
            ax1.set_title('Terrain Metrics Evolution with Transitions', fontweight='bold')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            
            # Plot 2: Mode Distribution by Terrain Type
            if transition_history:
                # Create mode distribution data
                mode_terrain_data = []
                
                for transition in transition_history:
                    timestamp = transition['timestamp']
                    # Find corresponding terrain analysis
                    analysis_idx = min(int(timestamp / 0.1), len(terrain_df) - 1)
                    
                    if analysis_idx < len(terrain_df):
                        terrain_metrics_at_transition = terrain_df.iloc[analysis_idx]
                        
                        # Classify terrain
                        if terrain_metrics_at_transition.get('max_elevation_ahead', 0) > 0.3:
                            terrain_type = 'obstacles'
                        elif terrain_metrics_at_transition.get('corridor_width', 2.0) < 0.8:
                            terrain_type = 'narrow_corridor'
                        elif terrain_metrics_at_transition.get('terrain_roughness', 0.1) > 0.2:
                            terrain_type = 'rough_terrain'
                        else:
                            terrain_type = 'flat_surface'
                        
                        # Get target mode
                        if hasattr(transition['to_pose'], 'name'):
                            target_mode = transition['to_pose'].name
                        else:
                            target_mode = str(transition['to_pose']).split('.')[-1]
                        
                        mode_terrain_data.append({
                            'terrain_type': terrain_type,
                            'target_mode': target_mode
                        })
                
                if mode_terrain_data:
                    # Create crosstab
                    mode_terrain_df = pd.DataFrame(mode_terrain_data)
                    crosstab = pd.crosstab(mode_terrain_df['terrain_type'], mode_terrain_df['target_mode'])
                    
                    # Normalize to percentages
                    crosstab_pct = crosstab.div(crosstab.sum(axis=1), axis=0) * 100
                    
                    # Create stacked bar plot
                    crosstab_pct.plot(kind='bar', stacked=True, ax=ax2, 
                                    color=[self.mode_colors.get(mode, '#808080') for mode in crosstab_pct.columns])
                    
                    ax2.set_xlabel('Terrain Type', fontweight='bold')
                    ax2.set_ylabel('Mode Distribution (%)', fontweight='bold')
                    ax2.set_title('Mode Selection by Terrain Type', fontweight='bold')
                    ax2.legend(title='Target Mode', bbox_to_anchor=(1.05, 1), loc='upper left')
                    ax2.tick_params(axis='x', rotation=45)
            
            # Plot 3: Terrain Metric Correlation Heatmap
            if len(available_metrics) >= 2:
                correlation_matrix = terrain_df[available_metrics].corr()
                
                im = ax3.imshow(correlation_matrix, cmap='coolwarm', vmin=-1, vmax=1, aspect='equal')
                ax3.set_title('Terrain Metrics Correlation Matrix', fontweight='bold')
                
                # Set ticks and labels
                ax3.set_xticks(range(len(available_metrics)))
                ax3.set_yticks(range(len(available_metrics)))
                ax3.set_xticklabels([metric.replace('_', '\n') for metric in available_metrics], rotation=45)
                ax3.set_yticklabels([metric.replace('_', '\n') for metric in available_metrics])
                
                # Add correlation values
                for i in range(len(available_metrics)):
                    for j in range(len(available_metrics)):
                        text = ax3.text(j, i, f'{correlation_matrix.iloc[i, j]:.2f}',
                                      ha='center', va='center', fontweight='bold',
                                      color='white' if abs(correlation_matrix.iloc[i, j]) > 0.5 else 'black')
                
                plt.colorbar(im, ax=ax3, label='Correlation Coefficient')
            
            # Plot 4: Transition Trigger Analysis
            if transition_history and available_metrics:
                # Analyze what terrain conditions trigger transitions
                trigger_data = {}
                
                for metric in available_metrics:
                    trigger_values = []
                    
                    for transition in transition_history:
                        timestamp = transition['timestamp']
                        analysis_idx = min(int(timestamp / 0.1), len(terrain_df) - 1)
                        
                        if analysis_idx < len(terrain_df):
                            trigger_values.append(terrain_df.iloc[analysis_idx][metric])
                    
                    if trigger_values:
                        trigger_data[metric] = trigger_values
                
                # Create box plots
                if trigger_data:
                    box_data = []
                    box_labels = []
                    
                    for metric, values in trigger_data.items():
                        box_data.append(values)
                        box_labels.append(metric.replace('_', '\n'))
                    
                    bp = ax4.boxplot(box_data, labels=box_labels, patch_artist=True)
                    
                    # Color boxes
                    colors = plt.cm.tab10(np.linspace(0, 1, len(box_data)))
                    for patch, color in zip(bp['boxes'], colors):
                        patch.set_facecolor(color)
                        patch.set_alpha(0.7)
                    
                    ax4.set_ylabel('Metric Value at Transition', fontweight='bold')
                    ax4.set_title('Terrain Conditions That Trigger Transitions', fontweight='bold')
                    ax4.grid(True, alpha=0.3, axis='y')
                    
                    # Add mean markers
                    for i, values in enumerate(box_data):
                        mean_val = np.mean(values)
                        ax4.plot(i+1, mean_val, 'ro', markersize=8, label='Mean' if i == 0 else "")
                    
                    if len(box_data) > 0:
                        ax4.legend()
            
            plt.tight_layout()
            
            # Save plot
            if save_path is None:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                save_path = f"origaker_terrain_correlation_{timestamp}.png"
            
            plt.savefig(save_path, dpi=self.figure_style['dpi'], bbox_inches='tight')
            print(f"✅ Terrain correlation analysis saved to {save_path}")
            
            if show_plot:
                plt.show()
            else:
                plt.close()
            
            return save_path
            
        except Exception as e:
            print(f"⚠️ Terrain correlation plot failed: {e}")
            return ""
    
    def create_comprehensive_report(self,
                                  transition_history: List[Dict],
                                  episode_duration: float,
                                  robot_trajectory: Optional[List[Tuple[float, float]]] = None,
                                  terrain_analysis_history: Optional[List[Dict]] = None,
                                  terrain_events: Optional[List[Dict]] = None,
                                  save_directory: str = "origaker_transition_reports") -> str:
        """
        Create comprehensive transition analysis report with all visualizations.
        
        Args:
            transition_history: List of transition records
            episode_duration: Total episode duration
            robot_trajectory: Optional trajectory data
            terrain_analysis_history: Optional terrain analysis data
            terrain_events: Optional terrain events
            save_directory: Directory to save the report
            
        Returns:
            Path to report directory
        """
        
        try:
            # Create report directory
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            report_dir = Path(save_directory) / f"origaker_report_{timestamp}"
            report_dir.mkdir(parents=True, exist_ok=True)
            
            print(f"📊 Generating comprehensive Origaker transition report...")
            print(f"   Report directory: {report_dir}")
            
            generated_plots = []
            
            # Generate timeline plot
            timeline_path = self.plot_transition_timeline(
                transition_history, episode_duration, robot_trajectory, terrain_events,
                save_path=str(report_dir / "01_transition_timeline.png"),
                show_plot=False
            )
            if timeline_path:
                generated_plots.append("Transition Timeline")
            
            # Generate performance matrix
            matrix_path = self.plot_transition_performance_matrix(
                transition_history,
                save_path=str(report_dir / "02_performance_matrix.png"),
                show_plot=False
            )
            if matrix_path:
                generated_plots.append("Performance Matrix")
            
            # Generate terrain correlation analysis
            if terrain_analysis_history:
                correlation_path = self.plot_terrain_correlation_analysis(
                    transition_history, terrain_analysis_history,
                    save_path=str(report_dir / "03_terrain_correlation.png"),
                    show_plot=False
                )
                if correlation_path:
                    generated_plots.append("Terrain Correlation")
            
            # Generate summary statistics
            stats = self._calculate_transition_statistics(transition_history, episode_duration)
            
            # Save statistics as JSON
            stats_file = report_dir / "transition_statistics.json"
            with open(stats_file, 'w') as f:
                # Convert any enum values to strings
                clean_stats = self._clean_stats_for_json(stats)
                json.dump(clean_stats, f, indent=2)
            
            # Generate text summary report
            summary_file = report_dir / "transition_summary.txt"
            with open(summary_file, 'w') as f:
                f.write("ORIGAKER MORPHOLOGY TRANSITION ANALYSIS REPORT\n")
                f.write("=" * 55 + "\n\n")
                f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"Episode Duration: {episode_duration:.2f} seconds\n")
                f.write(f"Total Transitions: {len(transition_history)}\n\n")
                
                f.write("PERFORMANCE SUMMARY:\n")
                f.write("-" * 20 + "\n")
                f.write(f"Transition Rate: {stats['transition_rate']:.2f} transitions/minute\n")
                f.write(f"Success Rate: {stats['success_rate']:.1f}%\n")
                f.write(f"Average Transition Cost: {stats['avg_transition_cost']:.2f}\n")
                f.write(f"Most Used Mode: {stats['most_used_mode']}\n")
                f.write(f"Most Reliable Transition: {stats['most_reliable_transition']}\n\n")
                
                f.write("MODE USAGE:\n")
                f.write("-" * 12 + "\n")
                for mode, percentage in stats['mode_usage_percentage'].items():
                    f.write(f"{mode}: {percentage:.1f}%\n")
                
                f.write(f"\nGenerated Visualizations: {', '.join(generated_plots)}\n")
            
            print(f"✅ Comprehensive report generated!")
            print(f"   📁 Location: {report_dir}")
            print(f"   📊 Plots: {len(generated_plots)}")
            print(f"   📈 Files: statistics.json, summary.txt")
            
            return str(report_dir)
            
        except Exception as e:
            print(f"⚠️ Report generation failed: {e}")
            return ""
    
    def _calculate_transition_statistics(self, transition_history: List[Dict], episode_duration: float) -> Dict:
        """Calculate comprehensive transition statistics."""
        
        stats = {
            'total_transitions': len(transition_history),
            'episode_duration': episode_duration,
            'transition_rate': 0.0,
            'success_rate': 0.0,
            'avg_transition_cost': 0.0,
            'mode_usage_percentage': {},
            'most_used_mode': 'SPREADER',
            'most_reliable_transition': 'None',
            'transition_frequency_by_hour': {}
        }
        
        if not transition_history:
            return stats
        
        # Basic rates
        stats['transition_rate'] = len(transition_history) / (episode_duration / 60)  # per minute
        
        # Success rate
        successes = sum(1 for t in transition_history if t.get('success', True))
        stats['success_rate'] = (successes / len(transition_history)) * 100
        
        # Average cost
        costs = [t.get('transition_cost', 2.0) for t in transition_history]
        stats['avg_transition_cost'] = np.mean(costs)
        
        # Mode usage analysis
        mode_durations = {'SPREADER': episode_duration}  # Start with full duration for starting mode
        current_mode = 'SPREADER'
        last_time = 0.0
        
        for transition in transition_history:
            # Calculate duration in current mode
            duration = transition['timestamp'] - last_time
            if current_mode not in mode_durations:
                mode_durations[current_mode] = 0
            mode_durations[current_mode] += duration
            
            # Update to new mode
            if hasattr(transition['to_pose'], 'name'):
                current_mode = transition['to_pose'].name
            else:
                current_mode = str(transition['to_pose']).split('.')[-1]
            
            last_time = transition['timestamp']
        
        # Final mode duration
        final_duration = episode_duration - last_time
        if current_mode not in mode_durations:
            mode_durations[current_mode] = 0
        mode_durations[current_mode] += final_duration
        
        # Convert to percentages
        for mode, duration in mode_durations.items():
            stats['mode_usage_percentage'][mode] = (duration / episode_duration) * 100
        
        # Most used mode
        stats['most_used_mode'] = max(mode_durations, key=mode_durations.get)
        
        return stats
    
    def _clean_stats_for_json(self, stats: Dict) -> Dict:
        """Clean statistics dictionary for JSON serialization."""
        
        clean_stats = {}
        for key, value in stats.items():
            if hasattr(value, 'name'):  # Enum value
                clean_stats[key] = value.name
            elif isinstance(value, dict):
                clean_stats[key] = self._clean_stats_for_json(value)
            else:
                clean_stats[key] = value
        
        return clean_stats


def test_origaker_transition_plotter():
    """Test the Origaker transition plotting system."""
    print("🧪 Testing Origaker Transition Plotter")
    print("=" * 40)
    
    try:
        # Initialize plotter
        plotter = OrigakerTransitionPlotter()
        
        # Create test data
        print("📊 Creating test data...")
        
        # Mock transition history
        transition_history = [
            {
                'timestamp': 5.0,
                'from_pose': 'SPREADER',
                'to_pose': 'CRAWLER', 
                'transition_cost': 2.0,
                'success': True
            },
            {
                'timestamp': 15.0,
                'from_pose': 'CRAWLER',
                'to_pose': 'HIGH_STEP',
                'transition_cost': 3.0,
                'success': True
            },
            {
                'timestamp': 25.0,
                'from_pose': 'HIGH_STEP',
                'to_pose': 'SPREADER',
                'transition_cost': 2.5,
                'success': False
            },
            {
                'timestamp': 35.0,
                'from_pose': 'SPREADER',
                'to_pose': 'ROLLING',
                'transition_cost': 3.5,
                'success': True
            }
        ]
        
        # Mock trajectory
        trajectory = []
        for i in range(100):
            x = i * 0.05
            y = 0.5 * np.sin(i * 0.1) + 0.02 * np.random.randn()
            trajectory.append((x, y))
        
        # Mock terrain analysis
        terrain_analysis = []
        for i in range(50):
            analysis = {
                'max_elevation_ahead': 0.1 + 0.3 * np.random.rand(),
                'corridor_width': 1.0 + 2.0 * np.random.rand(),
                'terrain_roughness': 0.05 + 0.3 * np.random.rand(),
                'obstacle_density': 0.1 * np.random.rand()
            }
            terrain_analysis.append(analysis)
        
        episode_duration = 40.0
        
        print("✅ Test data created")
        
        # Test timeline plot
        print("\n📈 Testing timeline plot...")
        timeline_path = plotter.plot_transition_timeline(
            transition_history, episode_duration, trajectory, show_plot=False
        )
        print(f"    ✓ Timeline plot: {timeline_path}")
        
        # Test performance matrix
        print("\n📊 Testing performance matrix...")
        matrix_path = plotter.plot_transition_performance_matrix(
            transition_history, show_plot=False
        )
        print(f"    ✓ Performance matrix: {matrix_path}")
        
        # Test terrain correlation
        print("\n🗺️ Testing terrain correlation...")
        correlation_path = plotter.plot_terrain_correlation_analysis(
            transition_history, terrain_analysis, show_plot=False
        )
        print(f"    ✓ Terrain correlation: {correlation_path}")
        
        # Test comprehensive report
        print("\n📋 Testing comprehensive report...")
        report_dir = plotter.create_comprehensive_report(
            transition_history, episode_duration, trajectory, terrain_analysis
        )
        print(f"    ✓ Comprehensive report: {report_dir}")
        
        print(f"\n✅ Origaker Transition Plotter Test Complete!")
        print(f"   - Timeline visualization: ✅")
        print(f"   - Performance analysis: ✅")
        print(f"   - Terrain correlation: ✅")
        print(f"   - Comprehensive reporting: ✅")
        
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    test_origaker_transition_plotter()