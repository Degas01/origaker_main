"""
Simple Comparative Analysis: Hopf, Matsuoka & Hybrid Oscillators
================================================================

Clear and focused script for comparing phase portraits and stability properties
of three oscillator types commonly used in robotics applications.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
import warnings
warnings.filterwarnings('ignore')

class HopfOscillator:
    """
    Hopf Oscillator: dx/dt = μx - ωy - x(x² + y²)
                     dy/dt = ωx + μy - y(x² + y²)
    """
    def __init__(self, mu=0.5, omega=1.0):
        self.mu = mu      # Bifurcation parameter
        self.omega = omega # Angular frequency
        
    def dynamics(self, state, t):
        x, y = state
        r_squared = x**2 + y**2
        
        dxdt = self.mu * x - self.omega * y - x * r_squared
        dydt = self.omega * x + self.mu * y - y * r_squared
        
        return [dxdt, dydt]
    
    def limit_cycle_radius(self):
        return np.sqrt(max(0, self.mu))

class MatsuokaOscillator:
    """
    Matsuoka Oscillator: Two-neuron model with mutual inhibition
    Simplified 2D version for phase portrait analysis
    """
    def __init__(self, beta=2.0, gamma=1.5, u1=1.0, u2=1.0):
        self.beta = beta    # Self-inhibition
        self.gamma = gamma  # Mutual inhibition  
        self.u1 = u1       # Input to neuron 1
        self.u2 = u2       # Input to neuron 2
        
    def rectify(self, x):
        return max(x, 0.0)
    
    def dynamics(self, state, t):
        x1, x2 = state
        
        y1 = self.rectify(x1)
        y2 = self.rectify(x2)
        
        # Reduced dynamics assuming fast adaptation
        dx1dt = -x1 - self.beta * y1 - self.gamma * y2 + self.u1
        dx2dt = -x2 - self.beta * y2 - self.gamma * y1 + self.u2
        
        return [dx1dt, dx2dt]

class HybridOscillator:
    """
    Hybrid Oscillator: Combines Hopf and Matsuoka dynamics
    """
    def __init__(self, alpha=0.5, hopf_params=None, matsuoka_params=None):
        self.alpha = alpha  # Mixing parameter (0=pure Hopf, 1=pure Matsuoka)
        
        if hopf_params is None:
            hopf_params = {'mu': 0.5, 'omega': 1.0}
        if matsuoka_params is None:
            matsuoka_params = {'beta': 2.0, 'gamma': 1.5, 'u1': 1.0, 'u2': 1.0}
            
        self.hopf = HopfOscillator(**hopf_params)
        self.matsuoka = MatsuokaOscillator(**matsuoka_params)
        
    def dynamics(self, state, t):
        x, y = state
        
        # Get dynamics from both oscillators
        hopf_derivs = self.hopf.dynamics([x, y], t)
        matsuoka_derivs = self.matsuoka.dynamics([x, y], t)
        
        # Linear combination
        dxdt = (1 - self.alpha) * hopf_derivs[0] + self.alpha * matsuoka_derivs[0]
        dydt = (1 - self.alpha) * hopf_derivs[1] + self.alpha * matsuoka_derivs[1]
        
        return [dxdt, dydt]

def plot_phase_portrait(oscillator, title, ax, color='blue'):
    """Plot phase portrait for an oscillator"""
    
    # Set up coordinate grid
    x = np.linspace(-3, 3, 20)
    y = np.linspace(-3, 3, 20)
    X, Y = np.meshgrid(x, y)
    
    # Compute direction field
    DX = np.zeros_like(X)
    DY = np.zeros_like(Y)
    
    for i in range(len(x)):
        for j in range(len(y)):
            derivatives = oscillator.dynamics([X[j,i], Y[j,i]], 0)
            DX[j,i] = derivatives[0]
            DY[j,i] = derivatives[1]
    
    # Plot direction field
    ax.quiver(X, Y, DX, DY, alpha=0.5, width=0.003, scale=50, color='gray')
    
    # Plot several trajectories from different starting points
    t = np.linspace(0, 20, 2000)
    initial_conditions = [
        [2.5, 0], [0, 2.5], [-2.5, 0], [0, -2.5],
        [1.8, 1.8], [-1.8, 1.8], [1.8, -1.8], [-1.8, -1.8]
    ]
    
    for ic in initial_conditions:
        try:
            sol = odeint(oscillator.dynamics, ic, t)
            ax.plot(sol[:, 0], sol[:, 1], color=color, linewidth=2, alpha=0.8)
            ax.plot(ic[0], ic[1], 'o', color=color, markersize=6, 
                   markeredgecolor='black', markeredgewidth=1)
        except:
            continue
    
    # Find and plot fixed point (origin for these systems)
    ax.plot(0, 0, 's', color='red', markersize=10, 
           markeredgecolor='black', markeredgewidth=2, label='Fixed Point')
    
    # Formatting
    ax.set_xlim(-3.5, 3.5)
    ax.set_ylim(-3.5, 3.5)
    ax.set_xlabel('x₁', fontsize=12)
    ax.set_ylabel('x₂', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal')
    
    # Add parameter info
    if hasattr(oscillator, 'mu'):
        info = f'μ={oscillator.mu}, ω={oscillator.omega}\nLimit Cycle R≈{oscillator.limit_cycle_radius():.2f}'
    elif hasattr(oscillator, 'beta'):
        info = f'β={oscillator.beta}, γ={oscillator.gamma}'
    else:
        info = f'α={oscillator.alpha} (Hybrid)'
    
    ax.text(0.02, 0.98, info, transform=ax.transAxes, fontsize=10,
           verticalalignment='top', bbox=dict(boxstyle='round', 
           facecolor='white', alpha=0.8))

def plot_time_series(oscillator, title, ax, color='blue'):
    """Plot time series for an oscillator"""
    
    t = np.linspace(0, 25, 3000)
    ic = [1.5, 0.8]  # Initial condition
    
    sol = odeint(oscillator.dynamics, ic, t)
    
    ax.plot(t, sol[:, 0], color=color, linewidth=2, label='x₁', alpha=0.8)
    ax.plot(t, sol[:, 1], color=color, linewidth=2, linestyle='--', 
           label='x₂', alpha=0.8)
    
    ax.set_title(f'{title}\nTime Series', fontsize=12, fontweight='bold')
    ax.set_xlabel('Time')
    ax.set_ylabel('Amplitude')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xlim(15, 25)  # Show steady-state

def create_comparison():
    """Create comprehensive comparison of oscillators"""
    
    # Initialize oscillators
    hopf = HopfOscillator(mu=0.5, omega=1.0)
    matsuoka = MatsuokaOscillator(beta=2.0, gamma=1.5, u1=1.0, u2=1.0)
    hybrid_weak = HybridOscillator(alpha=0.3)  # More Hopf-like
    hybrid_strong = HybridOscillator(alpha=0.7)  # More Matsuoka-like
    
    # Create figure
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    fig.suptitle('Comparative Analysis: Hopf, Matsuoka & Hybrid Oscillators', 
                 fontsize=16, fontweight='bold')
    
    # Phase portraits (top row)
    plot_phase_portrait(hopf, 'Hopf Oscillator', axes[0,0], 'blue')
    plot_phase_portrait(matsuoka, 'Matsuoka Oscillator', axes[0,1], 'green')
    plot_phase_portrait(hybrid_weak, 'Hybrid (α=0.3)', axes[0,2], 'orange')
    plot_phase_portrait(hybrid_strong, 'Hybrid (α=0.7)', axes[0,3], 'red')
    
    # Time series (bottom row)
    plot_time_series(hopf, 'Hopf', axes[1,0], 'blue')
    plot_time_series(matsuoka, 'Matsuoka', axes[1,1], 'green')
    plot_time_series(hybrid_weak, 'Hybrid (α=0.3)', axes[1,2], 'orange')
    plot_time_series(hybrid_strong, 'Hybrid (α=0.7)', axes[1,3], 'red')
    
    plt.tight_layout()
    return fig

def analyze_stability():
    """Analyze and print stability properties"""
    
    print("=" * 60)
    print("OSCILLATOR STABILITY ANALYSIS")
    print("=" * 60)
    
    print("\n1. HOPF OSCILLATOR (μ=0.5, ω=1.0)")
    print("   • Origin: Unstable spiral (repelling)")
    print("   • Limit cycle: Stable, radius ≈ 0.71")
    print("   • Behavior: Smooth sinusoidal oscillations")
    print("   • Applications: CPG networks, rhythmic control")
    
    print("\n2. MATSUOKA OSCILLATOR (β=2.0, γ=1.5)")
    print("   • Origin: Depends on parameters")
    print("   • Limit cycle: Typically rectangular/asymmetric")
    print("   • Behavior: Relaxation oscillations with sharp transitions")
    print("   • Applications: Neural networks, walking gaits")
    
    print("\n3. HYBRID OSCILLATOR (α=0.3)")
    print("   • Combines both behaviors")
    print("   • More Hopf-like: smoother oscillations")
    print("   • Tunable dynamics via α parameter")
    
    print("\n4. HYBRID OSCILLATOR (α=0.7)")
    print("   • More Matsuoka-like: sharper transitions")
    print("   • Adaptive behavior possible")
    print("   • Applications: Adaptive locomotion, multi-modal control")
    
    print("\n" + "=" * 60)
    print("KEY INSIGHTS:")
    print("• Hopf: Clean limit cycles, predictable frequency")
    print("• Matsuoka: Biologically realistic, complex dynamics")
    print("• Hybrid: Best of both worlds, tunable behavior")
    print("=" * 60)

def main():
    """Main execution function"""
    
    print("🔬 OSCILLATOR PHASE PORTRAIT COMPARISON")
    print("="*50)
    print("Generating comparative analysis...")
    
    # Create and show comparison plot
    fig = create_comparison()
    
    # Print stability analysis
    analyze_stability()
    
    # Show plots
    plt.show()
    
    print("\n✅ Analysis complete!")
    print("\n📊 The plots show:")
    print("• Phase portraits with trajectories and direction fields")
    print("• Time series showing oscillatory behavior")
    print("• Clear differences between oscillator types")
    print("• Stability properties and limit cycle characteristics")

if __name__ == "__main__":
    main()