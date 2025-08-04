"""
Hopf Oscillator Implementation

This module implements the Hopf oscillator, a nonlinear dynamical system that
exhibits stable limit cycle behavior. The Hopf oscillator is widely used in
robotics and neuroscience for generating smooth, stable oscillations.

The oscillator is based on the normal form of a supercritical Hopf bifurcation,
which creates a stable limit cycle when the bifurcation parameter μ > 0.

References:
- Righetti, L., & Ijspeert, A. J. (2006). Programmable central pattern generators: 
  an application to biped locomotion control. In Proceedings 2006 IEEE International 
  Conference on Robotics and Automation (pp. 1585-1590).
- Strogatz, S. H. (2018). Nonlinear dynamics and chaos: with applications to physics, 
  biology, chemistry, and engineering. CRC press.
"""

import numpy as np
import matplotlib.pyplot as plt


class HopfOscillator:
    """
    Hopf oscillator with stable limit cycle dynamics.
    
    The Hopf oscillator is a 2D nonlinear dynamical system that exhibits
    stable limit cycle behavior. It's particularly useful for generating
    smooth, continuous oscillations with controllable amplitude and frequency.
    
    Dynamics:
        dx/dt = (μ - (x² + y²)) * x - ω * y
        dy/dt = (μ - (x² + y²)) * y + ω * x
    
    where:
        r² = x² + y²  (squared distance from origin)
    
    Key Properties:
        - For μ > 0: Stable limit cycle with radius √μ
        - For μ = 0: Marginal stability (center)
        - For μ < 0: Stable fixed point at origin
        - ω controls the angular frequency of oscillation
    
    Parameters:
        mu (float): Bifurcation parameter controlling limit cycle radius.
                   - μ > 0: Stable limit cycle with radius √μ
                   - μ = 0: Marginal case
                   - μ < 0: Damped oscillations to origin
                   Typical range: 0.1-4.0 for stable oscillations
        
        omega (float): Natural angular frequency in rad/s.
                      Controls how fast the oscillator rotates around the limit cycle.
                      - Positive: Counter-clockwise rotation
                      - Negative: Clockwise rotation
                      Typical range: 1.0-20.0 rad/s
    
    State Variables:
        x (float): First state variable (can represent position, velocity, etc.)
        y (float): Second state variable (90° phase shifted from x)
    
    Behavioral Notes:
        - The system automatically converges to a circular limit cycle
        - No external forcing needed - self-sustaining oscillations
        - Amplitude is automatically regulated by the nonlinear term
        - Robust to perturbations - returns to limit cycle
        - Smooth sinusoidal outputs when on the limit cycle
    """
    
    def __init__(self, mu=1.0, omega=2*np.pi):
        """
        Initialize the Hopf oscillator with given parameters.
        
        Args:
            mu (float): Bifurcation parameter (controls limit cycle radius)
                       Default: 1.0 (creates unit circle limit cycle)
            omega (float): Angular frequency in rad/s
                          Default: 2π (1 Hz oscillation)
        """
        # Store parameters
        self.mu = mu
        self.omega = omega
        
        # Initialize state variables
        self.reset()
    
    def reset(self, x=None, y=None):
        """
        Reset the oscillator state to initial conditions.
        
        For Hopf oscillators, it's often useful to start with small random
        perturbations to avoid starting exactly at the unstable fixed point (0,0)
        when μ > 0.
        
        Args:
            x (float, optional): Initial x value. If None, uses small random value.
            y (float, optional): Initial y value. If None, uses small random value.
        """
        if x is None:
            # Small random perturbation to avoid starting at unstable origin
            self.x = np.random.normal(0, 0.1)
        else:
            self.x = float(x)
            
        if y is None:
            # Small random perturbation to avoid starting at unstable origin  
            self.y = np.random.normal(0, 0.1)
        else:
            self.y = float(y)
    
    def step(self, dt=0.01):
        """
        Advance the oscillator state by one time step using explicit Euler integration.
        
        Args:
            dt (float): Time step size for integration. Smaller values give more
                       accurate results. For Hopf oscillators, dt should be much
                       smaller than 1/ω for stability. Typical: 0.001-0.01
        
        Returns:
            tuple: (x, y) - The current state values
            
        Notes:
            The integration uses the standard Euler method:
            x_{n+1} = x_n + dt * dx/dt
            y_{n+1} = y_n + dt * dy/dt
        """
        # Compute squared radius r² = x² + y²
        r2 = self.x * self.x + self.y * self.y
        
        # Compute derivatives
        dx_dt = (self.mu - r2) * self.x - self.omega * self.y
        dy_dt = (self.mu - r2) * self.y + self.omega * self.x
        
        # Update state using explicit Euler integration
        self.x += dt * dx_dt
        self.y += dt * dy_dt
        
        return self.x, self.y
    
    def get_state(self):
        """
        Get the current complete state of the oscillator.
        
        Returns:
            dict: Dictionary containing state variables and derived quantities
        """
        r = np.sqrt(self.x*self.x + self.y*self.y)
        theta = np.arctan2(self.y, self.x)
        
        return {
            'x': self.x,
            'y': self.y,
            'radius': r,
            'phase': theta,
            'target_radius': np.sqrt(max(0, self.mu))
        }
    
    def set_parameters(self, **kwargs):
        """
        Update oscillator parameters.
        
        Args:
            **kwargs: Any of mu, omega
        """
        for param, value in kwargs.items():
            if hasattr(self, param):
                setattr(self, param, value)
            else:
                raise ValueError(f"Unknown parameter: {param}")
    
    def get_limit_cycle_radius(self):
        """
        Get the theoretical limit cycle radius.
        
        Returns:
            float: √μ if μ > 0, else 0
        """
        return np.sqrt(max(0, self.mu))
    
    def get_frequency_hz(self):
        """
        Get the oscillation frequency in Hz.
        
        Returns:
            float: Frequency in Hz (ω / 2π)
        """
        return self.omega / (2 * np.pi)


def test_hopf_oscillator():
    """
    Test the Hopf oscillator and visualize its behavior.
    """
    # Create oscillator with default parameters
    osc = HopfOscillator(mu=1.0, omega=2*np.pi)
    osc.reset()
    
    # Simulation parameters
    dt = 0.01
    total_time = 5.0
    steps = int(total_time / dt)
    
    # Storage for results
    time_points = []
    x_values = []
    y_values = []
    radius_values = []
    
    # Run simulation
    print("Running Hopf oscillator simulation...")
    print(f"Parameters: μ={osc.mu}, ω={osc.omega:.2f} rad/s ({osc.get_frequency_hz():.2f} Hz)")
    print(f"Expected limit cycle radius: {osc.get_limit_cycle_radius():.3f}")
    
    for i in range(steps):
        x, y = osc.step(dt)
        
        # Store results
        time_points.append(i * dt)
        x_values.append(x)
        y_values.append(y)
        radius_values.append(np.sqrt(x*x + y*y))
    
    print(f"Final state: x={x:.3f}, y={y:.3f}, radius={np.sqrt(x*x + y*y):.3f}")
    
    # Create visualization
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Time series plot
    axes[0,0].plot(time_points, x_values, label='x(t)', linewidth=2)
    axes[0,0].plot(time_points, y_values, label='y(t)', linewidth=2)
    axes[0,0].set_xlabel('Time (s)')
    axes[0,0].set_ylabel('State')
    axes[0,0].set_title('Hopf Oscillator Time Series')
    axes[0,0].legend()
    axes[0,0].grid(True, alpha=0.3)
    
    # Phase portrait
    axes[0,1].plot(x_values, y_values, 'b-', linewidth=2, alpha=0.7)
    axes[0,1].plot(x_values[0], y_values[0], 'go', markersize=8, label='Start')
    axes[0,1].plot(x_values[-1], y_values[-1], 'ro', markersize=8, label='End')
    
    # Draw theoretical limit cycle
    if osc.mu > 0:
        theta_circle = np.linspace(0, 2*np.pi, 100)
        r_circle = np.sqrt(osc.mu)
        x_circle = r_circle * np.cos(theta_circle)
        y_circle = r_circle * np.sin(theta_circle)
        axes[0,1].plot(x_circle, y_circle, 'k--', linewidth=2, alpha=0.5, label=f'Limit cycle (r={r_circle:.2f})')
    
    axes[0,1].set_xlabel('x')
    axes[0,1].set_ylabel('y')
    axes[0,1].set_title('Phase Portrait')
    axes[0,1].legend()
    axes[0,1].grid(True, alpha=0.3)
    axes[0,1].axis('equal')
    
    # Radius evolution
    axes[1,0].plot(time_points, radius_values, 'g-', linewidth=2, label='Actual radius')
    if osc.mu > 0:
        axes[1,0].axhline(y=np.sqrt(osc.mu), color='r', linestyle='--', 
                         linewidth=2, label=f'Target radius ({np.sqrt(osc.mu):.2f})')
    axes[1,0].set_xlabel('Time (s)')
    axes[1,0].set_ylabel('Radius')
    axes[1,0].set_title('Convergence to Limit Cycle')
    axes[1,0].legend()
    axes[1,0].grid(True, alpha=0.3)
    
    # Frequency analysis (simple)
    # Find peaks in x to estimate period
    x_array = np.array(x_values)
    t_array = np.array(time_points)
    
    # Simple zero-crossing detection for frequency estimation
    zero_crossings = []
    for i in range(1, len(x_array)):
        if x_array[i-1] <= 0 < x_array[i]:  # Positive-going zero crossing
            zero_crossings.append(t_array[i])
    
    if len(zero_crossings) >= 2:
        periods = np.diff(zero_crossings)
        avg_period = np.mean(periods)
        measured_freq = 1.0 / avg_period
        theoretical_freq = osc.get_frequency_hz()
        
        axes[1,1].text(0.1, 0.7, f'Theoretical frequency: {theoretical_freq:.3f} Hz', 
                      transform=axes[1,1].transAxes, fontsize=12)
        axes[1,1].text(0.1, 0.6, f'Measured frequency: {measured_freq:.3f} Hz', 
                      transform=axes[1,1].transAxes, fontsize=12)
        axes[1,1].text(0.1, 0.5, f'Error: {abs(measured_freq-theoretical_freq):.4f} Hz', 
                      transform=axes[1,1].transAxes, fontsize=12)
    
    axes[1,1].text(0.1, 0.3, f'μ = {osc.mu}', transform=axes[1,1].transAxes, fontsize=12)
    axes[1,1].text(0.1, 0.2, f'ω = {osc.omega:.2f} rad/s', transform=axes[1,1].transAxes, fontsize=12)
    axes[1,1].set_title('Parameters and Frequency Analysis')
    axes[1,1].axis('off')
    
    plt.tight_layout()
    plt.show()
    
    return time_points, x_values, y_values


def compare_parameter_effects():
    """
    Compare Hopf oscillators with different parameter settings.
    """
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Test different mu values (different amplitudes)
    mu_values = [0.5, 1.0, 2.0]
    colors = ['blue', 'red', 'green']
    
    for mu, color in zip(mu_values, colors):
        osc = HopfOscillator(mu=mu, omega=2*np.pi)
        osc.reset()
        
        x_vals, y_vals = [], []
        for _ in range(500):
            x, y = osc.step(0.01)
            x_vals.append(x)
            y_vals.append(y)
        
        axes[0,0].plot(x_vals, y_vals, color=color, linewidth=2, 
                      label=f'μ={mu} (r={np.sqrt(mu):.2f})')
    
    axes[0,0].set_title('Effect of μ on Limit Cycle Radius')
    axes[0,0].set_xlabel('x')
    axes[0,0].set_ylabel('y')
    axes[0,0].legend()
    axes[0,0].grid(True, alpha=0.3)
    axes[0,0].axis('equal')
    
    # Test different omega values (different frequencies)
    omega_values = [1*np.pi, 2*np.pi, 4*np.pi]
    
    for omega, color in zip(omega_values, colors):
        osc = HopfOscillator(mu=1.0, omega=omega)
        osc.reset()
        
        times, x_vals = [], []
        for i in range(300):
            x, y = osc.step(0.01)
            times.append(i * 0.01)
            x_vals.append(x)
        
        freq_hz = omega / (2*np.pi)
        axes[0,1].plot(times, x_vals, color=color, linewidth=2, 
                      label=f'ω={omega:.1f} ({freq_hz:.1f} Hz)')
    
    axes[0,1].set_title('Effect of ω on Oscillation Frequency')
    axes[0,1].set_xlabel('Time (s)')
    axes[0,1].set_ylabel('x')
    axes[0,1].legend()
    axes[0,1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    # Example usage following the task specification
    print("=== Hopf Oscillator Example Usage ===")
    
    # Create oscillator
    osc = HopfOscillator(mu=1.0, omega=2*np.pi)
    osc.reset()
    
    print(f"Created Hopf oscillator with μ={osc.mu}, ω={osc.omega:.2f}")
    print(f"Expected limit cycle radius: √μ = {osc.get_limit_cycle_radius():.3f}")
    print(f"Expected frequency: {osc.get_frequency_hz():.2f} Hz")
    
    # Run simulation
    print("\nRunning simulation...")
    for i in range(1000):
        x, y = osc.step(dt=0.01)
        
        if i % 200 == 0:
            r = np.sqrt(x*x + y*y)
            print(f"Step {i:4d}: x={x:.4f}, y={y:.4f}, radius={r:.4f}")
    
    final_radius = np.sqrt(x*x + y*y)
    print(f"\nFinal state: x={x:.4f}, y={y:.4f}")
    print(f"Final radius: {final_radius:.4f} (target: {osc.get_limit_cycle_radius():.4f})")
    
    # Test visualization
    print("\n=== Running Visualization Test ===")
    test_hopf_oscillator()
    
    print("\n=== Parameter Effect Comparison ===")
    compare_parameter_effects()
    
    print("\n=== Parameter Guidelines ===")
    print("μ: Bifurcation parameter")
    print("   μ > 0: Stable limit cycle with radius √μ")
    print("   μ = 0: Marginal stability") 
    print("   μ < 0: Stable fixed point at origin")
    print("ω: Angular frequency (rad/s)")
    print("   Positive: Counter-clockwise rotation")
    print("   Negative: Clockwise rotation")
    print("   Frequency in Hz = ω/(2π)")