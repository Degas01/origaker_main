import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

# ----------------------------- 
# Exact Replication of the Original Image
# ----------------------------- 

# Create the exact time axis shown in the image (0 to 6 seconds)
t = np.linspace(0, 6, 1200)

# ----------------------------- 
# MATSUOKA: Smooth exponential rise from ~0.1 to ~0.5 (NO oscillations)
# ----------------------------- 
matsuoka_output = 0.1 + 0.4 * (1 - np.exp(-t/1.5))

# ----------------------------- 
# HOPF: Clear sinusoidal oscillations from -1 to +1 at ~1Hz 
# ----------------------------- 
hopf_output = np.sin(2 * np.pi * 1.01 * t)

# ----------------------------- 
# FREQUENCY SPECTRUM: Create exactly what's shown in the image
# ----------------------------- 
# Frequency range from 0 to 5 Hz
freq_range = np.linspace(0, 5, 1000)

# Matsuoka spectrum: Very high peak at 0 Hz (175), rapid decay
matsuoka_spectrum = 175 * np.exp(-freq_range * 5) + 5

# Hopf spectrum: Sharp peak at 1 Hz (115), low elsewhere  
hopf_spectrum = 115 * np.exp(-((freq_range - 1.01)**2) / 0.01) + 2

# ----------------------------- 
# EXACT VALUES from the original image table
# ----------------------------- 
matsuoka_freq = 0.25
matsuoka_amp = 0.15 
matsuoka_stability = 0.00

hopf_freq = 1.01
hopf_amp = 0.93
hopf_stability = 0.99

# ----------------------------- 
# CREATE THE EXACT PLOT LAYOUT
# ----------------------------- 
plt.style.use('default')
fig = plt.figure(figsize=(15, 10))
fig.suptitle('Matsuoka vs Hopf Oscillator Comparison', fontsize=16, fontweight='bold', y=0.95)

# ----------------------------- 
# OUTPUT COMPARISON (Top Left) - Exact replication
# ----------------------------- 
ax1 = plt.subplot(2, 2, 1)
plt.plot(t, matsuoka_output, 'b-', linewidth=2, label='Matsuoka')
plt.plot(t, hopf_output, 'r-', linewidth=2, label='Hopf') 
plt.title('Output Comparison', fontsize=14, fontweight='bold')
plt.xlabel('Time (s)', fontsize=12)
plt.ylabel('Output', fontsize=12)
plt.legend(fontsize=11)
plt.grid(True, alpha=0.3)
plt.xlim(0, 6)
plt.ylim(-1.0, 1.0)

# ----------------------------- 
# PERFORMANCE COMPARISON (Top Right) - Exact bar chart
# ----------------------------- 
ax2 = plt.subplot(2, 2, 2)
categories = ['Frequency', 'Amplitude', 'Stability']
matsuoka_values = [matsuoka_freq, matsuoka_amp, matsuoka_stability]
hopf_values = [hopf_freq, hopf_amp, hopf_stability]

x = np.arange(len(categories))
width = 0.35

plt.bar(x - width/2, matsuoka_values, width, label='Matsuoka', color='steelblue', alpha=0.8)
plt.bar(x + width/2, hopf_values, width, label='Hopf', color='orange', alpha=0.8)

plt.title('Performance Comparison', fontsize=14, fontweight='bold')
plt.xlabel('Metrics', fontsize=12)
plt.ylabel('Values', fontsize=12)
plt.xticks(x, categories, fontsize=11)
plt.legend(fontsize=11)
plt.ylim(0, 1.1)
plt.grid(True, alpha=0.3)

# ----------------------------- 
# FREQUENCY SPECTRUM (Bottom Left) - Exact replication
# ----------------------------- 
ax3 = plt.subplot(2, 2, 3)
plt.plot(freq_range, matsuoka_spectrum, 'b-', linewidth=2, label='Matsuoka')
plt.plot(freq_range, hopf_spectrum, 'r-', linewidth=2, label='Hopf')
plt.title('Frequency Spectrum', fontsize=14, fontweight='bold')
plt.xlabel('Frequency (Hz)', fontsize=12)
plt.ylabel('Magnitude', fontsize=12)
plt.legend(fontsize=11)
plt.grid(True, alpha=0.3)
plt.xlim(0, 5)
plt.ylim(0, 180)

# ----------------------------- 
# RESULTS TABLE (Bottom Right) - Exact values from image
# ----------------------------- 
ax4 = plt.subplot(2, 2, 4)
ax4.axis('off')

# Exact table data from the original image
table_data = [
    ['Oscillator', 'Frequency (Hz)', 'Amplitude', 'Stability'],
    ['Matsuoka', '0.25', '0.15', '0.00'],
    ['Hopf', '1.01', '0.93', '0.99']
]

# Create table with exact styling
table = ax4.table(cellText=table_data, loc='center', cellLoc='center')
table.auto_set_font_size(False)
table.set_fontsize(12)
table.scale(1.2, 1.8)

# Style exactly like the original
for j in range(len(table_data[0])):
    table[(0, j)].set_facecolor('#D3D3D3')
    table[(0, j)].set_text_props(weight='bold')
    table[(0, j)].set_edgecolor('black')
    table[(0, j)].set_linewidth(1)

for i in range(1, len(table_data)):
    for j in range(len(table_data[0])):
        table[(i, j)].set_facecolor('white')
        table[(i, j)].set_edgecolor('black')
        table[(i, j)].set_linewidth(1)

plt.tight_layout()
plt.subplots_adjust(top=0.9, hspace=0.3, wspace=0.3)
plt.savefig('matsuoka_vs_hopf_exact_replication.png', dpi=300, bbox_inches='tight', 
            facecolor='white', edgecolor='none')
plt.show()

print("=== EXACT REPLICATION COMPLETE ===")
print("Reproduced exactly what is shown in the original image:")
print("- Matsuoka: Smooth exponential rise from 0.1 to 0.5 (NO oscillations)")
print("- Hopf: Clear sinusoidal oscillations -1 to +1 at 1.01 Hz")
print("- Frequency spectrum: Matsuoka peak at 0 Hz (175), Hopf peak at 1 Hz (115)")
print("- Table values: Exactly as shown in original image")