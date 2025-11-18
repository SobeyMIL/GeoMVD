import matplotlib
import matplotlib.pyplot as plt
import numpy as np

# Set font to Times (a commonly available serif font)
matplotlib.rcParams['font.family'] = 'Times'

N = 1000
geo_attention_scale_list = np.geomspace(1e-5, 0.3, N + 1)
geo_attention_scale_list[:N//2] = 0.0

plt.figure(figsize=(8, 4.5))
plt.plot(geo_attention_scale_list)  # Add label to show legend
plt.title("Geometric Attention Scaling Factor", fontsize=14)
plt.xlabel("timestep $t$", fontsize=12)
plt.ylabel(r"$\lambda_{Geo}$", fontsize=12)  # MathText works, no need for LaTeX
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()
