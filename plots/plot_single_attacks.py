import numpy as np
import matplotlib.pyplot as plt
import tikzplotlib

# Make a random dataset:
height = [0.0625, 0, 0, 1, 1]
bars = ('0.1', '0.2', '0.3', '0.4', '0.5')
y_pos = np.arange(len(bars))

# Create bars
plt.grid(alpha=0.2)
plt.bar(y_pos, height)

# Create names on the x-axis
plt.xticks(y_pos, bars)
plt.xlabel("Attack Strength (L0-Norm)")
plt.ylabel("Probability Impact")

# Show graphic
tikzplotlib.save("plots/single_attacks.tex")
