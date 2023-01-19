import numpy as np
import matplotlib.pyplot as plt
import tikzplotlib


labels = ['done1', 'done2', 'empty']
# done1, done2, empty
fgsm = [1,0.,1]
ffgsm = [1,0.625,0.0625]
deepfool = [1,0.125,0.875]

x = np.arange(len(labels))  # the label locations
width = 0.25  # the width of the bars

fig, ax = plt.subplots()

rects1 = ax.bar(x - width - 0.03, fgsm, width, label='FGSM', zorder=2)
rects2 = ax.bar(x, ffgsm, width, label='FFGSM', zorder=2)
rects3 = ax.bar(x + width + 0.03, deepfool, width, label='Deep Fool', zorder=2)

# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_ylabel('Probability')
#ax.set_title('Scores by group and gender')
ax.set_xticks(x, labels)
plt.legend(bbox_to_anchor=(1.03, 1), loc='upper left', borderaxespad=0.)

#ax.bar_label(rects1, padding=3)
#ax.bar_label(rects2, padding=3)
#ax.bar_label(rects3, padding=3)

specific_ticks = [0, 1, 2]

# Setting the xticks to specific positions
ax.set_xticks(specific_ticks)

# Set x-axis labels
ax.set_xticklabels(['done1',  'done2',  'empty'])

ax.grid(zorder=1)
fig.tight_layout()



# Show graphic
tikzplotlib.save("plots/single_attacks.tex")
