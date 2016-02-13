import matplotlib.pyplot as plt
import numpy as np

def visualize_weights(w, plot=False, ext='pdf', part=6):
    np.random.seed(0)
    ids = np.random.choice(w.shape[1], 10, False)
    ids.sort()
    fig, axes = plt.subplots(nrows=2, ncols=5)
    fig.suptitle('Weight Visualization', size=20)
    for i, ax in enumerate(axes.flat):
        heatmap = ax.imshow(w[:,ids[i]].reshape((28,28)), cmap = plt.cm.coolwarm)
        ax.set_title(i)
        ax.set_axis_off()
    fig.subplots_adjust(right=0.8)
    cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
    fig.colorbar(heatmap, cax=cbar_ax)
    plt.savefig('results/part%d.%s' %(part,ext), bbox_inches='tight')
    plt.show() if plot else plt.close()
    return ids