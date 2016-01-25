import matplotlib.pyplot as plt
from numpy import shape, std
from scipy.misc import imresize
from get_data import *
import os
import shutil

if len(sys.argv) == 3:
    save_ext = sys.argv[1]
    plot_graphs = (sys.argv[2] == '1')
else:
    save_ext = 'eps'
    plot_graphs = False

if os.path.exists("results/part 1/photos"):
    shutil.rmtree("results/part 1/photos")
os.makedirs("results/part 1/photos")

if os.path.exists("results/part 1/face overlay"):
    shutil.rmtree("results/part 1/face overlay")
os.makedirs("results/part 1/face overlay")


# Fetch dataset
act = ['Chris Klein', 'Gerard Butler', 'Leonardo DiCaprio', 'Jason Statham', 'Andy Richter']
_, photos, faces, _ = fetch_data("subset_actors.txt", act, 1)

# Print photos besides faces
for i in range(len(act)):
    plt.suptitle(act[i], fontsize=14)
    plt.subplot(1,2,1)
    plt.imshow(photos[i])
    plt.subplot(1,2,2)
    plt.imshow(faces[i])
    plt.savefig("results/part 1/photos/%d.%s"%(i, save_ext))
    plt.show() if plot_graphs else plt.close()

_, photos, faces, _ = fetch_data("subset_actors.txt", ['Richard Madden'], 10)
dims = map(shape, faces)
min_dim = min(dims)
faces_rd = [imresize(f, min_dim) for f in faces]

# Calculate standard deviation in dimensions
print "Dimension Standart Deviation: %f" %(std(dims))

# Print overlay of multiple faces
plt.title('Richard Madden')
for i in range(len(faces)):
    plt.imshow(faces_rd[i], alpha=0.1)
plt.savefig("results/part 1/face overlay/face overlay.%s" %(save_ext))
plt.show() if plot_graphs else plt.close()
