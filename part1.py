import matplotlib.pyplot as plt
from numpy import shape, std, vstack, mean, savetxt
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

if os.path.exists("results/part_1/photos"):
    shutil.rmtree("results/part_1/photos")
os.makedirs("results/part_1/photos")

if os.path.exists("results/part_1/face_overlay"):
    shutil.rmtree("results/part_1/face_overlay")
os.makedirs("results/part_1/face_overlay")


# Fetch dataset
act = ['Chris Klein', 'Gerard Butler', 'Leonardo DiCaprio', 'Jason Statham', 'Andy Richter']
_, photos, faces, _ = fetch_data("subset_actors.txt", act, 1)

# Print photos besides faces
for i in range(len(act)):
    plt.suptitle(act[i], fontsize=20)
    plt.subplot(1,2,1)
    plt.imshow(photos[i])
    plt.subplot(1,2,2)
    plt.imshow(faces[i])
    plt.savefig("results/part_1/photos/%d.%s"%(i, save_ext), bbox_inches='tight')
    plt.show() if plot_graphs else plt.close()

_, photos, faces, _ = fetch_data("subset_actors.txt", ['Richard Madden'], 15)
dims = map(shape, faces)
min_dim = min(dims)
faces_rd = [imresize(f, min_dim) for f in faces]

# Calculate standard deviation in dimensions
stdev = tuple([int(std(y)) for y in zip(*dims)][:-1])
avg = tuple([int(mean(y)) for y in zip(*dims)][:-1])
print "Dimension Standart Deviation: %s" %(stdev,)
print "Dimension Average: %s" %(avg,)
savetxt("results/part_1/faces_statistics.csv", vstack((stdev, avg)), fmt='%s')

# Print overlay of multiple faces
plt.title('Richard Madden')
for i in range(len(faces)):
    plt.imshow(faces_rd[i], alpha=0.05)
plt.savefig("results/part_1/face_overlay/face_overlay.%s" %(save_ext), bbox_inches='tight')
plt.show() if plot_graphs else plt.close()
