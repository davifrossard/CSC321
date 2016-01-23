import matplotlib.pyplot as plt
from numpy import shape, std
from scipy.misc import imresize
from get_data import *

# Fetch dataset
_, photos, faces, _ = fetch_data("subset_actors.txt", ['Chris Klein'], 5)

plt.gray()

dims = map(shape, faces)
min_dim = min(dims)
faces_rd = [imresize(f, min_dim) for f in faces]

# Calculate standart deviation in dimensions
print "Dimension Standart Deviation: %f" %(std(dims))

# Print photos besides faces
for i in range(3):
    plt.subplot(1,2,1)
    plt.imshow(photos[i])
    plt.subplot(1,2,2)
    plt.imshow(faces[i])
    plt.show()

# Print overlay of multiple faces
for i in range(len(faces)):
    plt.imshow(faces_rd[i], alpha=0.1)
plt.show(block=True)