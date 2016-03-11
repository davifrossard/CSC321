from multiprocessing import Pool
from socket import timeout
from hashlib import sha256
from scipy.ndimage import imread
from scipy.misc import imsave, imresize
from glob import glob
from io import BytesIO
from set_utils import to_one_hot
import numpy as np
import sys
import urllib2
import os

'''
DATA FETCHING SCRIPT FROM ASSIGNMENT 1
'''

def rgb2gray(image):
    if len(image.shape)==3:
        return np.dot(image[...,:3], [0.299, 0.587, 0.114])
    else:
        return image


def process_item(item, dtimeout=1):
    try:
        # Extract useful data from line
        name = item[0]
        url = item[3]
        ext = url.split('.')[-1]
        index = item[2]
        imhash = item[5].strip()
        facecoord = map(int, item[4].split(','))
        filename = name + "/" + name + '_' + str(index) + '.' + ext

        if os.path.exists("cropped/" + filename):
            # print "[EF] "+url+" ["+filename+"] already exists"
            return 0

        # Try fetching the url
        try:
            image = urllib2.urlopen(url, timeout=dtimeout)
        except urllib2.URLError, e:
            # print "[EF] "+url+" ["+filename+"] threw error "+str(e)
            return 0
        except timeout:
            # print "[TF] "+url+" ["+filename+"] timeouted"
            return 0

        image_data = image.read()

        # Calculate the hash and check against expected value
        downloaded_hash = sha256(image_data).hexdigest()
        if downloaded_hash != imhash:
            # print "[HF] "+url+" ["+filename+"] Hash check failed"
            return 0

        # Crop face and save as greyscale
        imarray = imread(BytesIO(image_data))
        face = imarray[0.95*facecoord[1]:1.05*facecoord[3], 0.95*facecoord[0]:1.05*facecoord[2]]
        face = imresize(face, (250,250))
        imsave("cropped/" + filename, face)

        # Everything went as expected
        if os.path.exists("cropped/" + filename):
            # print "[S] "+url+" ["+filename+"] Downloaded"
            return 1
        else:
            return 0
    except Exception, e:
        # print "Thread error"+e.message
        return 0


def fetch_data_files(source, targets, amount, numthreads=10, threadtimeout=1):
    data_lines = list([a.split("\t") for a in open(source).readlines()])
    pool = Pool(processes=numthreads)
    total_sucess = 0;
    for target in targets:

        # Fetches all lines of data from actor and shuffles it
        target_data = list([t_data for t_data in data_lines if t_data[0] == target])
        # shuffle(target_data)
        if len(target_data) == 0:
            print target + " not found in the source"
            continue
        if amount > len(target_data):
            print "Not enough data for " + target
            continue

        # Create artist's directory
        if not os.path.exists("cropped/" + target):
            os.makedirs("cropped/" + target)

        # Only download a determinate amount of images
        imsuccess = 0
        if amount > 0:
            last = 0
            print "Downloading images of " + target
            while imsuccess < amount:
                diff = amount - imsuccess
                if last > len(target_data):
                    raise ValueError('Not enough data for %s - %d requested, %d found' % (target, amount, imsuccess))
                processes = [pool.apply_async(process_item, [i, threadtimeout]) for i in target_data[last:last + diff]]
                last += diff

                for process in processes:
                    imsuccess += process.get()
                    ratio = (float(imsuccess) / amount) * 100
                    sys.stdout.write("\r%.2f%%" % ratio)
                    sys.stdout.flush()

        # Download all images
        else:
            print "Downloading images of " + target
            processes = [pool.apply_async(process_item, [i, threadtimeout]) for i in target_data]
            for process in processes:
                imsuccess += process.get()
                ratio = (float(imsuccess) / len(target_data)) * 100
                sys.stdout.write("\r%.2f%%" % ratio)
                sys.stdout.flush()

        print "\nDownloaded %d image[s] of %s" % (imsuccess, target)
        total_sucess += imsuccess

    pool.close()

    return total_sucess


def fetch_actors(source):
    data_lines = list([a.split("\t") for a in open(source).readlines()])
    return list(set([data[0] for data in data_lines]))

def fetch_data(source, targets, amount, numthreads=10, threadtimeout=1):
    faces = []
    fclass = []
    classes = to_one_hot(targets)
    for k,target in enumerate(targets):
        tfaces = sorted(set(glob("cropped/" + target + "/*")))
        if len(tfaces) < amount or amount == 0:
            fetch_data_files(source, [target], amount - len(tfaces), numthreads, threadtimeout)
            tfaces = sorted(set(glob("cropped/" + target + "/*")))

        for i in range(len(tfaces)):
            face = imread(tfaces[i], mode='RGB')
            faces.append(face)
            fclass.append(classes[k])
    return faces, fclass
