from random import shuffle
from multiprocessing import Pool, TimeoutError
from socket import timeout
from hashlib import sha256
from scipy.misc import imread, imsave
from numpy import dot
from io import BytesIO
import urllib2

import contextlib
import sys
import cStringIO

@contextlib.contextmanager
def nostdout():
    save_stdout = sys.stdout
    sys.stdout = cStringIO.StringIO()
    yield
    sys.stdout = save_stdout

def rgb2gray(image):
    return dot(image[...,:3], [0.299, 0.587, 0.114])

def process_item(item):
    try:
        # Extract useful data from line
        name = item[0]
        url = item[3]
        ext = url.split('.')[-1]
        index = item[-1]
        imhash = item[5].strip()
        facecoord = map(int, item[4].split(','))
        filename = name+"/"+name+'_'+str(index)+'.'+ext

        # Try fetching the url
        try:
            image = urllib2.urlopen(url, timeout=2)
        except urllib2.URLError,e:
            print "[EF] "+url+" ["+filename+"] threw error "+str(e)
            return [0]
        except timeout:
            print "[TF] "+url+" ["+filename+"] timeouted"
            return [0]
        image_data = image.read()

        # Calculate the hash and check against expected value
        downloaded_hash = sha256(image_data).hexdigest()
        if downloaded_hash != imhash:
            print "[HF] "+url+" ["+filename+"] Hash check failed"
            return [0]

        # Crop face and save as greyscale
        imarray = imread(BytesIO(image_data))
        face = imarray[facecoord[1]:facecoord[3],facecoord[0]:facecoord[2]].dot([0.299, 0.587, 0.114])

        # Everything went as expected
        print "[S] "+url+" ["+filename+"] downloaded"
        return [1, imarray, face]
    except Exception,e:
        print "Thread error: "+e.message
        return [0]

def fetch_data_artist(target_data, target, amount, numthreads=10, threadtimeout=3):
    download_queue = []
    pool = Pool(processes=numthreads)
    photos = []
    faces = []
    actors = []
    # Shuffles data from actor
    shuffle(target_data)
    if len(target_data) == 0:
        print target+" not found in the source"
        return [0]

    # Add indexes to entries (favor parallelism)
    i = 0
    for data in target_data:
        data.append(i)
        download_queue.append(data)
        i+=1

    # Only download a determinate amount of images
    imsuccess = 0
    if amount > 0:
        last = 0
        while imsuccess < amount:
            diff = amount - imsuccess
            processes = [pool.apply_async(process_item, [i]) for i in download_queue[last:last+diff]]
            last += diff
            for process in processes:
                try:
                    retval = process.get(timeout=threadtimeout)[0]
                    imsuccess += retval
                    if retval == 1:
                        photos.append(process.get(timeout=threadtimeout)[1])
                        faces.append(process.get(timeout=threadtimeout)[2])
                        actors.append(target)
                except TimeoutError,e:
                    print "Thread timeout"

    # Download all images
    else:
        processes = [pool.apply_async(process_item, [i]) for i in download_queue]
        for process in processes:
            try:
                print process.get(timeout=threadtimeout)
            except TimeoutError,e:
                print "Thread timeout"
    print "Downloaded %d image[s] of %s" %(imsuccess,target)
    total_success = imsuccess
    return total_success, photos, faces, actors


def fetch_data(source, targets, amount, numthreads=50, threadtimeout=2):
    data_lines = list([a.split("\t") for a in open(source).readlines()])
    total_success = 0;
    photos = []
    faces = []
    actors = []
    for target in targets:
        target_data = list([t_data for t_data in data_lines if t_data[0] == target])
        tsuccess, tphotos, tfaces, tactors = fetch_data_artist(target_data, target, amount, numthreads, threadtimeout)
        total_success += tsuccess
        photos.extend(tphotos)
        faces.extend(tfaces)
        actors.extend(tactors)

    return total_success, photos, faces, actors