from random import shuffle
from multiprocessing import Pool, TimeoutError
from socket import timeout
from hashlib import sha256
from scipy.misc import imread
from io import BytesIO
import urllib2
import sys

def process_item(item, dtimeout=2):
    try:
        # Extract useful data from line
        url = item[3]
        imhash = item[5].strip()
        facecoord = map(int, item[4].split(','))

        # Try fetching the url
        try:
            image = urllib2.urlopen(url, timeout=dtimeout)
        except urllib2.URLError,e:
            # print "[EF] "+url+" ["+filename+"] threw error "+str(e)
            return [0]
        except timeout:
            # print "[TF] "+url+" ["+filename+"] timeouted"
            return [0]
        image_data = image.read()

        # Calculate the hash and check against expected value
        downloaded_hash = sha256(image_data).hexdigest()
        if downloaded_hash != imhash:
            # print "[HF] "+url+" ["+filename+"] Hash check failed"
            return [0]

        # Crop face and save as greyscale
        imarray = imread(BytesIO(image_data),0)
        face = imarray[facecoord[1]:facecoord[3],facecoord[0]:facecoord[2]]

        # Everything went as expected
        # print "[S] "+url+" ["+filename+"] downloaded"
        return [1, imarray, face]
    except Exception,e:
        # print "Thread error: "+e.message
        return [0]

def fetch_data_artist(target_data, target, amount, numthreads=10, threadtimeout=3):
    pool = Pool(processes=numthreads)
    photos = []
    faces = []
    actors = []
    # Shuffles data from actor
    # shuffle(target_data)
    if len(target_data) == 0:
        print target+" not found in the source"
        return [0]

    # Only download a determinate amount of images
    imsuccess = 0
    print "Downloading images of "+target
    if amount > 0:
        last = 0
        while imsuccess < amount:
            diff = amount - imsuccess
            processes = [pool.apply_async(process_item, [i, threadtimeout]) for i in target_data[last:last+diff]]
            last += diff
            for process in processes:
                retval = process.get()
                imsuccess += retval[0]
                ratio = (float(imsuccess) / amount)*100
                sys.stdout.write("\r%.2f%%" % ratio)
                sys.stdout.flush()
                if retval[0] == 1:
                    photos.append(retval[1])
                    faces.append(retval[2])
                    actors.append(target)

    # Download all images
    else:
        processes = [pool.apply_async(process_item, [i]) for i in target_data]
        for process in processes:
            try:
                retval = process.get(timeout=threadtimeout)[0]
                imsuccess += retval
                ratio = (float(imsuccess) / len(target_data))*100
                sys.stdout.write("\r%.2f%%" % ratio)
                sys.stdout.flush()
                if retval == 1:
                    photos.append(retval[1])
                    faces.append(retval[2])
                    actors.append(target)
            except TimeoutError,e:
                pass
    print "\nDownloaded %d image[s] of %s" %(imsuccess,target)
    total_success = imsuccess
    return total_success, photos, faces, actors


def fetch_data(source, targets, amount, numthreads=10, threadtimeout=3):
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

