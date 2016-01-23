from random import shuffle
from multiprocessing import Pool, TimeoutError
from socket import timeout
from hashlib import sha256
from scipy.misc import imread, imsave
from glob import glob
import urllib2
import os
import shutil

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
            return 0
        except timeout:
            print "[TF] "+url+" ["+filename+"] timeouted"
            return 0
        image_data = image.read()

        # Calculate the hash and check against expected value
        downloaded_hash = sha256(image_data).hexdigest()
        if downloaded_hash != imhash:
            print "[HF] "+url+" ["+filename+"] Hash check failed"
            return 0

        # Crop face and save as greyscale
        output = open("uncropped/"+filename,"wb")
        output.write(image_data)
        output.close()
        imarray = imread("uncropped/"+filename, 1)
        face = imarray[facecoord[1]:facecoord[3],facecoord[0]:facecoord[2]]
        imsave("cropped/"+filename,face)

        # Everything went as expected
        print "[S] "+url+" ["+filename+"] downloaded"
        return 1
    except Exception, e:
        print "Thread error"+e.message
        return 0

def fetch_data_files(source, targets, amount, numthreads=10, threadtimeout=3):
    data_lines = list([a.split("\t") for a in open(source).readlines()])
    pool = Pool(processes=numthreads)
    total_sucess = 0;
    for target in targets:
        download_queue = []

        # Fetches all lines of data from actor and shuffles it
        target_data = list([t_data for t_data in data_lines if t_data[0] == target])
        shuffle(target_data)
        if len(target_data) == 0:
            print target+" not found in the source"
            continue
        # Create artist's directory, overwrite if existent
        if os.path.exists("cropped/"+target):
            shutil.rmtree("cropped/"+target)
        os.makedirs("cropped/"+target)
        if os.path.exists("uncropped/"+target):
            shutil.rmtree("uncropped/"+target)
        os.makedirs("uncropped/"+target)

        # Add indexes to entries (favor parallelism)
        i = 0
        for data in target_data:
            data.append(i)
            download_queue.append(data)
            i+=1

        # Only download a determinate amount of images
        imsuccess=0
        if amount > 0:
            last=0
            while imsuccess < amount:
                diff = amount - imsuccess
                processes = [pool.apply_async(process_item, [i]) for i in download_queue[last:last+diff]]
                last += diff
                for process in processes:
                    try:
                        imsuccess += process.get(timeout=threadtimeout)
                    except TimeoutError,e:
                        print "Thread timeout"

        # Download all images
        else:
            processes = [pool.apply_async(process_item, [i]) for i in download_queue]
            for process in processes:
                try:
                    imsuccess += process.get(timeout=threadtimeout)
                except TimeoutError,e:
                    print "Thread timeout"
        print "Downloaded %d image[s] of %s" %(imsuccess,target)
        total_sucess += imsuccess

    return total_sucess
    # Job finished
    print "Done"

def fetch_data(source, targets, amount):
    photos = []
    faces = []
    actors = []
    total_success = 0
    for target in targets:
        tphotos = glob("uncropped/"+target+"/*")
        tfaces = glob("cropped/"+target+"/*")
        if len(tphotos) < amount or len(tfaces) < amount:
            fetch_data_files(source, [target], amount)
            tphotos = glob("uncropped/"+target+"/*")
            tfaces = glob("cropped/"+target+"/*")
        for i in range(len(tphotos)):
            photos.append(imread(tphotos[i]))
            faces.append(imread(tfaces[i]))
            actors.append(target)
        total_success += i+1
        shuffle(photos)
        shuffle(faces)
    return total_success, photos, faces, actors
