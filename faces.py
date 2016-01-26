import subprocess
import os
import tempfile
import shutil
from glob import glob

save_ext = raw_input("Enter extension to save graphs [eps by default]: ") or 'eps'
show_graph = '1' if raw_input("Show graph windows [y/N]? ").lower() == 'y' else '0'

if os.path.exists("results"):
    if raw_input("Previous results found, keep backup [Y/n]? ").lower() != 'n':
        i=2
        while os.path.exists("results_%d" % i):
            i += 1
        os.rename("results", "results_%d"%(i))

if os.path.exists("cropped"):
    if raw_input("Faces folder found, re-dowload [y/N]? ").lower() == 'y':
        shutil.rmtree("cropped")

print("\n")

print "RUNNING PART 1"
subprocess.call("python part1.py %s %s" %(save_ext, show_graph), shell=True)

print "\n\nRUNNING PART 3"
subprocess.call("python part3.py %s %s" %(save_ext, show_graph), shell=True)

print "\n\nRUNNING PART 4"
subprocess.call("python part4.py %s %s" %(save_ext, show_graph), shell=True)

print "\n\nRUNNING PART 5"
subprocess.call("python part5.py %s %s" %(save_ext, show_graph), shell=True)

print "\n\nRUNNING PART 6"
subprocess.call("python part6.py %s %s" %(save_ext, show_graph), shell=True)
subprocess.call("python part6_2.py %s %s" %(save_ext, show_graph), shell=True)

print "\n\nCOMPILING REPORT"
current = os.getcwd()
temp = tempfile.mkdtemp()
shutil.copy('faces.tex', temp)
shutil.move('results', temp)
os.chdir(temp)
subprocess.call(['pdflatex', 'faces.tex'])
subprocess.call(['pdflatex', 'faces.tex'])
shutil.move('faces.pdf', current)
shutil.rmtree(temp)

