import os
import shutil
import subprocess
from os import listdir
from os.path import isfile, join

mypath = '/media/palm/62C0955EC09538ED/ptt/full_sized/plastic_bowl_plate'
onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f))]
outdir = os.path.join(mypath, 'output')

# Creates the output dir if it doesn't exists
if not os.path.exists(outdir):
    os.makedirs(outdir)

for file in onlyfiles:
    (name, ext) = os.path.basename(file).split('.')

    if ext == 'heic' or ext == 'HEIC':
        destination = os.path.join(outdir, name) + '.jpg'
        print(destination)
        source = os.path.join(mypath, file)
        print(source)
        # print ('converting   ',os.path.join(mypath, file))
        subprocess.call(['/home/palm/tifig/build/tifig', '-p', '-q', '100', source, destination])
        os.remove(source)
