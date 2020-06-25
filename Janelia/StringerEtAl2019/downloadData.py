import os
import re

# Setup data directory
os.system('mkdir data')

# Define data to download ('natimg2800_M1*.mat')
dataIDs = [12462683, 12462689, 12462701, 12462662, 12462686, 12462698, 12462704]

# Define function to get name of current file to download
def getFilename(info):
    filenameStartRegex = re.compile(r'\bX-Filename\b')
    start = filenameStartRegex.search(info).span()[1]+2
    cutInfo = info[start:]
    filenameEndRegex = re.compile(r'\bLocation\b')
    end = filenameEndRegex.search(cutInfo).span()[0]-1
    return cutInfo[:end]

# Get data
regex = re.compile(r'\bX-Filename\b')
for dataID in dataIDs:
    url = f'https://ndownloader.figshare.com/files/{dataID}'
    info = os.popen(f'curl -I {url}').read()
    filename = f'./data/{getFilename(info)}'
    if not(os.path.isfile(filename)):
        os.system(f'wget --no-check-certificate --output-document {filename} https://ndownloader.figshare.com/files/{dataID}')

# Get stimulus data
os.system('wget --no-check-certificate --output-document data/stimuli_class_assignment_confident.mat https://github.com/MouseLand/stringer-pachitariu-et-al-2018b/blob/master/classes/stimuli_class_assignment_confident.mat?raw=true')