import os
from urllib.request import urlopen
import ssl
import zipfile

# Get the model files (https://senselab.med.yale.edu/ModelDB/ShowModel?model=113732#tabs-1)
modelLink = 'https://senselab.med.yale.edu/modeldb/eavBinDown?o=113732&a=23&mime=application/zip'

u = urlopen(modelLink, context=ssl._create_unverified_context())
# Save data to .zip
localFile = open('SS-cortex.zip', 'wb')
localFile.write(u.read())
localFile.close()

# Unzip file
myzip = zipfile.ZipFile('SS-cortex.zip', 'r')
myzip.extractall('.')
myzip.close()

# Compile .hoc files
os.system('cd SS-cortex; nrnivmodl')