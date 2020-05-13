import os
from urllib.request import urlopen
import ssl
import zipfile

# Get the model files (https://senselab.med.yale.edu/ModelDB/ShowModel?model=113732#tabs-1)
modelLink = 'https://senselab.med.yale.edu/modeldb/eavBinDown?o=113732&a=23&mime=application/zip'
u = urlopen(modelLink, context=ssl._create_unverified_context())

# Save data to .zip
zipName = 'SS-cortex'
localFile = open('%s.zip' % (zipName), 'wb')
localFile.write(u.read())
localFile.close()

# Unzip file
myzip = zipfile.ZipFile('%s.zip' % (zipName), 'r')
myzip.extractall('.')
myzip.close()
os.system('rm %s.zip' % (zipName)) # delete .zip file

# Compile .hoc files
os.system('cd %s; nrnivmodl' % (zipName))