# ---------------------------------------------
# Download data
# ---------------------------------------------
import os
from urllib.request import urlopen
import ssl
import zipfile
def getData():

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


# ---------------------------------------------
# Make inhibitory cell template
# ---------------------------------------------
import re
def makeInhibitoryCellTemplate(templateName, templateDirectory):

    # Import text
    with open('%ssj3-cortex.hoc' % (templateDirectory), 'r') as f:
        templateText = f.readlines()

    # Define function to get segments and their indices
    def getSegment(text,startline,endline=None,returnIndices=False):
        startIndex, endIndex = False,False
        for i,line in enumerate(text):
            if startline in line:
                startIndex = i
            if not(endline==None):
                if endline in line:
                    endIndex = i
        if returnIndices==True:
            return startIndex,endIndex
        else:
            return templateText[startIndex:endIndex+1]

    # Get texts
    synapticMechanics = getSegment(templateText, "begintemplate AMPA", "endtemplate GABAB")
    inhibitoryTemplate = getSegment(templateText, "begintemplate Inhib", "endtemplate Inhib")

    # Remove link to external dynamics
    cutInx,_ = getSegment(inhibitoryTemplate, "external AMPA,NMDA,GABAA,GABAB", returnIndices=True)
    del inhibitoryTemplate[cutInx]

    # Initialise 'all' object
    cutInx,_ = getSegment(inhibitoryTemplate, "create cell", returnIndices=True)
    inhibitoryTemplate.insert(cutInx+1,'public all\nobjref all\n')

    # Add synapse dynamics
    for line in reversed(synapticMechanics):
        inhibitoryTemplate.insert(cutInx+2,line)

    # Create all section list
    cutInx,_ = getSegment(inhibitoryTemplate, "proc init()", returnIndices=True)
    inhibitoryTemplate.insert(cutInx+1,'   all = new SectionList()\n')

    # Append sections to all list
    cutInx,_ = getSegment(inhibitoryTemplate, "insert hh", returnIndices=True)
    inhibitoryTemplate.insert(cutInx+1,'                all.append()\n')

    # Replace words
    def replaceWord(pre,post):
        pattern  = re.compile(r'\b%s\b' % (pre), re.I)
        for i,line in enumerate(inhibitoryTemplate):
            if pre in line:
                indices = pattern.search(line).span()
                listString = list(line)
                listString[indices[0]:indices[1]] = list(post)
                inhibitoryTemplate[i] = ''.join(listString)

    replaceWord('Inhib','inhib') # Remove capital I from "Inhib"
    replaceWord('cell','soma') # Replace all "cell" with "soma"

    # Save result
    os.system("touch %s" % (templateName))
    with open(templateName, "w") as text_file:
        text_file.write(''.join(inhibitoryTemplate))