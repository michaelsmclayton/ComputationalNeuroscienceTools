
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# -------------------------------------------------
# Plot results
# -------------------------------------------------

# Load data
resultsAcrossStimulationType = np.load('resultsAcrossStimulationType.npy')

# set width of bar
barWidth = 0.25

# Initialise figure
plt.figure()

for index, startIndex in enumerate([0, 3, 6, 9]):

    # Set subplot
    plt.subplot(2,2,index+1)
 
    # set height of bar
    bars1 = resultsAcrossStimulationType[startIndex,]
    bars2 = resultsAcrossStimulationType[startIndex+2,]
    bars3 = resultsAcrossStimulationType[startIndex+1,]
    
    # Set position of bar on X axis
    r1 = np.arange(len(bars1))
    r2 = [x + barWidth for x in r1]
    r3 = [x + barWidth for x in r2]
    
    # Make the plot
    plt.bar(r1, bars1, color='#DA3932', width=barWidth, edgecolor='white', label='Excitatory')
    plt.bar(r2, bars2, color='#BEBF4E', width=barWidth, edgecolor='white', label='Balanced')
    plt.bar(r3, bars3, color='#3F539F', width=barWidth, edgecolor='white', label='Inhibitory')
    
    # Add xticks on the middle of the group bars
    plt.xlabel('group', fontweight='bold')
    plt.xticks([r + barWidth for r in range(len(bars1))], ['23e', '23i', '4e', '4i', '5e', '5i', '6e', '6i'])
    
    # Create legend & Show graphic
    # plt.legend()

plt.savefig('barGraph.png')


# Plot comparisons

# Initialise figure
plt.figure()

# set height of bar
bars1 = resultsAcrossStimulationType[2,]
bars2 = resultsAcrossStimulationType[6,]
bars3 = resultsAcrossStimulationType[-1,]

# Set position of bar on X axis
r1 = np.arange(len(bars1))
r2 = [x + barWidth for x in r1]
r3 = [x + barWidth for x in r2]

# Make the plot
plt.bar(r1, bars1, color='#DA3932', width=barWidth, edgecolor='white', label='Excitatory')
plt.bar(r2, bars2, color='#BEBF4E', width=barWidth, edgecolor='white', label='Balanced')
plt.bar(r3, bars3, color='#3F539F', width=barWidth, edgecolor='white', label='Inhibitory')

# Add xticks on the middle of the group bars
plt.xlabel('group', fontweight='bold')
plt.xticks([r + barWidth for r in range(len(bars1))], ['23e', '23i', '4e', '4i', '5e', '5i', '6e', '6i'])

# Create legend & Show graphic
# plt.legend()

plt.savefig('barGraphComp.png')

