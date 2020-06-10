# In-silico model of the contribution of the thalamus to cortical alpha power
Inspired by data collected by data collected in:<br>
<i>Lőrincz, M. L., Kékesi, K. A., Juhász, G., Crunelli, V., & Hughes, S. W. (2009). Temporal framing of thalamic relay-mode firing by phasic inhibition during the alpha rhythm. Neuron, 63(5), 683-696</i>

## Abstract
Alpha oscillations (~7-13 Hz) are commonly observed in human brain activity, especially in occipitoparietal regions. These oscillations reduce in amplitude during visual stimulation, and are known to be coordinated by activity in the thalamus. After performing a series of detailed experiments, Lőrincz et al. (2009; Neuron) proposed a model to explain how various cellular interactions lead to the generation of alpha oscillations in the lateral geniculate nucleus (see 'Original model' figure below). This Github repository contains code for an in-silico formulation of the model, demonstrating that such cellular interactions can indeed produce alpha oscillations that are attenuated by strong sensory inputs.

The model is composed of three cell types: high-threshold bursting (HT) neurons, interneurons, and relay model thalamocortical (TC) neurons. HT neurons display rhythmic bursting at alpha frequencies. However, when visual inputs are strong, the number of spikes for each burst is increased. In response to inputs from HT neurons, interneurons also spike at alpha frequencies, and in-phase with the HT neurons. However, when HT bursting is strong (due to strong visual inputs), interneurons also display spiking in the middle of the alpha cycle. In the absence of middle bursting, interneurons periodically inhibit TC neurons in phase with HT bursting, leading to rebound spikes that are in-phase with HT bursting. Such in-phase spiking leads to high synchronicity in the system, and therefore increases in overall alpha oscillations amplitudes. However, when the interneurons fire in the middle of the alpha cycle (due to strong visual inputs), interneurons periodically inhibit TC neurons anti-phasically with HT bursting, leading to rebound spikes that is out of phase with HT bursting. Such anti-phasic spiking leads to low synchronicity in the system, and therefore reductions in alpha oscillations amplitudes (see 'Output of computational model' figure below).


## Original model
<img width=80%, src="./figures/originalLorincz.png">

## Output of computational model
<img width=90% src="./figures/adexModelOutput.png">