
# TimeDeltaBayesianModel
An inverse model for alpine glacier ice thickness using one or multiple time intervals.
This work is related to a now submitted paper to the [Journal Of Glaciology](https://www.cambridge.org/core/journals/journal-of-glaciology), with the collaborative work of Gwenn Flowers, [Andrew Nolan](https://github.com/andrewdnolan), [Douglas Brinkerhoff](https://dbrinkerhoff.org/) and Étienne Berthier. 

It aims to slightly modify the work from Douglas Brinkerhoff in [this paper](https://www.frontiersin.org/articles/10.3389/feart.2016.00008/full) 
to be able to use multiple time intervals in a conjoint inversion using a mass conversation equation. This means that we are trying to use glacier satellite data showing different dynamics i.e. steady state and/or surging).
We then investigate the advantages of using satellite data from a slip-focused glacier-flow regime. 
This work aims to serve as a proof of concept that higher velocity data contains more information about basal topography, leading to higher quality inversions. 

Andrew Nolan generated the synthetic data, simulating a surging glacier. The code used to do so can be [found here](https://github.com/andrewdnolan/Harmonic_Beds) (with a sweet animation of the synthetic surge).
<p>

You can create a working environment (which might be a little messy, I must admit) with the command 

`conda env create -f environment.yml`.

You can then activate the environment with 

`conda activate tdbm`.

Examples showing the code are available in the files `synthetic_example.py` and `realdata_example.py`. The example with synthetic data highlights the various possibilities of the model and how it can use the different dynamics from a dataset for an inversion. The example with the real data showcases how we use velocities on a small surging glacier in Yukon to infer ice thickness. Those examples can show you can the basics of how the model work. The model is coded using the old version of [pymc](https://github.com/pymc-devs/pymc2) (with some documentation [here](https://buildmedia.readthedocs.org/media/pdf/pymcmc/stable/pymcmc.pdf) and [here](https://pymcmc.readthedocs.io/en/latest/index.html) and the documentation for their implementation of Gaussian processes [here](https://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.385.4366&rep=rep1&type=pdf). I don't plan to rewrite this model in the latest version of pymc anytime soon - it would prove quite interesting however to see what it would bring with it's new capabilities.

