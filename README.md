gpo-ifac2014
=====

Particle Bayesian optimisation for parameter inference in nonlinear state space models

This code was downloaded from < https://github.com/compops/gpo-ifac2014 > and contains the code used to produce some of the results in

J. Dahlin and F. Lindsten, **Particle filter-based Gaussian Process Optimisation for Parameter Inference**. In Proceedings of the 18th World Congress of the International Federation of Automatic Control (IFAC), Cape Town, South Africa, August 2014.

A pre-print of the paper is found at < http://arxiv.org/abs/1311.0689 >.

Requirements
--------------
The program is written in Matlab 2013a and uses the GPML toolbox for the Gaussian process modelling. The GMPL toolbox is available for download from http://www.gaussianprocess.org/gpml/code/matlab/doc/. The program also requires a DIRECT optimisation algorithm, available for download at < http://www4.ncsu.edu/~ctk/Finkel_Direct/ >.


Included files
--------------

**RUNME**
Executes the Matlab program that reproduces the plot in the paper for the Stochastic volaility model.

**pf**
Runs a bootstrap particle filter with systematic resampling to estimate the log-likelihood.

**datagen**
Generates data from a general state space model given function handles and the input.

**evalMu** 
Evaluates the posterior mean for the Gaussian processes (used by Direct)

**EI** 
Evaluates the expected improvment by using the Gaussian processes (used by Direct)
