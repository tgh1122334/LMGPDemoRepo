# LMGPDemoRepo
Data and code for LMGP

# IOzone data
The IOzone data is in `Data`:
- AllThroughputs.Rda: Throughputs on each configuration.
- AllThroughputs_summarized.Rda: Summary statistics for each configuration.
- initial_writers/random_readers/random_writers/re-readers/readers/rewriters.RData: fitted spline coefficients for configurations in each mode.

# Code
The R and C++ functions are in `src`:
- lib_funcategp_seppar.cpp contains the model code, for example, the likelihood function.
- lib_lmgp_final.R uses the C++ likelihood function and performs the maximum likelihood optimization.
- lib_simulation.R contains the code doing simulation. For example, it has wrap functions that perform the data train-test splitting, model fitting and predicting, and calculating error metrics.

# Example
The `example.R` provides an example using the functions to generate the train-test dataset, and train 4 models in the paper (GP, CGP, LMGP, and LMGP-S).
