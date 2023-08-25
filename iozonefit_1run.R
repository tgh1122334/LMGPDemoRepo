library(RcppArmadillo);
library(Rcpp);
library(tidyverse);
library(MICsplines);
library(lhs);
library(mvtnorm);
library(GauPro);
source("lib_simulation.R");
source("lib_lmgp_final.r");
sourceCpp("lib_funcategp_seppar.cpp");

temp = para_iozone_fit(
  paraidx = 1,
  train_ratio = 0.5,
  Rda_dir = "../../../../Rwork2021/tinkercliffs/varsysdata/",
  num_rmax = 3.0,
  adaptsvd = F
)
save(temp, file = "io1run.Rdata")
