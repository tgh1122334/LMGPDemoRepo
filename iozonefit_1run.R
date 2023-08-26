library(RcppArmadillo);
library(Rcpp);
library(tidyverse);
library(MICsplines);
library(lhs);
library(mvtnorm);
library(GauPro);
source("src/lib_simulation.R");
source("src/lib_lmgp_final.r");
sourceCpp("src/lib_funcategp_seppar.cpp");



train_ratio = 0.5; # percentage of data in training set
Rda_dir = "./Data/";
num_rmax = 3.0;
adaptsvd = F ##not fit all components, stop when 90% proportion is reached


ll_iozone_dat = iozone_split(train_ratio, Rda_dir) ##generate train-test set.
# browser()
int_svd_num = length(ll_iozone_dat$svd$d)
if(adaptsvd)
{
  int_svd_num = which((cumsum(ll_iozone_dat$svd$d)/sum(ll_iozone_dat$svd$d))>0.9)[1]
}
ll_iozone_pre = list()
for(ii in 1:int_svd_num)
{
  ## list(x = mat_X, z = vec_Z, y = vec_Y, num_lev = n_cate_vec)
  ll_train = list(
    x = ll_iozone_dat$train$X,
    z = ll_iozone_dat$train$Z,
    y = ll_iozone_dat$train$Ymat[,ii],
    num_lev = ll_iozone_dat$train$num_lev
  )
  ll_test = list(
    x = ll_iozone_dat$test$X,
    z = ll_iozone_dat$test$Z,
    y = ll_iozone_dat$test$Ymat[,ii],
    num_lev = ll_iozone_dat$test$num_lev
  )
  ##GP predicted on test set
  vec_y_pre_gp = gp_fit(datalist_train = ll_train, datalist_test = ll_test)
  ##CGP predicted on test set
  vec_y_pre_cgp = cgp_fit(datalist_train = ll_train, datalist_test = ll_test)
  
  ll_ini = ini_val(ll_train)
  ##LMGP trained parameters
  obj_lmgp = lmgp_EM(ll_train, 500, rmax = num_rmax, using_C = T, cgp_start = ll_ini$cgp, stop_early = T)
  ##LMGP-S trained parameters
  obj_lmgp_sep = lmgp_EM_seppar(ll_train, 500, rmax = num_rmax, using_C = T, cgp_start = ll_ini$cgp, stop_early = T)
  ##LMGP predicted on test set
  vec_y_pre = predict_lmgp_EM(obj_lmgp[[length(obj_lmgp)]], ll_train, ll_test, rmax = num_rmax)
  ##LMGP-S predicted on test set
  vec_y_pre_sep = predict_lmgp_em_seppar(obj_lmgp_sep[[length(obj_lmgp_sep)]], ll_train, ll_test, rmax = num_rmax)
  dat_pre = data.frame(
    lmgp = vec_y_pre,
    lmgp_sep = vec_y_pre_sep,
    gp = vec_y_pre_gp,
    cgp = vec_y_pre_cgp
  )
  ll_iozone_pre[[ii]] = dat_pre ##store the components
}

list(
  data = ll_iozone_dat,
  pre = ll_iozone_pre
)

