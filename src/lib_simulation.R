# some pdfs
rloglogis = function(n, num_location = 0, num_scale = 1)
{
  x = rlogis(n,num_location, num_scale)
  exp(x)
}
my_rweibull_v1 = function(n, num_location = 0, num_scale = 1)
{
  num_beta = 1/num_scale
  num_eta = exp(num_location)
  rweibull(n, shape = num_beta, scale = num_eta)
}

inv_psev = function(vec_u,num_location = 0, num_sclae = 1)
{
  vec_z = log(-log(1-vec_u))
  vec_z * num_sclae + num_location
}

my_rweibull_v2 = function(n, num_location = 0, num_scale = 1)
{
  vec_x = runif(n)
  vec_x = inv_psev(vec_x,num_location = num_location, num_sclae = num_scale)
  exp(vec_x)
}

# code for simulation study
dat_simu_distr = function(n_cate_vec = c(100,100,100))
{
  if(length(n_cate_vec)!=3)
  {
    stop("Simulation must have 3 categories")
  }
  #z0 lognormal
  #z1 loglogistic
  #z3 weibull
  #mu = 10*x1+ 25*x2^2
  #sigma = x3*
  require(tidyverse)
  require(lhs)
  int_N = sum(n_cate_vec)
  mat_distpara = matrix(ncol = 3+3, nrow = int_N)
  colnames(mat_distpara) = c("x1","x2","x3","z","mu","sigma")
  mat_distpara[,1:3] = randomLHS(int_N,3)
  vec_tmp = c(0, cumsum(n_cate_vec))
  for(j in 1:length(n_cate_vec))
  {
    mat_distpara[(vec_tmp[j]+1):vec_tmp[j+1],4] = j-1
  }
  rm(vec_tmp)
  mat_distpara[,5] = 3*mat_distpara[,1] + 2*mat_distpara[,2]^2
  mat_distpara[,6] = mat_distpara[,3]*2
  mat_sample = matrix(ncol = 6 + 1, nrow = 0)
  colnames(mat_sample) = c("x1","x2","x3","z","mu","sigma", "Y")
  ll_rfun = list(rnorm,rlnorm,rloglogis,my_rweibull_v1)
  mat_tmp = matrix(ncol = 7, nrow = 300)
  ll = 1
  for(j in 1:length(n_cate_vec))
  {
    rfun = ll_rfun[[j]]
    for(k in 1:n_cate_vec[j])
    {
      for(ii in 1:6)
      {
        mat_tmp[,ii] = mat_distpara[ll,ii]
      }
      mat_tmp[,7] = rfun(300,mat_distpara[ll,"mu"],mat_distpara[ll,"sigma"])
      mat_tmp[,7][mat_tmp[,7]>1e4] = 1e4
      mat_sample = rbind(mat_sample, mat_tmp)
      ll = ll + 1
    }
  }
  mat_sample
}

dat_simu_mvnorm = function(n_cate_vec = c(100,100,100), rmax = 3.0)
{
  require(lhs)
  require(mvtnorm)

  vec_thetag = c(1,2,3,0.3)
  num_sigma_epssq = 2
  vec_rho = c(0.1,0.5,0.7) * pi
  num_sigma_alpsq = 1
  num_mu = 2.0

  int_N = sum(n_cate_vec)
  mat_cted = diag(int_N) - matrix(1, ncol = int_N, nrow = int_N)/int_N
  # mat_X = randomLHS(int_N, length(n_cate_vec))
  mat_X = matrix(runif(int_N * length(n_cate_vec)), nrow = int_N)
  ncate = length(n_cate_vec)
  vec_Z = c()
  for(i in 1:ncate)
  {
    vec_Z = c(vec_Z, rep(i - 1, n_cate_vec[i]))
  }
  mat_eps = block_corrM(
    mat_X,
    vec_thetag[1:ncate],
    vec_thetag[ncate+1],
    n_cate_vec) * num_sigma_epssq
  
  mat_alp = sigma_alpha_inv_gen_CSK_W_KP_Cpp(
    mat_X,
    vec_Z,
    vec_rho,
    rmax,
    ncate,
    is_inverse = F) * num_sigma_alpsq
  
  
  # mat_cov = 0.5 * mat_cov + 0.5 * t(mat_cov)
  vec_Y = rmvnorm(1, mean = rep(num_mu, int_N), sigma = mat_eps)
  vec_Y = t(vec_Y) + mat_cted %*% t(rmvnorm(1, sigma = mat_alp))
  vec_Y = as.numeric(vec_Y)
  list(x = mat_X, z = vec_Z, y = vec_Y, num_lev = n_cate_vec,
    vec_thetag = vec_thetag,
    num_sigma_epssq = num_sigma_epssq,
    vec_rho = vec_rho,
    num_sigma_alpsq = num_sigma_alpsq,
    num_mu = num_mu)
}

dat_simu_mvnorm_sep = function(n_cate_vec = c(100,100,100), rmax = 3.0)
{
  require(lhs)
  require(mvtnorm)
  
  vec_theta = c(1,2,3,3,2,1,2,1,3)
  vec_nugg = c(0.1,0.4,0.8)
  vec_sigma_epssq = c(1,1.4,1.7)
  vec_rho = c(0.1,0.5,0.7) * pi
  num_sigma_alpsq = 1
  num_mu = c(1,3,5)

  ncate = length(n_cate_vec)
  int_N = sum(n_cate_vec)
  start_idx=1
  mat_cted = matrix(0.0, ncol = int_N, nrow = int_N)
  for(j in 1:ncate)
  {
    end_idx=sum(n_cate_vec[1:j])
    mat_cted[start_idx:end_idx,start_idx:end_idx] = diag(n_cate_vec[j]) -
      matrix(1, ncol = n_cate_vec[j], nrow = n_cate_vec[j])/n_cate_vec[j]
    start_idx=start_idx+n_cate_vec[j]
  }
  # mat_X = randomLHS(int_N, length(n_cate_vec))
  mat_X = matrix(runif(int_N * length(n_cate_vec)), nrow = int_N)
  vec_Z = c()
  for(i in 1:ncate)
  {
    vec_Z = c(vec_Z, rep(i - 1, n_cate_vec[i]))
  }
  # browser()
  mat_eps = block_corrM_seppar(mat_X, vec_theta, vec_nugg, n_cate_vec)
  vec_mean = rep(NA, int_N)
  start_idx=1
  for(j in 1:ncate)
  {
    
    end_idx=sum(n_cate_vec[1:j])
    
    mat_eps[start_idx:end_idx,start_idx:end_idx] = 
      mat_eps[start_idx:end_idx,start_idx:end_idx]*vec_sigma_epssq[j]
    vec_mean[start_idx:end_idx] = num_mu[i]
    start_idx=start_idx+n_cate_vec[j]
  }
  
  mat_alp = sigma_alpha_inv_gen_CSK_W_KP_Cpp(
    mat_X,
    vec_Z,
    vec_rho,
    rmax,
    ncate,
    is_inverse = F) * num_sigma_alpsq
  
  
  
  vec_Y = rmvnorm(1, mean = vec_mean, sigma = mat_eps)
  vec_Y = t(vec_Y) + mat_cted %*% t(rmvnorm(1, sigma = mat_alp))
  vec_Y = as.numeric(vec_Y)
  list(x = mat_X, z = vec_Z, y = vec_Y, num_lev = n_cate_vec,
       vec_theta = vec_theta,
       vec_nugg = vec_nugg,
       vec_sigma_epssq = vec_sigma_epssq,
       vec_rho = vec_rho,
       num_sigma_alpsq = num_sigma_alpsq,
       num_mu = num_mu)
}

para_simu_mvnorm = function(para_idx = 1, n_cate_vec = c(100,100,100), num_rmax = 3.0, save_res = F, file_name)
{
  vec_n_cate_pre = c(100, 100, 100) * 5
  llvar = dat_simu_mvnorm(n_cate_vec, rmax = num_rmax)
  llvar_pre = dat_simu_mvnorm(vec_n_cate_pre, rmax = num_rmax)
  obj_lmgp = lmgp_EM(llvar, 500, rmax = num_rmax)
  obj_lmgp_sep = lmgp_EM_seppar(llvar, 500, rmax = num_rmax)

  vec_y_pre = predict_lmgp_EM(obj_lmgp[[length(obj_lmgp)]], llvar, llvar_pre, rmax = num_rmax)
  vec_y_pre_sep = predict_lmgp_em_seppar(obj_lmgp_sep[[length(obj_lmgp_sep)]], llvar, llvar_pre, rmax = num_rmax)
  
  ll_true = list()
  ll_true$mod_nll1$par = llvar$vec_thetag
  ll_true$num_mu = llvar$num_mu
  ll_true$num_sigmasq_eps = llvar$num_sigma_epssq
  ll_true$num_sigmasq_alp = llvar$num_sigma_alpsq
  ll_true$mod_nll2$par = llvar$vec_rho
  vec_y_pre_true = predict_lmgp_EM(ll_true, llvar, llvar_pre, rmax = num_rmax)

  vec_y_pre_gp = gp_fit(datalist_train = llvar, datalist_test = llvar_pre)
  vec_y_pre_cgp = cgp_fit(datalist_train = llvar, datalist_test = llvar_pre)

  ##save the y_pre_true and y_pre as dataframce
  dat_pre = data.frame(
    y = vec_y_pre_true,
    lmgp = vec_y_pre,
    lmgp_sep = vec_y_pre_sep,
    gp = vec_y_pre_gp,
    cgp = vec_y_pre_cgp
  )
  if(save_res)
  {
    try(
      dir.create("save_res")
    )
    save(llvar,llvar_pre,obj_lmgp,obj_lmgp_sep,dat_pre,
      file = file_name)
  }
  list(
    lmgp = obj_lmgp,
    lmgp_sep = obj_lmgp_sep,
    data_pre = dat_pre
  )
}

para_simu_mvnorm = function(para_idx = 1, n_cate_vec = c(100,100,100), num_rmax = 3.0, save_res = F, file_name = NA)
{
  vec_n_cate_pre = c(100, 100, 100) * 5
  llvar = dat_simu_mvnorm(n_cate_vec, rmax = num_rmax)
  llvar_pre = dat_simu_mvnorm(vec_n_cate_pre, rmax = num_rmax)

  vec_y_pre_gp = gp_fit(datalist_train = llvar, datalist_test = llvar_pre)
  vec_y_pre_cgp = cgp_fit(datalist_train = llvar, datalist_test = llvar_pre)


  ll_ini = ini_val(llvar)

  obj_lmgp = lmgp_EM(llvar, 500, rmax = num_rmax, using_C = T, cgp_start = ll_ini$cgp, stop_early = F)
  obj_lmgp_sep =  lmgp_EM_seppar(llvar, 500, rmax = num_rmax, using_C = T, cgp_start = ll_ini$cgp, stop_early = F)
  
  vec_y_pre = predict_lmgp_EM(obj_lmgp[[length(obj_lmgp)]], llvar, llvar_pre, rmax = num_rmax)
  vec_y_pre_sep = predict_lmgp_em_seppar(obj_lmgp_sep[[length(obj_lmgp_sep)]], llvar, llvar_pre, rmax = num_rmax)
  
  ll_true = list()
  ll_true$mod_nll1$par = llvar$vec_thetag
  ll_true$num_mu = llvar$num_mu
  ll_true$num_sigmasq_eps = llvar$num_sigma_epssq
  ll_true$num_sigmasq_alp = llvar$num_sigma_alpsq
  ll_true$mod_nll2$par = llvar$vec_rho
  vec_y_pre_true = predict_lmgp_EM(ll_true, llvar, llvar_pre, rmax = num_rmax)
  
  
  
  ##save the y_pre_true and y_pre as dataframce
  dat_pre = data.frame(
    y = vec_y_pre_true,
    lmgp = vec_y_pre,
    lmgp_sep = vec_y_pre_sep,
    gp = vec_y_pre_gp,
    cgp = vec_y_pre_cgp
  )
  if(save_res)
  {
    try(
      dir.create("save_res")
    )
    save(llvar,llvar_pre,obj_lmgp,obj_lmgp_sep,dat_pre,
         file = file_name)
  }
  list(
    lmgp = obj_lmgp,
    lmgp_sep = obj_lmgp_sep,
    data_pre = dat_pre
  )
}


para_simu_mvnorm_sep = function(para_idx = 1, n_cate_vec = c(100,100,100), num_rmax = 3.0, save_res = F, file_name = NA)
{
  vec_n_cate_pre = c(100, 100, 100) * 5
  llvar = dat_simu_mvnorm_sep(n_cate_vec, rmax = num_rmax)
  llvar_pre = dat_simu_mvnorm_sep(vec_n_cate_pre, rmax = num_rmax)
  int_p = dim(llvar$x)[2]


  vec_y_pre_gp = gp_fit(datalist_train = llvar, datalist_test = llvar_pre)
  vec_y_pre_cgp = cgp_fit(datalist_train = llvar, datalist_test = llvar_pre)

  ll_ini = ini_val(llvar)
  
  obj_lmgp = lmgp_EM(llvar, 500, rmax = num_rmax, using_C = T, cgp_start = ll_ini$cgp, stop_early = F)
  obj_lmgp_sep =  lmgp_EM_seppar(llvar, 500, rmax = num_rmax, using_C = T, cgp_start = ll_ini$cgp, stop_early = F)
  
  vec_y_pre = predict_lmgp_EM(obj_lmgp[[length(obj_lmgp)]], llvar, llvar_pre, rmax = num_rmax)
  vec_y_pre_sep = predict_lmgp_em_seppar(obj_lmgp_sep[[length(obj_lmgp_sep)]], llvar, llvar_pre, rmax = num_rmax)
  
  ll_true = obj_lmgp_sep[[length(obj_lmgp_sep)]]
  for(i in 1:length(n_cate_vec))
  {
    ll_true$ll_mod_nll1[[i]]$par = c(llvar$vec_theta[(int_p * (i-1) + 1) : (int_p * i)],llvar$vec_nugg[i])
  }
  ll_true$muhat = llvar$num_mu
  ll_true$sigma_eps_sq = llvar$vec_sigma_epssq
  ll_true$sigma_alp_sq = llvar$num_sigma_alpsq
  ll_true$mod_nll2$par = llvar$vec_rho
  vec_y_pre_true = predict_lmgp_em_seppar(ll_true, llvar, llvar_pre, rmax = num_rmax)
  
  
  
  ##save the y_pre_true and y_pre as dataframce
  dat_pre = data.frame(
    y = vec_y_pre_true,
    lmgp = vec_y_pre,
    lmgp_sep = vec_y_pre_sep,
    gp = vec_y_pre_gp,
    cgp = vec_y_pre_cgp
  )
  if(save_res)
  {
    try(
      dir.create("save_res")
    )
    save(llvar,llvar_pre,obj_lmgp,obj_lmgp_sep,dat_pre,
         file = file_name)
  }
  list(
    lmgp = obj_lmgp,
    lmgp_sep = obj_lmgp_sep,
    data_pre = dat_pre
  )
}

iozone_split = function(train_ratio, Rda_dir = ".")
{
  combined_coef = data.frame()
  vec_files = c("initial_writers", "random_readers", "random_writers", "re-readers", "readers", "rewriters")
  ll = 1
  for (i in vec_files)
  {
    load(paste(Rda_dir, i, ".Rdata", sep = ""))
    coef_dat$z = ll - 1
    coef_dat$mode = str_split(i, pattern = "[.]")[[1]][1]
    print(str_split(i, pattern = "[.]")[[1]][1])
    ll = ll + 1
    combined_coef = rbind(combined_coef, coef_dat)
  }
  
  # str_wd = getwd()
  # ll_wd = strsplit(str_wd, split = "/")[[1]]
  # ll_wd = strsplit(ll_wd[[length(ll_wd)]], split = "")[[1]]
  # vec_mode = as.numeric(ll_wd)
  vec_mode = c(1,2,3)
  combined_sub_coef = filter(
    combined_coef,
    Record.Size == unique(combined_coef$Record.Size)[3],
    z %in% vec_mode)
  combined_sub_coef$z_coded = NA
  ll = 0
  for (i in vec_mode)
  {
    combined_sub_coef$z_coded[combined_sub_coef$z == i] = ll
    ll = ll + 1
  }
  
  vec_x_idx = c(1,2,4)
  num_r_max = 3.0
  ncates = length(unique(combined_sub_coef$z_coded))
  vec_train_idx = 1:dim(combined_sub_coef)[1]
  vec_train_idx = sample(vec_train_idx, round(length(vec_train_idx) * train_ratio))
  vec_train_idx = sort(vec_train_idx)
  
  subdat_train = combined_sub_coef[vec_train_idx,]
  subdat_test_int = combined_sub_coef[-vec_train_idx,]
  
  svd_obj = svd(subdat_train[, 5:25])
  w_train = as.matrix(subdat_train[, 5:25]) %*% svd_obj$v
  w_test_int = as.matrix(subdat_test_int[, 5:25]) %*% svd_obj$v
  
  ll_iozone = list()
  ll_iozone$train = list(
    X = as.matrix(subdat_train[, vec_x_idx]),
    Ymat = w_train,
    Z = subdat_train$z_coded,
    num_lev = table(subdat_train$z_coded)
  )
  ll_iozone$test = list(
    X = as.matrix(subdat_test_int[, vec_x_idx]),
    Z = subdat_test_int$z_coded,
    Ymat = w_test_int,
    num_lev = table(subdat_test_int$z_coded)
  )
  ll_iozone$svd = svd_obj
  ll_iozone
}

para_iozone_fit = function(paraidx=1, train_ratio = 0.7, Rda_dir = ".", num_rmax = 3.0, adaptsvd = T)
{
  ll_iozone_dat = iozone_split(train_ratio, Rda_dir)
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
    vec_y_pre_gp = gp_fit(datalist_train = ll_train, datalist_test = ll_test)
    vec_y_pre_cgp = cgp_fit(datalist_train = ll_train, datalist_test = ll_test)
    
    ll_ini = ini_val(ll_train)
    obj_lmgp = lmgp_EM(ll_train, 500, rmax = num_rmax, using_C = T, cgp_start = ll_ini$cgp, stop_early = T)
    obj_lmgp_sep = lmgp_EM_seppar(ll_train, 500, rmax = num_rmax, using_C = T, cgp_start = ll_ini$cgp, stop_early = T)
    vec_y_pre = predict_lmgp_EM(obj_lmgp[[length(obj_lmgp)]], ll_train, ll_test, rmax = num_rmax)
    vec_y_pre_sep = predict_lmgp_em_seppar(obj_lmgp_sep[[length(obj_lmgp_sep)]], ll_train, ll_test, rmax = num_rmax)
    dat_pre = data.frame(
      lmgp = vec_y_pre,
      lmgp_sep = vec_y_pre_sep,
      gp = vec_y_pre_gp,
      cgp = vec_y_pre_cgp
    )
    ll_iozone_pre[[ii]] = dat_pre
  }
  list(
    data = ll_iozone_dat,
    pre = ll_iozone_pre
  )
}