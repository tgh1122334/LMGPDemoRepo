lmgp_nll_eps = function(
  vec_new_para1,
  vec_cond_mean,
  mat_cond_sigma,
  datalist)
{
  # print(vec_new_para1)
  int_n = length(datalist$y)
  int_p = dim(datalist$x)[2]
  # browser()
  # vec_new_para1 = exp(vec_new_para1)
  # print(vec_new_para1)
  vec_new_nu = vec_new_para1[1:int_p]
  num_new_g = vec_new_para1[int_p+1]
  mat_omega_eps = block_corrM(
    datalist$x,
    vec_new_nu,
    num_new_g,
    datalist$num_lev)
  mat_omega_eps_inv = block_inv(
    mat_omega_eps,
    datalist$num_lev
  )
  num_mu = EXACTEM_nll1_mu(
    datalist$y,
    mat_omega_eps_inv,
    vec_cond_mean,
    mat_cond_sigma
  )
  num_sigmasq_eps = EXACTEM_nll1_sigsq(
    num_mu,
    datalist$y,
    mat_omega_eps_inv,
    vec_cond_mean,
    mat_cond_sigma
  )
  # print(c(num_mu,num_sigmasq_eps))
  # browser()
  num_nll = - determinant(mat_omega_eps)$modulus * 0.5
  # num_nll = -sum(log(diag(chol(mat_omega_eps))))
  # num_nll = - 2 * sum(log(diag(chol(mat_omega_eps)))) * 0.5
  num_nll = num_nll - log(num_sigmasq_eps)*int_n/2.0
  return(-num_nll)
}

lmgp_nll_alp = function(
  vec_new_alpha,
  vec_cond_mean,
  mat_cond_mat,
  datalist,
  rmax = 0.2)
{
  # browser()
  int_n = length(datalist$y)
  ncates = sum(datalist$num_lev!=0)
  # vec_new_alpha = pi/(1+exp(-vec_new_alpha))
  # vec_new_alpha[abs(vec_new_alpha-pi)<pi_adjust] = pi - pi_adjust
  # vec_new_alpha[abs(vec_new_alpha+pi)<pi_adjust] = -pi + pi_adjust
  mat_omega_alp = sigma_alpha_inv_gen_CSK_W(
    datalist$x,
    datalist$z,
    vec_new_alpha,
    rmax, ncates, F)
  # mat_omega_alp_inv = chol2inv(chol(mat_omega_alp))
  mat_omega_alp_inv = solve(mat_omega_alp)
  num_sigmasq_alp = EXACTEM_nll2_sigsq(
    mat_omega_alp_inv,
    vec_cond_mean,
    mat_cond_mat
  )
  num_nll = - determinant(mat_omega_alp)$modulus * 0.5
  # num_nll = -sum(log(diag(chol(mat_omega_alp))))
  # num_nll = - 2 * sum(log(diag(chol(mat_omega_alp)))) * 0.5
  num_nll = num_nll - log(num_sigmasq_alp) * int_n/2.0
  return(-num_nll)
}

lmgp_EM = function(
  datalist,
  num_iter = 500,
  rmax = 3.0,
  using_C = T,
  cgp_start,
  stop_early = F
)
{
  int_n = length(datalist$y)
  ncates = sum(datalist$num_lev!=0)
  int_p = dim(datalist$x)[2]
  # vec_para1 = c(rep(1e-3,int_p), 0.5)
  # vec_rho = rep(pi/2, ncates*(ncates-1)/2)
  ##use cgp start
  vec_para1 = cgp_start[1:(int_p+1)]
  vec_rho = cgp_start[(int_p+2):length(cgp_start)]
  ##create cov matrix, mu, sigma
  vec_nu = vec_para1[1:int_p]
  num_g = vec_para1[int_p + 1]
  num_sigmasq_eps = 1.0
  num_sigmasq_alp = 1.0
  num_mu = 0.0
  mat_cted = diag(int_n) - matrix(1, ncol = int_n, nrow = int_n)/int_n
  em_rep_list = list()
  vec_nll2 = rep(NA,num_iter)
  for(iter in 1:num_iter)
  {
    ##matrix
    mat_sigma_eps = block_corrM(
      datalist$x,
      vec_nu,
      num_g,
      datalist$num_lev) * num_sigmasq_eps
    mat_sigma_alpha = sigma_alpha_inv_gen_CSK_W(
      datalist$x,
      datalist$z,
      vec_rho,
      rmax, ncates, F) * num_sigmasq_alp
    ##cond dist alp
    vec_cond_mu = mat_sigma_alpha %*% 
      solve(mat_sigma_alpha + mat_sigma_eps, datalist$y - num_mu)
    mat_cond_mat = mat_sigma_alpha -
      mat_sigma_alpha %*% solve(mat_sigma_alpha + mat_sigma_eps) %*% 
      mat_sigma_alpha

    
    vec_cond_mu = mat_cted %*% vec_cond_mu
    mat_cond_mat = mat_cted %*% mat_cond_mat %*% mat_cted
    ###optim
    # browser()
    if(using_C)
    {
      optim_obj1 = optim(
        c(vec_nu,num_g),
        c_lmgp_nll_eps,
        method = "L-BFGS-B",
        lower = c(rep(1e-3, int_p), 1e-5),
        upper = c(rep(1000, int_p), 5.0),
        cond_mean = vec_cond_mu, 
        cond_sigma = mat_cond_mat,
        X = datalist$x,
        Y = datalist$y,
        Z = datalist$z,
        num_lev = datalist$num_lev
      )
      optim_obj2 = optim(
        vec_rho,
        c_lmgp_nll_alp,
        method = "L-BFGS-B",
        lower = rep(1e-3,length(vec_rho)),
        upper = rep(pi - 1e-3,length(vec_rho)),
        cond_mean = vec_cond_mu,
        cond_sigma = mat_cond_mat, 
        X = datalist$x,
        Y = datalist$y,
        Z = datalist$z,
        num_lev = datalist$num_lev,
        rmax = rmax
      )
    } else
    {
      optim_obj1 = optim(
        c(vec_nu,num_g),
        lmgp_nll_eps,
        method = "L-BFGS-B",
        lower = c(rep(1e-3, int_p), 1e-5),
        upper = c(rep(1000, int_p), 5.0),
        vec_cond_mean = vec_cond_mu, 
        mat_cond_sigma = mat_cond_mat,
        datalist = datalist
      )
      optim_obj2 = optim(
        vec_rho,
        lmgp_nll_alp,
        method = "L-BFGS-B",
        lower = rep(1e-3,length(vec_rho)),
        upper = rep(pi - 1e-3,length(vec_rho)),
        vec_cond_mean = vec_cond_mu,
        mat_cond_mat = mat_cond_mat, 
        datalist = datalist,
        rmax = rmax
      )
    }
    ##update for mu two sigma
    vec_nu = optim_obj1$par[1:int_p]
    num_g = optim_obj1$par[int_p + 1]
    vec_rho = optim_obj2$par
    vec_nll2[iter] = optim_obj2$value
    
    mat_omega_eps = block_corrM(
      datalist$x,
      vec_nu,
      num_g,
      datalist$num_lev)
    mat_omega_eps_inv = block_inv(
      mat_omega_eps, datalist$num_lev
    )
    mat_omega_alp = sigma_alpha_inv_gen_CSK_W(
      datalist$x,
      datalist$z,
      vec_rho,
      rmax, ncates, F)
    mat_omega_alp_inv = solve(mat_omega_alp)
    num_mu = EXACTEM_nll1_mu(
      datalist$y,
      mat_omega_eps_inv,
      vec_cond_mu,
      mat_cond_mat
    )
    num_sigmasq_eps = EXACTEM_nll1_sigsq(
      num_mu,
      datalist$y,
      mat_omega_eps_inv,
      vec_cond_mu,
      mat_cond_mat
    )
    num_sigmasq_alp = EXACTEM_nll2_sigsq(
      mat_omega_alp_inv,
      vec_cond_mu,
      mat_cond_mat
    )
    em_rep_list[[iter]] = list(
      mod_nll1 = optim_obj1,
      mod_nll2 = optim_obj2,
      num_mu = num_mu,
      num_sigmasq_eps = num_sigmasq_eps,
      num_sigmasq_alp = num_sigmasq_alp
    )
    num_rat = NA
    if(stop_early)
    {
      if((iter > 50))
      {
        num_rat = sd(vec_nll2[(iter - 40):iter])/abs(mean(vec_nll2[(iter - 40):iter]))
        if(num_rat < 0.0001)
        {
          return(em_rep_list)
          break
        }
      }
      cat("finish iterations\t",iter,"/",num_iter,"---",num_rat,"\r")
    } else{
      cat("finish iterations\t",iter,"/",num_iter,"\r")
    }
    
    # save(em_rep_list, file = "simudebug.RData")
  }
  em_rep_list
}

predict_lmgp_EM = function(
  EM_mod,
  datalist,
  newdatalist,
  rmax = 0.3
)
{
  vec_thetahat = EM_mod$mod_nll1$par
  vec_cthethat = EM_mod$mod_nll2$par
  num_sigma_eps_sq = EM_mod$num_sigmasq_eps
  num_sigma_alp_sq = EM_mod$num_sigmasq_alp
  num_muhat = EM_mod$num_mu
  
  ncates = length(datalist$num_lev)
  ndim = dim(datalist$x)[2]
  
  mat_sigma_eps_train = block_corrM(
    datalist$x,
    vec_thetahat[1:ndim],
    vec_thetahat[ndim+1],
    datalist$num_lev) * num_sigma_eps_sq
  
  mat_sigma_alp_train = sigma_alpha_inv_gen_CSK_W(
    datalist$x,
    datalist$z,
    vec_cthethat,
    rmax,
    ncates,
    is_inverse = F) * num_sigma_alp_sq
  
  mat_sigma_eps_test = block_corrM(
    newdatalist$x,
    vec_thetahat[1:ndim],
    vec_thetahat[ndim],
    newdatalist$num_lev
  ) * num_sigma_eps_sq
  
  
  mat_sigma_alp_test = sigma_alpha_inv_gen_CSK_W(newdatalist$x, newdatalist$z, vec_cthethat,
                                                 rmax, ncates, is_inverse = F) * num_sigma_alp_sq
  mat_sigma_eps_train_test = block_corrM_X1X2(
    datalist$x,
    newdatalist$x,
    vec_thetahat[1:ndim],
    datalist$num_lev,
    newdatalist$num_lev
  ) * num_sigma_eps_sq
  
  # mat_sigma_eps_train_test = convM_X1X2(
  #   datalist$x,
  #   newdatalist$x,
  #   vec_thetahat[1:ndim]
  # ) * num_sigma_eps_sq
  
  mat_sigma_alp_train_test = correlation_alpha_X1X2_CSK_W(
    datalist$x,
    newdatalist$x,
    datalist$z,
    newdatalist$z,
    vec_cthethat,
    rmax,ncates) * num_sigma_alp_sq
  # browser()
  vec_w_pre = num_muhat + t(mat_sigma_eps_train_test + mat_sigma_alp_train_test) %*%
    solve(mat_sigma_eps_train + mat_sigma_alp_train,
          datalist$y - num_muhat)
  return(vec_w_pre)
}

lmgp_EM_seppar = function(
  datalist,
  num_iter = 500,
  rmax = 3.0,
  using_C = T,
  cgp_start,
  stop_early = F
)
{
  int_n = length(datalist$y)
  int_p = dim(datalist$x)[2]
  ncates = length(datalist$num_lev)

  num_current_time = Sys.time()
  # vec_para_start = rep(c(rep(1e-3,int_p),0.5), ncates)
  # vec_para_start = c(vec_para_start, rep(pi/2,ncates*(ncates-1)/2))
  ###use cgp start
  vec_para_start = rep(cgp_start[1:(int_p+1)], ncates)
  vec_para_start = c(vec_para_start, cgp_start[(int_p+2):length(cgp_start)])
  require(mvtnorm)
  em_rep_list = list()
  vec_nll2 = rep(NA,num_iter)
  ##get a starting nlikeliother parameters
  muhat = rep(0, ncates)
  sigma_eps_sq = rep(1.0,ncates)
  sigma_alp_sq = 1.0
  # browser()
  start_idx=1
  mat_cted = matrix(0.0, ncol = int_n, nrow = int_n)
  for(j in 1:ncates)
  {
    end_idx=sum(datalist$num_lev[1:j])
    mat_cted[start_idx:end_idx,start_idx:end_idx] = diag(datalist$num_lev[j]) -
      matrix(1, ncol = datalist$num_lev[j], nrow = datalist$num_lev[j])/datalist$num_lev[j]
    start_idx=start_idx+datalist$num_lev[j]
  }
  # mat_cted = diag(int_n) - matrix(1, ncol = int_n, nrow = int_n)/int_n
  for (ii in 1:num_iter)
  {
    
    vec_nll_likeli1 = vec_para_start[1:(ncates*(int_p + 1))] ## the last one is nugget
    # print(vec_nll_likeli1)
    ###divide vec_theta and vec_nugg
    vec_theta = rep(NA, ncates*int_p)
    vec_nugg = rep(NA, ncates)
    for(j in 1:ncates)
    {
      vec_theta[(j*int_p-int_p+1):(j*int_p)] = vec_nll_likeli1[(j*(int_p+1)-int_p):(j*int_p+j-1)]
      vec_nugg[j] = vec_nll_likeli1[j*int_p+j]
    }

    vec_rho = vec_para_start[(ncates*(int_p + 1) + 1):length(vec_para_start)]
    # vec_rho[abs(vec_rho-pi)<1e-4] = pi - 1e-4
    mat_sigma_eps = block_corrM_seppar(datalist$x, vec_theta, vec_nugg, datalist$num_lev)
    start_idx=1
    for(j in 1:ncates)
    {
      
      end_idx=sum(datalist$num_lev[1:j])
      mat_sigma_eps[start_idx:end_idx,start_idx:end_idx] = 
        mat_sigma_eps[start_idx:end_idx,start_idx:end_idx]*sigma_eps_sq[j]
      start_idx=start_idx+datalist$num_lev[j]
    }
    mat_sigma_alpha = sigma_alpha_inv_gen_CSK_W(datalist$x, datalist$z, vec_rho, rmax, ncates, F) * 
      sigma_alp_sq
    vec_y_c=datalist$y
    start_idx=1
    for(j in 1:ncates)
    {
      end_idx=sum(datalist$num_lev[1:j])
      vec_y_c[start_idx:end_idx] = vec_y_c[start_idx:end_idx] - muhat[j]
      start_idx=start_idx+datalist$num_lev[j]
    }
    vec_alp_cond_mean = mat_sigma_alpha %*% solve(mat_sigma_alpha + mat_sigma_eps,vec_y_c)
    mat_alp_cond_covar = mat_sigma_alpha -
      mat_sigma_alpha %*% solve(mat_sigma_alpha + mat_sigma_eps) %*% mat_sigma_alpha
    
    vec_alp_cond_mean = mat_cted %*% vec_alp_cond_mean
    mat_alp_cond_covar = mat_cted %*% mat_alp_cond_covar %*% mat_cted

    ll_optim_nll1 = list()
    start_idx=1
    # browser()
    for(j in 1:ncates)
    {
      end_idx=sum(datalist$num_lev[1:j])
      subdatalist = list(
        x = datalist$x[start_idx:end_idx,],
        z = datalist$z[start_idx:end_idx],
        y = datalist$y[start_idx:end_idx],
        num_lev = c(end_idx-start_idx+1)
      )
      # browser()
      if(using_C)
      {
        # browser()
        optim_obj1 = optim(
          vec_para_start[(j*(int_p+1)-int_p):(j*int_p+j)],
          c_lmgp_nll_eps,
          method = "L-BFGS-B",
          lower = c(rep(1e-3, int_p),1e-5),
          upper = c(rep(1000, int_p),5.0),
          cond_mean = vec_alp_cond_mean[start_idx:end_idx], 
          cond_sigma = mat_alp_cond_covar[start_idx:end_idx,start_idx:end_idx],
          X = subdatalist$x,
          Y = subdatalist$y,
          Z = subdatalist$z,
          num_lev = subdatalist$num_lev
        )
      } else
      {
        optim_obj1 = optim(
          vec_para_start[(j*(int_p+1)-int_p):(j*int_p+j)],
          lmgp_nll_eps,
          method = "L-BFGS-B",
          lower = c(rep(1e-3, int_p),1e-5),
          upper = c(rep(1000, int_p),5.0),
          vec_cond_mean = vec_alp_cond_mean[start_idx:end_idx], 
          mat_cond_sigma = mat_alp_cond_covar[start_idx:end_idx,start_idx:end_idx],
          datalist = subdatalist
        )
      }
      ll_optim_nll1[[j]] =  optim_obj1
      vec_para_start[(j*(int_p+1)-int_p):(j*int_p+j)] = optim_obj1$par
      ###renew mu sigma
      vec_nu = optim_obj1$par[1:int_p]
      num_g = optim_obj1$par[int_p+1]
      mat_omega_eps = convM(
        subdatalist$x,
        vec_nu,
        num_g
      )
      mat_omega_eps_inv = solve(
        mat_omega_eps
      )
      muhat[j] = EXACTEM_nll1_mu(
        subdatalist$y,
        mat_omega_eps_inv,
        vec_alp_cond_mean[start_idx:end_idx],
        mat_alp_cond_covar[start_idx:end_idx,start_idx:end_idx]
      )
      sigma_eps_sq[j] = EXACTEM_nll1_sigsq(
        muhat[j],
        subdatalist$y,
        mat_omega_eps_inv,
        vec_alp_cond_mean[start_idx:end_idx],
        mat_alp_cond_covar[start_idx:end_idx,start_idx:end_idx]
      )
      start_idx=start_idx+datalist$num_lev[j]
    }
    
    if(using_C)
    {
      mod_nll2 =
        optim(
        vec_para_start[(ncates*(int_p+1) + 1):length(vec_para_start)],
        # rep(pi/2,length(vec_rho)),
        c_lmgp_nll_alp,
        method = "L-BFGS-B",
        lower = rep(1e-3,length(vec_rho)),
        upper = rep(pi - 1e-3,length(vec_rho)),
        cond_mean = vec_alp_cond_mean,
        cond_sigma = mat_alp_cond_covar, 
        X = datalist$x,
        Y = datalist$y,
        Z = datalist$z,
        num_lev = datalist$num_lev,
        rmax = rmax
      )
    } else
    {
      mod_nll2 =
        optim(
        vec_para_start[(ncates*(int_p+1) + 1):length(vec_para_start)],
        # rep(pi/2,length(vec_rho)),
        lmgp_nll_alp,
        method = "L-BFGS-B",
        lower = rep(1e-3,length(vec_rho)),
        upper = rep(pi - 1e-3,length(vec_rho)),
        vec_cond_mean = vec_alp_cond_mean,
        mat_cond_mat = mat_alp_cond_covar, 
        datalist = datalist,
        rmax = rmax
      )
    }
    # if(abs(mod_nll2$par[1] - 0.8016063)< 1e-7)
    # {
    #   browser()
    # }
    vec_nll2[ii] = mod_nll2$value
    ##renew sigmasq
    mat_omega_alp = sigma_alpha_inv_gen_CSK_W(
      datalist$x,
      datalist$z,
      mod_nll2$par,
      rmax, ncates, F)
    mat_omega_alp_inv = solve(mat_omega_alp)
    sigma_alp_sq = EXACTEM_nll2_sigsq(
      mat_omega_alp_inv,
      vec_alp_cond_mean,
      mat_alp_cond_covar
    )
    ##store results
    vec_para_start[(ncates*(int_p+1) + 1):length(vec_para_start)] = mod_nll2$par    
    em_rep_list[[ii]] = list(
      ll_mod_nll1 = ll_optim_nll1,
      mod_nll2 = mod_nll2,
      muhat = muhat,
      sigma_eps_sq = sigma_eps_sq,
      sigma_alp_sq = sigma_alp_sq,
      compute_time = Sys.time() - num_current_time
    )
    num_rat = NA
    if(stop_early)
    {
      if((ii > 50))
      {
        num_rat = sd(vec_nll2[(ii - 40):ii])/abs(mean(vec_nll2[(ii - 40):ii]))
        if(num_rat < 0.0001)
        {
          return(em_rep_list)
          break
        }
      }
      cat("finish iterations\t",ii,"/",num_iter,"---",num_rat,"\r")
    } else{
      cat("finish iterations\t",ii,"/",num_iter,"\r")
    }
  }
  em_rep_list
}

predict_lmgp_em_seppar = function(
  EM_mod,
  datalist,
  newdatalist,
  rmax = 0.3,
  bool_version2 = T
)
{
  ncates = length(datalist$num_lev)
  ndim = dim(datalist$x)[2]
  vec_theta = rep(NA, ncates*ndim)
  vec_nugg = rep(NA, ncates)
  vec_thetahat = rep(NA, ncates*(ndim+1))
  if(bool_version2)
  {
    for(j in 1:ncates)
    {
      vec_thetahat[(j*ndim+j-ndim):(j*ndim+j)] = EM_mod$ll_mod_nll1[[j]]$par
    }
  } else {
    vec_thetahat = EM_mod$mod_nll1$par
  }
  vec_cthethat = EM_mod$mod_nll2$par
  num_sigma_eps_sq = EM_mod$sigma_eps_sq
  num_sigma_alp_sq = EM_mod$sigma_alp_sq
  vec_muhat = EM_mod$muhat

  for(j in 1:ncates)
  {
    vec_theta[(j*ndim-ndim+1):(j*ndim)] = vec_thetahat[(j*(ndim+1)-ndim):(j*ndim+j-1)]
    vec_nugg[j] = vec_thetahat[j*ndim+j]
  }
  # browser()
  mat_sigma_eps_train = sigma_eps_seppar(
    datalist$x,
    vec_theta,
    vec_nugg,
    num_sigma_eps_sq,
    datalist$num_lev,
    is_inverse = FALSE
  )
  
  mat_sigma_alp_train = sigma_alpha_inv_gen_CSK_W(datalist$x, datalist$z, vec_cthethat,
                                             rmax, ncates, is_inverse = F) * num_sigma_alp_sq
  
  mat_sigma_eps_test = sigma_eps_seppar(
    newdatalist$x,
    vec_theta,
    vec_nugg,
    num_sigma_eps_sq,
    newdatalist$num_lev,
    is_inverse = FALSE
  )

  
  mat_sigma_alp_test = sigma_alpha_inv_gen_CSK_W(newdatalist$x, newdatalist$z, vec_cthethat,
                                            rmax, ncates, is_inverse = F) * num_sigma_alp_sq
  mat_sigma_eps_train_test = sigma_eps_x1x2_seppar(
    datalist$x,
    newdatalist$x,
    vec_theta,
    num_sigma_eps_sq,
    datalist$num_lev,
    newdatalist$num_lev
  )

  mat_sigma_alp_train_test = correlation_alpha_X1X2_CSK_W(
    datalist$x,
    newdatalist$x,
    datalist$z,
    newdatalist$z,
    vec_cthethat,
    rmax,ncates) * num_sigma_alp_sq
  ### pre version 0
  #### a union prediction
  vec_x2_mu=rep(0,sum(newdatalist$num_lev))
  start_idx=1
  for(j in 1:length(newdatalist$num_lev))
  {
    end_idx=sum(newdatalist$num_lev[1:j])
    if(end_idx < start_idx)
    {
      start_idx=start_idx+newdatalist$num_lev[j]
    } else
    {
      vec_x2_mu[start_idx:end_idx] = vec_muhat[j]
      start_idx=start_idx+newdatalist$num_lev[j]
    }
  }
  
  vec_x1_mu=rep(0,sum(datalist$num_lev))
  start_idx=1
  for(j in 1:length(datalist$num_lev))
  {
    end_idx=sum(datalist$num_lev[1:j])
    if(end_idx < start_idx)
    {
      start_idx=start_idx+datalist$num_lev[j]
    } else
    {
      vec_x1_mu[start_idx:end_idx] = vec_muhat[j]
      start_idx=start_idx+datalist$num_lev[j]
    }
  }
  vec_w_pre = vec_x2_mu + t(mat_sigma_eps_train_test + mat_sigma_alp_train_test) %*%
    solve(mat_sigma_eps_train + mat_sigma_alp_train,
          datalist$y - vec_x1_mu)
  return(vec_w_pre)
}



###baselines fit and predict
###gaussian process
gp_fit = function(datalist_train, datalist_test)
{
  int_p = dim(datalist_train$x)[2]
  vec_w_pre_gp_int = rep(NA, length(datalist_test$y))
  ncate = length(unique(datalist_train$z))
  for (i in 0:(ncate - 1))
  {
    # browser()
    vec_idx = datalist_train$z == i
    vec_idx_test_int = datalist_test$z == i
    optim_obj = optim(
      c(rep(1e-2, int_p), 1e-4),
      c_sep_gp_nll,
      method = "L-BFGS-B",
      lower = c(rep((1e-3), int_p),1e-8),
      upper = c(rep((1e3), int_p),5.0),
      X = datalist_train$x[vec_idx,],
      Y = datalist_train$y[vec_idx],
      para_coded = FALSE
    )
    # mod_gp = GauPro(X = datalist_train$x[vec_idx,],Z = datalist_train$y[vec_idx])
    vec_w_pre_gp_int[vec_idx_test_int] = c_sep_gp_pre(
      optim_obj$par,
      datalist_train$x[vec_idx, ],
      datalist_train$y[vec_idx],
      datalist_test$x[vec_idx_test_int, ],
      para_coded = FALSE)
  }
  vec_w_pre_gp_int
}

cgp_fit = function(datalist_train, datalist_test)
{
  ncate = length(unique(datalist_train$z))
  int_p = dim(datalist_train$x)[2]
  
  num_ctheta = ncate * (ncate - 1) / 2
  optim_obj = optim(
    c(rep(1e-2,int_p), 1e-4, rep(pi/2, num_ctheta)),
    c_cate_gp_nl,
    method = "L-BFGS-B",
    lower = c(rep(1e-3,int_p), 1e-5, rep(1e-3, num_ctheta)),
    upper = c(rep(1e3,int_p), 5.0, rep(pi - 1e-3, num_ctheta)),
    X = datalist_train$x,
    Y = datalist_train$y,
    Z = datalist_train$z,
    num_levels = ncate,
    para_coded = FALSE
  )
  c_cate_gp_pred(
    optim_obj$par,
    datalist_train$x,
    datalist_train$y,
    datalist_train$z,
    ncate,
    datalist_test$x,
    datalist_test$z,
    para_coded = FALSE
  )
}

###initial value provided by gp and cgp
ini_val = function(datalist_train)
{
  int_p = dim(datalist_train$x)[2]
  ncate = length(unique(datalist_train$z))
  mat_gp_par = matrix(nrow = ncate, ncol = int_p + 1)
  for (i in 0:(ncate - 1))
  {
    # browser()
    vec_idx = datalist_train$z == i
    gp_obj = optim(
      c(rep(1e-2, int_p), 1e-4),
      c_sep_gp_nll,
      method = "L-BFGS-B",
      lower = c(rep((1e-3), int_p),1e-8),
      upper = c(rep((1e3), int_p),5.0),
      X = datalist_train$x[vec_idx,],
      Y = datalist_train$y[vec_idx],
      para_coded = FALSE
    )
    mat_gp_par[i+1, ] = gp_obj$par
  }  
  num_ctheta = ncate * (ncate - 1) / 2
  cgp_obj = optim(
    c(rep(1e-2,int_p), 1e-4, rep(pi/2, num_ctheta)),
    c_cate_gp_nl,
    method = "L-BFGS-B",
    lower = c(rep(1e-3,int_p), 1e-5, rep(1e-3, num_ctheta)),
    upper = c(rep(1e3,int_p), 5.0, rep(pi - 1e-3, num_ctheta)),
    X = datalist_train$x,
    Y = datalist_train$y,
    Z = datalist_train$z,
    num_levels = ncate,
    para_coded = FALSE
  )
  list(gp = mat_gp_par, cgp = cgp_obj$par)
}

####gradient version
Amat = function(z,k,N)
{
    if(min(z) == 0)
    {
        z = z + 1
    }
    mat_A = matrix(ncol = N, nrow = 0)
    for(i in 1:k)
    {
        mat_temp = matrix(0.0, ncol = N, nrow = N)
        vec_k_cate = which(z == i)
        diag(mat_temp)[vec_k_cate] = 1.0
        mat_A = rbind(mat_A, mat_temp)
    }
    mat_A
}

#sigma_alpha_inv_gen_CSK_W(datalist$x,datalist$z,vec_rho,rmax, ncates, F)
sigma_alpha_inv_gen_CSK_W_KP = function(X,Z,rho,rmax,m,is_inverse)
{
    mat_Phi=CSK_W(X,rmax)
    mat_T = c_inverse_hsd(rho,m)
    mat_T = mat_T %*% t(mat_T)
    mat_A = Amat(Z,m, dim(X)[1])
    mat_sigma_alp = t(mat_A) %*% kronecker(mat_T, mat_Phi) %*% mat_A
    if(is_inverse)
    {
        solve(mat_sigma_alp)
    }
    mat_sigma_alp
}

#approximated grad for GP with mu and sigma
lmgp_nll_eps_grad = function(
  vec_new_para1,
  vec_cond_mean,
  mat_cond_sigma,
  datalist)
{
  vec_nll1_grad = rep(NA, length(vec_new_para1))
  int_p = length(vec_new_para1) - 1
  int_n = length(datalist$y)
  vec_new_nu = vec_new_para1[1:int_p]
  num_new_g = vec_new_para1[int_p+1]
  mat_omega_eps = block_corrM(
      datalist$x,
      vec_new_nu,
      num_new_g,
      datalist$num_lev)
  mat_omega_eps_inv = block_inv(
      mat_omega_eps,
      datalist$num_lev
  )
  num_mu = EXACTEM_nll1_mu(
      datalist$y,
      mat_omega_eps_inv,
      vec_cond_mean,
      mat_cond_sigma
  )
  num_sigmasq_eps = EXACTEM_nll1_sigsq(
      num_mu,
      datalist$y,
      mat_omega_eps_inv,
      vec_cond_mean,
      mat_cond_sigma
  )
  mat_dOmeEps_Dtheta = matrix(ncol = int_n, nrow = int_n)
  for(k in 1:int_p)
  {
    # browser()
    mat_omek = block_euclid_dist(as.matrix(datalist$x[,k]), datalist$num_lev)
    mat_dOmeEps_Dtheta = -mat_omega_eps * mat_omek
    mat_temp = mat_omega_eps_inv %*% mat_dOmeEps_Dtheta
    vec_nll1_grad[k] = sum(diag(mat_temp))
    dOmedThetak = -mat_temp %*% mat_omega_eps_inv 
    vec_nll1_grad[k] = vec_nll1_grad[k] + 
        t(datalist$y - num_mu) %*% dOmedThetak %*% (datalist$y - num_mu)/num_sigmasq_eps
    vec_nll1_grad[k] = vec_nll1_grad[k] -
        2 * t(datalist$y - num_mu) %*% dOmedThetak %*% vec_cond_mean/num_sigmasq_eps
    vec_nll1_grad[k] = vec_nll1_grad[k] + sum(diag(dOmedThetak %*% mat_cond_sigma))/num_sigmasq_eps
    vec_nll1_grad[k] = vec_nll1_grad[k] + t(vec_cond_mean) %*% dOmedThetak %*% vec_cond_mean/num_sigmasq_eps
    vec_nll1_grad[k] = vec_nll1_grad[k] * (0.5)
  }
  ##grad for nugg
  vec_nll1_grad[int_p+1] = sum(diag(mat_omega_eps_inv))
  dOmedThetak = -mat_omega_eps_inv %*% mat_omega_eps_inv
  vec_nll1_grad[int_p+1] = vec_nll1_grad[int_p+1] + 
      t(datalist$y - num_mu) %*% dOmedThetak %*% (datalist$y - num_mu)/num_sigmasq_eps
  vec_nll1_grad[int_p+1] = vec_nll1_grad[int_p+1] -
      2 * t(datalist$y - num_mu) %*% dOmedThetak %*% vec_cond_mean/num_sigmasq_eps
  vec_nll1_grad[int_p+1] = vec_nll1_grad[int_p+1] + sum(diag(dOmedThetak %*% mat_cond_sigma))/num_sigmasq_eps
  vec_nll1_grad[int_p+1] = vec_nll1_grad[int_p+1] + t(vec_cond_mean) %*% dOmedThetak %*% vec_cond_mean/num_sigmasq_eps
  vec_nll1_grad[int_p+1] = vec_nll1_grad[int_p+1] * (0.5)

  vec_nll1_grad
}

T_grad = function(vec_para, m)
{
  require(numDeriv)
  f = function(vv, ncates)
  {
    mat_T = c_inverse_hsd(vv, ncates)
    mat_T = mat_T %*% t(mat_T)
    vec_T = as.numeric(mat_T)
    vec_T
  }
  vec_dT = jacobian(func = f, x = vec_para, ncates = m)
  ll_dt = list()
  for(i in 1:(m*(m-1)/2))
  {
    ll_dt[[i]] = matrix(vec_dT[,i], ncol = m, byrow = T)
  }
  ll_dt
}


lmgp_nll_alp_v2 = function(
    vec_new_alpha,
    vec_cond_mean,
    mat_cond_mat,
    Phi, A, ncates,
    rmax = 0.2)
{
    # browser()
    n = length(vec_cond_mean)
    mat_T = c_inverse_hsd(vec_new_alpha,ncates)
    mat_T = mat_T %*% t(mat_T)
    mat_omega_alp = t(A) %*% kronecker(mat_T, Phi) %*% A
    mat_omega_alp_inv = solve(mat_omega_alp)
    num_sigmasq_alp = EXACTEM_nll2_sigsq(
    mat_omega_alp_inv,
    vec_cond_mean,
    mat_cond_mat
    )
    num_nll = - determinant(mat_omega_alp)$modulus * 0.5
    num_nll = num_nll - log(num_sigmasq_alp) * n/2.0
    return(-num_nll)
}
###gradient
lmgp_nll_alp_v2_grad = function(
    vec_new_alpha,
    vec_cond_mean,
    mat_cond_mat,
    Phi, A, ncates,
    rmax = 0.2)
{
    n = length(vec_cond_mean)
    mat_T = c_inverse_hsd(vec_new_alpha,ncates)
    mat_T = mat_T %*% t(mat_T)
    omega = t(A) %*% kronecker(mat_T, Phi) %*% A
    omega_inv = solve(omega)
    num_sigmasq_alp = EXACTEM_nll2_sigsq(
        omega_inv,
        vec_cond_mean,
        mat_cond_mat
    )
    vec_df = rep(NA, length(vec_new_alpha))
    ll_dt = T_grad(vec_new_alpha, ncates)
    for(i in 1:length(vec_new_alpha))
    {
        DomegaDrhoi = t(A) %*% kronecker(ll_dt[[i]], Phi) %*% A
        DomegainvDrhoi = -omega_inv %*% DomegaDrhoi %*% omega_inv
        DsigsqDrhor = (1/n) * t(vec_cond_mean) %*% DomegainvDrhoi %*% vec_cond_mean
        DsigsqDrhor = DsigsqDrhor + (1/n) * sum(diag(DomegainvDrhoi %*% mat_cond_mat))
        vec_df[i] = -0.5 * sum(diag(omega_inv %*% DomegaDrhoi)) - (n/2) * DsigsqDrhor / num_sigmasq_alp
    }
    -vec_df
}
