#define ARMA_USE_BLAS
#define ARMA_USE_LAPACK

#include <RcppArmadillo.h>
#include <iostream>
#include <Rcpp.h>
#include <chrono>

using namespace Rcpp;
using namespace std; 
using namespace arma;
//calculate diff(xAx)*-0.5
//[[Rcpp::depends(RcppArmadillo)]]
//[[Rcpp::export]]
mat convM(const mat& X, vec theta, double nugg)
{
	int n=X.n_rows;
	int p=X.n_cols;
	mat K(n,n, fill::zeros);
	K.fill(0.0);
	int i,j,k;
	//add openmp for each column to fill the row
	for(i=0;i<n;i++)
	{
		//cov matrix are symmetric
		for(j=0;j<i;j++)
		{
			for(k=0;k<p;k++)
			{
				// K(j,i) = K(j,i) + pow(X(i,k)-X(j,k),2.0) * theta(k);
				K(j,i) = K(j,i) + (X(i,k)-X(j,k))*(X(i,k)-X(j,k)) * theta(k);
			}
			//apply activation function
			K(j,i)=exp(-K(j,i));
			//symmatrix m
			K(i,j)=K(j,i);
		}
		//set diagonal value
		K(i,i)=1.0+nugg;//datum::eps;
	}
	return K;
}

//[[Rcpp::export]]
mat convM_X1X2(const mat& X1, const mat& X2, vec theta)
{
	int n1=X1.n_rows;
	int n2=X2.n_rows;
	int p=X1.n_cols;
	mat K(n1,n2);
	K.fill(0.0);
	int i,j,k;
	for(i=0;i<n1;i++)
	{
		//cov matrix are symmetric
		for(j=0;j<n2;j++)
		{
			for(k=0;k<p;k++)
			{
				K(i,j)+=pow(X1(i,k)-X2(j,k),2.0) * theta(k);
			}
			//apply activation function
			K(i,j)=exp(0.0-K(i,j));
		}
	}
	return K;
}

//[[Rcpp::export]]
mat block_corrM(const mat& X, vec theta, double nugg, Col<int> num_vec)
{
	int n=X.n_rows;
	mat blocked_mat(n,n,fill::zeros);
	int start_idx, end_idx;
    start_idx=0;
	for(int i=0;i<(int) num_vec.n_elem;i++)
	{
		end_idx=start_idx+num_vec(i)-1;
		if(end_idx < start_idx)
		{
			start_idx=start_idx+num_vec(i);
		} else
		{
			blocked_mat.submat( start_idx, start_idx, end_idx, end_idx)=convM(X.rows(start_idx, end_idx),theta,nugg);
			start_idx=start_idx+num_vec(i);
		}
	}
	return blocked_mat;
}


//block correlation matrix for x1 x2
//[[Rcpp::export]]
mat block_corrM_X1X2(const mat& X1, const mat& X2, vec theta, Col<int> numvec1, Col<int> numvec2)
{
	int n1=X1.n_rows;
	int n2=X2.n_rows;
	mat cor_mat_x1x2(n1,n2, fill::zeros);
	int start_idx_x1,end_idx_x1,start_idx_x2,end_idx_x2;
	int num_cates=(int) numvec1.n_elem;
	start_idx_x1=0;
	start_idx_x2=0;
	// cout<<theta<<endl;
	for(int i=0;i<num_cates;i++)
	{
		end_idx_x1=start_idx_x1+numvec1(i)-1;
		end_idx_x2=start_idx_x2+numvec2(i)-1;
		if((end_idx_x1 < start_idx_x1) || (end_idx_x2 < start_idx_x2))
		{
			start_idx_x1=start_idx_x1+numvec1(i);
			start_idx_x2=start_idx_x2+numvec2(i);
		} else
		{
			// cout<<start_idx_x1<<"\t"<<end_idx_x1<<endl;
			// cout<<start_idx_x2<<"\t"<<end_idx_x2<<endl;
			// cout<<X1.rows(start_idx_x1, end_idx_x1).max()<<endl;
			// cout<<X2.min()<<"\t"<<X2.max()<<endl;
			mat mat_subx1x2 = convM_X1X2(X1.rows(start_idx_x1, end_idx_x1),X2.rows(start_idx_x2, end_idx_x2),theta);
			cor_mat_x1x2.submat(start_idx_x1, start_idx_x2, end_idx_x1, end_idx_x2)=mat_subx1x2;
			// cout<<mat_subx1x2.min()<<"\t"<<mat_subx1x2.max()<<endl;
			start_idx_x1=start_idx_x1+numvec1(i);
			start_idx_x2=start_idx_x2+numvec2(i);
		}
	}
	// cout<<cor_mat_x1x2.min()<<"\t"<<cor_mat_x1x2.max()<<endl;
	return cor_mat_x1x2;
}
//[[Rcpp::export]]
mat block_inv(const mat& corM, Col<int> num_vec)
{
	//we require all subblocks of X to be sorted by Z before using this function!!!
    //calculate the block inverse
    //X.submat( first_row, first_col, last_row, last_col )
    int start_idx, end_idx;
	int n,num_cate;
	start_idx=0;
	n=corM.n_rows;
	num_cate=num_vec.n_elem;
    mat corM_inv(n,n,fill::zeros);
    for(int i =0;i<num_cate;i++)
    {
        end_idx=start_idx+num_vec(i)-1;
		if(end_idx < start_idx)
		{
			start_idx=start_idx+num_vec(i);
		} else
		{
			mat block_mat=corM.submat( start_idx, start_idx, end_idx, end_idx);
			block_mat=inv(block_mat);
			corM_inv.submat( start_idx, start_idx, end_idx, end_idx)=block_mat;//block_mat_inv;
			start_idx=start_idx+num_vec(i);
		}        
    }
    return corM_inv;
}

//[[Rcpp::export]]
mat c_inverse_hsd(vec para,int m)
{
	mat matL(m,m, fill::zeros);
	matL(0,0)=1;
	int start_idx=1;
	matL(1,0)=cos(para(0));
	matL(1,1)=sin(para(0));
	if(m>=3)
	{
		for(int r=3;r<m+1;r++)
		{
			//ctheta_slice=para[start_idx:(start_idx+r-2)]
			//V.subvec( first_index, last_index )
			vec ctheta_slice=para.subvec(start_idx,start_idx+r-2);
			matL(r-1,0)=cos(ctheta_slice(0));
			matL(r-1,r-1)=prod(sin(ctheta_slice));
			for(int s=2;s<r;s++)
			{
				matL(r-1,s-1)=prod(sin(ctheta_slice.subvec(0,(s-2))))*cos(ctheta_slice(s-1));
			}
			start_idx=start_idx+r-1;
		}
	}
	return matL;
}

/*
added 20191126
single separate para gp
*/
//[[Rcpp::export]]
double c_sep_gp_nll(vec vec_para, const mat &X, const vec &Y,bool para_coded=false)
{
    int n,d;
    n=size(X)[0];d=size(X)[1];
    if(para_coded)
    {
        vec_para=exp(vec_para);
    }
	if(vec_para.min() <= 0)
	{
		cout<<"invalid vec_para, return error code -1"<<endl;
		return -1;
	}
    mat mat_covar_XX=convM(X,vec_para.subvec(0,d-1),vec_para(d));
    vec n_ones = ones<vec>(n);
	// mat Ki=inv_sympd(mat_covar_XX);
	mat Ki=inv(mat_covar_XX);
	double mu,sigmasq;
	mu=as_scalar(n_ones.t()*Ki*Y)/as_scalar(n_ones.t() *Ki *n_ones);
	sigmasq=as_scalar((Y-mu).t()*Ki*(Y-mu))/(double) n;
	double nl;
	double val;double sign;
	log_det(val, sign, mat_covar_XX);
	nl=(n+0.0)/2.0*log(sigmasq)+0.5*val;
	// cout<<vec_para<<endl;
	// cout<<mu<<sigmasq<<nl<<endl;
    return nl;
}

//[[Rcpp::export]]
vec c_sep_gp_musigma(vec vec_para, const mat &X, const vec &Y,bool para_coded=false)
{
    int n,d;
    n=size(X)[0];d=size(X)[1];
    if(para_coded)
    {
        vec_para=exp(vec_para);
    }
	if(vec_para.min() <= 0)
	{
		cout<<"invalid vec_para, return error code -1"<<endl;
		vec vec_err(1);
		vec_err<<-1;
		return vec_err;
	}
    mat mat_covar_XX=convM(X,vec_para.subvec(0,d-1),vec_para(d));
    vec n_ones = ones<vec>(n);
	// mat Ki=inv_sympd(mat_covar_XX);
	mat Ki=inv(mat_covar_XX);
	double mu,sigmasq;
	mu=as_scalar(n_ones.t()*Ki*Y)/as_scalar(n_ones.t() *Ki *n_ones);
	sigmasq=as_scalar((Y-mu).t()*Ki*(Y-mu))/(double) n;
	vec vec_musigma(2);
	vec_musigma<<mu<<sigmasq;
    return vec_musigma;
}

//[[Rcpp::export]]
vec c_sep_gp_pre(vec vec_para, mat X1, vec Y1, mat X2, bool para_coded=false)
{
    int n1,n2,d;
    n1=size(X1)[0];n2=size(X2)[0];d=size(X1)[1];
    if(para_coded)
    {
        vec_para=exp(vec_para);
    }
	//double g=para(d);
	mat cate_convM_x1x2=convM_X1X2(X1,X2,vec_para.subvec(0,d-1));
	mat cate_convM_x1x1=convM(X1,vec_para.subvec(0,d-1),vec_para(d));
    vec  n_ones = ones<vec>(n1);
	// mat Ki=inv_sympd(cate_convM_x1x1);
	mat Ki=inv(cate_convM_x1x1);
	double mu;//,sigmasq;
	mu=as_scalar(n_ones.t()*Ki*Y1)/as_scalar(n_ones.t() *Ki *n_ones);
	// cout<<mu<<endl;
	vec y_pre(n2);
	y_pre=mu+cate_convM_x1x2.t()*Ki*(Y1-mu);
	// cout<<Y1-mu<<endl;
	return y_pre;
}

/*
added 20191124
baselines
categp
*/

// [[Rcpp::depends(RcppArmadillo)]]
// [[Rcpp::export]]
mat c_cate_convM(vec para, mat X, Col<int> Z, int num_levels)
{
	int n,d;
	n=size(X)[0];
	d=size(X)[1];
	vec theta=para.subvec(0,d-1);
	double g=para(d);
	mat x_conv=convM(X,theta,g);
	mat tauM(n,n, fill::ones);
	int start_idx=d+1;
	for (int i = 0; i < n; i++)
	{
		for(int j=0;j<=i;j++)
		{
			//different from cate_conv!
			//construct ctheta matrix
			vec ctheta_para=para.subvec(start_idx,start_idx+(num_levels-1)*num_levels/2-1);
			mat cate_mat_L=c_inverse_hsd(ctheta_para, num_levels);
			mat cate_mat_T=cate_mat_L * cate_mat_L.t();

			tauM(i,j)=cate_mat_T(Z(i),Z(j));
			tauM(j,i)=tauM(i,j);
		}
	}
	mat cate_x_covar=tauM%x_conv;
	return cate_x_covar;
}
// [[Rcpp::depends(RcppArmadillo)]]
// [[Rcpp::export]]
mat c_cate_convM_X1X2(vec para, mat X1, Col<int> Z1, int num_levels, mat X2, Col<int> Z2)
{
	int n1,n2,d;
	n1=size(X1)[0];
	n2=size(X2)[0];
	d=size(X1)[1];
	vec theta=para.subvec(0,d-1);
	mat x_conv=convM_X1X2(X1, X2, theta);
	mat tauM(n1,n2, fill::ones);
	int start_idx=d+1;
	for (int i = 0; i < n1; i++)
	{
		for(int j=0;j<n2;j++)
		{
			//different from cate_conv!
			//construct ctheta matrix
			vec ctheta_para=para.subvec(start_idx,start_idx+(num_levels-1)*num_levels/2-1);
			mat cate_mat_L=c_inverse_hsd(ctheta_para, num_levels);
			mat cate_mat_T=cate_mat_L * cate_mat_L.t();
			tauM(i,j)*=cate_mat_T(Z1(i),Z2(j));
		}
	}
	mat cate_x_covar=tauM%x_conv;
	return cate_x_covar;
}
//coded likelihood in cpp
// [[Rcpp::depends(RcppArmadillo)]]
// [[Rcpp::export]]
double c_cate_gp_nl(vec para, mat X, vec Y, Col<int> Z, int num_levels, double theta_max = 100.0, bool para_coded=false)
{
	// cout<<para<<endl;
	int n,d;
	n=size(X)[0];d=size(X)[1];
	if(para_coded)
	{
		para.subvec(0,d-1)=theta_max/(1+exp(-para.subvec(0,d-1)));
		para(d)=0.5/(1+exp(-para(d)))+1e-8;
		para.subvec(d+1,d+num_levels*(num_levels-1)/2)=\
		datum::pi/(1+exp(-1*para.subvec(d+1,d+num_levels*(num_levels-1)/2)));
	}
	mat cate_x_covar=c_cate_convM(para,X,Z,num_levels);
	//calculate likelihood
	vec  n_ones = ones<vec>(n);
	// mat Ki=inv_sympd(cate_x_covar);
	mat Ki=inv(cate_x_covar);
	double mu,sigmasq;
	mu=as_scalar(n_ones.t()*Ki*Y)/as_scalar(n_ones.t() *Ki *n_ones);
	//estimate sigmasq:	
	sigmasq=as_scalar((Y-mu).t()*Ki*(Y-mu))/(double) n;
	double nl;
	double val;double sign;
	log_det(val, sign, cate_x_covar); 	
	nl=(n+0.0)/2.0*log(sigmasq)+0.5*val;
	return nl;
}

// [[Rcpp::depends(RcppArmadillo)]]
// [[Rcpp::export]]
vec c_cate_gp_pred(vec para, 
	mat X1, vec Y, Col<int> Z1, int num_levels, 
	mat X2, Col<int> Z2, double theta_max = 100.0, bool para_coded=false)
{
	int n1=size(X1)[0];
	int n2=size(X2)[0];
	int d=size(X1)[1];
	if(para_coded)
	{
		para.subvec(0,d-1)=theta_max/(1+exp(-para.subvec(0,d-1)));
		para(d)=theta_max/(1+exp(-para(d)))+1e-8;
		para.subvec(d+1,d+num_levels*(num_levels-1)/2)=\
		datum::pi/(1+exp(-1*para.subvec(d+1,d+num_levels*(num_levels-1)/2)));
	}
	//double g=para(d);
	mat cate_convM_x1x2=c_cate_convM_X1X2(para,X1,Z1,num_levels,X2,Z2);
	mat cate_convM_x1x1=c_cate_convM(para,X1,Z1,num_levels);
	///calculate mu and sigma
	vec  n_ones = ones<vec>(n1);
	// mat Ki=inv_sympd(cate_convM_x1x1);
	mat Ki=inv(cate_convM_x1x1);
	double mu;//,sigmasq;
	mu=as_scalar(n_ones.t()*Ki*Y)/as_scalar(n_ones.t() *Ki *n_ones);
	vec y_pre(n2);
	y_pre=mu+cate_convM_x1x2.t()*Ki*(Y-mu);
	return y_pre;
}
// added 20200905
// return estimated mu and sigma
// [[Rcpp::export]]
vec c_cate_gp_musigma(vec para, mat X, vec Y, Col<int> Z, int num_levels, double theta_max = 100.0, bool para_coded=true)
{
	//cout<<para<<endl;
	int n,d;
	n=size(X)[0];d=size(X)[1];
	if(para_coded)
	{
		para.subvec(0,d-1)=theta_max/(1+exp(-para.subvec(0,d-1)));
		para(d)=theta_max/(1+exp(-para(d)))+1e-8;
		para.subvec(d+1,d+num_levels*(num_levels-1)/2)=\
		datum::pi/(1+exp(-1*para.subvec(d+1,d+num_levels*(num_levels-1)/2)));
	}
	cout<<para.t()<<endl;
	mat cate_x_covar=c_cate_convM(para,X,Z,num_levels);
	//calculate likelihood
	vec  n_ones = ones<vec>(n);
	mat Ki=inv_sympd(cate_x_covar);
	double mu,sigmasq;
	mu=as_scalar(n_ones.t()*Ki*Y)/as_scalar(n_ones.t() *Ki *n_ones);
	//estimate sigmasq:	
	sigmasq=as_scalar((Y-mu).t()*Ki*(Y-mu))/(double) n;
	vec vec_musigma(2);
	vec_musigma<<mu<<sigmasq;
	return vec_musigma;
}
/*
another version of cate gp
*/

// [[Rcpp::export]]
mat c_cate_convM_v2(vec para, mat X, Mat<int> Z, Col<int> vec_num_levels)
{
	int n,d,num_cates;
	n=size(X)[0];
	d=size(X)[1];
	num_cates=size(Z)[1];
	vec theta=para.subvec(0,d-1);
	double g=para(d);
	mat x_conv=convM(X,theta,g);
	mat tauM(n,n, fill::ones);
	int start_idx=d+1;
	for(int k=0;k<num_cates;k++)
	{
		for(int i = 0;i<n; i++)
		{
			for(int j=0;j<=i;j++)
			{
				vec ctheta_para=para.subvec(start_idx,start_idx+(vec_num_levels(k)-1)*vec_num_levels(k)/2-1);
				mat cate_mat_L=c_inverse_hsd(ctheta_para, vec_num_levels(k));
				mat cate_mat_T=cate_mat_L * cate_mat_L.t();
				//cout<<i<<"\t"<<j<<"\t"<<k<<endl;
				tauM(i,j)*=cate_mat_T(Z(i,k),Z(j,k));
				//cout<<"completed"<<endl;
				tauM(j,i)=tauM(i,j);
			}
		}
		if(k<num_cates-1)
		{
			start_idx+=(vec_num_levels(k)-1)*vec_num_levels(k)/2;
		}
	}
	mat cate_x_covar=tauM%x_conv;
	return cate_x_covar;
}
// [[Rcpp::depends(RcppArmadillo)]]
// [[Rcpp::export]]
mat c_cate_convM_X1X2_v2(vec para, mat X1, Mat<int> Z1, Col<int> vec_num_levels, mat X2, Mat<int> Z2)
{
	int n1,n2,d,num_cates;
	n1=size(X1)[0];
	n2=size(X2)[0];
	d=size(X1)[1];
	num_cates=size(Z1)[1];
	vec theta=para.subvec(0,d-1);
	mat x_conv=convM_X1X2(X1, X2, theta);
	mat tauM(n1,n2, fill::ones);
	int start_idx=d+1;
	for(int k=0;k<num_cates;k++)
	{
		for(int i = 0; i < n1; i++)
		{
			for(int j=0;j<n2;j++)
			{
				vec ctheta_para=para.subvec(start_idx,start_idx+(vec_num_levels(k)-1)*vec_num_levels(k)/2-1);
				mat cate_mat_L=c_inverse_hsd(ctheta_para, vec_num_levels(k));
				mat cate_mat_T=cate_mat_L * cate_mat_L.t();
				tauM(i,j)*=cate_mat_T(Z1(i,k),Z2(j,k));
			}
		}
		if(k<num_cates-1)
		{
			start_idx+=(vec_num_levels(k)-1)*vec_num_levels(k)/2;
		}
	}
	mat cate_x_covar=tauM%x_conv;
	return cate_x_covar;
}
//coded likelihood in cpp
// [[Rcpp::depends(RcppArmadillo)]]
// [[Rcpp::export]]
double c_cate_gp_nl_v2(vec para, mat X, vec Y, Mat<int> Z, Col<int> vec_num_levels, bool para_coded=false)
{
	//cout<<para<<endl;
	int n,d;
	n=size(X)[0];d=size(X)[1];
	if(para_coded)
	{
		para.subvec(0,d-1)=0.3/(1+exp(-para.subvec(0,d-1)));
		para(d)=0.5/(1+exp(-para(d)))+1e-8;
		int num_par=para.n_elem;
		para.subvec(d+1,num_par)=datum::pi/(1+exp(-1*para.subvec(d+1,num_par)));
	}
	mat cate_x_covar=c_cate_convM_v2(para,X,Z,vec_num_levels);
	//calculate likelihood
	vec n_ones = ones<vec>(n);
	mat Ki=inv_sympd(cate_x_covar);
	double mu,sigmasq;
	mu=as_scalar(n_ones.t()*Ki*Y)/as_scalar(n_ones.t() *Ki *n_ones);
	//estimate sigmasq:	
	sigmasq=as_scalar((Y-mu).t()*Ki*(Y-mu))/(double) n;
	double nl;
	double val;double sign;
	log_det(val, sign, cate_x_covar);
	nl=(n+0.0)/2.0*log(sigmasq)+0.5*val;
	return nl;
}

// [[Rcpp::depends(RcppArmadillo)]]
// [[Rcpp::export]]
vec c_cate_gp_pred_v2(vec para, mat X1, vec Y, Mat<int> Z1, Col<int> vec_num_levels, 
	mat X2, Mat<int> Z2, bool para_coded=false)
{
	int n1=size(X1)[0];
	int n2=size(X2)[0];
	int d=size(X1)[1];
	if(para_coded)
	{
		para.subvec(0,d-1)=0.3/(1+exp(-para.subvec(0,d-1)));
		para(d)=0.5/(1+exp(-para(d)))+1e-8;
		int num_par=para.n_elem;
		para.subvec(d+1,num_par)=datum::pi/(1+exp(-1*para.subvec(d+1,num_par)));
	}
	//double g=para(d);
	mat cate_convM_x1x2=c_cate_convM_X1X2_v2(para,X1,Z1,vec_num_levels,X2,Z2);
	mat cate_convM_x1x1=c_cate_convM_v2(para,X1,Z1,vec_num_levels);
	///calculate mu and sigma
	vec  n_ones = ones<vec>(n1);
	mat Ki=inv_sympd(cate_convM_x1x1);
	double mu;//,sigmasq;
	mu=as_scalar(n_ones.t()*Ki*Y)/as_scalar(n_ones.t() *Ki *n_ones);
	vec y_pre(n2);
	y_pre=mu+cate_convM_x1x2.t()*Ki*(Y-mu);
	return y_pre;
}

/*
functional GP begins here
*/

//[[Rcpp::export]]
mat euclid_dist_mat(const mat& X)
{
	int n=X.n_rows;int p=X.n_cols;
	mat K(n,n,fill::zeros);
	for(int i=0;i<n;i++)
	{
		for(int j=0;j<i;j++)
		{
			for(int k=0;k<p;k++)
			{
				K(i,j)+=pow(X(i,k)-X(j,k),2);
			}
			K(j,i)=K(i,j);
		}
		K(i,i)=0.0;
	}
	return K;
}
//[[Rcpp::export]]
mat block_euclid_dist(const mat& X, Col<int> num_vec)
{
	int n=X.n_rows;
	mat blocked_mat(n,n,fill::zeros);
	int start_idx, end_idx;
    start_idx=0;
	for(int i=0;i<(int) num_vec.n_elem;i++)
	{
		end_idx=start_idx+num_vec(i)-1;
		if(end_idx < start_idx)
		{
			start_idx=start_idx+num_vec(i);
		} else
		{
			blocked_mat.submat( start_idx, start_idx, end_idx, end_idx)=euclid_dist_mat(X.rows(start_idx, end_idx));
			start_idx=start_idx+num_vec(i);
		}
	}
	return blocked_mat;
}
//[[Rcpp::export]]
mat euclid_dist_mat_v2(const mat& X)
{
	int n=X.n_rows;int p=X.n_cols;
	vec pones=ones<vec>(p);
	mat K(n,n,fill::zeros);
	K=convM(X,pones,0.0);
	return -1*log(K);
}

//[[Rcpp::export]]
mat euclid_dist_mat_X1X2(const mat& X1, const mat& X2)
{
	int n1=X1.n_rows;
	int n2=X2.n_rows;
	int p=X1.n_cols;
	mat K(n1,n2,fill::zeros);
	for(int i=0;i<n1;i++)
	{
		for(int j=0;j<n2;j++)
		{
			for(int k=0;k<p;k++)
			{
				K(i,j)+=pow(X1(i,k)-X2(j,k),2);
			}
		}
	}
	return K;
}

//[[Rcpp::export]]
mat euclid_dist_mat_X1X2_v2(const mat& X1, const mat& X2)
{
	int n1=X1.n_rows;
	int n2=X2.n_rows;
	int p=X1.n_cols;
	vec pones=ones<vec>(p);
	mat K(n1,n2,fill::zeros);
	K=convM_X1X2(X1,X2,pones);
	return -1*log(K);
}

//discontinued cause no PSD guarantee
//[[Rcpp::export]]
mat CSK_B(const mat& X, double rmax)
{
    const double d_pi=3.14159265358979323846;
    int n=X.n_rows;
    mat cos_mat(n,n), sin_mat(n,n);
    mat distance_matrix=euclid_dist_mat(X);
	//distance_matrix.diag() +=std::numeric_limits<double>::epsilon();
	distance_matrix = sqrt(distance_matrix);
    cos_mat=(1.0-distance_matrix/rmax)%cos(d_pi*distance_matrix/rmax);
    sin_mat=1.0/d_pi*sin(d_pi*distance_matrix/rmax);
    mat return_correlation_mat=cos_mat+sin_mat;
    return_correlation_mat.elem(find(distance_matrix > rmax) ).zeros();
	//add a constant nugget to avoid singularity!
	// return_correlation_mat.diag()+=1e-7;
    return return_correlation_mat;
}

//compact support kernel correlation matrix for X1 X2
//[[Rcpp::export]]
mat CSK_B_X1X2(const mat& X1, const mat& X2, double rmax)
{
    const double d_pi=3.14159265358979323846;
    int n1=X1.n_rows;
	int n2=X2.n_rows;
    mat cos_mat(n1,n2), sin_mat(n1,n2);
    mat distance_matrix=euclid_dist_mat_X1X2(X1,X2);
	//distance_matrix.diag() +=std::numeric_limits<double>::epsilon();
	distance_matrix = sqrt(distance_matrix);
    cos_mat=(1.0-distance_matrix/rmax)%cos(d_pi*distance_matrix/rmax);
    sin_mat=1.0/d_pi*sin(d_pi*distance_matrix/rmax);
    mat return_correlation_mat=cos_mat+sin_mat;
    return_correlation_mat.elem(find(distance_matrix > rmax) ).zeros();
    return return_correlation_mat;
}

//get correlation matrix alpha for X1 X2
//[[Rcpp::export]]
mat correlation_alpha_X1X2_CSK_B(const mat& X1, const mat& X2, const Col<int> Z1, const Col<int> Z2,
vec cate_para, double rmax, int n_cates)
{
	mat cate_alpha_matrix_X1X2=CSK_B_X1X2(X1,X2,rmax);
	mat cate_hsd_mat=c_inverse_hsd(cate_para,n_cates);
	cate_hsd_mat=cate_hsd_mat * cate_hsd_mat.t();
	int n1=X1.n_rows;
	int n2=X2.n_rows;
	for(int i=0;i<n1;i++)
	{
		for(int j=0;j<n2;j++)//working on a symmetric matrix
		{
			cate_alpha_matrix_X1X2(i,j)=cate_alpha_matrix_X1X2(i,j)*cate_hsd_mat(Z1(i),Z2(j));
		}
	}
	return cate_alpha_matrix_X1X2;
}

//[[Rcpp::export]]
mat sigma_alpha_inv_gen_CSK_B(const mat& X, const Col<int> &Z,vec cate_para,double rmax, int m, 
    bool is_inverse=true)
{
	mat cate_alpha_matrix=CSK_B(X,rmax);
	mat cate_hsd_mat=c_inverse_hsd(cate_para,m);
	cate_hsd_mat=cate_hsd_mat * cate_hsd_mat.t();
	for(int i=0;i<(int) X.n_rows;i++)
	{
		for(int j=0;j<=i;j++)//working on a symmetric matrix
		{
			cate_alpha_matrix(i,j)=cate_alpha_matrix(i,j)*cate_hsd_mat(Z(i),Z(j));
			cate_alpha_matrix(j,i)=cate_alpha_matrix(i,j);
		}
		//cate_alpha_matrix(i,i)=1.0;//diagonal value==1
	}
	// cate_alpha_matrix.diag() += 2e-7;
	if(is_inverse)
	{
		return inv(cate_alpha_matrix);
	}
	else
	{
		return cate_alpha_matrix;
	}
}

/*
added 20201008
use Wendland’s Construction for CSK
add fungp version using separate theta for each cate
*/
//[[Rcpp::export]]
mat CSK_W(const mat& X, double rmax)
{
    // int n=X.n_rows;
	int p=X.n_cols;
	double v = ((double) p+2.0)/2.0;
    mat distance_matrix=euclid_dist_mat(X);
	//distance_matrix.diag() +=std::numeric_limits<double>::epsilon();
	distance_matrix = sqrt(distance_matrix);
	distance_matrix = distance_matrix/rmax;
	mat return_correlation_mat=distance_matrix;
	return_correlation_mat = pow(1.0-return_correlation_mat, v);
    return_correlation_mat.elem(find(distance_matrix >= 1.0) ).zeros();
	//add a constant nugget to avoid singularity!
	return_correlation_mat.diag()+=1e-7;
    return return_correlation_mat;
}

//compact support kernel correlation matrix for X1 X2
//[[Rcpp::export]]
mat CSK_W_X1X2(const mat& X1, const mat& X2, double rmax)
{
    // int n1=X1.n_rows;
	// int n2=X2.n_rows;
	int p=X1.n_cols;
	double v = ((double) p+1.0)/2.0;
    mat distance_matrix=euclid_dist_mat_X1X2(X1,X2);
	//distance_matrix.diag() +=std::numeric_limits<double>::epsilon();
	distance_matrix = sqrt(distance_matrix);
	distance_matrix = distance_matrix/rmax;
	mat return_correlation_mat=distance_matrix;
	return_correlation_mat = pow(1.0-return_correlation_mat, v);
    return_correlation_mat.elem(find(distance_matrix > 1.0) ).zeros();
    return return_correlation_mat;
}

//get correlation matrix alpha for X1 X2
//[[Rcpp::export]]
mat correlation_alpha_X1X2_CSK_W(const mat& X1, const mat& X2, const Col<int>& Z1, const Col<int>& Z2,
vec cate_para, double rmax, int n_cates)
{
	mat cate_alpha_matrix_X1X2=CSK_W_X1X2(X1,X2,rmax);
	mat cate_hsd_mat=c_inverse_hsd(cate_para,n_cates);
	cate_hsd_mat=cate_hsd_mat * cate_hsd_mat.t();
	int n1=X1.n_rows;
	int n2=X2.n_rows;
	for(int i=0;i<n1;i++)
	{
		for(int j=0;j<n2;j++)//working on a symmetric matrix
		{
			cate_alpha_matrix_X1X2(i,j)=cate_alpha_matrix_X1X2(i,j)*cate_hsd_mat(Z1(i),Z2(j));
		}
	}
	return cate_alpha_matrix_X1X2;
}

//[[Rcpp::export]]
mat sigma_alpha_inv_gen_CSK_W(const mat& X, const Col<int> &Z,vec cate_para,double rmax, int m, 
    bool is_inverse=true)
{
	mat cate_alpha_matrix=CSK_W(X,rmax);
	mat cate_hsd_mat=c_inverse_hsd(cate_para,m);
	cate_hsd_mat=cate_hsd_mat * cate_hsd_mat.t();
	for(int i=0;i<(int) X.n_rows;i++)
	{
		for(int j=0;j<i;j++)//working on a symmetric matrix
		{
			cate_alpha_matrix(j,i)=cate_alpha_matrix(j,i)*cate_hsd_mat(Z(i),Z(j));
			cate_alpha_matrix(i,j)=cate_alpha_matrix(j,i);
		}
		cate_alpha_matrix(i,i)=1.0;//diagonal value==1
	}
	cate_alpha_matrix.diag() += 1e-5;
	if(is_inverse)
	{
		return inv(cate_alpha_matrix);
	}
	else
	{
		return cate_alpha_matrix;
	}
}


//[[Rcpp::export]]
mat sigma_eps_inv(const mat& X, vec theta, double g, Col<int> num_vec)
{
	mat eps_mat=block_corrM(X,theta,g,num_vec);
	mat eps_mat_inv=block_inv(eps_mat,num_vec);
	return eps_mat_inv;
}

//[[Rcpp::export]]
double quadratic_multiple(const mat& A, const vec &b)
{
	double val=as_scalar(b.t()*A*b);
	return val;
}

//[[Rcpp::export]]
double quadratic_multiple_ab(const mat& A, const vec &a, const vec &b)
{
	double val=as_scalar(a.t()*A*b);
	return val;
}

//[[Rcpp::export]]
double nlog_likelihood_Q(double mu, const mat& Sigmamat_eps_inv, const mat& Sigmamat_alpha_inv, const vec &w, const vec& alpha)
{
	double nll;
	nll=quadratic_multiple(Sigmamat_eps_inv,w-mu-alpha)+quadratic_multiple(Sigmamat_alpha_inv,alpha);
	return nll;//actual get 2*nlog
}
//[[Rcpp::export]]
double mat_log_det(const mat& A)
{
	double Kdet;
	double sign;
	log_det(Kdet, sign, A); 
	return Kdet;
}
//[[Rcpp::export]]
double e_step_Q(double mu, const mat& Sigmamat_eps_inv, const mat& Sigmamat_alpha_inv, 
const vec &w, const mat &alpha_MH, int burn_out)
{
	//alpha_MH is sampled n B alpha from MH algorithm
	//return is MCMC intergal of nlog)likelihood_Q
	int B=alpha_MH.n_rows;
	vec val(B-burn_out);
	//we only use burnout MH samples
	//calculate matrix det
	double log_det_eps,log_det_alpha;
	//add negative to inverse matrix determint
	log_det_eps=-1.0*mat_log_det(Sigmamat_eps_inv);
	log_det_alpha=-1.0*mat_log_det(Sigmamat_alpha_inv); 
	int ll=0;
	for(int i=burn_out;i<B;i++)
	{
		val(ll)=nlog_likelihood_Q(mu,Sigmamat_eps_inv,Sigmamat_alpha_inv,w,alpha_MH.row(i).t())+log_det_eps+log_det_alpha;
		ll++;
	}
	return sum(val)/(double)(B-burn_out);
}

//20191023
//calculate E(alpha t(alpha))
//[[Rcpp::export]]
mat E_x_xt(const mat mat_X)
{
	int n=mat_X.n_rows;
	int p=mat_X.n_cols;
	mat mat_return(p,p,fill::zeros);
	for(int i=0;i<n;i++)
	{
		mat_return+=mat_X.row(i).t()*mat_X.row(i);
	}
	return mat_return/(double) n;
}

double tr_Sigma_E_x_xt(const mat& mat_sigma_inv, const mat& mat_alpha, bool E_x_xt_Computed=false)
{
	double val;
	if(E_x_xt_Computed)
	{
		val=trace(mat_sigma_inv*mat_alpha);
	} else
	{
		val=trace(mat_sigma_inv*E_x_xt(mat_alpha));
	}
	return val;
}

//another version for w mu alpha
double tr_Sigma_E_x_xt(const mat& mat_omega_inv, const mat& mat_alpha, double mu, const vec& vec_w)
{
	mat mat_alpha_copy=mat(mat_alpha);
	for(int i=0; i<(int) mat_alpha.n_rows;i++)
	{
		mat_alpha_copy.row(i)=vec_w.t()-mu-mat_alpha_copy.row(i);
	}
	double val=trace(mat_omega_inv*E_x_xt(mat_alpha_copy));
	return val;
}
//version 2 only uses warmed samples!!
//this is testing version
//[[Rcpp::export]]
double e_step_Q_v2_testing(double mu, const mat& mat_sigma_eps_inv, const mat& mat_sigma_alp_inv, 
const vec &vec_w, const mat &mat_alpha)
{
	//alpha_MH is sampled n B alpha from MH algorithm
	//return is MCMC intergal of nlog)likelihood_Q
	//int B=mat_alpha.n_rows;
	//we only use burnout MH samples
	//calculate matrix det
	double log_det_eps,log_det_alpha;
	//add negative to inverse matrix determint
	log_det_eps=-1.0*mat_log_det(mat_sigma_eps_inv);
	log_det_alpha=-1.0*mat_log_det(mat_sigma_alp_inv); 
	double val=log_det_eps+log_det_alpha;
	val+=tr_Sigma_E_x_xt(mat_sigma_alp_inv,mat_alpha);
	val+=tr_Sigma_E_x_xt(mat_sigma_eps_inv,mat_alpha,mu,vec_w);
	return val;
}

//return a matrix, each row is w-alp
//w-alpha=eta
//[[Rcpp::export]]
mat w_minus_alp(const vec& vec_w, const mat& mat_alp_sample)
{
	int n=mat_alp_sample.n_rows;
	int p=mat_alp_sample.n_cols;
	mat mat_return(n,p,fill::zeros);
	for(int i=0;i<n;i++)
	{
		mat_return.row(i)=vec_w.t()-mat_alp_sample.row(i);
	}
	return mat_return;
}

//return a matrix, E_eta_ones
//E (w-alpha)*t(ones)
//[[Rcpp::export]]
mat E_eta_tones(const mat& mat_eta_sample)
{
	int n=mat_eta_sample.n_rows;
	int p=mat_eta_sample.n_cols;
	vec vec_ones(p,fill::ones);
	mat mat_return(p,p,fill::zeros);
	for(int i=0;i<n;i++)
	{
		mat_return+=vec_ones*mat_eta_sample.row(i);
	}
	return mat_return/n;
}

//E(w-mu-alpha)*t(w-u-alpha)
//[[Rcpp::export]]
mat E_cw_cwt(double mu, const mat& mat_E_eta_teta, const mat& mat_E_eta_tones)
{
	int p=mat_E_eta_teta.n_cols;
	mat mat_ones(p,p,fill::ones);
	mat mat_return(p,p,fill::zeros);
	mat_return=mat_E_eta_teta-mu*(mat_E_eta_tones+mat_E_eta_tones.t())+mu*mu*mat_ones;
	return mat_return;
}

//production version that used calculated results
//invver function
//[[Rcpp::export]]
double e_step_Q_v2_production(double mu, 
double det_mat_sigma_eps_inv, double det_mat_sigma_alp_inv, 
const mat& mat_sigma_eps_inv, const mat& mat_sigma_alp_inv,
const mat& mat_E_alp_talp, const mat& mat_E_eta_teta, const mat& mat_E_eta_tones)
{
	double val=-1.0*det_mat_sigma_eps_inv+-1.0*det_mat_sigma_alp_inv;
	val+=trace(mat_sigma_alp_inv*mat_E_alp_talp);
	val+=trace(mat_sigma_eps_inv*E_cw_cwt(mu,mat_E_eta_teta,mat_E_eta_tones));
	return val;
}
//20191001
//now begins functions for cate GP predictions
//first we simple using matrix inverse to do prediction

//X1 is training data (n1), X2 is prediction data(n2), cormat_X1X2 is n1*n2 matrix
//return the condiction mean Y2|Y1
//[[Rcpp::export]]
vec GP_prediction_condmean(const vec& Y1, const vec& u1, const vec& u2, const mat& cormat_X1X1, const mat& cormat_X1X2)
{
	vec X2_condmean(u2.n_elem);
	X2_condmean=u2+cormat_X1X2.t()*solve(cormat_X1X1,Y1-u1);
	return X2_condmean;
}


/*
20191024
The following are code for separate likelihood procedure
*/

//inner function used in M step
// //[[Rcpp::export]]
// double E_likeli2_sigmaalp_hat_estimate(const mat& mat_sigma_alp_inv,const mat& mat_E_alp_talp)
// {
// 	int n=mat_sigma_alp_inv.n_rows;
// 	double val=trace(mat_sigma_alp_inv*mat_E_alp_talp);
// 	val/=n;
// 	return val;
// }//working on
// //[[Rcpp::export]]
// double E_likeli2(double det_mat_sigma_alp_inv,const mat& mat_sigma_alp_inv,const mat& mat_E_alp_talp)
// {
// 	double val-=det_mat_sigma_alp_inv;
// 	int n=mat_sigma_alp_inv.n_rows;
// 	val+=n*log(trace(mat_sigma_alp_inv*mat_E_alp_talp));
// 	return val;
// }//working on
//[[Rcpp::export]]
inline double E_likeli1_mu_hat_estimate(const mat& mat_omega_eps_inv,const vec& vec_w,const mat& mat_alpha_sample)
{
	double val;
	int n=mat_omega_eps_inv.n_cols;
	vec vec_ones=vec(n, fill::ones);
	val=as_scalar(vec_ones.t()*mat_omega_eps_inv*(vec_w-mean(mat_alpha_sample,0).t()))/as_scalar(vec_ones.t()*\
		mat_omega_eps_inv*vec_ones);
	return val;
}

//A1=E_x_xt(mat_alpha_samples)
//[[Rcpp::export]]
inline mat A2_mat_compute(const vec& vec_w, double mu, const mat& mat_A1, const mat& mat_alpha_sample)
{
	//mat_A1=E_x_xt(mat_alpha_sample)
	int n=mat_alpha_sample.n_cols;
	mat mat_A2(n,n, fill::zeros);
	mat_A2=(vec_w-mu)*(vec_w-mu).t()-2.0*(vec_w-mu)*mean(mat_alpha_sample,0)+mat_A1;
	return mat_A2;
}

//working on
//[[Rcpp::export]]
inline double E_likeli1_sigmaeps_estimate(const mat& mat_omega_eps_inv, const mat& mat_A2)
{
	int n=mat_omega_eps_inv.n_cols;
	return trace(mat_omega_eps_inv*mat_A2)/(double) n;
}

//[[Rcpp::export]]
inline mat A3_mat_compute(const mat& mat_omega_eps_inv,const mat& mat_A1,const mat& mat_alpha_sample, const vec& vec_w)
{
	double muhat=E_likeli1_mu_hat_estimate(mat_omega_eps_inv,vec_w,mat_alpha_sample);
	mat vec_alpha_sample_mean=mean(mat_alpha_sample,0).t();
	int n=mat_alpha_sample.n_cols;
	mat mat_A3(n,n,fill::ones);
	mat_A3=(vec_w-vec_alpha_sample_mean-muhat)*(vec_w-vec_alpha_sample_mean-muhat).t()+
		mat_A1-vec_alpha_sample_mean*vec_alpha_sample_mean.t();
	// mat_A3 = (vec_w-muhat)*(vec_w-muhat).t()-2.0*(vec_w-muhat)*mean(mat_alpha_sample,0)+mat_A1;
	return mat_A3;
}



//[[Rcpp::export]]
inline double E_likeli1_nll(const mat& mat_A3,const mat& mat_omega_eps_inv)
{
	double val;
	int n=mat_omega_eps_inv.n_cols;
	val= -1.0 * mat_log_det(mat_omega_eps_inv) + n*log(trace(mat_omega_eps_inv*mat_A3));
	return val;
}

//object function for optimization
//input is theta
//no nugger is provided!
//[[Rcpp::export]]
double E_likeli1_nll_optim_coded_nonugg(const vec& vec_par, const vec& vec_w, const mat& mat_X, 
const Col<int>& num_lev, const mat& mat_alpha_sample, const mat& mat_A1)
{
	vec vec_par_exp=exp(vec_par);
	mat mat_omega_eps_inv=sigma_eps_inv(mat_X,vec_par_exp,0.0,num_lev);
	mat mat_A3=A3_mat_compute(mat_omega_eps_inv,mat_A1,mat_alpha_sample,vec_w);
	return E_likeli1_nll(mat_A3,mat_omega_eps_inv);
}

//[[Rcpp::export]]
double E_likeli1_nll_optim_coded_nonugg_univtheta(double log_theta, const vec& vec_w, const mat& mat_X, 
const Col<int>& num_lev, const mat& mat_alpha_sample, const mat& mat_A1)
{
	int ndim=mat_X.n_cols;
	vec vec_par_exp(ndim,fill::ones);
	vec_par_exp*=exp(log_theta);
	mat mat_omega_eps_inv=sigma_eps_inv(mat_X,vec_par_exp,0.0,num_lev);
	mat mat_A3=A3_mat_compute(mat_omega_eps_inv,mat_A1,mat_alpha_sample,vec_w);
	return E_likeli1_nll(mat_A3,mat_omega_eps_inv);
}

//[[Rcpp::export]]
double E_likeli1_nll_optim_coded_nugg(const vec& vec_par, const vec& vec_w, const mat& mat_X, 
const Col<int>& num_lev, const mat& mat_alpha_sample, const mat& mat_A1, bool is_coded = true)
{
	vec vec_par_exp=vec_par;
	if(is_coded)
	{
		vec_par_exp = exp(vec_par_exp);
	}
	int ndim=mat_X.n_cols;
	mat mat_omega_eps_inv=sigma_eps_inv(mat_X,vec_par_exp.subvec(0,(ndim-1)),vec_par_exp(ndim),num_lev);
	mat mat_A3=A3_mat_compute(mat_omega_eps_inv,mat_A1,mat_alpha_sample,vec_w);
	return E_likeli1_nll(mat_A3,mat_omega_eps_inv);
}

//[[Rcpp::export]]
double E_likeli1_nll_optim_coded_nugg_v2(const vec& vec_par, const vec& vec_w, const mat& mat_X, 
const Col<int>& num_lev, const mat& mat_alpha_sample, const mat& mat_A1, double theta_max = 10, double nugg_max = 0.5, bool is_coded = true)
{
	vec vec_par_exp=vec_par;
	int ndim=mat_X.n_cols;
	if(is_coded)
	{
		vec_par_exp.subvec(0,(ndim-1))=theta_max/(1+exp(-vec_par.subvec(0,(ndim-1))));
		vec_par_exp(ndim) = nugg_max/(1+exp(-vec_par(ndim)));
	}
	
	mat mat_omega_eps_inv=sigma_eps_inv(mat_X,vec_par_exp.subvec(0,(ndim-1)),vec_par_exp(ndim),num_lev);
	mat mat_A3=A3_mat_compute(mat_omega_eps_inv,mat_A1,mat_alpha_sample,vec_w);
	return E_likeli1_nll(mat_A3,mat_omega_eps_inv);
}
//begin the functions for likeli2
//[[Rcpp::export]]
double E_likeli2_sigmaalp_estimate(const mat& mat_omega_alp_inv, const mat& mat_A1)
{
	int n=mat_omega_alp_inv.n_cols;
	return trace(mat_omega_alp_inv*mat_A1)/(double) n;
}
//[[Rcpp::export]]
double E_likeli2_nll(const mat& mat_A1,const mat& mat_omega_alp_inv)
{
	double val;
	int n=mat_omega_alp_inv.n_cols;
	val= -1.0 * mat_log_det(mat_omega_alp_inv) + n*log(trace(mat_omega_alp_inv*mat_A1));
	return val;
}

//[[Rcpp::export]]
double E_likeli2_nll_optim_coded_CSK_B(const vec vec_para, const mat& mat_A1,const mat& mat_X, const Col<int>& vec_Z,
double rmax, int n_cates)
{
	vec vec_para_scaled=datum::pi/(exp(-vec_para)+1.0);
	double val;
	mat mat_omega_alp_inv=sigma_alpha_inv_gen_CSK_B(mat_X,vec_Z,vec_para_scaled,rmax,n_cates);
	int n=mat_omega_alp_inv.n_cols;
	val=-1.0*mat_log_det(mat_omega_alp_inv)+n*log(trace(mat_omega_alp_inv*mat_A1));
	return val;
}

//[[Rcpp::export]]
double E_likeli2_nll_optim_coded_CSK_W(const vec vec_para, const mat& mat_A1,const mat& mat_X, const Col<int>& vec_Z,
double rmax, int n_cates)
{
	vec vec_para_scaled=datum::pi/(exp(-vec_para)+1.0);
	double val;
	mat mat_omega_alp_inv=sigma_alpha_inv_gen_CSK_W(mat_X,vec_Z,vec_para_scaled,rmax,n_cates);
	int n=mat_omega_alp_inv.n_cols;
	val=-1.0*mat_log_det(mat_omega_alp_inv)+n*log(trace(mat_omega_alp_inv*mat_A1));
	return val;
}

/*
followoing is profile likelihood estiamtion process
*/
//version 2 omega correlation matrix
/*
mat sigma_alpha_inv_gen_CSK_B(const mat& X, const Col<int> &Z,vec cate_para,double rmax, int m, 
    bool is_inverse=true)
*/
//[[Rcpp::export]]
mat omega_alpha_CSK_B_v2(const mat& mat_X, const Col<int>& col_Z, vec vec_cate_para,
	double rmax, int num_cates, bool bool_inverse=false)
{
	mat mat_cate_alpha=CSK_B(mat_X,rmax);
	mat mat_cate_hsd=c_inverse_hsd(vec_cate_para,num_cates);
	mat_cate_hsd=mat_cate_hsd * mat_cate_hsd.t();
	for(int i=0;i<(int) mat_X.n_rows;i++)
	{
		for(int j=0;j<=i;j++)//working on a symmetric matrix
		{
			mat_cate_alpha(i,j)=mat_cate_alpha(i,j)*mat_cate_hsd(col_Z(i),col_Z(j));
			mat_cate_alpha(j,i)=mat_cate_alpha(i,j);
		}
	}
	if(bool_inverse)
	{
		return inv_sympd(mat_cate_alpha);
	}
	else
	{
		return mat_cate_alpha;
	}
}



// 20200821
// test version for small size samples, avoid use it in production task
// distinued! PL model passed the estimation test
//[[Rcpp::export]]
double profile_likeli_internal(const vec& vec_w, const mat& mat_omega_eps, const mat& mat_omega_alp, double xi)
{
	int n=mat_omega_eps.n_cols;
	mat mat_omega=mat_omega_eps+mat_omega_alp*xi;
	mat mat_omega_inv=inv_sympd(mat_omega);
	vec vec_n_ones=vec(n,fill::ones);
	double muhat=as_scalar(vec_n_ones.t()*mat_omega_inv*vec_w)/as_scalar(vec_n_ones.t()*mat_omega_inv*vec_n_ones);
	mat mat_A=(vec_w-muhat*vec_n_ones)*(vec_w-muhat*vec_n_ones).t();
	double sigmahat=trace(mat_omega_inv*mat_A)/n;
	double nll=n*log(sigmahat)/2.0+mat_log_det(mat_omega)/2.0;
	return nll;
}
/*
structure of vec_para
c(xi, theta[ndim],ctheta)
*/
//[[Rcpp::export]]
double profile_likeli_optim_coded(const vec& vec_para, const mat& mat_X, const vec& vec_w, const Col<int>& col_z, 
	int ncates, Col<int> vec_num_levels, double rmax=0.1)
{
	int ndim=mat_X.n_cols;
	//represent parameters
	double xi=exp(vec_para(0));
	vec vec_theta=exp(vec_para.subvec(1,ndim));
	vec vec_ctheta=exp(vec_para.subvec(ndim+1,vec_para.n_elem-1));
	vec_ctheta=datum::pi*vec_ctheta/(vec_ctheta+1);
	//construct matrix
	mat mat_omega_eps=block_corrM(mat_X,vec_theta,0.0,vec_num_levels);
	mat mat_omega_alp=omega_alpha_CSK_B_v2(mat_X,col_z,vec_ctheta,rmax,ncates);
	double nll=profile_likeli_internal(vec_w,mat_omega_eps,mat_omega_alp,xi);
	return nll;
}

/*
xi seperate procedure
structure of vec_para
c(theta[ndim],ctheta)
*/
//[[Rcpp::export]]
double profile_likeli_optim_coded_xi(const vec& vec_para, const mat& mat_X, const vec& vec_w, const Col<int>& col_z, 
	int ncates, Col<int> vec_num_levels, double log_xi, double rmax=0.1)
{
	int ndim=mat_X.n_cols;
	//represent parameters
	double xi=exp(log_xi);
	vec vec_theta=exp(vec_para.subvec(0,ndim-1));
	vec vec_ctheta=exp(vec_para.subvec(ndim,vec_para.n_elem-1));
	vec_ctheta=datum::pi*vec_ctheta/(vec_ctheta+1);
	//construct matrix
	mat mat_omega_eps=block_corrM(mat_X,vec_theta,0.0,vec_num_levels);
	mat mat_omega_alp=omega_alpha_CSK_B_v2(mat_X,col_z,vec_ctheta,rmax,ncates);
	double nll=profile_likeli_internal(vec_w,mat_omega_eps,mat_omega_alp,xi);
	return nll;
}

/*
xi seperate procedure
structure of vec_para
c(theta,ctheta)
*/
//[[Rcpp::export]]
double profile_likeli_optim_coded_xi_sametheta(const vec& vec_para, const mat& mat_X, const vec& vec_w, const Col<int>& col_z, 
	int ncates, Col<int> vec_num_levels, double log_xi, double rmax=0.1)
{
	int ndim=mat_X.n_cols;
	//represent parameters
	double xi=exp(log_xi);
	vec vec_theta(ndim,fill::ones);
	vec_theta*=exp(vec_para[0]);
	vec vec_ctheta=exp(vec_para.subvec(1,vec_para.n_elem-1));
	vec_ctheta=datum::pi*vec_ctheta/(vec_ctheta+1);
	//construct matrix
	mat mat_omega_eps=block_corrM(mat_X,vec_theta,0.0,vec_num_levels);
	mat mat_omega_alp=omega_alpha_CSK_B_v2(mat_X,col_z,vec_ctheta,rmax,ncates);
	double nll=profile_likeli_internal(vec_w,mat_omega_eps,mat_omega_alp,xi);
	return nll;
}

/*
addded 20201010
separate theta and nugg for each cate
*/
//[[Rcpp::export]]
mat block_corrM_seppar(const mat& X, vec theta, vec nugg, Col<int> num_vec)
{
	int n=X.n_rows;
	int p=X.n_cols;
	mat blocked_mat(n,n,fill::zeros);
	int start_idx, end_idx;
    start_idx=0;
	for(int i=0;i<(int) num_vec.n_elem;i++)
	{
		end_idx=start_idx+num_vec(i)-1;
		if(end_idx < start_idx)
		{
			start_idx=start_idx+num_vec(i);
		} else
		{
			vec theta_iter=theta.subvec(p*i,p*i+p-1);
			blocked_mat.submat(start_idx,start_idx,end_idx,end_idx)=convM(X.rows(start_idx, end_idx),theta_iter,nugg(i));
			start_idx=start_idx+num_vec(i);
		}
	}
	return blocked_mat;
}

//block correlation matrix for x1 x2
//old version, for testing only!!!
//never call it in production code
//[[Rcpp::export]]
mat block_corrM_X1X2_seppar_test(const mat& X1, const mat& X2, vec theta, Col<int> numvec1, Col<int> numvec2)
{
	int n1=X1.n_rows;
	int n2=X2.n_rows;
	int p=X1.n_cols; 
	mat cor_mat_x1x2(n1,n2, fill::zeros);
	int start_idx_x1,end_idx_x1,start_idx_x2,end_idx_x2;
	int num_cates=(int) numvec1.n_elem;
	start_idx_x1=0;
	start_idx_x2=0;
	for(int i=0;i<num_cates;i++) 
	{
		end_idx_x1=start_idx_x1+numvec1(i)-1;
		end_idx_x2=start_idx_x2+numvec2(i)-1;
		if((end_idx_x1 < start_idx_x1) || (end_idx_x2 < start_idx_x2))
		{
			start_idx_x1=start_idx_x1+numvec1(i);
			start_idx_x2=start_idx_x2+numvec2(i);
		} else
		{
			vec theta_iter=theta.subvec(p*i,p*i+p-1);
			cor_mat_x1x2.submat(start_idx_x1, start_idx_x2, end_idx_x1, end_idx_x2)=
			convM_X1X2(X1.rows(start_idx_x1, end_idx_x1),X2.rows(start_idx_x2, end_idx_x2),theta_iter);
			start_idx_x1=start_idx_x1+numvec1(i);
			start_idx_x2=start_idx_x2+numvec2(i);
		}
	}
	return cor_mat_x1x2;
}
//[[Rcpp::export]]
mat block_corrM_X1X2_seppar(const mat& X1, const mat& X2, vec theta, Col<int> numvec1, Col<int> numvec2)
{
	int n1=X1.n_rows;
	int n2=X2.n_rows;
	int p=X1.n_cols; 
	mat cor_mat_x1x2(n1,n2, fill::zeros);
	int start_idx_x1,end_idx_x1,start_idx_x2,end_idx_x2;
	int num_cates=(int) numvec1.n_elem;
	start_idx_x1=0;
	start_idx_x2=0;
	for(int i=0;i<num_cates;i++) 
	{
		end_idx_x1=start_idx_x1+numvec1(i)-1;
		end_idx_x2=start_idx_x2+numvec2(i)-1;
		if((end_idx_x1 < start_idx_x1) || (end_idx_x2 < start_idx_x2))
		{
			start_idx_x1=start_idx_x1+numvec1(i);
			start_idx_x2=start_idx_x2+numvec2(i);
		} else
		{
			vec theta_iter=theta.subvec(p*i,p*i+p-1);
			cor_mat_x1x2.submat(start_idx_x1, start_idx_x2, end_idx_x1, end_idx_x2)=
			convM_X1X2(X1.rows(start_idx_x1, end_idx_x1),X2.rows(start_idx_x2, end_idx_x2),theta_iter);
			start_idx_x1=start_idx_x1+numvec1(i);
			start_idx_x2=start_idx_x2+numvec2(i);
		}
	}
	return cor_mat_x1x2;
}

//[[Rcpp::export]]
mat omega_eps_inv_seppar(const mat& X, vec theta, vec g, Col<int> num_vec)
{
	mat eps_mat=block_corrM_seppar(X,theta,g,num_vec);
	mat eps_mat_inv=block_inv(eps_mat,num_vec);
	return eps_mat_inv;
}

// using:
// E_likeli1_mu_hat_estimate(const mat& mat_omega_eps_inv,const vec& vec_w,const mat& mat_alpha_sample)
// X.submat( first_row, first_col, last_row, last_col )
//[[Rcpp::export]]
vec E_likeli1_mu_hat_estimate_seppar_v1(const mat& mat_omega_eps_inv,const vec& vec_w,const mat& mat_alpha_sample,
	const Col<int>& numvec)
{
	int n_cate=numvec.n_elem;
	vec vec_mu(n_cate, fill::zeros);
	int start_idx, end_idx;
    start_idx=0;
	for(int i=0;i<n_cate;i++)
	{
		end_idx=start_idx+numvec(i)-1;
		vec_mu(i) = E_likeli1_mu_hat_estimate(
			mat_omega_eps_inv.submat(start_idx,start_idx,end_idx,end_idx),
			vec_w.subvec(start_idx,end_idx),
			mat_alpha_sample.cols(start_idx,end_idx)
		);
		start_idx=start_idx+numvec(i);
	}
	return vec_mu;
}

// version 2 for verifying
//[[Rcpp::export]]
vec E_likeli1_mu_hat_estimate_seppar_v2(const mat& mat_omega_eps_inv,const vec& vec_w,const mat& mat_alpha_sample,
	const Col<int>& numvec)
{
	int p=numvec.n_elem;
	vec vec_mu(p, fill::zeros);
	int n=vec_w.n_elem;
	mat pones(n,p,fill::zeros);
	int start_idx, end_idx;
    start_idx=0;
	for(int i=0;i<p;i++)
	{
		end_idx=start_idx+numvec(i)-1;
		pones.submat(start_idx,i,end_idx,i).ones();
		start_idx=start_idx+numvec(i);
	}
	vec_mu = inv(pones.t()*mat_omega_eps_inv*pones)*(pones.t()*mat_omega_eps_inv*(vec_w-mean(mat_alpha_sample,0).t()));
	return vec_mu;
}

// version 3 for verifying and general procedure
//[[Rcpp::export]]
vec E_likeli1_mu_hat_estimate_seppar_v3(const mat& mat_omega_eps_inv,const vec& vec_w,const mat& mat_alpha_sample,
	const Col<int>& numvec)
{
	int n_cate=numvec.n_elem;
	int start_idx, end_idx;
    start_idx=0;
	vec vec_mu(n_cate, fill::ones);
	vec vec_alp_mean=mean(mat_alpha_sample,0).t();
	// val=as_scalar(vec_ones.t()*mat_omega_eps_inv*(vec_w-mean(mat_alpha_sample,0).t()))/as_scalar(vec_ones.t()*mat_omega_eps_inv*vec_ones);
	for(int i=0;i<n_cate;i++)
	{
		vec vec_ones(numvec(i), fill::ones);
		end_idx=start_idx+numvec(i)-1;
		mat submat_omega_eps_inv = mat_omega_eps_inv.submat( start_idx, start_idx, end_idx, end_idx);
		vec_mu(i)= as_scalar(vec_ones.t()*submat_omega_eps_inv*(vec_w.subvec(start_idx,end_idx)-vec_alp_mean.subvec(start_idx,end_idx)))/
			as_scalar(vec_ones.t()*submat_omega_eps_inv*vec_ones);
		start_idx=start_idx+numvec(i);
	}
	return vec_mu;
}

// sigma_eps formula return trace(mat_omega_eps_inv*mat_A2)/(double) n;
//[[Rcpp::export]]
vec E_likeli1_sigmaeps_estimate_seppar(const vec& vec_w, const mat& mat_omega_eps_inv, const mat& mat_alpha_sample, const Col<int>& vec_num)
{
	int n_cates=vec_num.n_elem;
	vec vec_sigma_eps(n_cates);
	int start_idx, end_idx;
    start_idx=0;
	vec vec_mu=E_likeli1_mu_hat_estimate_seppar_v1(
		mat_omega_eps_inv,
		vec_w,
		mat_alpha_sample,
		vec_num
	);
	for(int i=0;i<n_cates;i++)
	{
		end_idx=start_idx+vec_num(i)-1;

		mat submat_omega_eps_inv=mat_omega_eps_inv.submat(start_idx,start_idx,end_idx,end_idx);
		mat mat_A1=E_x_xt(mat_alpha_sample.cols(start_idx,end_idx));
		mat mat_A2 = A2_mat_compute(
			vec_w.subvec(start_idx,end_idx),
			vec_mu(i),
			mat_A1,
			mat_alpha_sample.cols(start_idx,end_idx)
		);
		vec_sigma_eps(i) = trace(submat_omega_eps_inv*mat_A2)/(double) vec_num(i);
		start_idx=start_idx+vec_num(i);
	}
	return vec_sigma_eps;
}


// working on
/*
mat A3_mat_compute_seppar(const mat& mat_omega_eps_inv,const mat& mat_A1,const mat& mat_alpha_sample, const vec& vec_w, const Col<int>& vec_num)
{
	vec vec_muhat=E_likeli1_mu_hat_estimate_seppar_v1(
		mat_omega_eps_inv,
		vec_w,
		mat_alpha_sample,
		vec_num
	);
	mat vec_alpha_sample_mean=mean(mat_alpha_sample,0).t();
	int n=mat_alpha_sample.n_cols;
	mat mat_A3(n,n,fill::ones);
	mat_A3=(vec_w-vec_alpha_sample_mean-muhat)*(vec_w-vec_alpha_sample_mean-muhat).t()+
		mat_A1-vec_alpha_sample_mean*vec_alpha_sample_mean.t();
	// mat_A3 = (vec_w-muhat)*(vec_w-muhat).t()-2.0*(vec_w-muhat)*mean(mat_alpha_sample,0)+mat_A1;
	return mat_A3;
}
*/
//[[Rcpp::export]]
inline double E_likeli1_nll_seppar(const mat& mat_A3,const mat& mat_omega_eps_inv)
{
	double val;
	int n=mat_omega_eps_inv.n_cols;
	val= -1.0 * mat_log_det(mat_omega_eps_inv) + n*log(trace(mat_omega_eps_inv*mat_A3));
	return val;
}

//object function for optimization
//input is theta
//no nugger is provided!
/*
double E_likeli1_nll_optim_coded_nugg_seppar_v1(const vec& vec_par, const vec& vec_w, const mat& mat_X, 
const Col<int>& num_lev, const mat& mat_alpha_sample, bool is_coded = true)
{
	//estimate vec_mu
	int n_cate=num_lev.n_elem;
	int p=mat_X.n_cols;
	vec vec_theta(n_cate*p);
	vec vec_nugg(n_cate);
	for(int i=0;i<n_cate;i++)
	{
		vec_theta.subvec(i*(p+1),i*(p+1)+p-1)=vec_par.subvec();
		vec_nugg(i)=vec_par(i*(p+1)+p);
	}
	mat mat_omega_inv=omega_eps_inv_seppar(
		mat_X,
		vec_theta,
		vec_nugg,
		num_lev
	);
	vec vec_mu=E_likeli1_mu_hat_estimate_seppar_v1(
		mat_omega_inv,
		vec_w,
		mat_alpha_sample,
		num_lev
	);
	//TBD not finished yet
	return -1;
}
*/
// naive version for correction and checking
/*
double E_likeli1_nll_optim_coded_nugg(const vec& vec_par, const vec& vec_w, const mat& mat_X, 
const Col<int>& num_lev, const mat& mat_alpha_sample, const mat& mat_A1, bool is_coded = true)
*/
//[[Rcpp::export]]
double E_likeli1_nll_optim_coded_nugg_seppar_v2(const vec& vec_par, const vec& vec_w, const mat& mat_X, 
const Col<int>& num_lev, const mat& mat_alpha_sample, bool is_coded = true)
{
	int n_cate=num_lev.n_elem;
	int p=mat_X.n_cols;
	vec vec_theta(n_cate*p);
	vec vec_nugg(n_cate);
	for(int i=0;i<n_cate;i++)
	{
		vec_theta.subvec(i*p,i*p+p-1)=vec_par.subvec(i*(p+1),i*(p+1)+p-1);
		vec_nugg(i)=vec_par(i*(p+1)+p);
	}
	if(is_coded)
	{
		vec_theta=exp(vec_theta);
		vec_nugg=exp(vec_nugg);
	}
	int start_idx, end_idx;
	double num_nll1=0.0;
	start_idx=0;
	for(int i=0;i<n_cate;i++)
	{
		end_idx=start_idx+num_lev(i)-1;
		vec subvec_par(p+1);
		subvec_par.subvec(0,p-1)=vec_theta.subvec(i*p,i*p+p-1);
		subvec_par(p)=vec_nugg(i);
		Col<int> subnum_lev(1);
		subnum_lev(0)=num_lev(i);
		num_nll1+=E_likeli1_nll_optim_coded_nugg(
			subvec_par,
			vec_w.subvec(start_idx,end_idx),
			mat_X.rows(start_idx,end_idx), 
			subnum_lev,
			mat_alpha_sample.cols(start_idx,end_idx),
			E_x_xt(mat_alpha_sample.cols(start_idx,end_idx)),
			false);
		start_idx=start_idx+num_lev(i);
	}
	return num_nll1;
}


/*
other function for convenience and speed
within seppar verion
*/

//[[Rcpp::export]]
mat sigma_eps_seppar(const mat& X, vec theta, vec g, vec vec_sigmaepssq, Col<int> num_lev, bool is_inverse = true)
{
	int n_cate=num_lev.n_elem;
	mat mat_sigma_eps=block_corrM_seppar(X,theta,g,num_lev);
	int start_idx, end_idx;
	start_idx=0;
	for(int i=0;i<n_cate;i++)
	{
		end_idx=start_idx+num_lev(i)-1;
		if(end_idx < start_idx)
		{
			start_idx=start_idx+num_lev(i);
		} else
		{
			mat_sigma_eps.submat(start_idx,start_idx,end_idx,end_idx) = mat_sigma_eps.submat(start_idx,start_idx,end_idx,end_idx)*vec_sigmaepssq(i);
			start_idx=start_idx+num_lev(i);
		}
	}
	if(is_inverse)
	{
		mat_sigma_eps=block_inv(mat_sigma_eps,num_lev);
	}
	return mat_sigma_eps;
}

//[[Rcpp::export]]
mat sigma_eps_x1x2_seppar(const mat& X1, const mat& X2, vec theta, vec vec_sigmaepssq, Col<int> num_lev1, Col<int> num_lev2)
{
	int n_cate=num_lev1.n_elem;
	mat mat_sigma_eps_X1X2=block_corrM_X1X2_seppar(
		X1,
		X2,
		theta,
		num_lev1,
		num_lev2
	);
	int start_idx1, end_idx1;
	int start_idx2, end_idx2;
	start_idx1=0;
	start_idx2=0;
	for(int i=0;i<n_cate;i++)
	{
		end_idx1=start_idx1+num_lev1(i)-1;
		end_idx2=start_idx2+num_lev2(i)-1;
		if((end_idx1 < start_idx1) || (end_idx2 < start_idx2))
		{
			start_idx1=start_idx1+num_lev1(i);
			start_idx2=start_idx2+num_lev2(i);
		} else
		{
			mat_sigma_eps_X1X2.submat(start_idx1,start_idx2,end_idx1,end_idx2) = mat_sigma_eps_X1X2.submat(start_idx1,start_idx2,end_idx1,end_idx2)*vec_sigmaepssq(i);
			start_idx1=start_idx1+num_lev1(i);
			start_idx2=start_idx2+num_lev2(i);
		}
	}
	return mat_sigma_eps_X1X2;
}

// used in conditional mean part
//[[Rcpp::export]]
vec center_vec_w(const vec& vec_w, const vec& vec_mu, Col<int> num_lev)
{
	vec vec_wc = vec_w;
	int n_cate=num_lev.n_elem;
	int start_idx, end_idx;
	start_idx=0;
	for(int i=0;i<n_cate;i++)
	{
		end_idx=start_idx+num_lev(i)-1;
		vec_wc.subvec(start_idx,end_idx) = vec_wc.subvec(start_idx,end_idx) - vec_mu(i);
		start_idx=start_idx+num_lev(i);
	}
	return vec_wc;
}


/*
added 20201206
exact em funs
*/
//[[Rcpp::export]]
double EXACTEM_nll1_mu(const vec& vec_w, const mat& mat_omega_eps_inv, const vec& vec_cond_mean, const mat& mat_cond_sigma)
{
	double num_muhat;
	int n = vec_w.n_elem;
	vec vec_nones(n, fill::ones);
	num_muhat = quadratic_multiple_ab(mat_omega_eps_inv, vec_nones,vec_w-vec_cond_mean);
	num_muhat = num_muhat/quadratic_multiple(mat_omega_eps_inv, vec_nones);
	return num_muhat;
}

//[[Rcpp::export]]
double EXACTEM_nll1_sigsq(double num_muhat, const vec& vec_w, const mat& mat_omega_eps_inv,const vec& vec_cond_mean, const mat& mat_cond_sigma)
{
	double num_sigmasq;
	int n = vec_w.n_elem;
	num_sigmasq = quadratic_multiple(mat_omega_eps_inv,vec_w - num_muhat);
	num_sigmasq -= 2.0 * quadratic_multiple_ab(mat_omega_eps_inv, vec_w - num_muhat,vec_cond_mean);
	num_sigmasq += trace(mat_omega_eps_inv*mat_cond_sigma);
	num_sigmasq += quadratic_multiple(mat_omega_eps_inv,vec_cond_mean);
	num_sigmasq /= (double) (n);
	return num_sigmasq;
}

//[[Rcpp::export]]
double EXACTEM_nll2_sigsq(const mat& mat_omega_alp_inv,const vec& vec_cond_mean, const mat& mat_cond_sigma)
{
	double num_sigmasq;
	int n = vec_cond_mean.n_elem;
	num_sigmasq = trace(mat_omega_alp_inv*mat_cond_sigma);
	num_sigmasq += quadratic_multiple(mat_omega_alp_inv,vec_cond_mean);
	num_sigmasq /= (double) (n);
	return num_sigmasq;
}

/*
added 20211003
kron NR for alp matrix
*/
//[[Rcpp::export]]
mat Amat_Cpp(const Col<int>& Z, int k, int N)
{
	mat A(0,N);
	for(int i=0;i<k;i++)
	{
		mat temp(N,N, fill::zeros);
		uvec idx = find(Z == i);
		for(int j=0;j<(int) idx.n_elem;j++)
		{
			temp(idx(j),idx(j)) = 1.0;
		}
		A = join_cols(A, temp);
	}
	return A;
}


//sigma_alpha_inv_gen_CSK_W_KP = function(X,Z,rho,rmax,m,is_inverse)
//[[Rcpp::export]]
mat sigma_alpha_inv_gen_CSK_W_KP_Cpp(const mat& X, const Col<int>& Z, const vec& rho, double rmax, int m, bool is_inverse)
{
	int N = Z.n_elem;
	mat Phi=CSK_W(X,rmax);
	mat mat_L = c_inverse_hsd(rho,m);
	mat_L = mat_L * mat_L.t();
	mat A = Amat_Cpp(Z,m, N);
	mat sigma_alp = A.t() * kron(mat_L, Phi) * A;
	if(is_inverse)
	{
	  return inv(sigma_alp);
	}
	return sigma_alp;
}


///final likelihood version
//[[Rcpp::export]]
double c_lmgp_nll_eps(const vec& para, const vec& cond_mean, const mat& cond_sigma,
	const mat& X, const vec& Y, const Col<int> Z, const Col<int> num_lev)
{
	int n = Y.n_elem;
	int p = X.n_cols;
	vec theta = para.subvec(0, p - 1);
	double g = para(p);
	mat omega_eps = block_corrM(
		X,
		theta,
		g,
		num_lev);
	mat omega_eps_inv = block_inv(
		omega_eps,
		num_lev
	);
	double mu = EXACTEM_nll1_mu(
		Y,
		omega_eps_inv,
		cond_mean,
		cond_sigma
	);
	double sigmasq_eps = EXACTEM_nll1_sigsq(
		mu,
		Y,
		omega_eps_inv,
		cond_mean,
		cond_sigma
	);
	double num_nll, sign;
	log_det(num_nll, sign, omega_eps);
	num_nll = num_nll * (-0.5);
	num_nll = num_nll - log(sigmasq_eps)*n/2.0;
	return -num_nll;
}

//[[Rcpp::export]]
double c_lmgp_nll_alp(const vec& para, const vec& cond_mean, const mat& cond_sigma,
	const mat& X, const vec& Y, const Col<int> Z, const Col<int> num_lev, double rmax = 3.0)
{
	int n = Y.n_elem;
	int ncates = num_lev.n_elem;
	mat omega_alp = sigma_alpha_inv_gen_CSK_W(
	X,
	Z,
	para,
	rmax, ncates, false);
	mat omega_alp_inv = inv_sympd(omega_alp);
	double sigmasq_alp = EXACTEM_nll2_sigsq(
	omega_alp_inv,
	cond_mean,
	cond_sigma
	);
	double num_nll, sign;
	log_det(num_nll, sign, omega_alp);
	num_nll = num_nll * (-0.5);
	num_nll = num_nll - log(sigmasq_alp) * n/2.0;
	return (-num_nll);
}