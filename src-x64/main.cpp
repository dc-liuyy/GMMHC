// [[Rcpp::depends(RcppEigen)]]
#include <Rcpp.h>
#include <RcppEigen.h>
#include <iostream>
#include <exception>
#include <math.h>
#include <limits>
#define _MATH_DEFINES_DEFINED
using namespace Rcpp;
using namespace Eigen;
using namespace std;
using Eigen::Map;
using Eigen::MatrixXd;
using Eigen::VectorXd;
using Rcpp::as;

const double epsilon = 0.00000001;

// NumericMatrix => MatrixXd
Map<MatrixXd> asMatrixXd(NumericMatrix x){
  return as<Map<MatrixXd> >(x);
}

// NumericVector => VectorXd
Map<VectorXd> asVectorXd(NumericVector x){
  return as<Map<VectorXd> >(x);
}

// record the position of NAs
MatrixXi mask_na(NumericMatrix X){
  int N = X.rows();
  int p = X.cols();
  MatrixXi mask;
  mask.setOnes(N,p);
  for(int i=0; i<N; i++){
    for(int j=0; j<p; j++){
      if (NumericMatrix::is_na(X(i,j))){
        mask(i,j) = 0;   // NA -- 0   observed -- 1
      }
    }
  }
  return mask;
}

VectorXd sliceVec(NumericMatrix X,int rc, int i,VectorXi index){
  /*
  X       input matrix
  rc      0 -- row, 1 -- column
  i       slice the ith row/column
  index   0 -- omit, 1 -- maintain
  */
  VectorXd out = VectorXd(index.sum());
  if (rc == 1){
    X = wrap(as<Map<MatrixXd> >(X).transpose());
  }
  int p = X.cols();
  int l=0;
  for (int j=0;j<p;j++){
    if (index(j)==1){
      out(l) = X(i,j);
      l++;
    }
  }
  return out;
}


MatrixXd sliceMat(NumericMatrix X,VectorXi rindex,VectorXi cindex){
  /*
  X       input matrix
  rindex  row, 0 -- omit, 1 -- maintain
  cindex  column, 0 -- omit, 1 -- maintain
  */
  MatrixXd out = MatrixXd(rindex.sum(),cindex.sum());
  int N = X.rows();
  int p = X.cols();
  if (rindex.size() != N){
    throw "sliceMat: unmatched rindex";
  }
  if (cindex.size() != p){
    throw "sliceMat: unmatched cindex";
  }
  int l1=0;
  for (int i=0;i<N;i++){
    if (rindex(i)==1){
      int l2=0;
      for (int j=0;j<p;j++){
        if (cindex(j)==1){
          out(l1,l2) = X(i,j);
          l2++;
        }
      }
      l1++;
    }
  }
  return out;
}

MatrixXd replaceMat(MatrixXd X, MatrixXd rep, VectorXi rindex, VectorXi cindex){
  int N = X.rows();
  int p = X.cols();
  if (rindex.size() != N){
    throw "replaceMat: unmatched rindex for X";
  }
  if (cindex.size() != p){
    throw "replaceMat: unmatched cindex for X";
  }
  if (rindex.sum() != rep.rows()){
    throw "replaceMat: unmatched rindex for rep";
  }
  if (cindex.sum() != rep.cols()){
    throw "replaceMat: unmatched cindex for rep";
  }

  int l1=0;
  for (int i=0;i<N;i++){
    if (rindex(i)==1){
      int l2=0;
      for (int j=0;j<p;j++){
        if (cindex(j)==1){
          X(i,j) = rep(l1,l2);
          l2++;
        }
      }
      l1++;
    }
  }
  return X;
}

VectorXd replaceVec(VectorXd X, VectorXd rep, VectorXi index){
  int N = X.size();
  if (index.size() != N){
    throw "replaceVec: unmatched index for X";
  }
  if (index.sum() != rep.size()){
    throw "replaceVec: unmatched index for rep";
  }
  int l1=0;
  for (int i=0;i<N;i++){
    if (index(i)==1){
      X(i) = rep(l1);
      l1++;
    }
  }
  return X;
}


double mydmnorm(VectorXd x,
                VectorXd mean, MatrixXd var){
  /* densities of multivariate normal distribution
  * input  - x     a vector with dimension d
  *        - mean  the mean vector with dimension d
  *        - var   the variance matrix
  */
  int d = mean.size();
  // check the dimensions

  if (x.size() != d) throw "mydmnorm: mismatch of dimensions of 'x' and 'mean'";
  if (var.cols() != var.rows()) throw "mydmnorm: 'var' must be a square matrix";
  if (var.cols() != d) throw "mydmnorm: mismatch of dimensions of 'x' and 'var'";
  double var_det = var.determinant();
  if (var_det< 0 || var_det==0)  {
    cout << "var_det="<<var_det<<endl;
    throw "mydmnorm: 'var' must be positive definite";
  }

  // compute the densities
  VectorXd x_mean = x - mean;
  MatrixXd I = MatrixXd::Identity(d,d);
  double den = pow(2*M_PI,-d/2) * pow(var_det,-0.5)
    * exp(-0.5 * x_mean.transpose() * var.ldlt().solve(I) * x_mean);
  return den;
}


VectorXd fws_fun(int i, const NumericMatrix X, const MatrixXi X_mask,const VectorXi X_mask_row,
  NumericMatrix mu_0,List sig2_0,VectorXd phi_0){
  int K = sig2_0.size();
  int p = X_mask.cols();
  VectorXd fs(K);
  VectorXd X_obs;
  if (X_mask_row(i)==p){
    X_obs = (as<Map<MatrixXd> >(X)).row(i);
  }else{
    X_obs = sliceVec(X,0,i,X_mask.row(i));
  }

  for(int k=0; k<K; k++){
    VectorXd mu_obs;
    MatrixXd sig_obs;
    if (X_mask_row(i)==p){
      mu_obs = (as<Map<MatrixXd> >(mu_0)).col(k);
      sig_obs = sig2_0[k];
    }else{
      mu_obs = sliceVec(mu_0,1,k,X_mask.row(i));
      sig_obs = sliceMat(sig2_0[k],X_mask.row(i),X_mask.row(i));
    }
    fs(k) = mydmnorm(X_obs, mu_obs, sig_obs);

  }
  VectorXd fs_wgt = phi_0.array()*fs.array();
  return fs_wgt;
}
// [[Rcpp::export]]
NumericMatrix impute_with_colmean(const NumericMatrix X){
  const MatrixXi X_mask = mask_na(X);
  const int N = X.rows();
  const int p = X.cols();
  MatrixXd complete(N,p);
  try{
    Map<MatrixXd> X_ = as<Map<MatrixXd> >(X);
    for(int j=0;j<p;j++){
      int obsnum = X_mask.col(j).sum();
      if (obsnum == N){
        complete.col(j) = X_.col(j);
      }else{
        VectorXd rep = VectorXd::Ones(N-obsnum) * sliceVec(X,1,j,X_mask.col(j)).sum()/X_mask.col(j).sum();
        complete.col(j) = replaceVec(X_.col(j), rep, 1-X_mask.col(j).array());
      }
    }
  }catch(const char* &e){
    cout << e << endl;
  }
  return wrap(complete);
}
// [[Rcpp::export]]
List calculate_paras_with_class(const NumericMatrix X, const NumericVector class0){
  // range of class0 is 1 to K


  // phi_0 <- tapply(class0,class0,length)/N
  //   mu_0 <- do.call(cbind,by(X0,class0,colMeans))
  //     mycov <- function(x){
  //       y <- cov(x)+diag(1e-6,p)
  //       return(y)
  //     }
  //   sig2_0 <- array(do.call(cbind,by(X0,class0,mycov)),dim=c(p,p,K))
  int N = X.rows();
  int p = X.cols();
  int K = max(class0);
  assert(X.rows() != class0.length());

  Map<MatrixXd> X_ = as<Map<MatrixXd> >(X);
  VectorXd phi = VectorXd::Zero(K) ;
  MatrixXd mu = MatrixXd::Zero(p,K);
  List sig2(K);
  for(int k=0; k<K; k++){
    sig2[k] = MatrixXd::Zero(p,p);
  }
  for(int i=0;i<N;i++){
    int k = class0(i)-1;
    phi(k) += 1;
    mu.col(k) = mu.col(k) + X_.row(i).transpose();
    sig2[k] = asMatrixXd(sig2[k]) + X_.row(i).transpose() * X_.row(i);
  }
  for(int k=0;k<K;k++){
    mu.col(k) = mu.col(k)/phi(k);
    sig2[k] = (asMatrixXd(sig2[k]) - phi(k)*mu.col(k)*mu.col(k).transpose())/(phi(k)-1);
    phi(k) = phi(k)/N;
  }

  return(List::create(Named("phi")=wrap(phi),
                      Named("mu")=mu,Named("sig2")=sig2));
}


// [[Rcpp::export]]
List myalgorithm(const NumericMatrix X, List para_0,const int max_iter,
                 const double epsilon_dif=0.000001, const double epsilon_var=0.000001) {
  /* main body of algorithm
   * input
     - X             data matrix: matrix(N,p)
     - para_0        initial parameters: list
      - para_0[0]    probabilities of components: vector(K)
      - para_0[1]    means of components: matrix(p,K)
      - para_0[2]    covariances of components: list(K)
        - para_0[2][k]  matrix(p,p)

   */
  const int N = X.rows();
  const int p = X.cols();

  // initialize the parameters
  VectorXd phi_0 = para_0[0];
  NumericMatrix mu_0 = para_0[1];
  const int K = phi_0.size();
  if (mu_0.rows() != p) {
    stop("mismatch of dimensions of 'X' and 'mu_0'\n");
  }
  if (mu_0.cols() != K){
    stop("mismatch of dimensions of 'phi_0' and 'mu_0'\n");
  }
  List sig2_0 = para_0[2];
  if (sig2_0.size() != K){
    stop("mismatch of dimensions of 'phi_0' and 'sig2_0'\n");
  }
  for(int k=0; k<K; k++){
    MatrixXd sig2_0_ = sig2_0[k];
    if (sig2_0_.rows()!=p || sig2_0_.cols()!=p){
      stop("mismatch of dimensions of 'mu_0' and 'sig2_0'\n");
    }
  }

  VectorXd phi(K);
  NumericMatrix mu(p,K);
  List sig2(K);
  NumericMatrix sig2_(p,p);
  for(int k=0; k<K; k++){
    sig2[k] = sig2_;
  }

  const MatrixXi X_mask = mask_na(X);
  const VectorXi X_mask_row = X_mask.rowwise().sum();  // number of observed variabels of each row
  int iter = 0;
  bool stopping = false;

  // main body
  try{
    while(!stopping){
      double llh_0 = 0;
      double llh = 0;
      iter += 1;
      // E-step
      MatrixXd w(N,K);
      for (int i=0;i<N;i++){
        VectorXd fs_wgt = fws_fun(i,X,X_mask,X_mask_row,mu_0,sig2_0,phi_0);
        w.row(i) = fs_wgt.array() *  MatrixXd::Ones(K,1).array()/fs_wgt.sum();
        llh_0 = log(fs_wgt.sum())+llh_0;
      }//checked
      // M-step
      MatrixXd mu_(p,K);
      for (int j=0;j<K;j++){
        VectorXd num_mu = VectorXd::Zero(p);  //matrix or vector?
        MatrixXd num_sig2 = MatrixXd::Zero(p,p);
        for (int i=0;i<N;i++){
          VectorXd x0 = VectorXd::Zero(p);
          MatrixXd sig0_1 = MatrixXd::Zero(p,p);
          MatrixXd sig0_2 = MatrixXd::Zero(X_mask_row(i), X_mask_row(i));
          if (X_mask_row(i) == p){
            x0 = (as<Map<MatrixXd> >(X)).row(i);
          }else {
            MatrixXd I = MatrixXd::Identity(X_mask_row(i),X_mask_row(i));
            sig0_2 = sliceMat(sig2_0[j],X_mask.row(i),X_mask.row(i)).ldlt().solve(I);
            // sig2_0[Ina,Ina,j]-sig2_0[Ina,I,j]%*%sig0_2%*%sig2_0[I,Ina,j]
            MatrixXd temp1 = sliceMat(sig2_0[j],1-X_mask.row(i).array(),1-X_mask.row(i).array()) -
              sliceMat(sig2_0[j],1-X_mask.row(i).array(),X_mask.row(i).array()) *
              sig0_2 * sliceMat(sig2_0[j],X_mask.row(i).array(),1-X_mask.row(i).array());
            sig0_1 = replaceMat(sig0_1, temp1, 1-X_mask.row(i).array(), 1-X_mask.row(i).array());
            VectorXd cen = sliceVec(X,0,i,X_mask.row(i)) - sliceVec(mu_0,1,j,X_mask.row(i));
            MatrixXd temp2 = sliceVec(mu_0,1,j,1-X_mask.row(i).array()) +
              sliceMat(sig2_0(j),1-X_mask.row(i).array(),X_mask.row(i)) * sig0_2 * cen;
            x0 = replaceVec(x0,temp2,1-X_mask.row(i).array());
            x0 = replaceVec(x0,sliceVec(X,0,i,X_mask.row(i)),X_mask.row(i));
          }
          num_mu = num_mu.array() + w(i,j)* (x0.array());
          num_sig2 = num_sig2 + w(i,j)*(x0 * x0.transpose() + sig0_1);

        }
        MatrixXd sig2j_(p,p);
        mu_.col(j) = num_mu.array() / w.col(j).sum();
        phi(j) = w.col(j).sum() / N;
        sig2j_ = (num_sig2 - num_mu * mu_.col(j).transpose() - mu_.col(j) * num_mu.transpose() + w.col(j).sum() * mu_.col(j) * mu_.col(j).transpose())/w.col(j).sum();
        sig2[j] = wrap((sig2j_ + sig2j_.transpose()) / 2 + MatrixXd::Identity(p,p)*epsilon_var);


      }
      mu = wrap(mu_);

      //stopping
      for (int i=0;i<N;i++){
        VectorXd fs_wgt = fws_fun(i,X,X_mask,X_mask_row,mu,sig2,phi);
        llh = log(fs_wgt.sum())+llh;
      }
      double dif = abs((llh_0-llh)/llh_0);

      mu_0 = mu;
      phi_0 = phi;
      sig2_0 = sig2;
      if ((dif < epsilon_dif)|(iter>=max_iter)) stopping = true;
    }

  }catch(const char* &e){
    cout << "stop at iter:"<<iter<<endl;
    cout << e << endl;
    MatrixXd w(N,K);
    for (int i=0;i<N;i++){
      VectorXd fs_wgt = fws_fun(i,X,X_mask,X_mask_row,mu_0,sig2_0,phi_0);
      w.row(i) = fs_wgt.array() *  MatrixXd::Ones(K,1).array()/fs_wgt.sum();
    }

    // compute the index - Vk
    // mu_x:
    MatrixXd mu_x(N, K);
    for (int i=0;i<N;i++){
      VectorXd x0(p);
      MatrixXd sig0_1 = MatrixXd::Zero(p,p);
      MatrixXd sig0_2 = MatrixXd::Zero(X_mask_row(i),X_mask_row(i));
      MatrixXd I = MatrixXd::Identity(X_mask_row(i),X_mask_row(i));
      for (int j=0; j<K; j++){
        if (X_mask_row(i)==p){
          x0 = (as<Map<MatrixXd> >(X)).row(i);
        }else{
          sig0_2 = sliceMat(sig2_0[j],X_mask.row(i),X_mask.row(i)).ldlt().solve(I);
          MatrixXd temp1 = sliceMat(sig2_0[j],1-X_mask.row(i).array(),1-X_mask.row(i).array()) -
            sliceMat(sig2_0[j],1-X_mask.row(i).array(),X_mask.row(i).array()) *
            sig0_2 * sliceMat(sig2_0[j],X_mask.row(i).array(),1-X_mask.row(i).array());
          sig0_1 = replaceMat(sig0_1, temp1, 1-X_mask.row(i).array(), 1-X_mask.row(i).array());
          VectorXd cen = sliceVec(X,0,i,X_mask.row(i)) - sliceVec(mu_0,1,j,X_mask.row(i));
          MatrixXd temp2 = sliceVec(mu_0,1,j,1-X_mask.row(i).array()) +
            sliceMat(sig2_0(j),1-X_mask.row(i).array(),X_mask.row(i)) * sig0_2 * cen;
          x0 = replaceVec(x0,temp2,1-X_mask.row(i).array());
          x0 = replaceVec(x0,sliceVec(X,0,i,X_mask.row(i)),X_mask.row(i));
        }
        VectorXd mu_j = (as<Map<MatrixXd> >(mu_0)).col(j);
        mu_x(i,j) = ((mu_j - x0).array().square()).sum();
      }
    }


    Map<MatrixXd> mu_ = as<Map<MatrixXd> >(mu_0);
    double min_mu_mu = (numeric_limits<double>::max)();
    for(int l1=0; l1<K-1; l1++){
      for(int l2=l1+1; l2<K;l2++){
        double mu_mu = ((mu_.col(l1)-mu_.col(l2)).array().square()).sum();
        if(mu_mu<min_mu_mu){
          min_mu_mu = mu_mu;
        }
      }
    }
    VectorXd colmeanX(p);
    for(int j=0;j<p;j++){
      colmeanX(j) = sliceVec(X,1,j,X_mask.col(j)).sum()/X_mask.col(j).sum();
    }
    MatrixXd v = colmeanX * MatrixXd::Ones(1,K) - mu_;
    double dis_mu=0;
    for(int k=0;k<K;k++){
      dis_mu = dis_mu + (v.col(k).array().square()).sum();
    }
    double Vk;
    Vk = ((w.array().square() * mu_x.array()).sum()+dis_mu/K)/min_mu_mu;
    return List::create(Named("paras_est") = List::create(Named("phi")=wrap(phi),
                              Named("mu")=mu,Named("sig2")=sig2),
                              Named("w")=wrap(w), Named("index_Vk")=Vk);
  }
  //compute the probability matrix w
  MatrixXd w(N,K);
  for (int i=0;i<N;i++){
    VectorXd fs_wgt = fws_fun(i,X,X_mask,X_mask_row,mu_0,sig2_0,phi_0);
    w.row(i) = fs_wgt.array() *  MatrixXd::Ones(K,1).array()/fs_wgt.sum();
  }

  // compute the index - Vk
  // mu_x:
  MatrixXd mu_x(N, K);
  for (int i=0;i<N;i++){
    VectorXd x0(p);
    MatrixXd sig0_1 = MatrixXd::Zero(p,p);
    MatrixXd sig0_2 = MatrixXd::Zero(X_mask_row(i),X_mask_row(i));
    MatrixXd I = MatrixXd::Identity(X_mask_row(i),X_mask_row(i));
    for (int j=0; j<K; j++){
      if (X_mask_row(i)==p){
        x0 = (as<Map<MatrixXd> >(X)).row(i);
      }else{
        sig0_2 = sliceMat(sig2_0[j],X_mask.row(i),X_mask.row(i)).ldlt().solve(I);
        MatrixXd temp1 = sliceMat(sig2_0[j],1-X_mask.row(i).array(),1-X_mask.row(i).array()) -
          sliceMat(sig2_0[j],1-X_mask.row(i).array(),X_mask.row(i).array()) *
          sig0_2 * sliceMat(sig2_0[j],X_mask.row(i).array(),1-X_mask.row(i).array());
        sig0_1 = replaceMat(sig0_1, temp1, 1-X_mask.row(i).array(), 1-X_mask.row(i).array());
        VectorXd cen = sliceVec(X,0,i,X_mask.row(i)) - sliceVec(mu_0,1,j,X_mask.row(i));
        MatrixXd temp2 = sliceVec(mu_0,1,j,1-X_mask.row(i).array()) +
          sliceMat(sig2_0(j),1-X_mask.row(i).array(),X_mask.row(i)) * sig0_2 * cen;
        x0 = replaceVec(x0,temp2,1-X_mask.row(i).array());
        x0 = replaceVec(x0,sliceVec(X,0,i,X_mask.row(i)),X_mask.row(i));
      }
      VectorXd mu_j = (as<Map<MatrixXd> >(mu_0)).col(j);
      mu_x(i,j) = ((mu_j - x0).array().square()).sum();
    }
  }


  Map<MatrixXd> mu_ = as<Map<MatrixXd> >(mu_0);
  double min_mu_mu = (numeric_limits<double>::max)();
  for(int l1=0; l1<K-1; l1++){
    for(int l2=l1+1; l2<K;l2++){
      double mu_mu = ((mu_.col(l1)-mu_.col(l2)).array().square()).sum();
      if(mu_mu<min_mu_mu){
        min_mu_mu = mu_mu;
      }
    }
  }
  VectorXd colmeanX(p);
  for(int j=0;j<p;j++){
    colmeanX(j) = sliceVec(X,1,j,X_mask.col(j)).sum()/X_mask.col(j).sum();
  }
  MatrixXd v = colmeanX * MatrixXd::Ones(1,K) - mu_;
  double dis_mu=0;
  for(int k=0;k<K;k++){
    dis_mu = dis_mu + (v.col(k).array().square()).sum();
  }
  double Vk;
  Vk = ((w.array().square() * mu_x.array()).sum()+dis_mu/K)/min_mu_mu;
  return List::create(Named("paras_est") = List::create(Named("phi")=wrap(phi),
                            Named("mu")=mu,Named("sig2")=sig2),
                      Named("w")=wrap(w), Named("index_Vk")=Vk);
}



