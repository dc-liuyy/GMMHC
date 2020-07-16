# GMMHC

## Installation

```R
#install.packages("devtools")
library(devtools)
install_github("dc-liuyy/GMMHC")
```

## Example

```R
library(mnormt)
library(GMMHC)
generate_data <- function(N,p,d,mr){
  X_array <- array(dim=c(N, p, d))
  phi_real <- c(1/3,1/3,1/3)
  num_real <- rmultinom(1,N,phi_real)
  class_real <- c(rep(1,num_real[1]),rep(2,num_real[2]),rep(3,num_real[3]))
    
  rho <- 0.5
  col1 <- cumprod(c(1,rep(rho,p-1)))
  col2 <- cumprod(c(1,rep(1/rho,p-1)))    
  cov_X <- col1 %*% t(col2)
  cov_X[upper.tri(cov_X)] <- rev(cov_X[lower.tri(cov_X)])
    
  for (i in 1:d){   
    X_array[1:num_real[1],,i] <- rmnorm(num_real[1], rep(90, p), cov_X*9 )
    
    X_array[(1:num_real[2])+num_real[1],,i] <- rmnorm(num_real[2], rep(100, p), cov_X*9)
    
    X_array[(1:num_real[3])+(num_real[2]+num_real[1]),,i] <- rmnorm(num_real[3], rep(110, p), cov_X*9)
  }
  
  nm <- round(N*p*d*mr) 
  m <- cbind(sample(1:N,nm,replace = TRUE),sample(1:p,nm,replace = TRUE),sample(1:d,nm,replace = TRUE))
  X_array[m] <-NA
  return(list(X_array=X_array, class_real=class_real))
}


N <- 1000
p <- 5
d <- 3
mr <- 0.1
X_gen <- generate_data(N,p,d,mr)
X_array <- X_gen$X_array

ks <- c(2,3,4,5)
result <- GMMHC(X_array, ks, min.nc=2, max.nc=5)
```

