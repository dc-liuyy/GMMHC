\name{GMMHC}
\alias{GMMHC}
%- Also NEED an '\alias' for EACH other topic documented here.
\title{ GMMHC}
\description{
  The function provides a clustering method combining the agglomerative hierarchical clustering method and the Gaussian mixture model. 
}
\usage{
GMMHC(X_array, ks, min.nc=2, max.nc=15, maxiter = 100, 
                  distance = "euclidean", method="average",index_nb = "kl")
}
%- maybe also 'usage' for other objects documented here.
\arguments{
  \item{X_array}{ a 3-d array}
  \item{ks}{ the range of number of clusters for gaussian mixture model}
  \item{min.nc}{ minimal number of clusters for NbClust}
  \item{max.nc}{ maximal number of clusters for NbClust}
  \item{maxiter}{ the maximal steps for solving the gaussian mixture model with EM algorithm}
  \item{distance}{ parameter for NbClust. The distance measure to be used to compute the dissimilarity matrix. This must be one of: "euclidean", "maximum", "manhattan", "canberra", "binary", "minkowski" or "NULL". By default, distance="euclidean". If the distance is "NULL", the dissimilarity matrix (diss) should be given by the user. If distance is not "NULL", the dissimilarity matrix should be "NULL".}
  \item{method}{ parameter for NbClust. The cluster analysis method to be used. This should be one of: "ward.D", "ward.D2", "single", "complete", "average", "mcquitty", "median", "centroid", "kmeans".}
  \item{index_nb}{ parameter for NbClust. The index to be calculated. This should be one of : "kl", "ch", "hartigan", "ccc", "scott", "marriot", "trcovw", "tracew", "friedman", "rubin", "cindex", "db", "silhouette", "duda", "pseudot2", "beale", "ratkowsky", "ball", "ptbiserial", "gap", "frey", "mcclain", "gamma", "gplus", "tau", "dunn", "hubert", "sdindex", "dindex", "sdbw", "all" (all indices except GAP, Gamma, Gplus and Tau), "alllong" (all indices with Gap, Gamma, Gplus and Tau included).}
}
\value{
  \item{best_num }{ the best number of clusters}
  \item{class_est }{ partition that corresponds to the best number of clusters}
}
\note{ }
\seealso{}
\keyword{ Statistics}

