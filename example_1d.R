##one-dimensional functional data example
library(keras)
library(tensorflow)
#sample size
n0=n1=n2=300
n0.train=ceiling(n0*0.7); n1.train=ceiling(n1*0.7); n2.train=ceiling(n2*0.7)
n0.test=n0-n0.train; n1.test=n1-n1.train; n2.test=n2-n2.train
##generate data
#gererate grid points
S=seq(0, 1, length.out=50)
#rate for exponential distribution
r0=c(0.1, 0.3, 0.5)
#df and non-central parameters for t distribution
df1=c(3, 5, 7)
ncp1=c(3,3,3)
#sd and mean for normal distribution
eigen2=c(2.5, 2, 1.5)
mu2=c(0,0,0)
#generate projection scores
xi0=cbind(rexp(n0, r0[1]), rexp(n0, r0[2]), rexp(n0, r0[3]))#exponential 
xi1=cbind(rt(n1, df1[1], ncp=ncp1[1]), rt(n1, df1[2], ncp=ncp1[2]), rt(n1, df1[3], ncp=ncp1[3]))#student's t 
xi2=cbind(rnorm(n2, mu2[1], eigen2[1]), rnorm(n2, mu2[2], eigen2[2]), rnorm(n2, mu2[3], eigen2[3]))#normal
#generate basis functions
BB1=log10(S+2); BB2=S; BB3=S^3
BB=rbind(BB1, BB2, BB3)
#generate discretely observed curves
Data=list(as.matrix(xi0%*%BB), as.matrix(xi1%*%BB), as.matrix(xi2%*%BB))
#Randomly assign training samples
index.0=sample(1:n0, n0.train)
index.1=sample(1:n1, n1.train)
index.2=sample(1:n2, n2.train)
#create train and test data list
D.train=D.test=list()
D.train[[1]]=Data[[1]][index.0, ]
D.train[[2]]=Data[[2]][index.1, ]
D.train[[3]]=Data[[3]][index.2, ]
D.test[[1]]=Data[[1]][-index.0, ]
D.test[[2]]=Data[[2]][-index.1, ]
D.test[[3]]=Data[[3]][-index.2, ]



#Call mfdnn function
source("dnn_1d_par.R")
source("dnn_1d.R")
#set up candidates for hyperparameters 
J=c(10,20); L=c(2,3); p=c(300,500); s=c(0.1,0.3)
#selection for hyperparameters
r1.v=mfdnn.1d.par(D.train, J, S, L, p, s, epoch=300, batch=20)
optimal=which(r1.v$error == min(r1.v$error), arr.ind = TRUE)[1,]
J=J[optimal[1]]; L=L[optimal[2]]; p=p[optimal[3]]; s=s[optimal[4]]
#fit mfdnn model
r1=mfdnn.1d(D.train, D.test, J, S, L, p, s, epoch=300, batch=20)
r1$error