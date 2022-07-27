##two-dimensional functional data example
library(keras)
library(tensorflow)
#sample size
n0=n1=n2
n0.train=ceiling(n0*0.7); n1.train=ceiling(n1*0.7); n2.train=ceiling(n2*0.7)
n0.test=n0-n0.train; n1.test=n1-n1.train; n2.test=n2-n2.train
#Generate data
S1=seq(0, 1, length.out=50)
S2=seq(0, 1, length.out=50)
#rate for exponential distribution
r0=c(0.1, 0.3, 0.5, 0.7, 0.9)
#df and non-central parameters for t distribution
df1=c(3, 5, 7, 9, 11)
ncp1=c(3,3,3,3,3)
#sd and mean for normal distribution
eigen2=c(2.5, 2, 1.5, 1, .5)
mu2=c(0,0,0,0,0)
#generate curve
xi0=cbind(rexp(n0, r0[1]), rexp(n0, r0[2]), rexp(n0, r0[3]), rexp(n0, r0[4]), rexp(n0, r0[5]))
xi1=cbind(rt(n1, df1[1], ncp=ncp1[1]), rt(n1, df1[2], ncp=ncp1[2]), rt(n1, df1[3], ncp=ncp1[3]), rt(n1, df1[4], ncp=ncp1[4]), rt(n1, df1[4], ncp=ncp1[5]))
xi2=cbind(rnorm(n2, mu2[1], eigen2[1]), rnorm(n2, mu2[2], eigen2[2]), rnorm(n2, mu2[3], eigen2[3]), rnorm(n2, mu2[4], eigen2[4]), rnorm(n2, mu2[5], eigen2[5]))


SS1=rep(S1, 50); SS2=rep(S2, each=50)


BB1=SS1; BB2=SS2; BB3=(SS1)*(SS2); BB4=(SS1)^2; BB5=(SS2)^2 


BB=rbind(BB1, BB2, BB3, BB4, BB5)


Data=list(as.matrix(xi0%*%BB), as.matrix(xi1%*%BB), as.matrix(xi2%*%BB))

#Randomly assign training samples
index.0=sample(1:n0, n0.train)
index.1=sample(1:n1, n1.train)
index.2=sample(1:n2, n2.train)

D.train=D.test=list()
D.train[[1]]=Data[[1]][index.0, ]
D.train[[2]]=Data[[2]][index.1, ]
D.train[[3]]=Data[[3]][index.2, ]

D.test[[1]]=Data[[1]][-index.0, ]
D.test[[2]]=Data[[2]][-index.1, ]
D.test[[3]]=Data[[3]][-index.2, ]



#Call M_dnn function
source("dnn_2d_par.R")
source("dnn_2d.R")
J1=J2=c(5,6); L=c(2,3); p=c(300,500); s=c(0.2,0.3)
epoch=300; batch=20
r1.cv=mfdnn.2d.par(D.train, J1, J2, S1, S2, L, p, s, epoch, batch)
optimal=which(r1.cv$error == min(r1.cv$error), arr.ind = TRUE)[1,]
J1=J1[optimal[1]]; J2=J2[optimal[2]]; L=L[optimal[3]]; p=p[optimal[4]]; s=s[optimal[5]]
r1=mfdnn.2d(D.train, D.test, J1, J2, S1, S2, L, p, s, epoch, batch)
r1$error

