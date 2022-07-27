###two-dimensional FDNN classification
#################################################
##########Fourier basis function#################
#################################################
Fourier=function(s, M, j){
  k=j %/% 2
  
  if(j==1){
    return(1)
  }else if(j %% 2 == 0){
    return(sqrt(2/M)*cos(2*pi*k*s))
  }else if(j %% 2 != 0){
    return(sqrt(2/M)*sin(2*pi*k*s))
  }
}

##input
#S1: a vector of all grid points with length M1 for the 1st dimension
#S2: a vector of all grid points with length M2 for the 2nd dimension
#J1: number of truncated eigenvalues for the 1st dimension
#J2: number of truncated eigenvalues for the 2nd dimension
#D.train: training data list of K, each element is a data matrix nk.train by M1*M2
#D.test: testing data list of K, each element is a data matrix nk.test by M1*M2
#L: length of the DNN
#p: width of the DNN
#s: dropout rate
#epoch: epoch number
#batch: batch size
##return
#error: misclassification rate of the testing set

mfdnn.2d=function(D.train, D.test, J1, J2, S1, S2, L, p, s, epoch, batch){
  J=J1*J2; M1=length(S1); M2=length(S2); M=M1*M2
  n.train=sapply(D.train, function(x) dim(x)[1]); n.test=sapply(D.test, function(x) dim(x)[1])
  
  phi1=matrix(NA, M1, J1)
  for(m in 1:M1){
    for(j in 1:J1){
      phi1[m, j]=Fourier(S1[m], M1, j)
    }
  }
  phi2=matrix(NA, M2, J2)
  for(m in 1:M2){
    for(j in 1:J2){
      phi2[m, j]=Fourier(S2[m], M2, j)
    }
  }
  
  phi=t(kronecker(t(phi2), t(phi1))) 
  
  
  K=length(D.train)
  
  
  C.train=lapply(D.train, FUN = function(x) (x/M) %*% phi)
  C.test=lapply(D.test, FUN = function(x) (x/M) %*% phi)
  
  
  x_train=x_test=y_train=y_test=c()
  
  for(k in 1:K){
    x_train=rbind(x_train, C.train[[k]])
    x_test=rbind(x_test, C.test[[k]])
    y_train=c(y_train, rep(k-1, n.train[k]))
    y_test=c(y_test, rep(k-1, n.test[k]))
  }
  
  
  
  y_train=keras::to_categorical(matrix(y_train))
  y_test=keras::to_categorical(matrix(y_test))
  
 
  
  model=keras::keras_model_sequential()
  model %>% keras::layer_dense(units=p, activation = "relu", input_shape = c(J), kernel_initializer = "normal", constraint_maxnorm(max_value = 1, axis = 0))%>% 
    layer_dropout(rate = s)
  for(xx in 1:L){
    model %>% keras::layer_dense(units=p, activation = "relu", kernel_initializer = "normal", constraint_maxnorm(max_value = 1, axis = 0))%>% 
      layer_dropout(rate = s)
  }
  model %>% keras::layer_dense(units =K, activation = "softmax")  
  
  model %>% keras::compile(
    loss="categorical_crossentropy",
    optimizer=optimizer_adam(),
    metrics=c('accuracy')
  )
  
  
  history = model %>% keras::fit(
    x_train, y_train,
    epochs=epoch, batch_size=batch
  )
  
  y.pred=model %>% predict(x_test) %>% k_argmax()
  
  scores <- model %>% evaluate(x_test, y_test)
  
  E=1-scores[2]
  
  attributes(E)=NULL
  
  
  list(error=E)
}
