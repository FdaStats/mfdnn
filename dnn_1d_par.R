###one-dimensional FDNN classification: cross validation for hyperparameters selection
#################################################
##########Fourier basis function#################
#################################################
Fourier=function(s, M, j){
  k=j %/% 2
  
  if(j==1){
    return(rep(1,M))
  }else if(j %% 2 == 0){
    return(sqrt(2/M)*cos(2*pi*k*s))
  }else if(j %% 2 != 0){
    return(sqrt(2/M)*sin(2*pi*k*s))
  }
}
##input
#S: a vector of all grid points with length M
#J: candidate number of truncated eigenvalues
#D.train: training data list of K, each element is a data matrix nk.train by M1*M2
#L: candidate vector for length of the DNN
#p: candidate vector for width of the DNN
#s: candidate vector for dropout rate
#epoch: epoch number
#batch: batch size
##return
#error: misclassification rate of the testing set

mfdnn.1d.par=function(D.train, J, S, L, p, s, epoch, batch){
  M=length(S); K=length(D.train)
  
  n.train=lapply(D.train, function(x) dim(x)[1])
  ind.train=lapply(n.train, function(x) sample(1:x, floor(0.8*x)))
  n.train.cv=sapply(ind.train, length)
  n.test.cv=unlist(n.train)-n.train.cv
  
  D.train.cv=D.test.cv=vector("list", K)
  x_train.cv=x_test.cv=y_train.cv=y_test.cv=c()
  for(k in 1:K){
    D.train.cv[[k]]=D.train[[k]][ind.train[[k]],]
    D.test.cv[[k]]=D.train[[k]][-ind.train[[k]],]
  }
  
    
  l1=length(J); l2=length(L); l3=length(p); l4=length(s) 
  
  error=array(NA, c(l1,l2,l3,l4))
  

  for(ii in 1:l1){
    for(jj in 1:l2){
      for(kk in 1:l3){
        for(ll in 1:l4){
          J.cv=J[ii]; L.cv=L[jj]; p.cv=p[kk]; s.cv=s[ll]
          
          
          phi.cv=c()
          for(j in 1:J.cv){
            phi.cv=cbind(phi.cv, Fourier(S,M,j))
          }
          
          C.train.cv=lapply(D.train.cv, FUN = function(x) (x/M) %*% phi.cv)
          C.test.cv=lapply(D.test.cv, FUN = function(x) (x/M) %*% phi.cv)
          
          
          x_train.cv=x_test.cv=y_train.cv=y_test.cv=c()
          
          for(k in 1:K){
            x_train.cv=rbind(x_train.cv, C.train.cv[[k]])
            x_test.cv=rbind(x_test.cv, C.test.cv[[k]])
            y_train.cv=c(y_train.cv, rep(k-1, n.train.cv[k]))
            y_test.cv=c(y_test.cv, rep(k-1, n.test.cv[k]))
          }
          
          
          
          y_train.cv=keras::to_categorical(matrix(y_train.cv))
          y_test.cv=keras::to_categorical(matrix(y_test.cv))
          
          
          model=keras::keras_model_sequential()
          model %>% keras::layer_dense(units=p.cv, activation = "relu", input_shape = c(J.cv), kernel_initializer = "normal", constraint_maxnorm(max_value = 1, axis = 0))%>% 
            layer_dropout(rate = s.cv)
          for(xx in 1:L.cv){
            model %>% keras::layer_dense(units=p.cv, activation = "relu", kernel_initializer = "normal", constraint_maxnorm(max_value = 1, axis = 0))%>% 
              layer_dropout(rate = s.cv)
          }
          model %>% keras::layer_dense(units =K, activation = "softmax")  
          
          model %>% keras::compile(
            loss="categorical_crossentropy",
            optimizer=optimizer_adam(),
            metrics=c('accuracy')
          )
          
          
          history = model %>% keras::fit(
            x_train.cv, y_train.cv,
            epochs=epoch, batch_size=batch
          )
          
          y.pred.cv=model %>% predict(x_test.cv) %>% k_argmax()
          
          scores.cv <- model %>% evaluate(x_test.cv, y_test.cv)
          
          E.cv=1-scores.cv[2]
          
          attributes(E.cv)=NULL
          
          
          error[ii,jj,kk,ll]=E.cv
         }
        }
      }
    }
  
  
  list(error=error)
}
