library(SemiPar)
library(mgcv)
# library(refund)

yvec <- read.csv(file='/N/u/liyuny/Carbonate/cnode_ffr_main/results/deterministic_lv/vnode/sim_b2/y.csv',header=FALSE)
t <- read.csv(file='/N/u/liyuny/Carbonate/cnode_ffr_main/results/deterministic_lv/vnode/sim_b2/t.csv',header=FALSE)
s <- read.csv(file='/N/u/liyuny/Carbonate/cnode_ffr_main/results/deterministic_lv/vnode/sim_b2/s.csv',header=FALSE)
odeX <- read.csv(file='/N/u/liyuny/Carbonate/cnode_ffr_main/results/deterministic_lv/vnode/sim_b2/odeX.csv',header=FALSE)

yvec <- as.vector(yvec$V1)
n = 300
t <- t$V1
s <- s$V1
tngrid = length(t)
sngrid = length(s)
# fit the model
tmat1<- matrix(t,nrow=tngrid, nc=sngrid, byrow=FALSE)
tmat2<- matrix(t,nrow=tngrid, nc=sngrid, byrow=FALSE)

smat<- matrix(s,nrow=tngrid,nc=sngrid, byrow=TRUE)
rmat<- matrix(s,nrow=tngrid,nc=sngrid, byrow=TRUE)

Lmat1 <- matrix(rep(odeX$V1,tngrid),nc=sngrid)
Lmat2 <- matrix(rep(odeX$V2,tngrid),nc=sngrid)

tvec <- matrix(t$V1,nc=1,byrow=FALSE)

m100 <- bam(yvec ~s(tvec,bs="ps",k=12)+te(tmat1,smat,by=Lmat1, bs="ps")+te(tmat2,rmat,by=Lmat2, bs="ps",k=7),method="REML")

################################################################
##              for  METHOD 1    pffr                         ##
################################################################

###### obtain INTERCEPT 

term0 <- m100$smooth[[1]]

x=t

predDat <- data.frame(x)

names(predDat) <- c(term0$term)

PX0 <- PredictMat(term0, dat=predDat)

m100_hatBeta0_func<- matrix(PX0%*%m100$coefficients[term0$first.para:term0$last.para], ncol=length(t))

m100_hatBeta0<-m100_hatBeta0_func+m100$coefficients[1]

###### obtain BETA1

term1 <- m100$smooth[[2]]

x=rep(t, rep(length(s), length(t)))

predDat <- data.frame(x=rep(t, rep(length(s), length(t))), y=rep(s, length(t)), by=rep(1, length(t)*length(x)))

names(predDat) <- c(term1$term, term1$by)

PX <- PredictMat(term1, dat=predDat)


m100_hatBeta1_sparse <- matrix(PX%*%m100$coefficients[term1$first.para:term1$last.para], nrow=length(s), ncol=length(t))

plot(m100, select=2, scheme=1, theta=35, phi=32, col='grey80',ticktype = "detailed",
     xlab='t',ylab='s',main= "beta1(t,s)")

plot(m100, select=3, scheme=1, theta=35, phi=32, col='grey80',ticktype = "detailed",
     xlab='t',ylab='s',main= "beta2(t,s)")
#######################################################################
######################### Pointwise CI for trueBeta0(t)  ##################
#######################################################################


Sigma_Beta0<-m100$Vp[term0$first.para:term0$last.para,term0$first.para:term0$last.para]
PX0<-as.matrix(PX0)

varBeta0<-matrix(0,nrow=length(t),ncol=1)
lBoundBeta0<-matrix(0,nrow=length(t),ncol=1)
uBoundBeta0<-matrix(0,nrow=length(t),ncol=1)

for(i in 1:(length(t))){
  varBeta0[i]<-t(PX0[i,])%*%Sigma_Beta0%*%PX0[i,]
  lBoundBeta0[i] <- as.vector(m100_hatBeta0[b,])[i] - 1.96*sqrt(varBeta0[i])
  uBoundBeta0[i] <- as.vector(m100_hatBeta0[b,])[i] + 1.96*sqrt(varBeta0[i])
}


