# All the model for f(u,Du, D2u) driver
#  u_t + f(u,Du,D2u)= 0

import numpy as np
import tensorflow as tf
from scipy.stats import norm

class Bounded:
    
    # initial constructor
    def __init__(self, Sig, T, xInit, muScal, alpha) :
        self.Sig = Sig #  scalar
        self.d= np.shape(xInit)[0] # dimension
        self.T =T
        self.Init = xInit
        self.sigScal =  Sig

        self.rescal = 0.5

        self.muScal = muScal
        self.alpha = alpha

        self.iu = 2
        self.idu = 2


    # modify exploration sigma
    def modifySigExplore( self, SigExplore):
        self.SigExplore = SigExplore
        
    # driver
    # t    
    #
    # u     (nbSample , 1)
    # Du    (nbSample, d )
    # return (nbSample)
    def fDW( self, t, x, u, Du, D2u):
        expMat = tf.exp(self.alpha*(self.T-t))
        xSum=tf.reduce_sum(x,axis=-1)
        nbSamp = tf.shape(x)[-2]
        return  (tf.cos(xSum)*(self.alpha+ 0.5*self.Sig * self.Sig*self.d ) + tf.sin(xSum)*self.muScal*self.d)*expMat   \
            - self.rescal*tf.multiply( tf.pow(tf.cos(xSum)*expMat,self.iu),tf.pow(-tf.sin(xSum)*expMat,self.idu)) \
            + self.rescal*tf.multiply( tf.pow(u,self.iu), tf.pow(tf.reduce_mean(Du,axis=-1),self.idu))

    # numpy eqivalent of f
    def fNumpy( self, t, x, u, Du):
        expMat = np.exp(self.alpha*(self.T-t))
        xSum= np.sum(x,axis=-1)
        nbSamp = np.shape(x)[-2]
        ret = (np.cos(xSum)*(self.alpha+ 0.5*self.Sig * self.Sig*self.d ) + np.sin(xSum)*self.muScal*self.d)*expMat   \
            - self.rescal*np.multiply( np.power(np.cos(xSum)*expMat,self.iu),np.power(-np.sin(xSum)*expMat,self.idu))+\
            self.rescal*np.multiply( np.power(u,self.iu), np.power(np.mean(Du,axis=-1),self.idu))
        return  ret.squeeze()

    # final  
    # x   (nbsample,d,nbSim)
    # return  (nbSample,nbSim))
    def gTf(self,x):
        return tf.cos(tf.reduce_sum(x, axis=-1))
   
    # finla derivative
    def DgTf(self,x):
        a = tf.range(1, self.d + 1, dtype=tf.float32)
        return -tf.tile(tf.expand_dims(tf.sin(tf.reduce_sum(x, axis=-1)),axis=1),[1,self.d])
    
    # final  Numpy 
    # x   (nbSim,d)
    # return  (nbSim))
    def g(self,x):
        return np.cos(np.sum(x, axis=-1))

    def Dg(self,x):
        return -np.sin(np.sum(x, axis=-1))
    
    # sol for one point
    def SolPoint(self,t,x):
        return np.cos(np.sum(x, axis=-1))*np.exp(self.alpha*(self.T-t))
    
    def Sol( self,t,x):
        return np.cos(np.sum(x, axis=-1))*np.exp(self.alpha*(self.T-t))
          
    def derSol( self,t,x):
        return -np.tile(np.expand_dims(np.sin(np.sum(x, axis=-1))*np.exp(self.alpha*(self.T-t)), axis=1),[1,self.d])
     
    # same for Feyman Kac
    def fNumpyFeyn( self, t, x, u, Du):
        return self.fNumpy(t, x, u, Du)
    
 
    # renormalization trend
    def renormTrend(self, t):
        return self.muScal*t*tf.ones(shape= [self.d], dtype=tf.float32)

    # renormalizae sig
    def renormSigma(self, t):
        return self.Sig*np.sqrt(t)*tf.ones(shape= [self.d], dtype=tf.float32)

    # calculate the present value with an Euler scheme
    def getValues(self, istep , TStep, Init ,batchSize):
        return np.tile(np.expand_dims(Init, axis=0),[batchSize,1]) + \
            self.muScal*TStep*np.ones([batchSize,self.d])*istep + \
            np.sqrt(TStep*istep)*np.random.normal(size=[batchSize,self.d])*self.Sig
    
    def der2Sol( self,t,x):
        return -np.tile(np.expand_dims(np.sin(np.sum(x, axis=-1))*np.exp(self.alpha*(self.T-t)), axis=1),[self.d,self.d])

    # get  Sigma^T D g
    def getVolByRandom(self,t , x ,random):
        return random*self.Sig
    

    # sigma at a given date (for Euler schem)
    # t time double
    # x  array size (nbsample,d)
    def sig(self,t, x):
        return self.Sig*tf.ones(tf.shape(x))
 
    # permits to truncate the SDE value
    def truncate(self, istep , TStep, Init, quantile):
        expAA = np.exp( np.diag(self.A*istep*TStep))
        sigAA = self.Sig*np.sqrt((expAA*expAA-np.ones(self.d))/(2.*np.diag(self.A)))
        xMin = norm.ppf(1-quantile)
        xMax = norm.ppf(quantile)
        return  expAA* Init + sigAA* xMin,  expAA* Init + sigAA* xMax
    
     # one step of values
    def getOneStepFromValues(self,xPrev, t, TStep):
        return xPrev + self.muScal*TStep*np.ones([np.shape(xPrev)[0],self.d]) + np.sqrt(TStep)*np.random.normal(size=[np.shape(xPrev)[0],self.d])*self.sigScal

    def getOneStepFromValuesDw(self,xPrev, t, TStep, wSigT):
        return xPrev + self.muScal*TStep*np.ones([np.shape(xPrev)[0],self.d]) +  wSigT
