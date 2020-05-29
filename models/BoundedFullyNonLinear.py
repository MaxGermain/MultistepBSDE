# All the model for f(u,Du, D2u) driver
#  u_t + f(u,Du,D2u)= 0

import numpy as np
import tensorflow as tf
from scipy.stats import norm

# model for PDE  article XW 
class BoundedFNL:
    
    # initial constructor
    def __init__(self, Init, muScal, sigScal, rescal, alpha ,d, T):
        self.muScal = muScal
        self.sigScal = sigScal
        self.rescal = rescal
        self.d = d # dimension
        self.alpha =alpha
        self.T =T
        self.Init= Init
        self.rho =0.
        self.name = 'bounded FNL'        
        
    
    # driver for dW modelization
    # t    
    #
    # u     (nbSample , 1)
    # Du    (nbSample, d )
    # return (nbSample)
    def fDW( self, t, x, u, Du, D2u):
        expMat = tf.exp(self.alpha*(self.T-t))
        xSum=tf.reduce_sum(x,axis=-1)
        tfCos = tf.cos(xSum)
        GamDiag = tf.linalg.diag_part(D2u)
        return  tfCos*self.alpha*expMat + self.rescal* tfCos* tfCos*expMat *expMat + self.rescal *  u * tf.reduce_mean(GamDiag,axis=1)  - self.muScal*tf.reduce_sum(Du, axis=-1)
    
    def fDWNumpy( self, t, x, u, Du, D2u):
        expMat = np.exp(self.alpha*(self.T-t))
        xSum=np.sum(x,axis=-1)
        tfCos = np.cos(xSum)
        return  tfCos*self.alpha*expMat + self.rescal* tfCos* tfCos*expMat *expMat + self.rescal *  u * np.einsum("jii->j",D2u)/self.d - self.muScal*np.sum(Du, axis=-1)
    

   
    # finla derivative
    def Dg(self,x):
        return -np.tile(np.expand_dims(np.sin(np.sum(x, axis=-1)),axis=1),[1,self.d])

    # finla derivative
    def DgTf(self,x):
        return -tf.tile(tf.expand_dims(tf.sin(tf.reduce_sum(x, axis=-1)),axis=1),[1,self.d])
    
    # finla derivative
    def D2g(self,x):
        return -np.tile(np.expand_dims(np.tile(np.expand_dims(np.cos(np.sum(x, axis=-1)),axis=1),[1,self.d]), axis=2),[1,1,self.d])
    def D2gTf(self,x):
        return -tf.tile(tf.expand_dims(tf.tile(tf.expand_dims(tf.cos(tf.reduce_sum(x, axis=-1)),axis=1),[1,self.d]), axis=2),[1,1,self.d])

    # permits to truncate the SDE value
    def truncate(self, istep , TStep, xInit, quantile):
        xMin = norm.ppf(1-quantile)
        xMax = norm.ppf(quantile)
        return  np.sqrt(TStep*istep)*xMin*self.sigScal , np.sqrt(TStep*istep)*xMax*self.sigScal    

    # x   (nbSim,d)
    # return  (nbSim))
    def g(self,x):
        return np.cos(np.sum(x, axis=-1))
    def gTf(self,x):
        return tf.cos(tf.reduce_sum(x, axis=-1))

    def Sol( self,t,x):
        return np.cos(np.sum(x, axis=-1))*np.exp(self.alpha*(self.T-t))

    def derSol( self,t,x):
        return -np.tile(np.expand_dims(np.sin(np.sum(x, axis=-1))*np.exp(self.alpha*(self.T-t)), axis=1),[1,self.d])
    
    def der2Sol( self,t,x):
        return -np.tile(np.expand_dims(np.tile(np.expand_dims(np.cos(np.sum(x, axis=-1))*np.exp(self.alpha*(self.T-t)), axis=1),[1,self.d]), axis=2),[1,1, self.d]) 

    # renormalization trend
    def renormTrend(self, t):
        return self.muScal*t*tf.ones(shape= [self.d], dtype=tf.float32)

    # renormalizae sig
    def renormSigma(self, t):
        return self.sigScal*np.sqrt(t)*tf.ones(shape= [self.d], dtype=tf.float32)

    # sigma
    # t time doubl
    # x  array size (nbsample,d)
    def sig(self,t, x):
        return self.sigScal*tf.ones(shape= tf.shape(x), dtype=tf.float32)

    def getValuesExt(self, t):
        randExt = np.tile( np.expand_dims(np.array([-3.,3.]),axis=1), [1, self.d])
        return np.tile(np.expand_dims(self.Init, axis=0),[2,1]) + \
            self.muScal*t*np.ones([2,self.d]) + \
            np.sqrt(t)*randExt*self.sigScal
    
    # calculate the present value with an Euler scheme
    def getValues(self, istep , TStep, xInit ,batchSize):
        return np.tile(np.expand_dims(xInit, axis=0),[batchSize,1]) + \
            self.muScal*TStep*np.ones([batchSize,self.d])*istep + \
            np.sqrt(TStep*istep)*np.random.normal(size=[batchSize,self.d])*self.sigScal


    # get  Sigma^T D g
    def getVolByRandom(self,t , x ,random):
        return random*self.sigScal
    

    def getOneStepFromValuesDw(self,xPrev, t, TStep, wSigT):
        return xPrev + self.muScal*TStep*np.ones([np.shape(xPrev)[0],self.d]) +  wSigT

