# All the model for f(u,Du, D2u) driver
#  u_t + f(u,Du,D2u)= 0

import numpy as np
import tensorflow as tf
from scipy.stats import norm

class ModelMongeAmpere:
    
    # initial constructor
    def __init__(self, xyInit, muScal, sigScal, T, lamb, d):
        self.name = 'noleverage'
        self.muScal = muScal
        self.sigScal = sigScal
        self.T =T
        self.Init= xyInit     
        self.lamb  = lamb  
        self.d = d
    
    # driver for dW modelization
    # t    
    # u     (nbSample , 1)
    # Du    (nbSample, d )
    # return (nbSample)
    def fDW( self, t, x, u, Du, D2u):           
        return  tf.linalg.det(D2u) + 1 - tf.linalg.det(self.D2gTf(x))

    def fDWNumpy( self, t, x, u, Du, D2u):       
        return  np.linalg.det(D2u) + np.ones(x.shape[0]) - np.linalg.det(self.D2g(x))

    def drift(self, y):
        return 0. * y

    def driftTf(self, y):
        return 0. * y

    # finla derivative
    def Dg(self,x):        
        return np.einsum('ij,i->ij', x, -np.sin(np.sum(x, axis=1)/np.sqrt(self.d)))/np.sqrt(self.d)
    
    def ones(self,i):
        res = [0] * self.d
        res[i] = 1
        return res
    
    def DgTf(self,x):
        return tf.einsum('ij,i->ij', x,  -1.0*tf.sin(tf.reduce_sum(x, axis=1)/np.sqrt(self.d)))/np.sqrt(self.d)
    
    # finla derivative
    def D2g(self,x):        
        return np.einsum('jk,i->ijk', np.eye(self.d), -np.sin(np.sum(x, axis=1)/np.sqrt(self.d)))/np.sqrt(self.d) + np.einsum('ij,ik->ijk', x, np.einsum('ij,i->ij', x, -np.cos(np.sum(x, axis=1)/np.sqrt(self.d))))/(self.d)
        
    def D2gTf(self,x):
        return tf.einsum('jk,i->ijk', tf.eye(self.d), -1.0*tf.sin(tf.reduce_sum(x, axis=1)/np.sqrt(self.d)))/np.sqrt(self.d) + tf.einsum('ij,ik->ijk', x, tf.einsum('ij,i->ij', x,  -1.0*tf.cos(tf.reduce_sum(x, axis=1)/np.sqrt(self.d))))/(self.d)

    # permits to truncate the SDE value
    def truncate(self, istep , TStep, xInit, quantile):
        xMin = norm.ppf(1-quantile)
        xMax = norm.ppf(quantile)
        return  np.sqrt(TStep*istep)*xMin*self.sigScal , np.sqrt(TStep*istep)*xMax*self.sigScal

    # x   (nbSim,d)
   # return  (nbSim))
    def g(self,x):
        return np.cos(np.sum(x, axis=1)/np.sqrt(self.d))

    def gTf(self,x):
        return tf.cos(tf.reduce_sum(x, axis=1)/np.sqrt(self.d))

    def Sol( self,t,x):
        return self.g(x) + (self.T-t) * np.ones(x.shape[0])
    
    def SolTf( self,t,x):
        return  self.gTf(x) + (self.T-t) * tf.tile(tf.ones(1), (tf.shape(x)[0]))

    def derSol( self,t,x):
        return self.Dg(x)
    
    def der2Sol( self,t,x):
        return self.D2g(x)


    def control(self, t, x, u, Du, D2u):
        return 0. * x

    # renormalization trend
    def renormTrend(self, t):
        return np.float32(0. * self.muScal)

    # renormalize sig
    def renormSigma(self, t):
        return self.sigScal[0]*np.sqrt(t)*tf.ones(shape= [self.d], dtype=tf.float32)

    # sigma
    # t time doubl
    # x  array size (nbsample,d)
    def sig(self,t, x):
        return tf.einsum('j,i->ij',tf.constant(self.sigScal, dtype=tf.float32),tf.ones(shape= tf.shape(x)[0], dtype=tf.float32))

    def forwardExpect(self, t, batchSize):
        return 0. * tf.ones((batchSize,self.d))

    
    # calculate the present value with an Euler scheme
    def getValues(self, istep , TStep, xInit ,batchSize):
        x0 = np.tile(np.expand_dims(xInit, axis=0),[batchSize,1])\
            + np.sqrt(istep*TStep)*np.einsum('ij,j->ij',np.random.normal(size=[batchSize,self.d]),self.sigScal)

        return x0        

    # get  Sigma^T D g
    def getVolByRandom(self,t , x ,random): 
        return np.einsum('ij,j->ij', random, self.sigScal)

    def getOneStepFromValuesDw(self,xPrev, t, TStep, wSigT):
        return xPrev +   wSigT
