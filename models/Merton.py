# All the model for f(u,Du, D2u) driver
#  u_t + f(u,Du,D2u)= 0

import numpy as np
import tensorflow as tf
from scipy.stats import norm

class ModelMerton:
    
    # initial constructor
    def __init__(self, Init, muScal, sigScal, T, theta, lamb, eta):
        self.name = 'merton'
        self.muScal = muScal
        self.sigScal = sigScal
        self.d = 1
        self.T =T
        self.Init= Init
        self.theta = theta
        self.lamb= lamb
        self.R = np.sum(self.lamb **2 * self.theta ** 2)
        self.eta = eta           
    
    # driver for dW modelization
    # t    
    #
    # u     (nbSample , 1)
    # Du    (nbSample, d )
    # return (nbSample)
    def fDW( self, t, x, u, Du, D2u):        
        return  -self.R/2 * tf.einsum('i,i->i' ,tf.einsum('ij,ij->i', Du,Du), tf.reshape(tf.math.reciprocal(D2u), [tf.shape(D2u)[0]])) - tf.einsum('j,ij->i',tf.constant(self.muScal, dtype= tf.float32), Du)
    
    def fDWNumpy( self, t, x, u, Du, D2u):
        return  -self.R/2 * np.einsum('i,ijk->i' ,np.einsum('ij,ij->i', Du,Du), 1/D2u)  - np.einsum('j,ij->i',self.muScal, Du)
   
    # finla derivative
    def Dg(self,x):
        return -self.eta * self.g(x).reshape([x.shape[0],1])
    
    def DgTf(self,x):
        return -self.eta * tf.reshape(self.gTf(x),[tf.shape(x)[0],1])
    
    # finla derivative
    def D2g(self,x):
        return self.eta**2 * self.g(x).reshape([x.shape[0],1,1])
    def D2gTf(self,x):
        return self.eta**2 * tf.reshape(self.gTf(x), [tf.shape(x)[0],1,1])

    # permits to truncate the SDE value
    def truncate(self, istep , TStep, xInit, quantile):
        xMin = norm.ppf(1-quantile)
        xMax = norm.ppf(quantile)
        return  np.sqrt(TStep*istep)*xMin*self.sigScal , np.sqrt(TStep*istep)*xMax*self.sigScal    

    # x   (nbSim,d)
    # return  (nbSim))
    def g(self,x):
        return -np.exp(-self.eta*x).reshape([x.shape[0]])
    def gTf(self,x):
        return -1.0*tf.reshape(tf.math.exp(-self.eta*x),[tf.shape(x)[0]])

    def Sol( self,t,x):
        return self.g(x)*np.exp(-(self.T-t)*self.R/2)

    def derSol( self,t,x):
        return self.Dg(x).reshape([x.shape[0],1])*np.exp(-(self.T-t)*self.R/2)
    
    def der2Sol( self,t,x):
        return self.D2g(x)*np.exp(-(self.T-t)*self.R/2)

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
        return tf.einsum('j,i->ij',tf.constant(self.sigScal, dtype=tf.float32),tf.ones(shape= tf.shape(x)[0], dtype=tf.float32))

    def getValuesExt(self, t):
        randExt = np.tile( np.expand_dims(np.array([-3.,3.]),axis=1), [1, self.d])
        return np.tile(np.expand_dims(self.Init, axis=0),[2,1]) + \
            t*np.einsum('i,j->ij',np.ones([2]),self.muScal)+ \
            np.sqrt(t)*randExt*np.einsum('i,j->ij',np.ones([2]),self.sigScal)
    
    # calculate the present value with an Euler scheme
    def getValues(self, istep , TStep, xInit ,batchSize):
       return np.tile(np.expand_dims(xInit, axis=0),[batchSize,1]) + \
            TStep*istep*np.einsum('i,j->ij',np.ones([batchSize]),self.muScal) + \
            np.sqrt(TStep*istep)*np.einsum('ij,j->ij',np.random.normal(size=[batchSize,self.d]),self.sigScal)

    # get  Sigma^T D g
    def getVolByRandom(self,t , x ,random): 
        return np.einsum('ij,j->ij', random, self.sigScal)
    

    def getOneStepFromValuesDw(self,xPrev, t, TStep, wSigT):
        return xPrev + TStep*np.einsum('j,i->ij',self.muScal,np.ones([np.shape(xPrev)[0]])) +  wSigT

