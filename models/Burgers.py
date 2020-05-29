# All the model for f(u,Du, D2u) driver
#  u_t + f(u,Du,D2u)= 0

import numpy as np
import tensorflow as tf

# model for PDE  article XW 
class Burgers:
    
    # initial constructor
    def __init__(self, Init, muScal, sigScal, T, nu):
        self.name = 'burgers'
        self.muScal = muScal
        self.sigScal = sigScal
        self.d = 1
        self.T =T
        self.Init= Init
        self.nu = nu         
    
    # driver for dW modelization
    # t    
    #
    # u     (nbSample , 1)
    # Du    (nbSample, d )
    # return (nbSample)
    def fDW( self, t, x, u, Du, D2u):        
        return -1.0 * tf.einsum('i,ij->i',u,Du)
    
    def fDWNumpy( self, t, x, u, Du, D2u):
        return -1.0 *np.einsum('i,ij->i',u,Du)
   
    # finla derivative
    def DgNumpy(self,x):
        return -1.0*np.sin(x).reshape([x.shape[0],1])
    
    def DgTf(self,x):
        return -1.0*tf.reshape(tf.sin(x),[tf.shape(x)[0],1])
    
    # finla derivative
    def D2g(self,x):
        return -np.cos(x).reshape([x.shape[0],1,1])
    def D2gTf(self,x):
        return tf.reshape(-np.cos(x), [tf.shape(x)[0],1,1])    

    def drift(self, y):
        return 0. * y

    def driftTf(self, y):
        return 0. * y

    def forwardExpect(self, t, batchSize):
        return 0. * tf.ones((batchSize,self.d))

    # x   (nbSim,d)
    # return  (nbSim))
    def gNumpy(self,x):
        return np.cos(x) 
    def gTf(self,x):
        return tf.reshape(tf.cos(x),[tf.shape(x)[0]])

    def Sol( self,t,x, N = 100000):
        if t < self.T:
            values = np.random.normal(np.einsum('jk,i->ji',x,np.ones(N)), self.sigScal * np.sqrt( (self.T -t)),size=[x.shape[0],N])
            f = np.mean((np.einsum('jk,i->ji',x,np.ones(N)) - values)/(self.T -t) * np.exp(-np.sin(values)/self.sigScal**2), axis = -1)
            return f/np.mean(np.exp(-np.sin(values)/self.sigScal**2), axis = -1)    
        else:
            return self.gNumpy(x)
    
    def SolTf( self,t,x):
        return self.gTf(x)

    def derSol( self,t,x, N = 100000):
        if t < self.T:
            values = np.random.normal(np.einsum('jk,i->ji',x,np.ones(N)), self.sigScal * np.sqrt( (self.T -t)),size=[x.shape[0],N])
            f = np.mean((1/(self.T -t)-(np.einsum('jk,i->ji',x,np.ones(N)) -\
             values)**2/(self.sigScal*(self.T -t)**2))  * np.exp(-np.sin(values)/self.sigScal**2), axis = -1) * np.mean(np.exp(-np.sin(values)/self.sigScal**2), axis = -1) + np.mean((np.einsum('jk,i->ji',x,np.ones(N)) -\
             values)/(self.T -t) * np.exp(-np.sin(values)/self.sigScal**2), axis = -1)**2
            return np.expand_dims(f/np.mean(np.exp(-np.sin(values)/self.sigScal**2), axis = -1)**2,-1)
        else:
            return self.DgNumpy(x)
    
    def der2Sol( self,t,x):
        return self.D2g(x)

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

