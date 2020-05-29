# All the model for f(u,Du, D2u) driver
#  u_t + f(u,Du,D2u)= 0

import numpy as np
import tensorflow as tf
from scipy.stats import norm

class SemiUnbounded:
    
    # initial constructor
    def __init__(self,  Sig, T, xInit) :
        self.Sig = Sig #  scalar
        self.d= np.shape(xInit)[0] # dimension
        self.T =T
        self.Init = xInit
        self.sigScal=  Sig
        self.muScal = 0.

        self.rescal = 0.5 


    # modify exploration sigma
    def modifySigExplore( self, sigScal):
        self.sigScal = sigScal
        
    
    def fDW( self, t, x, u, Du, D2u):
        # u time derivative
        Ut = - tf.reduce_mean(tf.where(x < 0,  tf.sin(x),x), axis=-1)
        # u value
        xSum= tf.einsum("ij,j->i",x,tf.convert_to_tensor(np.arange(1.,self.d+1.),dtype=tf.float32))
        cosU= tf.cos(xSum)
        UVal = -(self.T-t) * Ut+ cosU
        # U X derivarive (sum)
        DUVal = (self.T-t)*tf.reduce_mean(tf.where(x < 0,  tf.cos(x),tf.ones(tf.shape(x))), axis=-1) - self.d*(self.d+1.)/2.* tf.sin(xSum)
        # sum of diag of Hessian
        D2UVal = -cosU* self.d*(self.d+1)*(2*self.d+1)/6. - (self.T-t)*tf.reduce_mean(tf.where(x < 0,  tf.sin(x),tf.zeros(tf.shape(x))), axis=-1)
       
        return  - Ut - 0.5*self.Sig*self.Sig*D2UVal  - self.rescal*(UVal*DUVal/self.d + UVal*UVal)+ self.rescal*( tf.pow(u,2.) + tf.multiply(u,tf.reduce_mean(Du,axis=-1))) 
        
    # final  
    # x   (nbsample,d,nbSim)
    # return  (nbSample,nbSim))
    def gTf(self,x):
        a = tf.range(1, self.d + 1, dtype=tf.float32)
        return tf.cos(tf.einsum("li,i->l ", x, a))
   
    # finla derivative
    def DgTf(self,x):
        a = tf.range(1, self.d + 1, dtype=tf.float32)
        return tf.einsum("l,i->li ", tf.sin(tf.einsum("li,i->l ", x, a)), -a)
    
    # final  Numpy 
    # x   (nbSim,d)
    # return  (nbSim))
    def gNumpy(self,x):
        a = 1.0 * np.arange(1, self.d + 1)
        return np.cos(np.einsum("li,i->l ", x, a))

    def DgNumpy(self,x):
        a = 1.0 * np.arange(1, self.d + 1)
        return np.einsum("l,i->li ", np.sin(np.einsum("li,i->l ", x, a)), -a)
    
    # sol for one point
    def SolPoint(self,t,x):
        a = 1.0 * np.arange(1, self.d + 1)        
        return (self.T-t) * np.mean(np.where(x < 0,  np.sin(x),x), axis=-1) + np.cos(np.einsum("i,i-> ", x, a))
    
    def Sol( self,t,x):
        a = 1.0 * np.arange(1, self.d + 1)        
        return (self.T-t) * np.mean(np.where(x < 0,  np.sin(x),x), axis=-1) + np.cos(np.einsum("ji,i->j ", x, a))

          
    def derSol( self,t,x):
        a =np.arange(1,self.d+1)
        xSum= np.matmul(x,a)
        res=  (self.T-t) * np.where(x < 0,  np.cos(x),1.)/self.d  - np.tile(np.expand_dims(np.asarray(np.sin(xSum)), axis=1), [1,self.d])*a
        return res

    # numpy eqivalent of f
    def fNumpy( self, t, x, u, Du):
        # u time derivative
        Ut = - np.mean(np.where(x < 0,  np.sin(x),x), axis=-1)
        # u value
        xSum= np.einsum("ij,j->i",x,np.arange(1.,self.d+1.))
        cosU= np.cos(xSum)
        UVal = -(self.T-t) * Ut+ cosU
        # U X derivarive (sum)
        DUVal = (self.T-t)*np.mean(np.where(x < 0,  np.cos(x),np.ones(np.shape(x))), axis=-1) - self.d*(self.d+1.)/2.* np.sin(xSum)
        # sum of diag of Hessian
        D2UVal = -cosU* self.d*(self.d+1)*(2*self.d+1)/6. - (self.T-t)*np.reduce_mean(np.where(x < 0,  np.sin(x),np.zeros(np.shape(x))), axis=-1)
       
        ret =  - Ut- 0.5*self.Sig*self.Sig*D2UVal - self.rescal*(UVal*DUVal/self.d + UVal*UVal)+ self.rescal*( np.pow(u,2.) + np.multiply(u,np.reduce_mean(Du,axis=-1)))
        return  ret.squeeze()
                            

    # same for Feyman Kac
    def fNumpyFeyn( self, t, x, u, Du):
        return self.fNumpy(t, x, u, Du)
    
 
    # renormalization trend
    def renormTrend(self, t):
        return tf.zeros(shape= [self.d], dtype=tf.float32)

    # renormalizae sig
    def renormSigma(self, t):
        return self.Sig*np.sqrt(t)*tf.ones(shape= [self.d], dtype=tf.float32)

    # calculate the present value with an Euler scheme
    def getValues(self, istep , TStep, Init ,batchSize):
        return np.tile(np.expand_dims(Init, axis=0),[batchSize,1]) + np.sqrt(TStep*istep)*np.random.normal(size=[batchSize,self.d])*self.Sig

    
    def der2Sol( self,t,x):
        xSum = np.einsum("lj,j->l",x,np.arange(1.,self.d+1.))
        cosU = np.cos(xSum)
        return np.tile(-cosU* self.d*(self.d+1)*(2*self.d+1)/6. - (self.T-t)*np.mean(np.where(x < 0,  np.sin(x),np.zeros(x.shape)), axis=-1),[1,1]).T

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
    
    # one step of values sing external indreements giben by getVolByRandom
    def getOneStepFromValuesDw(self,xPrev, t, TStep, wSigT):
       return xPrev +  wSigT
