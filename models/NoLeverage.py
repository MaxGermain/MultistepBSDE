# All the model for f(u,Du, D2u) driver
#  u_t + f(u,Du,D2u)= 0

import numpy as np
import tensorflow as tf
from scipy.stats import norm

class ModelNoLeverage:
    
    # initial constructor
    def __init__(self, xyInit, muScal, sigScal, T, theta, sigma, lamb, eta, gamma, kappa):
        self.name = 'noleverage'
        self.muScal = muScal
        self.sigScal = sigScal
        self.d = xyInit.shape[0]
        self.T =T
        self.Init= xyInit
        self.rho = 0.
        self.theta = theta
        self.sigma = sigma
        self.lamb = lamb
        self.eta = eta
        self.gamma = gamma
        self.kappa = kappa           

    
    # driver for dW modelization
    # t    
    #
    # u     (nbSample , 1)
    # Du    (nbSample, d )
    # return (nbSample)
    def fDW( self, t, x, u, Du, D2u):  
        y = x[:,1:]  
        grady = Du[:,1:]
        hessy = tf.linalg.diag_part(D2u[:,1:,1:])

        return  -0.5 * tf.einsum('i,i->i',tf.einsum('j,ij->i',tf.constant(self.lamb**2, dtype= tf.float32),tf.einsum('ij,ij->ij',y,y)), tf.einsum('i,i->i' ,tf.einsum('i,i->i', Du[:,0],Du[:,0]), tf.math.reciprocal(D2u[:,0,0]))) \
        + tf.einsum('j,ij->i',tf.einsum('i,i->i',tf.constant(self.kappa, dtype= tf.float32),tf.constant(self.theta, dtype= tf.float32)),grady) - tf.einsum('ij,ij->i',tf.einsum('j,ij->ij',tf.constant(self.kappa, dtype= tf.float32),y), grady)\
        - tf.einsum('j,ij->i', tf.constant(self.muScal, dtype= tf.float32), Du) + 0.5 * tf.einsum('j,ij->i',tf.constant(self.gamma**2, dtype= tf.float32), hessy) 
    
    def fDWNumpy( self, t, x, u, Du, D2u):
        y = x[:,1:]  
        grady = Du[:,1:]
        hessy = np.array([[D2u[k,j,j] for j in range(1,D2u.shape[1])] for k in range(D2u.shape[0])])

        return  -0.5 * np.einsum('i,i->i',np.einsum('j,ij->i',self.lamb**2,y**2), np.einsum('i,i->i' ,np.einsum('i,i->i', Du[:,0],Du[:,0]), 1/D2u[:,0,0])) \
        + np.einsum('j,ij->i',np.einsum('i,i->i',self.kappa,self.theta),grady) - np.einsum('ij,ij->i',np.einsum('j,ij->ij',self.kappa,y), grady)\
         - np.einsum('j,ij->i', self.muScal, Du) + 0.5 * np.einsum('j,ij->i',self.gamma**2, hessy)
   
    # final derivative
    def Dg(self,x):
        gx = -self.eta * self.g(x)
        gtotal = np.zeros(x.shape)
        gtotal[:,0] = gx
        return gtotal
    
    def DgTf(self,x):
        gx = tf.expand_dims(-self.eta * self.gTf(x),1)
        gtotal = tf.zeros((tf.shape(x)[0],self.d-1), dtype= tf.float32)
        return tf.concat([gx, gtotal], 1)
    
    # final derivative
    def D2g(self,x):
        gx = self.eta**2 * self.g(x)
        gtotal = np.zeros((x.shape[0],self.d,self.d))
        gtotal[:,0,0] = gx
        return gtotal

    def D2gTf(self,x):
        gx = self.eta**2 * self.gTf(x)
        zeroOne = np.zeros((self.d,self.d), dtype= np.float32)
        zeroOne[0,0] = 1         
        return tf.einsum('jk,ijk->ijk', tf.constant(zeroOne),tf.tile(tf.expand_dims(tf.expand_dims(gx,axis = -1),axis = -1), [1, self.d, self.d]))

    # permits to truncate the SDE value
    def truncate(self, istep , TStep, xInit, quantile):
        xMin = norm.ppf(1-quantile)
        xMax = norm.ppf(quantile)
        return  np.sqrt(TStep*istep)*xMin*self.sigScal , np.sqrt(TStep*istep)*xMax*self.sigScal

    # x   (nbSim,d)
    # return  (nbSim))
    def g(self,x):
        return -np.exp(-self.eta*x[:,0])

    def gTf(self,x):
        return -1.0*tf.reshape(tf.math.exp(-self.eta*x[:,0]),[tf.shape(x)[0]])

    def w(self, t,y):
        s = np.linspace(t, self.T, 10000)
        ds = s[1] - s[0]
        chi = np.sum(np.einsum('i,ij->ij',self.kappa*self.theta,self.psi(s)) + np.einsum('i,ij->ij',self.gamma**2/2,(self.phi(s) - (1-self.rho**2)*self.psi(s)**2)), axis=1)*ds
        return np.exp(-1.0*  np.einsum('jk,ij->i',self.phi(np.array([t])), y**2/2) - np.einsum('jk,ij->i',self.psi(np.array([t])),y) - np.sum(chi, axis = 0))

    def phi(self, t):        
        kappabar = self.kappa + self.rho * self.gamma * self.lamb
        kappahat = np.sqrt(self.kappa**2 + 2*self.rho*self.gamma*self.lamb*self.kappa + self.gamma**2*self.lamb**2)
        return np.einsum('i,ij->ij',self.lamb**2, np.sinh(np.einsum('i,j->ij',kappahat, (self.T-t)))/(np.einsum('i,ij->ij',\
            kappabar,np.sinh(np.einsum('i,j->ij',kappahat, (self.T-t)))) \
            + np.einsum('i,ij->ij',kappahat,np.cosh(np.einsum('i,j->ij',kappahat, (self.T-t))))))

    def psi(self, t):        
        kappabar = self.kappa + self.rho * self.gamma * self.lamb
        kappahat = np.sqrt(self.kappa**2 + 2*self.rho*self.gamma*self.lamb*self.kappa + self.gamma**2*self.lamb**2)
        return np.einsum('i,ij->ij',self.lamb**2*self.kappa*self.theta/kappahat,(np.cosh(np.einsum('i,j->ij',kappahat,\
             (self.T-t))) - 1)/(np.einsum('i,ij->ij',kappabar,np.sinh(np.einsum('i,j->ij',kappahat, (self.T-t)))) \
                 + np.einsum('i,ij->ij',kappahat,np.cosh(np.einsum('i,j->ij',kappahat, (self.T-t))))))

    def Dw(self, t,y):
        phi = self.phi(np.array([t]))
        psi = self.psi(t*np.ones(y.shape[0]))
        
        grady = np.einsum('ij,i->ij',np.einsum('jk,ij->ij',-1.0*phi,y) - psi.T,self.w(t,y))
        total = np.zeros((y.shape[0],self.d))
        total[:,1:] = grady
        return total

    def D2w(self, t,y):
        phi = self.phi(np.array([t]))
        psi = self.psi(t*np.ones(y.shape[0]))
        
        hessy = np.einsum('jk,i->ijk',np.diag(-phi[:,0]),self.w(t,y)) + np.einsum('ij,ik->ijk',np.einsum('jk,ij->ij',-1.0*phi,y) - psi.T,np.einsum('ij,i->ij',np.einsum('jk,ij->ij',-1.0*phi,y) - psi.T,self.w(t,y)))
        total = np.zeros((y.shape[0],self.d,self.d))
        total[:,1:,1:] = hessy
        return total

    def wTf(self, t,y):
        return 1.

    def Sol( self,t,x):
        return self.g(x)*self.w(t,x[:,1:])

    def derSol( self,t,x):
        return np.einsum('ij,i->ij',self.Dg(x),self.w(t,x[:,1:])) + np.einsum('i,ij->ij',self.g(x),self.Dw(t,x[:,1:]))
    def der2Sol( self,t,x):
    
        return np.einsum('ijk,i->ijk',self.D2g(x),self.w(t,x[:,1:])) + np.einsum('ij,ik->ijk',self.Dg(x),self.Dw(t,x[:,1:])) \
            + np.einsum('i,ijk->ijk',self.g(x),self.D2w(t,x[:,1:])) + np.einsum('ij,ik->ikj',self.Dg(x),self.Dw(t,x[:,1:]))

    def control(self, t, x, u, Du, D2u):
        return -1/self.sigma * (np.einsum('ij,i->ij',np.einsum('j,ij->ij',self.lamb, x[:,1:]), np.einsum('i,i->i' ,Du[:,0], 1/D2u[:,0,0])))

    # renormalization trend
    def renormTrend(self, t):
        return t * self.muScal

    # renormalize sig
    def renormSigma(self, t):
        return self.sigScal*np.sqrt(t)

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

