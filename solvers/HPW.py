# Equation    \f$ u_t + \mu(t,x)  u_x +f (t,x,u,u_x, u_xx) = 0 \f$

import numpy as np
import tensorflow as tf
import sys, traceback
import math
import os
from tensorflow.contrib.slim import fully_connected as fc
import time
from tensorflow.python.tools import inspect_checkpoint as chkp
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pylab as P



class HPWbase:

    # initial constructor
    def __init__(self, xInit , model , T, nbStep,   networkUZ,  initialLearningRate = 1e-2,  initialLearningRateStep  =  0.005):

        # save model : SDE coeff, final function, non linearity
        self.model = model
        self.d = len(xInit) # dimension
        self.xInit = xInit
        self.T= T
        self.nbStep = nbStep
        self.TStep = T/ nbStep
        self.initialLearningRate= initialLearningRate
        self.initialLearningRateStep= initialLearningRateStep # for UZ
        self.networkUZ=networkUZ  # for U and Z 
        

    # first routine to build derivatibes if not provided
    def buildG(self, iStep):
        self.learning_rate = tf.compat.v1.placeholder(tf.float32, shape=[], name="learning_rate")
        self.XPrev=   tf.compat.v1.placeholder(dtype=tf.float32, shape=[None, self.d], name='X')
        # Sample size
        sample_size = tf.shape(self.XPrev)[0]
        #
        rescale =tf.tile(tf.expand_dims(self.model.renormSigma(self.T),axis=0),[sample_size,1])
        normX0 = (self.XPrev-  tf.convert_to_tensor(self.xInit,dtype=tf.float32)  -  tf.tile(tf.expand_dims(self.model.renormTrend(self.T),axis=0),[sample_size,1]))/ rescale
        # Y0 and other variabel to initiliaze weights
        ULoc, DULoc = self.networkUZ.createNetwork(normX0, iStep, rescale)
        self.U = ULoc
        self.Z = DULoc
        self.G = self.model.gTf(self.XPrev)
        self.DG = self.model.DgTf(self.XPrev)
        self.norm = normX0
        self.total_loss=tf.reduce_mean(tf.pow(ULoc- self.model.gTf(self.XPrev),2))+ tf.reduce_mean(tf.pow(DULoc- self.model.DgTf(self.XPrev),2))
        self.train=tf.compat.v1.train.AdamOptimizer(learning_rate = self.learning_rate).minimize(self.total_loss)

 
    # first build
    def build0(self):                                         
        self.learning_rate=tf.compat.v1.placeholder(tf.float32, shape=[], name="learning_rate")
        self.dWSig= tf.compat.v1.placeholder(dtype=tf.float32, shape=[None, self.d], name='DWT') 
        self.UNext=tf.compat.v1.placeholder(dtype=tf.float32, shape=[None], name='UNext')
        sample_size = tf.shape(self.UNext)[0]
        self.Y0 = tf.get_variable("Y0", [], tf.float32, self.Y0_initializer)
        self.Z0 = tf.get_variable("Z0", None, tf.float32, tf.random_uniform([1,self.d], minval=-.05, maxval=.05, dtype=tf.float32, seed=2))
        Y = self.Y0*tf.ones([sample_size])
        Z = tf.tile(self.Z0,[sample_size, 1])
        # trend vector
        tXInit= tf.tile(tf.expand_dims(tf.convert_to_tensor(self.xInit, dtype=tf.float32), axis=0),[sample_size,1])
        # vol
        sig = self.model.sig(0,tXInit)
        YNext = Y  + self.TStep *(- self.model.fDW(0.,tXInit, Y, Z, 0.)) + tf.einsum("ij,ij->i",Z,self.dWSig)
        self.total_loss= tf.reduce_mean(tf.pow(self.UNext- YNext,2))
       
        self.train = tf.compat.v1.train.AdamOptimizer(learning_rate = self.learning_rate).minimize(self.total_loss)   
             
    # general build
    def build(self, iStep):
                                                                                   
        self.learning_rate=tf.compat.v1.placeholder(tf.float32, shape=[], name="learning_rate")
        self.XPrev = tf.compat.v1.placeholder(dtype=tf.float32, shape=[None, self.d], name='XPrev')
        self.UNext=tf.compat.v1.placeholder(dtype=tf.float32, shape=[None], name='UNext')
        self.dWSig = tf.compat.v1.placeholder(dtype=tf.float32, shape=[None, self.d], name='dW')
        # Sample size
        sample_size = tf.shape(self.XPrev)[0]
        #
        rescale = tf.tile(tf.expand_dims(self.model.renormSigma(self.TStep*iStep),axis=0),[sample_size,1])
        normX0 = (self.XPrev- self.xInit -  tf.tile(tf.expand_dims(self.model.renormTrend(self.TStep*iStep),axis=0),[sample_size,1]))/ rescale
        ULoc, ZLoc = self.networkUZ.createNetworkWithInitializer(normX0,iStep,self.CWeightUZ, self.CbiasUZ, rescale)
        # stores
        self.U= ULoc
        self.Z = ZLoc
        # vol
        sig = self.model.sig(iStep*self.TStep,self.XPrev)
        # Y
        YNext =  ULoc + self.TStep *(- self.model.fDW(iStep*self.TStep,self.XPrev, ULoc, ZLoc,0.))+ tf.reduce_sum(tf.multiply(ZLoc,self.dWSig),axis=1)
        # Loss
        self.total_loss=    tf.reduce_mean(tf.pow(self.UNext- YNext,2))
        self.train = tf.compat.v1.train.AdamOptimizer(learning_rate = self.learning_rate).minimize(self.total_loss)

# use adaptive  learning rate
# No truncation
class HPW(HPWbase):

  
    def BuildAndtrainML( self, batchSize, batchSizeVal, num_epoch=100, num_epochExt=100, num_epochExt0= 400, min_decrease_rate=0.05,nbOuterLearning=20,thePlot= "", saveDir = "./" , baseFile =""):
        
        tf.compat.v1.reset_default_graph()
        plt.style.use('seaborn-whitegrid')

        # first calculate g projection and its derivative
        #################################################
        # projection of the final
        gGDeriv =  tf.Graph()
        with gGDeriv.as_default():
            # build
            start_time = time.time()
            sess = tf.compat.v1.Session()
            #initialize
            self.theLearningRate = self.initialLearningRate
            # graph for derivative
            self.buildG(self.nbStep)
            # graph for weights
            self.weightLoc, self.biasLoc =  self.networkUZ.getBackWeightAndBias(self.nbStep) 
            # initialize variables
            sess.run(tf.compat.v1.global_variables_initializer())
            end_time = time.time()
            rtime = end_time-start_time 
            print( "Graph for projection took ", rtime)
            # history loss
            loss_hist_stop_criterion = []
            #time for SDE
            for iout in range(num_epochExt0):
                start_time = time.time()
                for epoch in range(num_epoch):
                     # get model value at a given time step
                     xValPrev = self.model.getValues(self.nbStep, self.TStep, self.xInit,batchSize)
                     sess.run(self.train, feed_dict ={self.XPrev : xValPrev, self.learning_rate : self.theLearningRate})
                end_time = time.time()
                rtime = end_time-start_time            # validate loss with trainin
                start_time = end_time
                xValPrev = self.model.getValues(self.nbStep, self.TStep, self.xInit,batchSizeVal)
                valLoss,Z =  sess.run([self.total_loss,self.Z], feed_dict= {  self.XPrev : xValPrev,  self.learning_rate :self.theLearningRate  })
                #if (iout%10==0):
                print("Loss is ", valLoss)
                if (math.isnan(float(valLoss))):
                    return  valLoss, valLoss 
                loss_hist_stop_criterion.append(valLoss)
                if (iout%nbOuterLearning==0):
                    mean_loss_from_last_check = np.mean(loss_hist_stop_criterion)
                    if (iout>0):
                        decrease_rate = (last_loss_check - mean_loss_from_last_check) / last_loss_check
                        if decrease_rate < min_decrease_rate:
                            self.theLearningRate = self.theLearningRate/2.
                            print("Projection, and derivative,  Learning decrease to "  ,self.theLearningRate)       
                    last_loss_check = mean_loss_from_last_check
                    loss_hist_stop_criterion=[]
                if (self.theLearningRate  < 1e-5):
                       break
        # get back weight
        self.CWeightUZ, self.CbiasUZ  = self.networkUZ.getWeights(sess,self.weightLoc, self.biasLoc)
        # save old session
        sessPrev = sess
        # sAVE u and placeholder used
        UPrev = self.U
        XPrev_prev = self.XPrev
            
        if ((thePlot != "") and (self.d ==1)):               
            U, Z =  sess.run([self.U, self.Z ], feed_dict= {  self.XPrev : xValPrev })
            UAnal = self.model.Sol(self.T, xValPrev)
            DUAnal = self.model.derSol(self.T, xValPrev)
            if ( self.d ==1):
                plt.plot(xValPrev[:,0],U,'.',label ="U",markersize=1)
                plt.plot(xValPrev[:,0],UAnal,'.',label ="U Analytic",markersize=1)
                P.xlabel(r"$x$")
                P.ylabel(r"$u(x)$")
                P.legend(loc='upper left')
                plt.savefig(thePlot+"_U_Last.png")
                P.figure()

                plt.plot(xValPrev[:,0],Z[:,0],'.',label ="Z",markersize=1)
                plt.plot(xValPrev[:,0],DUAnal[:,0],'.',label ="Z Analytic",markersize=1)
                P.xlabel(r"$x$")
                P.ylabel(r"$D_xu(x)$") 
                P.legend(loc='upper left')
                plt.savefig(thePlot+"_Z_Last.png")
                P.figure()   


        # backward resolutions
        #####################
        for  istep in range(self.nbStep-1,0,-1):
            print("STEP" , istep)
            print("#############")
            print("Graph for  UZ")
            # new graph for U, Z calculation
            #initialize
            gCurr = tf.Graph()
            with gCurr.as_default():
                sess = tf.compat.v1.Session()
                #initialize
                self.theLearningRate = self.initialLearningRateStep
                # build
                self.build(istep)
                # graph for weights
                self.weightLoc, self.biasLoc = self.networkUZ.getBackWeightAndBias(istep)
                 # initialize variables
                sess.run(tf.compat.v1.global_variables_initializer())
                #time for SDE
                timeValue = self.TStep*istep
                # history loss
                loss_hist_stop_criterion = []
                for iout in range(num_epochExt):
                    start_time = time.time()
                    for epoch in range(num_epoch):
                        xPrev = self.model.getValues(istep, self.TStep, self.xInit,batchSize)
                        wSigT = self.model.getVolByRandom(timeValue, xPrev ,np.random.normal(size=[batchSize,self.d]))*np.sqrt(self.TStep)
                        # estimate gamma
                        xNext= self.model.getOneStepFromValuesDw(xPrev, timeValue, self.TStep, wSigT)
                        # Estimate unext
                        UNext = sessPrev.run(UPrev,feed_dict ={XPrev_prev :xNext})
                        # calculate U DU
                        sess.run(self.train, feed_dict ={self.UNext : UNext.squeeze(),  self.XPrev: xPrev, self.dWSig : wSigT ,  self.learning_rate :self.theLearningRate})

                    xValPrev = self.model.getValues(istep, self.TStep, self.xInit,batchSizeVal)
                    wSigTVal = self.model.getVolByRandom(timeValue,xValPrev, np.random.normal(size=[batchSizeVal,self.d]))*np.sqrt(self.TStep)
                    xValNext = self.model.getOneStepFromValuesDw(xValPrev, timeValue, self.TStep, wSigTVal)
                    # evaluate at next step
                    UNext  = sessPrev.run(UPrev ,feed_dict ={XPrev_prev  :xValNext})
                    # calculate U DU
                    valLoss =  sess.run(self.total_loss, feed_dict= {   self.UNext : UNext.squeeze(), self.XPrev : xValPrev,
                                                                                    self.dWSig :wSigTVal,   self.learning_rate :self.theLearningRate  })
                    #if (iout%50==0):
                    print("Opt Loss UZ ", valLoss)
                    if (math.isnan(float(valLoss))):
                        return  valLoss, valLoss 
                    loss_hist_stop_criterion.append(valLoss)
                    if (iout%nbOuterLearning==0):
                        mean_loss_from_last_check = np.mean(loss_hist_stop_criterion)
                        if (iout>0):
                            decrease_rate = (last_loss_check - mean_loss_from_last_check) / last_loss_check
                            if decrease_rate < min_decrease_rate:
                                self.theLearningRate = self.theLearningRate/2.
                                print("Projection, and derivative,  Learning decrease to "  ,self.theLearningRate)       
                        last_loss_check = mean_loss_from_last_check
                        loss_hist_stop_criterion=[]
                    if (self.theLearningRate  < 1e-5):
                        break
                optVal =valLoss
                # sAVE u
                UPrev = self.U
                XPrev_prev = self.XPrev
                # get back weight
                self.CWeightUZ, self.CbiasUZ = self.networkUZ.getWeights(sess,self.weightLoc, self.biasLoc)
                # store associated session
                sessPrev= sess
                      
            if ((thePlot != "") and (self.d==1) and ((istep == 1) or (istep == 60) )):    
                xValPrev = xValPrev[::10]                  
                U, Z  =  sess.run([self.U, self.Z ], feed_dict= {  self.XPrev : xValPrev })
                UAnal = self.model.Sol(self.TStep*istep, xValPrev)
                DUAnal = self.model.derSol(self.TStep*istep, xValPrev)

                if ( self.d ==1):
                    plt.plot(xValPrev[:,0],U,'.',label ="U",markersize=1)
                    plt.plot(xValPrev[:,0],UAnal,'.',label ="U Analytic",markersize=1)                    
                    P.xlabel(r"$x$")
                    P.ylabel(r"$u(x)$")
                    P.legend(loc='upper left')
                    plt.savefig(thePlot+"_U_"+str(istep)+".png")
                    P.figure()

                    plt.plot(xValPrev[:,0],Z[:,0],'.',label ="Z",markersize=1)
                    plt.plot(xValPrev[:,0],DUAnal[:,0],'.',label ="Z Analytic",markersize=1)                    
                    P.xlabel(r"$x$")
                    P.ylabel(r"$D_xu(x)$") 
                    P.legend(loc='upper left')
                    plt.savefig(thePlot+"_Z_"+str(istep)+".png")
                    P.figure() 
  
        # last run
        print("Last step")
        print("##########")
        print("Graph init for  UZ")
        # new graph for U, Z calculation
        #initialize
        gCurr = tf.Graph()
        with gCurr.as_default():
            sess = tf.compat.v1.Session()
            #initialize
            self.theLearningRate = self.initialLearningRate
            # initial value for Y0 expectation of UN
            xNext= self.model.getValues(1, self.TStep, self.xInit,batchSizeVal)
            UNext = sessPrev.run(UPrev,feed_dict ={XPrev_prev :xNext})
            Y0_init = np.mean(UNext)
            self.Y0_initializer = tf.constant_initializer(Y0_init)
            # build graph
            self.build0()
            # initialize variables
            sess.run(tf.compat.v1.global_variables_initializer())
            xInit= np.tile(np.expand_dims(self.xInit, axis=0),[batchSize,1])
            xInitVal= np.tile(np.expand_dims(self.xInit, axis=0),[batchSizeVal,1])
            # history loss
            loss_hist_stop_criterion = []
            for iout in range(num_epochExt):
                start_time = time.time()
                for epoch in range(num_epoch):
                    wSigT = self.model.getVolByRandom(0., xInit ,np.random.normal(size=[batchSize,self.d]))*np.sqrt(self.TStep)
                    xNext=  self.model.getOneStepFromValuesDw(xInit, 0. , self.TStep, wSigT)
                    # estimate U at x (next step)
                    UNext = sessPrev.run( UPrev ,feed_dict ={XPrev_prev :xNext})
                    sess.run(self.train, feed_dict ={self.UNext : UNext.squeeze(), self.dWSig : wSigT,  self.learning_rate :self.theLearningRate })
                    
                wSigTVal = self.model.getVolByRandom(0., xInitVal,np.random.normal(size=[batchSizeVal,self.d]))*np.sqrt(self.TStep)
                xValNext =  self.model.getOneStepFromValuesDw(xInitVal, 0. , self.TStep, wSigTVal)
                UNext = sessPrev.run(UPrev ,feed_dict ={XPrev_prev :xValNext})
                valLoss =  sess.run(self.total_loss  , feed_dict= {  self.UNext : UNext.squeeze(), self.dWSig :wSigTVal,   self.learning_rate :self.theLearningRate   })
                #if (iout%50==0):
                print("Opt Loss UZ  init ", valLoss)
                if (math.isnan(float(valLoss))):
                    return  valLoss, valLoss
                loss_hist_stop_criterion.append(valLoss)
                if (iout%nbOuterLearning==0):
                    mean_loss_from_last_check = np.mean(loss_hist_stop_criterion)
                    if (iout>0):
                        decrease_rate = (last_loss_check - mean_loss_from_last_check) / last_loss_check
                        if decrease_rate < min_decrease_rate:
                            self.theLearningRate = self.theLearningRate/2.
                            print("Projection, and derivative,  Learning decrease to "  ,self.theLearningRate)       
                    last_loss_check = mean_loss_from_last_check
                    loss_hist_stop_criterion=[]
                # get out if learning rate too small
                if (self.theLearningRate < 1e-5):
                    break
            Y0, Z0 = sess.run([ self.Y0, self.Z0 ])
        return  Y0, Z0
