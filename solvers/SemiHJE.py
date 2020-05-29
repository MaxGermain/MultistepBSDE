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
plt.rcParams.update({'figure.max_open_warning': 0})
import pylab as P
from mpl_toolkits.mplot3d import axes3d, Axes3D 

class SemiHJE:
    # initial constructor
    def __init__(self,  model ,  T, nbStepUDU,   networkUZ,  initialLearningRateLast = 1e-2,  initialLearningRateNoLast  =  0.005):
        # save model : SDE coeff, final function, non linearity
        self.model = model
        self.d =networkUZ.d # dimension
        self.xInit = model.Init
        self.T= T
        self.nbStepUDU = nbStepUDU
        self.TStepUDU = T/nbStepUDU
        self.initialLearningRateLast= initialLearningRateLast
        self.initialLearningRateNoLast= initialLearningRateNoLast # for UZ
        self.networkUZ=networkUZ  # for  Z  for BSDE

              
    # general BSDE resolution
    def buildBSDEUStep(self):
        dic = {}
        dic["LRate"] = tf.compat.v1.placeholder(tf.float32, shape=[], name="learning_rate")
        dic["RandG"] = tf.compat.v1.placeholder(dtype=tf.float32, shape=[None, self.d,self.nbStepUDU], name='randG')
        # Sample size
        sample_size = tf.shape(dic["RandG"])[0]
        sig = self.model.sigScal
        mu= self.model.muScal
        sqrtDt = np.sqrt(self.TStepUDU)

        X0 = self.xInit
        X = tf.tile(tf.expand_dims(X0, axis = 0), [sample_size,1])
        Y0 = tf.get_variable("Y0",None, tf.float32,tf.random_uniform([], minval = -1, maxval = 1,dtype= tf.float32))
        Y = tf.tile(tf.expand_dims(Y0, axis = 0), [sample_size])
        Z0 = tf.get_variable("Z0",None, tf.float32,tf.random_uniform([self.d], minval = -1, maxval = 1,dtype= tf.float32))
        Z = tf.tile(tf.expand_dims(Z0, axis = 0), [sample_size,1])
        dic["Z"] = Z
        self.XYZlist = [[X,Y0,Z0]]

        for iStep in range(self.nbStepUDU):
            Y =  Y + self.TStepUDU * (-self.model.fDW(iStep*self.TStepUDU, X, Y,  dic["Z"],  0.)) + tf.reduce_sum(tf.multiply(dic["Z"],tf.einsum('j,ij->ij',tf.constant(sig, dtype= tf.float32),dic["RandG"][:,:,iStep]*sqrtDt)), axis=1)
            X = X + mu*self.TStepUDU + tf.einsum('j,ij->ij',tf.constant(sig, dtype= tf.float32),dic["RandG"][:,:,iStep]*sqrtDt)
            self.XYZlist.append([X,Y,dic["Z"]])

            normX0 = tf.einsum('ij,j->ij',(X - tf.tile(tf.expand_dims(self.xInit + mu*self.TStepUDU*(iStep+1), axis = 0), [sample_size,1])), tf.math.reciprocal(tf.constant(sig*np.sqrt(self.TStepUDU*(iStep+1)), dtype= tf.float32)))
            dic["Z"] = self.networkUZ.createNetworkZ(normX0, iStep+1, 0.)   
            
        # Loss
        dic["Loss"] = tf.reduce_mean(tf.pow(Y -self.model.gTf(X),2))
        dic["train"] = tf.compat.v1.train.AdamOptimizer(learning_rate = dic["LRate"]).minimize(dic["Loss"])
        return dic
    
    # plot
    def plot(self, iStep, XYZ, thePlot):
        x =  XYZ[0][::10]
        uVal = XYZ[1][::10]
        zVal = XYZ[2] [::10]

        UAnal = self.model.Sol(self.TStepUDU*iStep, x)
        DUAnal = self.model.derSol(self.TStepUDU*iStep, x)

        fig = plt.figure()
        plt.plot(x[:,0],zVal[:,0],'.' ,label ="Z   ",markersize=1)
        plt.plot(x[:,0],DUAnal[:,0],'.' ,label ="Z Analytic  ",markersize=1)
        P.xlabel(r"$x$")
        P.ylabel(r"$D_{x} u(x)$")
        P.legend(loc='upper left')
        plt.savefig(thePlot+"_Z_"+str(iStep)+".png")
        plt.close()

        fig = plt.figure()
        plt.plot(x[:,0], uVal,'.' ,label ="U  ",markersize=1)
        plt.plot(x[:,0],UAnal,'.' ,label ="U Analytic  ",markersize=1)
        P.xlabel(r"$x$")
        P.ylabel(r"$u(x)$")
        P.legend(loc='upper left')
        plt.savefig(thePlot+"_U_"+str(iStep)+".png")
        plt.close()

    # buikd and train
    def BuildAndtrainML( self, batchSize, batchSizeVal, num_epoch=100, num_epochExtNoLast=100, min_decrease_rate=0.05,nbOuterLearning=20,thePlot= "", saveDir = "./" , baseFile =""):
        
        tf.compat.v1.reset_default_graph()
        plt.style.use('seaborn-whitegrid')        
        
        gCurr = tf.Graph()
        with gCurr.as_default():
            sessBSDE = tf.Session()
            #initialize
            theLearningRate = self.initialLearningRateNoLast
            num_epochExt = num_epochExtNoLast
            
            dicoBSDE= self.buildBSDEUStep()
            # initialize variables
            sessBSDE.run(tf.compat.v1.global_variables_initializer())
            print("GRAPH CREATED")
            #saver
            saver = tf.compat.v1.train.Saver()
            LossMin = 1e6
            # history loss
            loss_hist_stop_criterion = []
            for iout in range(num_epochExt):
                start_time = time.time()
                for epoch in range(num_epoch):
                    NG = np.random.normal(size=[batchSize, self.d,self.nbStepUDU])
                    # calculate U DU
                    sessBSDE.run(dicoBSDE["train"], feed_dict ={dicoBSDE["RandG"]:NG, dicoBSDE["LRate"] : theLearningRate})

                NG = np.random.normal(size=[batchSizeVal, self.d,self.nbStepUDU]) 
                # calculate U DU
                valLoss, XYZlist =  sessBSDE.run([dicoBSDE["Loss"], self.XYZlist ], feed_dict={dicoBSDE["RandG"]:NG })
                
                if (valLoss < LossMin):
                    LossMin= valLoss
                    saver.save(sessBSDE, os.path.join(saveDir,baseFile+"BSDE"))
                    print(str((iout+1)*num_epoch)+" Opt Loss BSDE  U ", valLoss)
                    print('Y0', XYZlist[0][1])
                if (math.isnan(float(valLoss))):
                    print('NAN values in validation loss')
                loss_hist_stop_criterion.append(valLoss)
                if (iout%nbOuterLearning==0):
                    mean_loss_from_last_check = np.mean(loss_hist_stop_criterion)
                    if (iout>0):
                        decrease_rate = (last_loss_check - mean_loss_from_last_check) / last_loss_check
                        if decrease_rate < min_decrease_rate:
                            theLearningRate = np.maximum(1e-6,theLearningRate/2.)
                            print("Projection, and derivative,  Learning decrease to "  ,theLearningRate)       
                    last_loss_check = mean_loss_from_last_check
                    loss_hist_stop_criterion=[]

            
                if (thePlot != "" and iout%10 == 0 and self.d==1):
                    for iStep in [1,self.nbStepUDU//2]:
                        self.plot( iStep, XYZlist[iStep], thePlot)

        Y0, Z0  = XYZlist[0][1], XYZlist[0][2]
        return  Y0, Z0
