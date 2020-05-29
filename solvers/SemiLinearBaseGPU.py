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
import tensorflow_probability as tfp

class PDESLSolveBaseGPU:
    # initial constructor
    def __init__(self,  model ,  T, nbStepUDU,  networkUZ,  initialLearningRateLast = 1e-2,  initialLearningRateNoLast  =  0.005):
        # save model : SDE coeff, final function, non linearity
        self.model = model
        self.d =networkUZ.d # dimension
        self.xInit = model.Init
        self.T= T
        self.nbStepUDU = nbStepUDU
        self.TStepUDU = T/nbStepUDU
        self.initialLearningRateLast= initialLearningRateLast
        self.initialLearningRateNoLast= initialLearningRateNoLast # for UZ
        self.networkUZ=networkUZ  # for U and Z  for BSDE
              
    # general BSDE resolution
    def buildBSDEUStep(self, iStep,  ListWeightUZ , ListBiasUZ, cut):
        dic = {}
        dic["LRate"] = tf.compat.v1.placeholder(tf.float32, shape=[], name="learning_rate")
        dic["XPrev"]=  tf.compat.v1.placeholder(dtype=tf.float32, shape=[None, self.d], name='XPrev')
        dic["RandG"] = tfp.distributions.Normal(tf.zeros(shape=[self.d,self.nbStepUDU-iStep]), tf.ones(shape=[self.d,self.nbStepUDU-iStep])).sample(tf.shape(dic["XPrev"])[0])

        # Sample size
        sample_size = tf.shape(dic["XPrev"])[0]
        sig = self.model.sigScal
        mu= self.model.muScal
        rescale= sig*np.sqrt(self.TStepUDU*iStep)
        normX0 = (dic["XPrev"]- self.xInit - mu*self.TStepUDU*iStep)/ rescale
        if (iStep < (self.nbStepUDU-1)):
            dic["U"], dic["Z"]  =self.networkUZ.createNetworkWithInitializer(normX0,iStep,ListWeightUZ[-1], ListBiasUZ[-1], rescale)
        else:
            dic["U"], dic["Z"]  =self.networkUZ.createNetwork(normX0, iStep, rescale)
        dic["weightLoc"], dic["biasLoc"] =  self.networkUZ.getBackWeightAndBias(iStep)
        sqrtDt = np.sqrt(self.TStepUDU)
        # Y
        YNext =  dic["U"] + self.TStepUDU * ( -self.model.fDW(iStep*self.TStepUDU,dic["XPrev"], dic["U"],  dic["Z"], 0.))+ tf.reduce_sum(tf.multiply(dic["Z"],sig*dic["RandG"][:,:,0]*sqrtDt),axis=1)

        XNext = dic["XPrev"] + mu*self.TStepUDU + sig*dic["RandG"][:,:,0]*sqrtDt
        iStepLoc = iStep+1
        for   i in  range( len( ListWeightUZ)-1,-1,-1):
            iReverse = len( ListWeightUZ)-i
            normX0 = (XNext- self.xInit - mu*self.TStepUDU*iStepLoc)/ (sig*np.sqrt(self.TStepUDU*iStepLoc))
            U, Z=self.networkUZ.createNetworkNotTrainable(normX0,iStepLoc,ListWeightUZ[i],ListBiasUZ[i])
            YNext = YNext  + self.TStepUDU * ( -self.model.fDW(iStepLoc*self.TStepUDU,XNext,U , Z, 0.))+ tf.reduce_sum(tf.multiply(Z,sig*dic["RandG"][:,:,iReverse]*sqrtDt),axis=1) 
            XNext = XNext + mu*self.TStepUDU + sig*dic["RandG"][:,:,iReverse]*sqrtDt
            iStepLoc = iStepLoc+1
         # Loss
        if (cut and iStep < self.nbStepUDU/2):         
            print('Alternative loss')   
            normXNext = (XNext - self.xInit - mu*self.TStepUDU*self.nbStepUDU/2)/ sig*np.sqrt(self.TStepUDU*self.nbStepUDU/2)
            terminal, _  =self.networkUZ.createNetworkNotTrainable(normXNext,self.nbStepUDU/2,ListWeightUZ[0], ListBiasUZ[0])            
            dic["Loss"]=    tf.reduce_mean(tf.pow(YNext - terminal,2))
        else:
            dic["Loss"]=    tf.reduce_mean(tf.pow(YNext -self.model.gTf(XNext),2))
        dic["train"] = tf.compat.v1.train.AdamOptimizer(learning_rate = dic["LRate"]).minimize(dic["Loss"])
        return dic


    # BSDE at time 0
    def buildBSDEUStep0(self,   ListWeightUZ , ListBiasUZ, Y0_initializer, Z0_initializer, cut):
        dic ={}
        dic["LRate"] = tf.compat.v1.placeholder(tf.float32, shape=[], name="learning_rate")
        dic["XPrev"]=  tf.compat.v1.placeholder(dtype=tf.float32, shape=[None, self.d], name='XPrev')

        dic["RandG"] = tfp.distributions.Normal(tf.zeros(shape=[self.d,self.nbStepUDU]), tf.ones(shape=[self.d,self.nbStepUDU]), name='rand0').sample(tf.shape(dic["XPrev"])[0])

        sample_size = tf.shape(dic["RandG"])[0]
        dic["Y0"] = tf.get_variable("Y0", [], tf.float32, Y0_initializer)
        dic["Z0"] = tf.get_variable("Z0", [self.d], tf.float32, Z0_initializer)
        Y = dic["Y0"]*tf.ones([sample_size])
        Z = tf.tile(tf.expand_dims(dic["Z0"], axis=0), [sample_size,1])
        # trend vector
        tXInit= tf.tile(tf.expand_dims(tf.convert_to_tensor(self.xInit, dtype=tf.float32), axis=0),[sample_size,1])
        sig = self.model.sigScal
        mu= self.model.muScal
        sqrtDt = np.sqrt(self.TStepUDU)
        # Y
        YNext =  Y + self.TStepUDU * ( -self.model.fDW(0.,tXInit, Y , Z, 0.))+ tf.reduce_sum(tf.multiply(Z,sig*dic["RandG"][:,:,0]*sqrtDt),axis=1) 

        XNext = tXInit + mu*self.TStepUDU + sig*dic["RandG"][:,:,0]*sqrtDt
        iStepLoc = 1
        for   i in  range( len( ListWeightUZ)-1,-1,-1):
            iReverse = len( ListWeightUZ)-i
            normX0 = (XNext- self.xInit - mu*self.TStepUDU*iStepLoc)/ (sig*np.sqrt(self.TStepUDU*iStepLoc))
            U, Z=self.networkUZ.createNetworkNotTrainable(normX0,iStepLoc,ListWeightUZ[i],ListBiasUZ[i])
            YNext = YNext  + self.TStepUDU * ( -self.model.fDW(iStepLoc*self.TStepUDU,XNext,U , Z, 0.))+ tf.reduce_sum(tf.multiply(Z,sig*dic["RandG"][:,:,iReverse]*sqrtDt),axis=1) 
            XNext = XNext + mu*self.TStepUDU + sig*dic["RandG"][:,:,iReverse]*sqrtDt
            iStepLoc = iStepLoc+1
         # Loss
        if (cut):         
            print('Alternative loss')   
            normXNext = (XNext - self.xInit - mu*self.TStepUDU*self.nbStepUDU/2)/ sig*np.sqrt(self.TStepUDU*self.nbStepUDU/2)
            terminal, _  =self.networkUZ.createNetworkNotTrainable(normXNext,self.nbStepUDU/2,ListWeightUZ[0], ListBiasUZ[0])            
            dic["Loss"]=    tf.reduce_mean(tf.pow(YNext - terminal,2))
        else:
            dic["Loss"]=    tf.reduce_mean(tf.pow(YNext -self.model.gTf(XNext),2))       
        dic["train"] = tf.compat.v1.train.AdamOptimizer(learning_rate = dic["LRate"]).minimize(dic["Loss"])
        return dic

    # plot
    def plot(self, iStep, sessUPrev, dicoUPrev, batchSizeVal,thePlot):
        x =  np.sort(self.model.getValues(iStep, self.TStepUDU, self.xInit,batchSizeVal), axis=0)[::10]
        uVal, zVal= sessUPrev.run([dicoUPrev["U"],dicoUPrev["Z"]] , feed_dict= {dicoUPrev["XPrev"] : x})
        UAnal = self.model.Sol(self.TStepUDU*iStep, x)
        DUAnal = self.model.derSol(self.TStepUDU*iStep, x)

        fig = plt.figure()
        plt.plot(x[:,0],zVal[:,0],'.' ,label ="Z   ",markersize=1)
        plt.plot(x[:,0],DUAnal[:,0],'.' ,label ="Z Analytic  ",markersize=1)
        P.xlabel(r"$x$")
        P.ylabel(r"$D_x u(x)$") 
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
        
    # plot
    def plotGam(self, iStep, sessGam, dicoGam,batchSizeVal,thePlot):
        x =  np.sort(self.model.getValues(iStep, self.TStepUDU, self.xInit,batchSizeVal), axis=0)
        gamVal= sessGam.run(dicoGam["Gam"] , feed_dict= {dicoGam["XPrev"] : x})
        D2UAnal = self.model.der2Sol(self.TStepUDU*iStep, x)

        plt.plot(x[:,0],D2UAnal[:,0,0] ,label ="Gam Analytic  ",markersize=1)
        plt.plot(x[:,0],gamVal[:,0,0] ,label ="Gam   ",markersize=1)
        P.xlabel("X")
        P.ylabel("D2f(X)")
        P.legend(loc='upper left')
        plt.savefig(thePlot+"_Gam_"+str(iStep)+".png")
        P.figure() 

    # buiLd and train
    def BuildAndtrainML( self, batchSize, batchSizeVal, num_epoch=100, num_epochExtNoLast=100, num_epochExtLast= 400, min_decrease_rate=0.05,nbOuterLearning=20,thePlot= "", saveDir = "./" , baseFile ="", cut = False):
        
        tf.compat.v1.reset_default_graph()
        plt.style.use('seaborn-whitegrid')

        ListWeightUZ = []
        ListBiasUZ = []
        ListWeightGam= []
        ListBiasGam = []
        
        for iStepLoc in range(self.nbStepUDU-1):
            iStep = -iStepLoc -1 + self.nbStepUDU
            # test
            if (iStep > 0):
                # Now solve  U BSDE
                gCurr = tf.Graph()
                with gCurr.as_default():
                    sessBSDE = tf.compat.v1.Session()
                    #initialize
                    if (iStep < self.nbStepUDU -1):
                        theLearningRate = self.initialLearningRateNoLast
                        num_epochExt = num_epochExtNoLast
                    else:
                        theLearningRate = self.initialLearningRateLast
                        num_epochExt = num_epochExtLast
                    if (iStep == (self.nbStepUDU/2 - 1) and cut):
                        ListWeightUZ = [ListWeightUZ[-1]]
                        ListBiasUZ = [ListBiasUZ[-1]]
                    dicoBSDE= self.buildBSDEUStep(iStep, ListWeightUZ,ListBiasUZ, cut)
                    # initialize variables
                    sessBSDE.run(tf.compat.v1.global_variables_initializer())
                    print("CREATED BSDE ", iStep)
                    #saver
                    saver = tf.compat.v1.train.Saver()
                    #time for SDE
                    timeValue = self.TStepUDU*iStep
                    LossMin = 1e6
                    # history loss
                    loss_hist_stop_criterion = []
                    for iout in range(num_epochExt):
                        start_time = time.time()
                        for epoch in range(num_epoch):
                            xPrev = self.model.getValues(iStep, self.TStepUDU, self.xInit,batchSize)
                            # calculate U DU
                            sessBSDE.run(dicoBSDE["train"], feed_dict ={dicoBSDE["XPrev"]: xPrev, dicoBSDE["LRate"] : theLearningRate}) 

                        xPrev = self.model.getValues(iStep, self.TStepUDU, self.xInit,batchSizeVal)
                        # calculate U DU
                        valLoss =  sessBSDE.run(dicoBSDE["Loss"], feed_dict={dicoBSDE["XPrev"]: xPrev})
                        
                        if (valLoss < LossMin):
                            LossMin= valLoss
                            saver.save(sessBSDE, os.path.join(saveDir,baseFile+"BSDE_"+str(iStep)))
                            print("Opt Loss BSDE  U ", valLoss)
                        if (math.isnan(float(valLoss))):
                            return  valLoss, valLoss 
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

                    # restore best
                    saver.restore(sessBSDE, os.path.join(saveDir,baseFile+"BSDE_"+str(iStep)))
                    # get back weight
                    CWeightU, CbiasU = self.networkUZ.getWeights(sessBSDE, dicoBSDE["weightLoc"], dicoBSDE["biasLoc"])
                    # store session
                    ListWeightUZ.append(CWeightU)
                    ListBiasUZ.append(CbiasU)
                    
                    if ((thePlot != "") and ((iStepLoc == self.nbStepUDU-2) or (iStepLoc == (self.nbStepUDU-2)/2))):
                        if (self.d==1):
                            self.plot( iStep, sessBSDE, dicoBSDE, batchSizeVal,thePlot)

                if (iStep==1):
                    # store estimate
                    U, Z = sessBSDE.run([dicoBSDE["U"],dicoBSDE["Z"]], feed_dict ={dicoBSDE["XPrev"] :xPrev})
                    uEstim = np.mean(U)
                    zEstim = np.mean(Z, axis=0)
                tf.compat.v1.reset_default_graph()

        # last run
        print("Last step")
        print("##########")
        print("Graph init for Last step")
       
        tf.compat.v1.reset_default_graph()
        # new graph for U, Z calculation
        Y0_initializer = tf.constant_initializer(uEstim) 
        Z0_initializer = tf.constant_initializer(zEstim) 
                   
        gCurr = tf.Graph()
        with gCurr.as_default():
            sessBSDE = tf.compat.v1.Session()
            #initialize
            theLearningRate = self.initialLearningRateNoLast
            num_epochExt = num_epochExtNoLast
            # for Gam
            # build graph
            dicoBSDE = self.buildBSDEUStep0(ListWeightUZ,ListBiasUZ, Y0_initializer,Z0_initializer, cut )
            # initialize variables
            sessBSDE.run(tf.compat.v1.global_variables_initializer())
            # saver
            saver = tf.train.Saver()
            LossMin =1e6
            # history loss
            loss_hist_stop_criterion = []
            for iout in range(num_epochExt):
                start_time = time.time()
                for epoch in range(num_epoch):
                    # calculate U DU
                    Xshape = np.ones((batchSize, self.d))
                    sessBSDE.run(dicoBSDE["train"], feed_dict ={dicoBSDE["LRate"] : theLearningRate, dicoBSDE["XPrev"]: Xshape}) 

                # calculate U DU
                Xshape = np.ones((batchSizeVal, self.d))
                valLoss =  sessBSDE.run(dicoBSDE["Loss"], feed_dict={dicoBSDE["XPrev"]: Xshape}) 
                
                if (valLoss < LossMin):
                    LossMin= valLoss
                    saver.save(sessBSDE, os.path.join(saveDir,baseFile+"BSDE0"))
                    print("Opt Loss BSDE  U ", valLoss)
                if (math.isnan(float(valLoss))):
                    return  valLoss, valLoss 
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

        # restore best
        saver.restore(sessBSDE, os.path.join(saveDir,baseFile+"BSDE0"))

        Y0, Z0  = sessBSDE.run( [dicoBSDE["Y0" ], dicoBSDE["Z0" ]] )
        return  Y0, Z0 
