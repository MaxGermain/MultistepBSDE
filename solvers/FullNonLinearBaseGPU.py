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
from mpl_toolkits.mplot3d import axes3d, Axes3D 

class PDEFNLSolveBaseGPU:
    # initial constructor
    def __init__(self,  model ,  T, nbStepUDU,  nbStepGam,   networkUZ, networkGam,  initialLearningRateLast = 1e-2,  initialLearningRateNoLast  =  0.005):
        # save model : SDE coeff, final function, non linearity
        self.model = model
        self.d =networkUZ.d # dimension
        self.xInit = model.Init
        self.T= T
        self.nbStepUDU = nbStepUDU
        self.nbStepGam= nbStepGam
        self.nbStepGamStab = int(nbStepUDU/nbStepGam)
        print("nbStepGamStab" ,self.nbStepGamStab)
        self.TStepUDU = T/nbStepUDU
        self.TStepGam = T/nbStepGam
        self.initialLearningRateLast= initialLearningRateLast
        self.initialLearningRateNoLast= initialLearningRateNoLast # for UZ
        self.networkUZ=networkUZ  # for U and Z  for BSDE
        self.networkGam = networkGam  # gamma

              
    # general BSDE resolution
    def buildBSDEUStep(self, iStep,  ListWeightUZ , ListBiasUZ, ListWeightGam, ListBiasGam):
        dic = {}
        dic["LRate"] = tf.compat.v1.placeholder(tf.float32, shape=[], name="learning_rate")
        dic["XPrev"]=  tf.compat.v1.placeholder(dtype=tf.float32, shape=[None, self.d], name='XPrev')
        dic["RandG"] = tf.compat.v1.placeholder(dtype=tf.float32, shape=[None, self.d,self.nbStepUDU-iStep], name='randG')
        # Sample size
        sample_size = tf.shape(dic["XPrev"])[0]
        sig = self.model.sigScal
        mu= self.model.muScal
        rescale= sig*np.sqrt(self.TStepUDU*iStep)
        normX0 = tf.einsum('ij,j->ij',(dic["XPrev"]- self.xInit - mu*self.TStepUDU*iStep), tf.math.reciprocal(tf.constant(rescale, dtype= tf.float32)))
        if (iStep < (self.nbStepUDU-1)):
            dic["U"], dic["Z"]  =self.networkUZ.createNetworkWithInitializer(normX0,iStep,ListWeightUZ[-1], ListBiasUZ[-1], 0.)
        else:
            dic["U"], dic["Z"]  =self.networkUZ.createNetwork(normX0, iStep, 0.)
        dic["weightLoc"], dic["biasLoc"] =  self.networkUZ.getBackWeightAndBias(iStep)
        GamExt = self.networkGam.createNetworkNotTrainable(normX0,iStep, ListWeightGam[-1], ListBiasGam[-1])
        sqrtDt = np.sqrt(self.TStepUDU)
        # Y
        YNext =  dic["U"] + self.TStepUDU * (0.5*tf.reduce_sum(tf.einsum('j,ij->ij',tf.constant(sig**2, dtype= tf.float32) ,tf.linalg.diag_part(GamExt)),axis=1) -self.model.fDW(iStep*self.TStepUDU,dic["XPrev"], dic["U"],  dic["Z"], GamExt))+ tf.reduce_sum(tf.multiply(dic["Z"],tf.einsum('j,ij->ij',tf.constant(sig, dtype= tf.float32),dic["RandG"][:,:,0]*sqrtDt)), axis=1)
        
        XNext = dic["XPrev"] + mu*self.TStepUDU + tf.einsum('j,ij->ij',tf.constant(sig, dtype= tf.float32),dic["RandG"][:,:,0]*sqrtDt)
        iStepLoc = iStep+1
        for   i in  range( len( ListWeightUZ)-1,-1,-1):
            iReverse = len( ListWeightUZ)-i
            normX0 = tf.einsum('ij,j->ij',XNext- self.xInit - mu*self.TStepUDU*iStepLoc, tf.math.reciprocal(tf.constant(sig*np.sqrt(self.TStepUDU*iStepLoc), dtype= tf.float32)))
            U, Z=self.networkUZ.createNetworkNotTrainable(normX0,iStepLoc,ListWeightUZ[i],ListBiasUZ[i])
            idecGam = int((self.nbStepUDU -(iStep+len( ListWeightUZ)-i))/self.nbStepGamStab)
            Gam =self.networkGam.createNetworkNotTrainable(normX0,iStepLoc, ListWeightGam[idecGam], ListBiasGam[idecGam])
            YNext = YNext  + self.TStepUDU * (0.5*tf.reduce_sum(tf.einsum('j,ij->ij',tf.constant(sig**2, dtype= tf.float32),tf.linalg.diag_part(Gam)),axis=1) -self.model.fDW(iStepLoc*self.TStepUDU,XNext,U , Z, Gam))+ tf.reduce_sum(tf.multiply(Z,tf.einsum('j,ij->ij',tf.constant(sig, dtype= tf.float32),dic["RandG"][:,:,iReverse]*sqrtDt)), axis=1)
            XNext = XNext + mu*self.TStepUDU + tf.einsum('j,ij->ij',tf.constant(sig, dtype= tf.float32),dic["RandG"][:,:,iReverse]*sqrtDt)
            iStepLoc = iStepLoc+1
         # Loss
        dic["Loss"]=    tf.reduce_mean(tf.pow(YNext -self.model.gTf(XNext),2))
        dic["train"] = tf.compat.v1.train.AdamOptimizer(learning_rate = dic["LRate"]).minimize(dic["Loss"])
        return dic

    # BSDE at time 0
    def buildBSDEUStep0(self,   ListWeightUZ , ListBiasUZ, ListWeightGam, ListBiasGam,Y0_initializer, Z0_initializer):
        dic ={}
        dic["LRate"] = tf.compat.v1.placeholder(tf.float32, shape=[], name="learning_rate")
        dic["RandG"] = tf.compat.v1.placeholder(dtype=tf.float32, shape=[None, self.d,self.nbStepUDU], name='randG')
        dic["Gam0"]=  tf.compat.v1.placeholder(dtype=tf.float32, shape=[self.d, self.d], name='Gam0')
        sample_size = tf.shape(dic["RandG"])[0]
        dic["Y0"] = tf.compat.v1.get_variable("Y0", [], tf.float32, Y0_initializer)
        dic["Z0"] = tf.compat.v1.get_variable("Z0", [self.d], tf.float32, Z0_initializer)
        Y = dic["Y0"]*tf.ones([sample_size])
        Z = tf.tile(tf.expand_dims(dic["Z0"], axis=0), [sample_size,1])
        gam = tf.tile(tf.expand_dims(dic["Gam0"], axis=0), [sample_size,1,1])
        # trend vector
        tXInit= tf.tile(tf.expand_dims(tf.convert_to_tensor(self.xInit, dtype=tf.float32), axis=0),[sample_size,1])
        sig = self.model.sigScal
        mu= self.model.muScal
        sqrtDt = np.sqrt(self.TStepUDU)
        # Y
        YNext =  Y + self.TStepUDU * (0.5*tf.reduce_sum(tf.einsum('j,ij->ij',tf.constant(sig**2, dtype= tf.float32),tf.linalg.diag_part(gam)),axis=1) -self.model.fDW(0.,tXInit, Y , Z, gam))+ tf.reduce_sum(tf.multiply(Z,tf.einsum('j,ij->ij',tf.constant(sig, dtype= tf.float32),dic["RandG"][:,:,0]*sqrtDt)),axis=1)

        XNext = tXInit + mu*self.TStepUDU + tf.einsum('j,ij->ij',tf.constant(sig, dtype= tf.float32),dic["RandG"][:,:,0]*sqrtDt)
        iStepLoc = 1
        for   i in  range( len( ListWeightUZ)-1,-1,-1):
            iReverse = len( ListWeightUZ)-i
            normX0 = tf.einsum('ij,j->ij',(XNext- self.xInit - mu*self.TStepUDU*iStepLoc), tf.math.reciprocal(tf.constant(sig*np.sqrt(self.TStepUDU*iStepLoc), dtype= tf.float32)))
            U, Z=self.networkUZ.createNetworkNotTrainable(normX0,iStepLoc,ListWeightUZ[i],ListBiasUZ[i])
            idecGam = int((self.nbStepUDU -(len( ListWeightUZ)-i))/self.nbStepGamStab)
            Gam =self.networkGam.createNetworkNotTrainable(normX0,iStepLoc, ListWeightGam[idecGam], ListBiasGam[idecGam])
            YNext = YNext  + self.TStepUDU * (0.5*tf.reduce_sum(tf.einsum('j,ij->ij',tf.constant(sig**2, dtype= tf.float32),tf.linalg.diag_part(Gam)),axis=1) -self.model.fDW(iStepLoc*self.TStepUDU,XNext,U , Z, Gam))+ tf.reduce_sum(tf.multiply(Z,sig*dic["RandG"][:,:,iReverse]*sqrtDt),axis=1)
            XNext = XNext + mu*self.TStepUDU + tf.einsum('j,ij->ij',tf.constant(sig, dtype= tf.float32),dic["RandG"][:,:,iReverse]*sqrtDt)
            iStepLoc = iStepLoc+1
         # Loss
        dic["Loss"]=    tf.reduce_mean(tf.pow(YNext -self.model.gTf(XNext),2))       
        dic["train"] = tf.compat.v1.train.AdamOptimizer(learning_rate = dic["LRate"]).minimize(dic["Loss"])
        return dic
 
    # plot
    def plot(self, iStep, sessUPrev, dicoUPrev, batchSizeVal,thePlot,gamVal):
        x =  np.sort(self.model.getValues(iStep, self.TStepUDU, self.xInit,batchSizeVal), axis=0)
        uVal, zVal= sessUPrev.run([dicoUPrev["U"],dicoUPrev["Z"]] , feed_dict= {dicoUPrev["XPrev"] : x})
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

        np.savetxt(thePlot+'x'+str(iStep),x[:,0])
        np.savetxt(thePlot+'u'+str(iStep),uVal)
        np.savetxt(thePlot+'trueu'+str(iStep),UAnal)
        np.savetxt(thePlot+'z'+str(iStep),zVal[:,0])
        np.savetxt(thePlot+'truez'+str(iStep),DUAnal[:,0])

        if self.model.name == "merton":
            for i in range(self.model.lamb.shape[0]):
                plt.plot(x[:,0],-self.model.lamb[i]*self.model.theta[i]/np.exp(self.model.theta[i]) * zVal[:,0]/gamVal[:,0,0],'.' ,label ="Control   ",markersize=1)
                plt.plot(x[:,0],self.model.lamb[i]*self.model.theta[i]/np.exp(self.model.theta[i])/self.model.eta* np.ones(x.shape[0]),'.' ,label ="Control Analytic",markersize=1)
                P.xlabel(r"$x$")
                P.ylabel(r"$\alpha_" + str(i) + '(x)$')
                P.legend(loc='upper left')
                P.ylim(0., 2*self.model.lamb[i]*self.model.theta[i]/np.exp(self.model.theta[i])/self.model.eta) 
                plt.savefig(thePlot+"_Alpha0_"+str(iStep)+".png")
                P.figure()
                np.savetxt(thePlot+'control'+str(iStep),-self.model.lamb[i]*self.model.theta[i]/np.exp(self.model.theta[i]) * zVal[:,0]/gamVal[:,0,0])
                np.savetxt(thePlot+'truecontrol'+str(iStep),self.model.lamb[i]*self.model.theta[i]/np.exp(self.model.theta[i])/self.model.eta* np.ones(x.shape[0]))
        
    # plot
    def plotGam(self, iStep, sessGam, dicoGam,batchSizeVal,thePlot):
        x =  np.sort(self.model.getValues(iStep, self.TStepUDU, self.xInit,batchSizeVal), axis=0)
        gamVal= sessGam.run(dicoGam["Gam"] , feed_dict= {dicoGam["XPrev"] : x})
        D2UAnal = self.model.der2Sol(self.TStepUDU*iStep, x)

        plt.plot(x[:,0],gamVal[:,0,0],'.' ,label ="Gamma  ",markersize=1)
        plt.plot(x[:,0],D2UAnal[:,0,0],'.' ,label ="Gamma Analytic  ",markersize=1)
        P.xlabel(r"$x$")
        P.ylabel(r"$D^2_{x} u(x)$")
        P.legend(loc='upper left')
        plt.savefig(thePlot+"_Gam_"+str(iStep)+".png")
        P.figure()

        np.savetxt(thePlot+'xGAM'+str(iStep),x[:,0])
        np.savetxt(thePlot+'gam'+str(iStep),gamVal[:,0,0])
        np.savetxt(thePlot+'truegam'+str(iStep),D2UAnal[:,0,0])

        return gamVal

    # buikd and train
    def BuildAndtrainML( self, batchSize, batchSizeVal, num_epoch=100, num_epochExtNoLast=100, num_epochExtLast= 400, min_decrease_rate=0.05,nbOuterLearning=20,thePlot= "", saveDir = "./" , baseFile =""):
        
        tf.compat.v1.reset_default_graph()
        plt.style.use('seaborn-whitegrid')

        ListWeightUZ = []
        ListBiasUZ = []
        ListWeightGam= []
        ListBiasGam = []
        
        for  iGam in range(self.nbStepGam,0,-1):
            
            gCurrGam = tf.Graph()
            with gCurrGam.as_default():
                sessGam = tf.compat.v1.Session()
                #initialize
                if iGam == self.nbStepGam:
                    theLearningRate = self.initialLearningRateLast
                    num_epochExt = num_epochExtLast
                else:                        
                    theLearningRate = self.initialLearningRateNoLast
                    num_epochExt = num_epochExtNoLast
                dicoGam= self.buildGamStep(iGam,ListWeightUZ,ListBiasUZ,ListWeightGam,ListBiasGam )
                print("CREATED GAM ", iGam)
                # initialize variables
                sessGam.run(tf.compat.v1.global_variables_initializer())
                #time for SDE
                timeValue = iGam*self.TStepGam
                #saver
                saver = tf.compat.v1.train.Saver()
                LossMin = 1e6
                # history loss
                loss_hist_stop_criterion = []
                for iout in range(num_epochExt):
                    start_time = time.time()
                    for epoch in range(num_epoch):
                        GamTraj =  np.zeros([batchSize,self.d,self.d])
                        xPrev = self.model.getValues(iGam, self.TStepGam, self.xInit,batchSize)
                        NG = np.random.normal(size=[batchSize,self.d, self.sizeNRG(iGam)])
                        # calculate Gamma
                        sessGam.run(dicoGam["train"], feed_dict ={dicoGam["XPrev"]: xPrev,dicoGam["RandG"]:NG,
                                                               dicoGam["LRate"] : theLearningRate})

                    xPrev = self.model.getValues(iGam, self.TStepGam, self.xInit,batchSizeVal)
                    NG = np.random.normal(size=[batchSizeVal,self.d, self.sizeNRG(iGam)])
                    # calculate Gamma
                    valLoss =  sessGam.run(dicoGam["Loss"], feed_dict={dicoGam["RandG"]:NG,dicoGam["XPrev"]: xPrev })
                    if (valLoss < LossMin):
                        LossMin= valLoss
                        saver.save(sessGam, os.path.join(saveDir,baseFile+"Gam_"+str(iGam)))
                        print(" Gam", valLoss, " iout" , iout)
                    if (math.isnan(float(valLoss))):
                        print('NAN values in validation loss', valLoss)
                        #return  valLoss, valLoss 
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
                saver.restore(sessGam, os.path.join(saveDir,baseFile+"Gam_"+str(iGam)))
                # get back weight
                CWeightGam, CbiasGam = self.networkGam.getWeights(sessGam, dicoGam["weightLoc"], dicoGam["biasLoc"])
                # store weight
                ListWeightGam.append(CWeightGam)
                ListBiasGam.append(CbiasGam)

                if (thePlot != "" and (iGam*self.nbStepGamStab == 1 or iGam*self.nbStepGamStab == 60 or iGam*self.nbStepGamStab == 4)):
                    if (self.d==1):
                        gamval = self.plotGam( iGam*self.nbStepGamStab,  sessGam, dicoGam,batchSizeVal,thePlot)

            if (iGam==1):
                # store average
                gam0Estim = np.mean(sessGam.run(dicoGam["Gam"], feed_dict ={dicoGam["XPrev"] :xPrev}), axis=0)

                
            tf.compat.v1.reset_default_graph()
                
            if (iGam ==self.nbStepGam):
                idec= -1
            else:
                idec=0
            for iStepLoc in range(self.nbStepGamStab+idec):
                iStep = iGam*self.nbStepGamStab-iStepLoc +idec
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
                        dicoBSDE= self.buildBSDEUStep(iStep, ListWeightUZ,ListBiasUZ,ListWeightGam,ListBiasGam)
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
                                NG = np.random.normal(size=[batchSize, self.d,self.nbStepUDU-iStep])
                                # calculate U DU
                                sessBSDE.run(dicoBSDE["train"], feed_dict ={dicoBSDE["RandG"]:NG, dicoBSDE["XPrev"]: xPrev, dicoBSDE["LRate"] : theLearningRate})

                            xPrev = self.model.getValues(iStep, self.TStepUDU, self.xInit,batchSizeVal)
                            NG = np.random.normal(size=[batchSizeVal, self.d,self.nbStepUDU-iStep]) 
                            # calculate U DU
                            valLoss =  sessBSDE.run(dicoBSDE["Loss"], feed_dict={dicoBSDE["XPrev"]: xPrev,dicoBSDE["RandG"]:NG })
                            
                            if (valLoss < LossMin):
                                LossMin= valLoss
                                saver.save(sessBSDE, os.path.join(saveDir,baseFile+"BSDE_"+str(iStep)))
                                print("Opt Loss BSDE  U ", valLoss)
                            if (math.isnan(float(valLoss))):
                                print('NAN values in validation loss', valLoss)
                                #return  valLoss, valLoss 
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
                       
                        if ((thePlot != "") and ((iStep == self.nbStepUDU/2) or (iStep == 1))): #-1 in Xavier's code
                            if (self.d==1):
                                self.plot( iStep, sessBSDE, dicoBSDE, batchSizeVal,thePlot,gamval)

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
        # initial value for Y0 expectation of UN
        Gam0_initializer = tf.constant_initializer( gam0Estim)
        # for gamma
        gCurrGam = tf.Graph()
        with gCurrGam.as_default():
            sessGam = tf.compat.v1.Session()
            #initialize
            theLearningRate = self.initialLearningRateNoLast
            num_epochExt = num_epochExtNoLast
            dicoGam= self.buildGamStep0(ListWeightUZ,ListBiasUZ,ListWeightGam,ListBiasGam, Gam0_initializer)
            # initialize variables
            sessGam.run(tf.compat.v1.global_variables_initializer())
            #saver
            saver = tf.compat.v1.train.Saver()
            LossMin = 1e6
            # history loss
            loss_hist_stop_criterion = []
            for iout in range(num_epochExt):
                start_time = time.time()
                for epoch in range(num_epoch):
                    # calculate Gamma
                    NG = np.random.normal(size=[batchSize, self.d,self.sizeNRG(0)])
                    sessGam.run(dicoGam["train"], feed_dict ={dicoGam["RandG"]:NG, dicoGam["LRate"] : theLearningRate})

                NG = np.random.normal(size=[batchSizeVal, self.d,self.sizeNRG(0)])
                # calculate Gamma
                valLoss =  sessGam.run(dicoGam["Loss"], feed_dict ={dicoGam["RandG"]:NG})
                
                if (valLoss < LossMin):
                    LossMin= valLoss
                    saver.save(sessGam, os.path.join(saveDir,baseFile+"Gam0"))
                    print(" Gam", valLoss, " iout" , iout)
                if (math.isnan(float(valLoss))):
                    print('NAN values in validation loss', valLoss) 
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
        saver.restore(sessGam, os.path.join(saveDir,baseFile+"Gam0"))
        Gam0= sessGam.run( dicoGam["Gam0"])
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
            dicoBSDE = self.buildBSDEUStep0(ListWeightUZ,ListBiasUZ,ListWeightGam,ListBiasGam, Y0_initializer,Z0_initializer )
            # initialize variables
            sessBSDE.run(tf.compat.v1.global_variables_initializer())
            # saver
            saver = tf.compat.v1.train.Saver()
            LossMin =1e6
            # history loss
            loss_hist_stop_criterion = []
            for iout in range(num_epochExt):
                start_time = time.time()
                for epoch in range(num_epoch):
                    # calculate U DU
                    NG = np.random.normal(size=[batchSize, self.d,self.nbStepUDU])
                    sessBSDE.run(dicoBSDE["train"], feed_dict ={dicoBSDE["Gam0"] :Gam0,dicoBSDE["RandG"]:NG, dicoBSDE["LRate"] : theLearningRate})

                NG = np.random.normal(size=[batchSizeVal, self.d,self.nbStepUDU])
                # calculate U DU
                valLoss =  sessBSDE.run(dicoBSDE["Loss"], feed_dict={dicoBSDE["Gam0"] :Gam0,dicoBSDE["RandG"]:NG})
                
                if (valLoss < LossMin):
                    LossMin= valLoss
                    saver.save(sessBSDE, os.path.join(saveDir,baseFile+"BSDE0"))
                    print("Opt Loss BSDE  U ", valLoss)
                if (math.isnan(float(valLoss))):
                    print('NAN values in validation loss', valLoss)
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
        return  Y0, Z0, Gam0
