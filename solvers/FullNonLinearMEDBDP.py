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

class PDEFNLSolveSimpleLSExp:
    # initial constructor
    def __init__(self,  model ,  T, nbStep,    networkUZGam,   initialLearningRateLast = 1e-2,  initialLearningRateNoLast  =  0.005):

        # save model : SDE coeff, final function, non linearity
        self.model = model
        self.d =networkUZGam.d # dimension
        self.xInit = model.Init
        self.T= T
        self.nbStep = nbStep
        self.TStep = T/ nbStep
        self.initialLearningRateLast= initialLearningRateLast
        self.initialLearningRateNoLast= initialLearningRateNoLast # for UZ
        self.networkUZGam=networkUZGam  # for U and Z  for BSDE   

    # general BSDE resolution
    def buildBSDEUStep(self, iStep,  CWeightU =0, CbiasU= 0):
        dic = {}
        dic["LRate"] = tf.compat.v1.placeholder(tf.float32, shape=[], name="learning_rate")
        dic["XPrev"]=  tf.compat.v1.placeholder(dtype=tf.float32, shape=[None, self.d], name='XPrev')
        dic["UNext"]=  tf.compat.v1.placeholder(dtype=tf.float32, shape=[None], name='UNext')
        dic["GamExt"]=  tf.compat.v1.placeholder(dtype=tf.float32, shape=[None, self.d, self.d], name='GamExt')
        dic["dWSig"] = tf.compat.v1.placeholder(dtype=tf.float32, shape=[None, self.d], name='dW')
        # Sample size
        sample_size = tf.shape(dic["XPrev"])[0]
        rescale = tf.tile(tf.expand_dims(self.model.renormSigma(self.TStep*iStep),axis=0),[sample_size,1])
        normX0 = (dic["XPrev"]- self.xInit -  tf.tile(tf.expand_dims(self.model.renormTrend(self.TStep*iStep),axis=0),[sample_size,1]))/ rescale
        if (iStep < (self.nbStep-1)):
            dic["U"], dic["Z"], dic["Gam"]  =self.networkUZGam.createNetworkWithInitializer(normX0,iStep,CWeightU, CbiasU, rescale)
        else:
             dic["U"], dic["Z"], dic["Gam"]  =self.networkUZGam.createNetwork(normX0, iStep, rescale)
        dic["weightLoc"], dic["biasLoc"] =  self.networkUZGam.getBackWeightAndBias(iStep) 
        # vol
        sig = self.model.sig(iStep*self.TStep,dic["XPrev"])
        # Y
        YNext =  dic["U"] + self.TStep * (0.5*tf.einsum('ij,ij->i',sig ,tf.linalg.diag_part(dic["GamExt"])) -self.model.fDW(iStep*self.TStep,dic["XPrev"], dic["U"],  dic["Z"], dic["GamExt"]))+ tf.reduce_sum(tf.multiply(dic["Z"],dic["dWSig"]),axis=1)
        # Loss
        dic["Loss"]=    tf.reduce_mean(tf.pow(dic["UNext"] - YNext,2))
        dic["train"] = tf.compat.v1.train.AdamOptimizer(learning_rate = dic["LRate"]).minimize(dic["Loss"])
        dic["DU"] = tf.gradients(dic["U"], dic["XPrev"])[0]
        return dic



    # BSDE at time 0
    def buildBSDEUStep0(self, Y0_initializer, Z0_initializer, Gam0_initializer):
        dic ={}
        dic["LRate"] = tf.compat.v1.placeholder(tf.float32, shape=[], name="learning_rate")
        dic["UNext"]=  tf.compat.v1.placeholder(dtype=tf.float32, shape=[None], name='UNext')
        dic["GamExt"]=  tf.compat.v1.placeholder(dtype=tf.float32, shape=[None, self.d, self.d], name='GamExt')
        dic["dWSig"] = tf.compat.v1.placeholder(dtype=tf.float32, shape=[None, self.d], name='dW')
        sample_size = tf.shape(dic["UNext"])[0]
        dic["Y0"] = tf.compat.v1.get_variable("Y0", [], tf.float32, Y0_initializer)
        dic["Z0"] = tf.compat.v1.get_variable("Z0", [self.d], tf.float32, Z0_initializer)
        dic["Gam0"] = tf.compat.v1.get_variable("Gam0", [self.d, self.d], tf.float32, Gam0_initializer)
        Y = dic["Y0"]*tf.ones([sample_size])
        Z = tf.tile(tf.expand_dims(dic["Z0"], axis=0), [sample_size,1])
        Gam = tf.tile(tf.expand_dims(dic["Gam0"], axis=0), [sample_size,1,1])
        # trend vector
        tXInit= tf.tile(tf.expand_dims(tf.convert_to_tensor(self.xInit, dtype=tf.float32), axis=0),[sample_size,1])
        # vol
        sig = self.model.sig(0,tXInit)
         # Y
        YNext =  Y + self.TStep * (0.5*tf.einsum('ij,ij->i',sig ,tf.linalg.diag_part(dic["GamExt"])) -self.model.fDW(0.,tXInit, Y , Z, dic["GamExt"]))+ tf.reduce_sum(tf.multiply(Z,dic["dWSig"]),axis=1)
        dic["Loss"]= tf.reduce_mean(tf.pow(dic["UNext"]- YNext,2))       
        dic["train"] = tf.compat.v1.train.AdamOptimizer(learning_rate = dic["LRate"]).minimize(dic["Loss"])
        return dic


    # plot
    def plot(self, iStep, sessUPrev, dicoUPrev,batchSizeVal,thePlot):
        x =  np.sort(self.model.getValues(iStep, self.TStep, self.xInit,batchSizeVal), axis=0)
        uVal, zVal, gamVal= sessUPrev.run([dicoUPrev["U"],dicoUPrev["Z"],dicoUPrev["Gam"]] , feed_dict= {dicoUPrev["XPrev"] : x})
        UAnal = self.model.Sol(self.TStep*iStep, x)
        DUAnal = self.model.derSol(self.TStep*iStep, x)
        D2UAnal = self.model.der2Sol(self.TStep*iStep, x)

        plt.plot(x[:,0],zVal[:,0],'.' ,label ="Z   ",markersize=1)
        plt.plot(x[:,0],DUAnal[:,0],'.' ,label ="Z Analytic  ",markersize=1)
        P.xlabel(r"$x$")
        P.ylabel(r"$D_x u(x)$")
        P.legend(loc='upper left')
        plt.savefig(thePlot+"_Z_"+str(iStep)+".png")
        P.figure()

        plt.plot(x[:,0],gamVal[:,0,0],'.' ,label ="Gamma   ",markersize=1)
        plt.plot(x[:,0],D2UAnal[:,0,0],'.' ,label ="Gamma Analytic  ",markersize=1)
        P.xlabel(r"$x$")
        P.ylabel(r"$D^2_{xx} u(x)$")
        P.legend(loc='upper left')
        plt.savefig(thePlot+"_Gam_"+str(iStep)+".png")
        P.figure()

        plt.plot(x[:,0], uVal,'.' ,label ="U  ",markersize=1)
        plt.plot(x[:,0],UAnal,'.' ,label ="U Analytic  ",markersize=1)
        P.xlabel(r"$x$")
        P.ylabel(r"$u(x)$")
        P.legend(loc='upper left')
        plt.savefig(thePlot+"_U_"+str(iStep)+".png")
        P.figure()

        np.savetxt(thePlot+'x'+str(iStep),x[:,0])
        np.savetxt(thePlot+'gam'+str(iStep),gamVal[:,0,0])
        np.savetxt(thePlot+'truegam'+str(iStep),D2UAnal[:,0,0])

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
        
    # buikd and train
    def BuildAndtrainML( self, batchSize, batchSizeVal, num_epoch=100, num_epochExtNoLast=100, num_epochExtLast= 400, min_decrease_rate=0.05,nbOuterLearning=20,thePlot= "", saveDir = "./" , baseFile =""):
        
        tf.compat.v1.reset_default_graph()
        plt.style.use('seaborn-whitegrid')

        dicoList = []
        sessList = []
        
        # first time step
        #################

        # Now solve  U BSDE
        gCurr = tf.Graph()
        with gCurr.as_default():
            sess = tf.compat.v1.Session()
            #initialize
            theLearningRate = self.initialLearningRateLast
            num_epochExt = num_epochExtLast
            dicoBSDE= self.buildBSDEUStep(self.nbStep-1)
            # initialize variables
            sess.run(tf.compat.v1.global_variables_initializer())
            #time for SDE
            timeValue = (self.nbStep-1)*self.TStep
            #saver
            saver = tf.compat.v1.train.Saver()
            LossMin = 1e6
            # history loss
            loss_hist_stop_criterion = []
            for iout in range(num_epochExt):
                start_time = time.time()
                for epoch in range(num_epoch):
                    xPrev = self.model.getValues(self.nbStep-1, self.TStep, self.xInit,batchSize)
                    wSigT = self.model.getVolByRandom(timeValue, xPrev ,np.random.normal(size=[batchSize,self.d]))*np.sqrt(self.TStep)
                    xNext= self.model.getOneStepFromValuesDw(xPrev, timeValue, self.TStep, wSigT)
                    
                    
                    # Estimate unext
                    UNext =  self.model.g(xNext)
                    GamExt = self.model.D2g(xPrev)
                    # calculate U DU
                    sess.run(dicoBSDE["train"], feed_dict ={dicoBSDE["UNext"] : UNext,
                                                            dicoBSDE["GamExt"] :GamExt,
                                                            dicoBSDE["XPrev"]: xPrev,
                                                            dicoBSDE["dWSig"] : wSigT ,
                                                            dicoBSDE["LRate"] : theLearningRate})

                xPrev = self.model.getValues(self.nbStep-1, self.TStep, self.xInit,batchSizeVal)
                wSigT = self.model.getVolByRandom(timeValue, xPrev ,np.random.normal(size=[batchSizeVal,self.d]))*np.sqrt(self.TStep)
                xNext= self.model.getOneStepFromValuesDw(xPrev, timeValue, self.TStep, wSigT)
                # evaluate at next step
                UNext  =  self.model.g(xNext)
                GamExt = self.model.D2g(xPrev)
                # calculate U DU
                valLoss =  sess.run(dicoBSDE["Loss"], feed_dict={dicoBSDE["UNext"] : UNext,
                                                                 dicoBSDE["GamExt"] :GamExt,
                                                                 dicoBSDE["XPrev"]: xPrev,
                                                                 dicoBSDE["dWSig"] : wSigT  })
                
                if (valLoss < LossMin):
                    
                    LossMin= valLoss
                    saver.save(sess, os.path.join(saveDir,baseFile+"BSDE_"+str(self.nbStep-1)))
                    print("Opt Loss BSDE U ", valLoss, " iout" , iout)
                if (math.isnan(float(valLoss))):
                    
                    print(valLoss)
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
            saver.restore(sess, os.path.join(saveDir,baseFile+"BSDE_"+str(self.nbStep-1)))
            # get back weight
            CWeightU, CbiasU = self.networkUZGam.getWeights(sess, dicoBSDE["weightLoc"], dicoBSDE["biasLoc"])
            # store session
            sessList.append(sess)
            dicoList.append(dicoBSDE)
            
        if (thePlot != ""):
            if (self.d==1):
                self.plot( self.nbStep-1 , sess, dicoBSDE, batchSizeVal,thePlot) 
 
                    
        # backward resolutions
        #####################
        for  iStep in range(self.nbStep-2,0,-1):
            print("STEP" , iStep)
            print("#############")
            print("Graph for  BSDE resolution")

            # Now solve  U BSDE
            gCurr = tf.Graph()
            with gCurr.as_default():
                sess = tf.compat.v1.Session()
                #initialize
                theLearningRate = self.initialLearningRateNoLast
                num_epochExt = num_epochExtNoLast
                dicoBSDE= self.buildBSDEUStep(iStep, CWeightU, CbiasU)
                # initialize variables
                sess.run(tf.compat.v1.global_variables_initializer())
                #saver
                saver = tf.compat.v1.train.Saver()
                #time for SDE
                timeValue = self.TStep*iStep
                LossMin = 1e6
                # history loss
                loss_hist_stop_criterion = []
                for iout in range(num_epochExt):
                    start_time = time.time()
                    for epoch in range(num_epoch):
                        UNext =  np.zeros(batchSize)
                        xPrev = self.model.getValues(iStep, self.TStep, self.xInit,batchSize)
                        wSigT = self.model.getVolByRandom(timeValue, xPrev ,np.random.normal(size=[batchSize,self.d]))*np.sqrt(self.TStep)
                        xNext= self.model.getOneStepFromValuesDw(xPrev, timeValue, self.TStep, wSigT)
                        xPrevStor= xNext
                        for i in range(len(dicoList)-1,-1,-1):
                            tLoc= timeValue + (len(dicoList)-i)*self.TStep
                            wSigTLoc = self.model.getVolByRandom(tLoc, xPrevStor ,np.random.normal(size=[batchSize,self.d]))*np.sqrt(self.TStep)
                             # Estimate unext
                            U, Z, Gam =  sessList[i].run( [dicoList[i]["U"] , dicoList[i]["Z"], dicoList[i]["Gam"]],  feed_dict ={dicoList[i]["XPrev"]: xPrevStor})
                            sig2 = self.model.sigScal*self.model.sigScal
                            UNext = UNext  -self.TStep * (0.5* np.einsum('i,jii->j',sig2,Gam) -self.model.fDWNumpy(tLoc,xPrevStor,U, Z,Gam))-  np.sum(np.multiply(Z,wSigTLoc),axis=1)
                            
                            xPrevStor=self.model.getOneStepFromValuesDw(xPrevStor, tLoc, self.TStep, wSigTLoc)
                        # Estimate unext
                        UNext = UNext +self.model.g(xPrevStor)
                        GamExt = sessList[-1].run( dicoList[-1]["Gam"],  feed_dict ={dicoList[-1]["XPrev"]: xPrev}) # previous date
                        # calculate U DU
                        sess.run(dicoBSDE["train"], feed_dict ={dicoBSDE["UNext"] : UNext,
                                                                dicoBSDE["GamExt"] :GamExt,
                                                                dicoBSDE["XPrev"]: xPrev,
                                                                dicoBSDE["dWSig"] : wSigT ,
                                                                dicoBSDE["LRate"] : theLearningRate})


                    UNext =  np.zeros(batchSizeVal)
                    xPrev = self.model.getValues(iStep, self.TStep, self.xInit,batchSizeVal)
                    wSigT = self.model.getVolByRandom(timeValue, xPrev ,np.random.normal(size=[batchSizeVal,self.d]))*np.sqrt(self.TStep)
                    xNext= self.model.getOneStepFromValuesDw(xPrev, timeValue, self.TStep, wSigT)
                    xPrevStor= xNext
                    for i in range(len(dicoList)-1,-1,-1):
                        tLoc= timeValue + (len(dicoList)-i)*self.TStep
                        wSigTLoc = self.model.getVolByRandom(tLoc, xPrevStor ,np.random.normal(size=[batchSizeVal,self.d]))*np.sqrt(self.TStep)
                         # Estimate unext
                        U, Z, Gam =  sessList[i].run( [dicoList[i]["U"] , dicoList[i]["Z"], dicoList[i]["Gam"]],  feed_dict ={dicoList[i]["XPrev"]: xPrevStor})
                        sig2 = self.model.sigScal*self.model.sigScal
                        UNext = UNext  -self.TStep * (0.5* np.einsum('i,jii->j',sig2,Gam) -self.model.fDWNumpy(tLoc,xPrevStor,U, Z,Gam))-  np.sum(np.multiply(Z,wSigTLoc),axis=1)

                        xPrevStor=self.model.getOneStepFromValuesDw(xPrevStor, tLoc, self.TStep, wSigTLoc)
                    # Estimate unext
                    UNext = UNext +self.model.g(xPrevStor)
                    GamExt = sessList[-1].run( dicoList[-1]["Gam"],  feed_dict ={dicoList[-1]["XPrev"]: xPrev}) # previous date
    
                    # calculate U DU
                    valLoss =  sess.run(dicoBSDE["Loss"], feed_dict={dicoBSDE["UNext"] : UNext,
                                                                     dicoBSDE["GamExt"] :GamExt,
                                                                     dicoBSDE["XPrev"]: xPrev,
                                                                     dicoBSDE["dWSig"] : wSigT  }) 
                    if (valLoss < LossMin):
                        LossMin= valLoss
                        saver.save(sess, os.path.join(saveDir,baseFile+"BSDE_"+str(iStep)))
                        print("Opt Loss BSDE  U ", valLoss)
                    if (math.isnan(float(valLoss))):
                        print(valLoss)
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
                saver.restore(sess, os.path.join(saveDir,baseFile+"BSDE_"+str(iStep)))
                # get back weight
                CWeightU, CbiasU = self.networkUZGam.getWeights(sess, dicoBSDE["weightLoc"], dicoBSDE["biasLoc"])
                # store session
                sessList.append(sess)
                dicoList.append(dicoBSDE)
 
        
            if (thePlot != "" and (iStep == self.nbStep//2 or iStep ==1)):
                if (self.d==1):
                    self.plot( iStep, sess, dicoBSDE, batchSizeVal,thePlot)

  
        # last run
        print("Last step")
        print("##########")
        print("Graph init for Last step")
        # new graph for U, Z calculation                                                                         
        gCurr = tf.Graph()
        with gCurr.as_default():
            sess = tf.compat.v1.Session()
            #initialize
            theLearningRate = self.initialLearningRateNoLast
            num_epochExt = num_epochExtNoLast
            # initial value for Y0 expectation of UN
            xNext= self.model.getValues(1, self.TStep, self.xInit,batchSizeVal)
            uNext , ZNext, GamNext = sessList[-1].run([dicoList[-1]["U"],dicoList[-1]["Z"],dicoList[-1]["Gam"]], feed_dict ={dicoList[-1]["XPrev"] :xNext})
            Y0_initializer = tf.constant_initializer(np.mean(uNext, axis=0))
            Z0_initializer = tf.constant_initializer(np.mean(ZNext, axis=0))
            Gam0_initializer = tf.constant_initializer( np.mean(GamNext, axis=0))
            # build graph
            dicoBSDE = self.buildBSDEUStep0(Y0_initializer,Y0_initializer, Gam0_initializer )
            # initialize variables
            sess.run(tf.compat.v1.global_variables_initializer())
            # saver
            saver = tf.compat.v1.train.Saver()
            xInit= np.tile(np.expand_dims(self.xInit, axis=0),[batchSize,1])
            xInitVal= np.tile(np.expand_dims(self.xInit, axis=0),[batchSizeVal,1])
            LossMin =1e6
            # history loss
            loss_hist_stop_criterion = []
            for iout in range(num_epochExt):
                start_time = time.time()
                for epoch in range(num_epoch):
                    UNext =  np.zeros(batchSize)
                    wSigT = self.model.getVolByRandom(0., xInit ,np.random.normal(size=[batchSize,self.d]))*np.sqrt(self.TStep)
                    xNext= self.model.getOneStepFromValuesDw(xInit, 0., self.TStep, wSigT)
                    xPrevStor= xNext
                    for i in range(len(dicoList)-1,-1,-1):
                        tLoc=  (len(dicoList)-i)*self.TStep
                        wSigTLoc = self.model.getVolByRandom(tLoc, xPrevStor ,np.random.normal(size=[batchSize,self.d]))*np.sqrt(self.TStep)
                         # Estimate unext
                        U, Z, Gam =  sessList[i].run( [dicoList[i]["U"] , dicoList[i]["Z"], dicoList[i]["Gam"]],  feed_dict ={dicoList[i]["XPrev"]: xPrevStor})
                        sig2 = self.model.sigScal*self.model.sigScal
                        UNext = UNext  -self.TStep * (0.5* np.einsum('i,jii->j',sig2,Gam) -self.model.fDWNumpy(tLoc,xPrevStor,U, Z,Gam))-  np.sum(np.multiply(Z,wSigTLoc),axis=1)

                        xPrevStor=self.model.getOneStepFromValuesDw(xPrevStor, tLoc, self.TStep, wSigTLoc)
                    # Estimate unext
                    UNext = UNext +self.model.g(xPrevStor)
                    GamExt = sessList[-1].run( dicoList[-1]["Gam"],  feed_dict ={dicoList[-1]["XPrev"]: xInit}) # previous date
                   
                    sess.run(dicoBSDE["train"], feed_dict ={dicoBSDE["UNext"] : UNext,
                                                            dicoBSDE["GamExt"] :GamExt,
                                                            dicoBSDE["dWSig"] : wSigT ,
                                                            dicoBSDE["LRate"] : theLearningRate})

                UNext =  np.zeros(batchSizeVal)
                wSigT = self.model.getVolByRandom(0., xInitVal ,np.random.normal(size=[batchSizeVal,self.d]))*np.sqrt(self.TStep)
                xNext= self.model.getOneStepFromValuesDw(xInitVal, 0., self.TStep, wSigT)
                xPrevStor= xNext
                for i in range(len(dicoList)-1,-1,-1):
                    tLoc=  (len(dicoList)-i)*self.TStep
                    wSigTLoc = self.model.getVolByRandom(tLoc, xPrevStor ,np.random.normal(size=[batchSizeVal,self.d]))*np.sqrt(self.TStep)
                     # Estimate unext
                    U, Z, Gam =  sessList[i].run( [dicoList[i]["U"] , dicoList[i]["Z"], dicoList[i]["Gam"]],  feed_dict ={dicoList[i]["XPrev"]: xPrevStor})
                    sig2 = self.model.sigScal*self.model.sigScal
                    UNext = UNext  -self.TStep * (0.5* np.einsum('i,jii->j',sig2,Gam) -self.model.fDWNumpy(tLoc,xPrevStor,U, Z,Gam))-  np.sum(np.multiply(Z,wSigTLoc),axis=1)

                    xPrevStor=self.model.getOneStepFromValuesDw(xPrevStor, tLoc, self.TStep, wSigTLoc)
                # Estimate unext
                UNext = UNext +self.model.g(xPrevStor)
                GamExt = sessList[-1].run( dicoList[-1]["Gam"],  feed_dict ={dicoList[-1]["XPrev"]: xInitVal}) # previous date
                    
                valLoss =  sess.run( dicoBSDE["Loss"] , feed_dict= {dicoBSDE["UNext"] : UNext,
                                                                    dicoBSDE["GamExt"] :GamExt,
                                                                    dicoBSDE["dWSig"] : wSigT})
                if (valLoss < LossMin):
                    LossMin= valLoss
                    saver.save(sess, os.path.join(saveDir,baseFile+"BSDE_0"))
                if (iout%10==0):
                    print("Opt Loss UZ  init ", valLoss)
                if (math.isnan(float(valLoss))):
                    print(valLoss)
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
        saver.restore(sess, os.path.join(saveDir,baseFile+"BSDE_0"))
        Y0, Z0, Gam0  = sess.run( [dicoBSDE["Y0" ], dicoBSDE["Z0" ],dicoBSDE["Gam0"]] )
        return  Y0, Z0, Gam0
