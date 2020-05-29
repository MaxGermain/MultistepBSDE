import numpy as np
import tensorflow as tf
import os
import solvers  as solv
import networks as net
import models as mod
import multiprocessing
import sys
import time

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

_MULTIPROCESSING_CORE_COUNT = multiprocessing.cpu_count()
print("args", sys.argv)

d = 1
xInit= np.ones(d,dtype=np.float32) 
# nb neuron
nbNeuron = 10 +d 
print("nbNeuron " , nbNeuron)
# nb layer
nbLayer= 2  
print("nbLayer " ,nbLayer)
layerSize= nbNeuron*np.ones((nbLayer,), dtype=np.int32)
sig = np.array([float(1/np.sqrt(float(d))) ], dtype=np.float32)
print("Sig ",  sig)
rescal=1.
muScal =0.
sigScal=1
alpha= 1 
T=1.

batchSize= 1000
batchSizeVal= 10000
num_epoch=400
num_epochExtNoLast =10
num_epochExtLast= 200
initialLearningRateLast = 1e-2
initialLearningRateNoLast = 1e-3
nbOuterLearning =10
nTest = 10

# create the model
model = mod.BoundedFNL(xInit, muScal, sigScal, rescal, alpha ,d,T)

print(" WILL USE " + str(_MULTIPROCESSING_CORE_COUNT) + " THREADS ")
print("REAL IS ", model.Sol(0.,xInit), " DERIV", model.derSol(0.,xInit))


theNetwork = net.FeedForwardUZDZ(d,layerSize,tf.nn.tanh)
ndt = [120]

print("PDE BoundedFNL EMDBDP  Dim ", d, " layerSize " , layerSize,  " rescal " ,rescal, "T ", T , "batchsize ",batchSize, " batchSizeVal ", batchSizeVal, "num_epoch " , num_epoch, " num_epochExtNoLast ", num_epochExtNoLast  , "num_epochExtLast " , num_epochExtLast,   "alpha ", alpha ,"VOL " , sigScal, "initialLearningRateLast" , initialLearningRateLast , "initialLearningRateNoLast " , initialLearningRateNoLast)

# nest on ndt
for indt  in ndt:

    print("NBSTEP",indt)
    # create graph
    resol =  solv.PDEFNLSolveSimpleLSExp(model, T, indt, theNetwork , initialLearningRateLast=initialLearningRateLast, initialLearningRateNoLast = initialLearningRateNoLast)
        

    baseFile = "BoundedFNLEMDBDPd"+str(d)+"nbNeur"+str(layerSize[0])+"nbHL"+str(len(layerSize))+"ndt"+str(indt)+"Alpha"+str(int(alpha*100))
    plotFile = "pictures/"+baseFile
    
    Y0List = []
    for i in range(nTest):
        # train
        t0 = time.time()
        Y0, Z0, Gamma0    = resol.BuildAndtrainML( batchSize, batchSizeVal, num_epochExtNoLast=num_epochExtNoLast, num_epochExtLast= num_epochExtLast ,num_epoch=num_epoch,  nbOuterLearning=nbOuterLearning, thePlot= plotFile ,  baseFile = baseFile, saveDir= "save/")
        t1 = time.time()
        print(" NBSTEP", indt, " EstimMC Val is " , Y0,  " REAL IS ", model.Sol(0.,xInit),    " Z0 ", Z0," DERREAL IS  ",model.derSol(0.,xInit), "Gamma0 " , Gamma0,t1-t0)

        Y0List.append(Y0)
        print(Y0List)

    print("Y0", Y0List)
    yList = np.array(Y0List)
    yMean = np.mean(yList)
    print(" DNT ", indt , "MeanVal ", yMean, " Etyp ", np.sqrt(np.mean(np.power(yList-yMean,2.))))
