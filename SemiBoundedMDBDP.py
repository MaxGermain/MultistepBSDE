import numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import tensorflow as tf
import solvers  as solv
import networks as net
import models as mod
import multiprocessing
import sys
import time

_MULTIPROCESSING_CORE_COUNT = multiprocessing.cpu_count()
print("args", sys.argv)

d = 1
xInit= np.ones(d,dtype=np.float32) 
# nb neuron
nbNeuron = 10 + d 
print("nbNeuron " , nbNeuron)
# nb layer
nbLayer= 2  
print("nbLayer " ,nbLayer)
layerSize= nbNeuron*np.ones((nbLayer,), dtype=np.int32)
sig = 1 
print("Sig ",  sig)
rescal=1.
muScal = 0.2/d 
sigScal=sig/np.sqrt(d)
alpha= 0.5 
T=1.

batchSize= 1000
batchSizeVal= 50000
num_epoch= 50 
num_epochExtNoLast = 100 
num_epochExtLast = 1000 
initialLearningRateLast = 1e-2
initialLearningRateNoLast = 1e-3
nbOuterLearning = 50 
nTest = 1 

# create the model
model = mod.Bounded( sigScal, T, xInit, muScal, alpha)

print(" WILL USE " + str(_MULTIPROCESSING_CORE_COUNT) + " THREADS ")
print("REAL IS ", model.SolPoint(0., xInit), " DERIV", model.derSol(0., xInit))
 
theNetwork = net.FeedForwardUZ(d,layerSize,tf.nn.tanh)

ndt = [120]
cut = False

print("PDE Bounded MDBDP Dim ", d, " layerSize " , layerSize,  " rescal " ,rescal, "T ", T , "batchsize ",batchSize, " batchSizeVal ", batchSizeVal, "num_epoch " , num_epoch, " num_epochExtNoLast ", num_epochExtNoLast  , "num_epochExtLast " , num_epochExtLast,   "alpha ", alpha ,"VOL " , sigScal, "initialLearningRateLast" , initialLearningRateLast , "initialLearningRateNoLast " , initialLearningRateNoLast)
print('cut',cut)

# nest on ndt
for indt  in ndt:

    print("NBSTEP",indt)
    # create graph
    resol =  solv.PDESLSolveBaseGPU(model, T, indt, theNetwork, initialLearningRateLast=initialLearningRateLast, initialLearningRateNoLast = initialLearningRateNoLast)
        
    baseFile = "BoundedMDBDP"+str(d)+"nbNeur"+str(layerSize[0])+"nbHL"+str(len(layerSize))+"ndt"+str(indt)+"Sig"+str(int(100.*sig))+"Alpha"+str(int(alpha*100))+str(cut)
    thePlot = "pictures/"+baseFile

    Y0List = []
    for i in range(nTest):
        # train
        t0 = time.time()
        Y0, Z0  = resol.BuildAndtrainML( batchSize, batchSizeVal, num_epochExtNoLast=num_epochExtNoLast, num_epochExtLast= num_epochExtLast ,num_epoch=num_epoch,  nbOuterLearning=nbOuterLearning, thePlot= thePlot , baseFile = baseFile, cut = cut, saveDir= "save/")
        t1 = time.time()                
        print(" NBSTEP", indt, " EstimMC Val is " , Y0,  " REAL IS ", model.SolPoint(0., xInit),    " Z0 ", Z0," DERREAL IS  ",model.derSol(0., xInit),"Duration", t1 - t0)

        Y0List.append(Y0)
        print(Y0List)

    print("Y0", Y0List)
    yList = np.array(Y0List)
    yMean = np.mean(yList)
    print(" DNT ", indt , "MeanVal ", yMean, " Etyp ", np.sqrt(np.mean(np.power(yList-yMean,2.))))
