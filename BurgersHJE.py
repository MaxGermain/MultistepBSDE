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

nbLayer= 2  
print("nbLayer " ,nbLayer)

rescal=1.
T=1.

batchSize= 1000
batchSizeVal= 50000 
num_epoch= 50 
num_epochExtNoLast = 200
initialLearningRateLast = 1e-2
initialLearningRateNoLast = 1e-3
nbOuterLearning =10
nTest = 10

d = 1
nbNeuron = 11
sigScal = np.ones(d, dtype=np.float32)
muScal = np.zeros(d, dtype=np.float32)
nu = 1

layerSize= nbNeuron*np.ones((nbLayer,), dtype=np.int32)

xInit = np.ones(d, dtype=np.float32) 

# create the model
model = mod.Burgers(xInit, muScal, sigScal, T, nu)

print(" WILL USE " + str(_MULTIPROCESSING_CORE_COUNT) + " THREADS ")

print("REAL IS ", model.Sol(0.,xInit.reshape(1,d),1000000), " DERIV", model.derSol(0.,xInit.reshape(1,d),1000000))

theNetwork = net.FeedForwardUZ(d,layerSize,tf.nn.tanh)

ndt = [ 120 ] 


print("PDE Burgers HJE  Dim ", d, " layerSize " , layerSize,  " rescal " ,rescal, "T ", T , "batchsize ",batchSize, " batchSizeVal ", batchSizeVal, "num_epoch " , num_epoch, " num_epochExtNoLast ", num_epochExtNoLast  ,"VOL " , sigScal, "initialLearningRateNoLast " , initialLearningRateNoLast)

# nest on ndt
for indt  in ndt:

    print("NBSTEP",indt)
    # create graph
    resol =  solv.SemiHJE(model, T, indt, theNetwork, initialLearningRateNoLast = initialLearningRateNoLast)
    
    baseFile = "BurgersHJEd"+str(d)+"nbNeur"+str(layerSize[0])+"nbHL"+str(len(layerSize))+"ndt"+str(indt)
    plotFile = "pictures/"+baseFile
    saveFolder = "save/"

    Y0List = []
    for i in range(nTest):
        # train
        t0 = time.time()
        Y0, Z0  = resol.BuildAndtrainML( batchSize, batchSizeVal, num_epochExtNoLast=num_epochExtNoLast ,num_epoch=num_epoch,  nbOuterLearning=nbOuterLearning, thePlot= plotFile , baseFile = baseFile, saveDir= saveFolder)
        t1 = time.time()
        print(" NBSTEP", indt, " EstimMC Val is " , Y0,  " REAL IS ", model.Sol(0.,xInit.reshape(1,d)),    " Z0 ", Z0," DERREAL IS  ",model.derSol(0.,xInit.reshape(1,d)),t1-t0)

        Y0List.append(Y0)
        print(Y0List)

    print("Y0", Y0List)
    yList = np.array(Y0List)
    yMean = np.mean(yList)
    print(" DNT ", indt , "MeanVal ", yMean, " Etyp ", np.sqrt(np.mean(np.power(yList-yMean,2.))))
