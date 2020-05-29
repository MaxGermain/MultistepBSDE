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

nbLayer= 3 
print("nbLayer " ,nbLayer)
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

d = 15
nbNeuron = d + 10
sigScal =  0.5*np.ones(d)
muScal = np.zeros(d)
lamb = 0.5

quant = 1.0 

layerSize= nbNeuron*np.ones((nbLayer,), dtype=np.int32)

xyInit = np.ones(d) 

# create the model
model = mod.ModelMongeAmpere(xyInit, muScal, sigScal, T, lamb, d)

print(" WILL USE " + str(_MULTIPROCESSING_CORE_COUNT) + " THREADS ")
print("REAL IS ", model.Sol(0.,xyInit.reshape(1,d)), " DERIV", model.derSol(0.,xyInit.reshape(1,d)), " GAMMA", model.der2Sol(0.,xyInit.reshape(1,d)))

theNetwork = net.FeedForwardUZ(d,layerSize,tf.nn.tanh)

ndt = [120] 

print("PDE Monge Ampere PWG Dim ", d, " layerSize " , layerSize,   "T ", T , "batchsize ",batchSize, " batchSizeVal ", batchSizeVal, "num_epoch " , num_epoch, " num_epochExtNoLast ", num_epochExtNoLast  , "num_epochExtLast " , num_epochExtLast,  "VOL " , sigScal, "initialLearningRateLast" , initialLearningRateLast , "initialLearningRateNoLast " , initialLearningRateNoLast, "quantile", quant)

# nest on ndt
for indt  in ndt:

    print("NBSTEP",indt)
    # create graph
    resol =  solv.PDEFullNLExplicitGamAdapt(xyInit,model, T, indt, theNetwork , initialLearningRate=initialLearningRateLast, initialLearningRateStep = initialLearningRateNoLast)
        
    baseFile = "MongeAmperePWGd"+str(d)+"nbNeur"+str(layerSize[0])+"nbHL"+str(len(layerSize))+"ndt"+str(indt)+"quantile"+str(quant)
    plotFile = "pictures/"+baseFile
    saveFolder = "save/"

    Y0List = []
    for i in range(nTest):
        # train
        t0 = time.time()
        Y0, Z0  = resol.BuildAndtrainML( batchSize, batchSizeVal, num_epochExt=num_epochExtNoLast, num_epoch=num_epoch,  nbOuterLearning=nbOuterLearning, thePlot= plotFile , baseFile = baseFile, saveDir= saveFolder, quantile = quant)
        t1 = time.time()
        print(" NBSTEP", indt, " EstimMC Val is " , Y0,  " REAL IS ", model.Sol(0.,xyInit.reshape(1,d)),    " Z0 ", Z0," DERREAL IS  ",model.derSol(0.,xyInit.reshape(1,d)),t1-t0)

        Y0List.append(Y0)
        print(Y0List)

    print("Y0", Y0List)
    yList = np.array(Y0List)
    yMean = np.mean(yList)
    print(" DNT ", indt , "MeanVal ", yMean, " Etyp ", np.sqrt(np.mean(np.power(yList-yMean,2.))))
