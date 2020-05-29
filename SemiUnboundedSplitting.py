import numpy as np
import tensorflow as tf
import os
import solvers  as solv
import networks as net
import models as mod
import multiprocessing
import math
import sys

def str_to_bool(s):
    if s == 'True':
         return True
    elif s == 'False':
         return False
    else:
         raise ValueError
     
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

_MULTIPROCESSING_CORE_COUNT = multiprocessing.cpu_count()

print("args", sys.argv)
    
# dimension
d = 1
# vol min
sigMin = 1.
# nb neuron
nbNeuron = d+ 10
# nb layer
nbLayer= 2

#initial
xInit = 0.5 * np.ones([d])
T = 1.

#forward process
SigMin = sigMin/np.sqrt(d)

batchSize= 1000
batchSizeVal= 50000
nTest = 1 
# number of iter epoch
num_epoch = 50
# number of externla iteration for gamma and first step
num_epochExt = 100 
num_epochExt0 = 1000
nbOuterLearning = 10
initialLearningRate = 0.01 # for final....
initialLearningRateStep = 0.001 # for UZ
print(" WILL USE " + str(_MULTIPROCESSING_CORE_COUNT) + " THREADS ")

#layer
layerSize= nbNeuron*np.ones((nbLayer,), dtype=np.int32)
print("typ", type(layerSize), "and ", type(layerSize[0]))
# create the model
model = mod.SemiUnbounded(sigMin, T, xInit)

# Solution
sol = model.SolPoint(0., xInit)
print("Solution", sol)

# first for U Z
theNetworkU = net.FeedForwardU(d,layerSize,tf.nn.tanh)

ndt = [120] 

print("PDE unbounded splitting SigMin", sigMin,  "  Dim ", d, " layerSize " , layerSize,   "T ", T ,  "batchsize ",batchSize, " batchSizeVal ", batchSizeVal,   "nbOuterLearning" , nbOuterLearning , "num_epochExt ", num_epochExt , " num_epochExt0 " ,  num_epochExt0 ,   "initialLearningRateStep" ,initialLearningRateStep  )

listOfVal =[]
listOfValOfVal = []
# nest on ndt
for indt  in ndt:

    print("NBSTEP",indt)
    # create graph
    resol =  solv.Splitting(xInit, model, T, indt, theNetworkU, initialLearningRate= initialLearningRate,initialLearningRateStep =  initialLearningRateStep)

    
    Y0List = []
    i=0
    baseFile = "UnboudedSplittingSigMinDim"+str(d)+"Step"+str(indt)+"Neur"+str(nbNeuron)+"NbLay"+str(nbLayer)
    while (i < nTest):
        if (i==0):
            thePlot="pictures/"+baseFile
        else:
            thePlot=""
        # train
        Y0 = resol.BuildAndtrainML( batchSize, batchSizeVal,  num_epoch=num_epoch, num_epochExt=num_epochExt, nbOuterLearning=nbOuterLearning,thePlot=thePlot, baseFile = baseFile)
        
        print(" NBSTEP", indt, " EstimMC Val is " , Y0)
        if ( not math.isnan(float(Y0))):
            listOfValOfVal.append([indt, Y0])
            Y0List.append(Y0)
            i= i+1

    print("Y0", Y0List)
    print("listOfValOfVal", listOfValOfVal)
    yList = np.array(Y0List)
    yMean = np.mean(yList)
    etyp = np.sqrt(np.mean(np.power(yList-yMean,2.)))
    print(" DNT ", indt , "MeanVal ", yMean, " Etyp ",  etyp)
    valLoc = [ indt, yMean, etyp]
    listOfVal.append(valLoc)

print("NDT, VAL, ETYP",  listOfVal)   
print("ListOfVal ", listOfValOfVal)
# Solution
sol = model.SolPoint(0., xInit)
print("Solution", sol)
print("PDE unbounded  SigMin", sigMin,  "  Dim ", d, " layerSize " , layerSize,   "T ", T ,  "batchsize ",batchSize, " batchSizeVal ", batchSizeVal,   "nbOuterLearning" , nbOuterLearning , "num_epochExt ", num_epochExt , " num_epochExt0 " ,  num_epochExt0 ,   "initialLearningRateStep" ,initialLearningRateStep  )
