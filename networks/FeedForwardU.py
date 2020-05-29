# create all the networks necessary
import numpy as np
import tensorflow as tf
from tensorflow.contrib.slim import fully_connected as fc

# Feed forward classical
class FeedForwardU:

    def __init__(self, d,  layerSize, activation):
        self.d=d
        self.layerSize = layerSize
        self.activation= activation


    # create the netwrok fow scratch
    # X generic input (nbsamp, dim)
    # calculate U and DU
    def createNetwork(self, X,iStep, renormalizeFactor):
        with tf.compat.v1.variable_scope("NetWorkU"+str(iStep) , reuse=tf.compat.v1.AUTO_REUSE):
            fPrev= fc(X, int(self.layerSize[0]), scope='enc_fc1', activation_fn=self.activation)
            for i in np.arange(len(self.layerSize)-1):
                scopeName='enc_fc'+str(i+2)
                f = fc(fPrev,int(self.layerSize[i+1]), scope=scopeName, activation_fn=self.activation)
                fPrev = f
            UZ  = fc(fPrev,1, scope='UZ',activation_fn= None)
        return  UZ[:,0]



    # create the network with weight initialiazers
    # X      :  generic input
    # weightInit : W matrix used to initialize
    # biasInit   :  bias used to initialize
    def createNetworkWithInitializer(self, X, iStep, weightInit, biasInit, renormalizeFactor, trainable= True):
        with tf.compat.v1.variable_scope("NetWorkU"+str(iStep) , reuse=tf.compat.v1.AUTO_REUSE):
            cMinW =0
            cMinB= 0
            fPrev= fc(X, int(self.layerSize[0]), scope='enc_fc1', activation_fn=self.activation,  weights_initializer=tf.constant_initializer(np.reshape(weightInit[:self.d*self.layerSize[0]],[self.d,self.layerSize[0]])), biases_initializer= tf.constant_initializer(biasInit[0:self.layerSize[0]]) , trainable= True)
            cMinW += self.d*self.layerSize[0]
            cMinB +=  self.layerSize[0]
            for i in np.arange(len(self.layerSize)-1):
                scopeName='enc_fc'+str(i+2)
                f = fc(fPrev,int(self.layerSize[i+1]), scope=scopeName, activation_fn=self.activation,  weights_initializer=tf.constant_initializer(np.reshape(weightInit[cMinW : cMinW+self.layerSize[i]*self.layerSize[i+1]],[self.layerSize[i],self.layerSize[i+1]])) , biases_initializer= tf.constant_initializer(biasInit[cMinB:cMinB+self.layerSize[i+1]]), trainable= True)
                cMinW+= self.layerSize[i]*self.layerSize[i+1]
                cMinB+= self.layerSize[i+1]
                fPrev = f
            UDU  = fc(fPrev,1, scope='UZ',activation_fn= None, weights_initializer=tf.constant_initializer(np.reshape(weightInit[cMinW:cMinW+self.layerSize[len(self.layerSize)-1]*(1)],[self.layerSize[len(self.layerSize)-1],1])), biases_initializer= tf.constant_initializer(biasInit[cMinB:cMinB+(1)]), trainable= True)
        return  UDU[:,0]


    # get back weights and bias associated to the network
    def getBackWeightAndBias(self,iStep):
        Weights = []
        Bias = []
        with tf.compat.v1.variable_scope("NetWorkU"+str(iStep), reuse=tf.compat.v1.AUTO_REUSE):
            Weights.append(tf.compat.v1.get_variable("enc_fc1/weights", [self.d,self.layerSize[0]]))
            Bias.append(tf.compat.v1.get_variable("enc_fc1/biases", [self.layerSize[0]]))
            for i in np.arange(len(self.layerSize)-1):
                scopeName='enc_fc'+str(i+2)
                Weights.append(tf.compat.v1.get_variable(scopeName+"/weights", [self.layerSize[i],self.layerSize[i+1]]))
                Bias.append(tf.compat.v1.get_variable(scopeName+"/biases", [self.layerSize[i+1]]))
            Weights.append(tf.compat.v1.get_variable("UZ/weights",[self.layerSize[len(self.layerSize)-1],1]))
            Bias.append(tf.compat.v1.get_variable("UZ/biases",[1]))
        return Weights, Bias

    # transfrom the list of weight in a single weight array (idem for bias)
    def getWeights( self, sess, weightLoc, biasLoc):
        # get back weight
        weights =sess.run(weightLoc)
        bias =  sess.run(biasLoc)
        return np.concatenate([ x.flatten() for x in weights]), np.concatenate([x.flatten() for x in bias])
        
