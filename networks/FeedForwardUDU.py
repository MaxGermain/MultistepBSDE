import numpy as np
import tensorflow as tf
from tensorflow.contrib.slim import fully_connected as fc

# Feed forward classical
class FeedForwardUDU:

    def __init__(self, d,  layerSize, activation):
        self.d=d
        self.layerSize = layerSize
        self.activation= activation

    # create the netwrok fow scratch
    # X generic input (nbsamp, dim)
    # calculate U 
    def createNetworkUOnly(self, X,iStep):
        with tf.compat.v1.variable_scope("NetWork"+str(iStep) , reuse=tf.compat.v1.AUTO_REUSE):
            fPrev= fc(X, int(self.layerSize[0]), scope='enc_fc1', activation_fn=self.activation)
            for i in np.arange(len(self.layerSize)-1):
                scopeName='enc_fc'+str(i+2)
                f = fc(fPrev,int(self.layerSize[i+1]), scope=scopeName, activation_fn=self.activation)
                fPrev = f
            UZ  = fc(fPrev,1, scope='uPDu',activation_fn= None)
        return  UZ[:,0]

    # create the netwrok fow scratch
    # X generic input (nbsamp, dim)
    # calculate U and DU
    def createNetwork(self, X,iStep, renormalizeFactor):
        with tf.compat.v1.variable_scope("NetWork"+str(iStep) , reuse=tf.compat.v1.AUTO_REUSE):
            fPrev= fc(X, int(self.layerSize[0]), scope='enc_fc1', activation_fn=self.activation)
            for i in np.arange(len(self.layerSize)-1):
                scopeName='enc_fc'+str(i+2)
                f = fc(fPrev,int(self.layerSize[i+1]), scope=scopeName, activation_fn=self.activation)
                fPrev = f
            UZ  = fc(fPrev,self.d+1, scope='uPDu',activation_fn= None)
        return  UZ[:,0], UZ[:,1:]

    # create network for control only : so DU
    def createNetworkDU(self, X, iStep):
        with tf.compat.v1.variable_scope("NetWork"+str(iStep), reuse=tf.compat.v1.AUTO_REUSE):
            fPrev= fc(X, int(self.layerSize[0]), scope='enc_fc1', activation_fn=self.activation)
            for i in np.arange(len(self.layerSize)-1):
                scopeName='enc_fc'+str(i+2)
                f = fc(fPrev,int(self.layerSize[i+1]), scope=scopeName, activation_fn=self.activation)
                fPrev = f
            Z  = fc(fPrev,self.d, scope='uPDu',activation_fn= None)
        return Z

    # create the network with weight initialiazers
    # X      :  generic input
    # weightInit : W matrix used to initialize
    # biasInit   :  bias used to initialize
    def createNetworkWithInitializer(self, X, iStep, weightInit, biasInit, renormalizeFactor):
        with tf.compat.v1.variable_scope("NetWork"+str(iStep) , reuse=tf.compat.v1.AUTO_REUSE):
            cMinW =0
            cMinB= 0
            fPrev= fc(X, int(self.layerSize[0]), scope='enc_fc1', activation_fn=self.activation,  weights_initializer=tf.constant_initializer(np.reshape(weightInit[:self.d*int(self.layerSize[0])],[self.d,self.layerSize[0]])), biases_initializer= tf.constant_initializer(biasInit[0:int(self.layerSize[0])]) )
            cMinW += self.d*int(self.layerSize[0])
            cMinB +=  int(self.layerSize[0])
            for i in np.arange(len(self.layerSize)-1):
                scopeName='enc_fc'+str(i+2)
                f = fc(fPrev,int(self.layerSize[i+1]), scope=scopeName, activation_fn=self.activation,  weights_initializer=tf.constant_initializer(np.reshape(weightInit[cMinW : cMinW+int(self.layerSize[i])*int(self.layerSize[i+1])],[int(self.layerSize[i]),int(self.layerSize[i+1])])) , biases_initializer= tf.constant_initializer(biasInit[cMinB:cMinB+int(self.layerSize[i+1])]))
                cMinW+= int(self.layerSize[i])*int(self.layerSize[i+1])
                cMinB+= int(self.layerSize[i+1])
                fPrev = f
            UDU  = fc(fPrev,self.d+1, scope='uPDu',activation_fn= None, weights_initializer=tf.constant_initializer(np.reshape(weightInit[cMinW:cMinW+int(self.layerSize[len(self.layerSize)-1])*(self.d+1)],[int(self.layerSize[len(self.layerSize)-1]),self.d+1])), biases_initializer= tf.constant_initializer(biasInit[cMinB:cMinB+(self.d+1)]))
        return  UDU[:,0], UDU[:,1:]
    
    # get back weights and bias associated to the network
    def getBackWeightAndBias(self,iStep):
        Weights = []
        Bias = []
        with tf.compat.v1.variable_scope("NetWork"+str(iStep), reuse=tf.compat.v1.AUTO_REUSE):
            Weights.append(tf.get_variable("enc_fc1/weights", [self.d,int(self.layerSize[0])]))
            Bias.append(tf.get_variable("enc_fc1/biases", [int(self.layerSize[0])]))
            for i in np.arange(len(self.layerSize)-1):
                scopeName='enc_fc'+str(i+2)
                Weights.append(tf.get_variable(scopeName+"/weights", [int(self.layerSize[i]),int(self.layerSize[i+1])]))
                Bias.append(tf.get_variable(scopeName+"/biases", [int(self.layerSize[i+1])]))
            Weights.append(tf.get_variable("uPDu/weights",[int(self.layerSize[len(self.layerSize)-1]),self.d+1]))
            Bias.append(tf.get_variable("uPDu/biases",[self.d+1]))
        return Weights, Bias

    # transfrom the list of weight in a single weight array (idem for bias)
    def getWeights( self, sess, weightLoc, biasLoc):
        # get back weight
        weights =sess.run(weightLoc)
        bias =  sess.run(biasLoc)
        return np.concatenate([ x.flatten() for x in weights]), np.concatenate([x.flatten() for x in bias])
        

# Feed forward classical, but automatic diff for u
class FeedForwardUDUAutoDiff:

    def __init__(self, d,  layerSize, activation):
        self.d=d
        self.layerSize = layerSize
        self.activation= activation
 

    # create the netwrok fow scratch
    # X generic input (nbsamp, dim)
    # calculate U and DU
    def createNetwork(self, X,iStep,renormalizeFactor):
        with tf.compat.v1.variable_scope("NetWork"+str(iStep) , reuse=tf.compat.v1.AUTO_REUSE):
            fPrev= fc(X, int(self.layerSize[0]), scope='enc_fc1', activation_fn=self.activation)
            for i in np.arange(len(self.layerSize)-1):
                scopeName='enc_fc'+str(i+2)
                f = fc(fPrev,int(self.layerSize[i+1]), scope=scopeName, activation_fn=self.activation)
                fPrev = f
            U  = fc(fPrev,1, scope='U',activation_fn= None)
            DU = tf.gradients(U,X)
        return  U[:,0], DU[0]/renormalizeFactor



    # create the network with weight initialiazers
    # X      :  generic input
    # weightInit : W matrix used to initialize
    # biasInit   :  bias used to initialize
    def createNetworkWithInitializer(self, X, iStep, weightInit, biasInit, renormalizeFactor):
        with tf.compat.v1.variable_scope("NetWork"+str(iStep) , reuse=tf.compat.v1.AUTO_REUSE):
            cMinW =0
            cMinB= 0
            fPrev= fc(X, int(self.layerSize[0]), scope='enc_fc1', activation_fn=self.activation,  weights_initializer=tf.constant_initializer(np.reshape(weightInit[:self.d*int(self.layerSize[0])],[self.d,int(self.layerSize[0])])), biases_initializer= tf.constant_initializer(biasInit[0:int(self.layerSize[0])]) )
            cMinW += self.d*int(self.layerSize[0])
            cMinB += int( self.layerSize[0])
            for i in np.arange(len(self.layerSize)-1):
                scopeName='enc_fc'+str(i+2)
                f = fc(fPrev,int(self.layerSize[i+1]), scope=scopeName, activation_fn=self.activation,  weights_initializer=tf.constant_initializer(np.reshape(weightInit[cMinW : cMinW+int(self.layerSize[i])*int(self.layerSize[i+1])],[int(self.layerSize[i]),int(self.layerSize[i+1])])) , biases_initializer= tf.constant_initializer(biasInit[cMinB:cMinB+int(self.layerSize[i+1])]))
                cMinW+= int(self.layerSize[i])*int(self.layerSize[i+1])
                cMinB+= int(self.layerSize[i+1])
                fPrev = f
            U  = fc(fPrev,1, scope='U',activation_fn= None, weights_initializer=tf.constant_initializer(np.reshape(weightInit[cMinW:cMinW+int(self.layerSize[len(self.layerSize)-1])],
                                                                                                                      [int(self.layerSize[len(self.layerSize)-1]),1])),
                    biases_initializer= tf.constant_initializer(biasInit[cMinB:cMinB+1]))
            DU = tf.gradients(U,X)
        return  U[:,0], DU[0]/renormalizeFactor
    
    # get back weights and bias associated to the network
    def getBackWeightAndBias(self,iStep):
        Weights = []
        Bias = []
        with tf.compat.v1.variable_scope("NetWork"+str(iStep), reuse=tf.compat.v1.AUTO_REUSE):
            Weights.append(tf.compat.v1.get_variable("enc_fc1/weights", [self.d,int(self.layerSize[0])]))
            Bias.append(tf.compat.v1.get_variable("enc_fc1/biases", [int(self.layerSize[0])]))
            for i in np.arange(len(self.layerSize)-1):
                scopeName='enc_fc'+str(i+2)
                Weights.append(tf.compat.v1.get_variable(scopeName+"/weights", [int(self.layerSize[i]),int(self.layerSize[i+1])]))
                Bias.append(tf.compat.v1.get_variable(scopeName+"/biases", [int(self.layerSize[i+1])]))
            Weights.append(tf.compat.v1.get_variable("U/weights",[int(self.layerSize[len(self.layerSize)-1]),1]))
            Bias.append(tf.compat.v1.get_variable("U/biases",[1]))
        return Weights, Bias

    # transfrom the list of weight in a single weight array (idem for bias)
    def getWeights( self, sess, weightLoc, biasLoc):
        # get back weight
        weights =sess.run(weightLoc)
        bias =  sess.run(biasLoc)
        return np.concatenate([ x.flatten() for x in weights]), np.concatenate([x.flatten() for x in bias])



    
    
# Feed forward classical
# the network takes X and time as input
class FeedForwardUDUTime:

    def __init__(self, d,  layerSize, activation):
        self.d=d
        self.layerSize = layerSize
        self.activation= activation

    # create the netwrok fow scratch
    # t time
    # X generic input (nbsamp, dim)
    def createNetwork(self, t, x):
        time_and_X = tf.concat([t,x], axis=-1) 
        print("TIMAN ", time_and_X )
        with tf.variable_scope("NetWork" , reuse=tf.AUTO_REUSE):
            fPrev= fc(time_and_X, self.layerSize[0], scope='enc_fc1', activation_fn=self.activation)
            for i in np.arange(len(self.layerSize)-1):
                scopeName='enc_fc'+str(i+2)
                f = fc(fPrev,self.layerSize[i+1], scope=scopeName, activation_fn=self.activation)
                fPrev = f
            UDU  = fc(fPrev,self.d+1, scope='uPDu',activation_fn= None)
        return  UDU

# Feed forward classical
# the network takes X and time as input
# Use automatic differentiaton for  derivatives
class FeedForwardUDUTimeAutoDiff:

    def __init__(self, d,  layerSize, activation, model):
        self.d=d
        self.layerSize = layerSize
        self.activation= activation
        self.model = model # save model

    # create the netwrok fow scratch
    # t time
    # X generic input (nbsamp, dim)
    def createNetwork(self, t, x):
        time_and_X = tf.concat([t,x], axis=-1)
        with tf.variable_scope("NetWork" , reuse=tf.AUTO_REUSE):
            fPrev= fc(time_and_X, self.layerSize[0], scope='enc_fc1', activation_fn=self.activation)
            for i in np.arange(len(self.layerSize)-1):
                scopeName='enc_fc'+str(i+2)
                f = fc(fPrev,self.layerSize[i+1], scope=scopeName, activation_fn=self.activation)
                fPrev = f
            U  = fc(fPrev,1, scope='uPDu',activation_fn= None)
            DU = tf.gradients(U/self.model.renormalizeFactor(),x)
        return  tf.concat([U, DU[0]],axis=-1)

    
# Feed forward classical
# the network takes X and time as input
class FeedForwardUTime:

    def __init__(self, d,  layerSize, activation):
        self.d=d
        self.layerSize = layerSize
        self.activation= activation

    # create the netwrok fow scratch
    # t time
    # X generic input (nbsamp, dim)
    def createNetwork(self, t, x):
        time_and_X = tf.concat([t,x], axis=-1) 
        with tf.variable_scope("NetWork" , reuse=tf.AUTO_REUSE):
            fPrev= fc(time_and_X, self.layerSize[0], scope='enc_fc1', activation_fn=self.activation)
            for i in np.arange(len(self.layerSize)-1):
                scopeName='enc_fc'+str(i+2)
                f = fc(fPrev,self.layerSize[i+1], scope=scopeName, activation_fn=self.activation)
                fPrev = f
            U  = fc(fPrev,1, scope='uPDu',activation_fn= None)
        return  U

