import tensorflow as tf
import tensorflow_probability as tfp
tfd = tfp.distributions
psd_kernels = tfp.math.psd_kernels
import pandas as pd
import numpy as np
from tensorflow_probability import distributions as tfd

def placeinducingpoints(X,S,C,M=16):
    """
    set up inducing point locations - evenly spaced.
    
    """
    Z = np.linspace(np.min(X[:,0]),np.max(X[:,0]),M)[None,:]
    Z = np.repeat(Z,S,0)
    Z = Z.flatten()[:,None]
    #duplicate for different sensors...
    Z = np.c_[Z,np.repeat(np.arange(S),M)]
    #add different components...
    newZ = np.zeros([0,3])
    for compi in range(C):
        newZ = np.r_[newZ,np.c_[Z,np.full(len(Z),compi)]]
    return newZ
    
    
class Kernel():
    def __init__(self,gpflowkernel):
        self.gpflowkernel = gpflowkernel
    def matrix(self,X1,X2):    
        cov = self.gpflowkernel.K(X1,X2) * ((X2[:,1][None,:]==X1[:,1][:,None]) & (X2[:,2][None,:]==X1[:,2][:,None]))
        return cov.numpy().astype(np.float32)
        
        
class EQ(Kernel):
    def __init__(self,ls=20,v=0.5):
        """
        An EQ kernel built so that they return zero unless both the second
        and third column are equal between the two inputs.
        """
        self.ls = ls
        self.v = v
    def k(self,x1,x2):
        """
        For two points, x1 and x2, computes their covariance for this kernel
        """
        return self.v*np.exp(-(x1-x2)**2/(2*self.ls**2))  #+0.5**2
    
    def matrix(self,X1,X2):
        """For locations defined by X1 and X2 compute covariance matrix"""    
        try:
            X1 = X1.numpy()
            X2 = X2.numpy()
        except AttributeError:
            pass
        r = np.zeros([X1.shape[0],X2.shape[0]])
        for i1,x1 in enumerate(X1):
            for i2,x2 in enumerate(X2):
                    if (x1[1]==x2[1]) and (x1[2]==x2[2]): #if same sensor and component
                        r[i1,i2] = self.k(x1[0],x2[0])
        return tf.Variable(r,dtype=tf.float32)
        

        
        
class SparseModel():
    def __init__(self,X,Z,C,k):
        """
        Precompute matrices for VI.
        X = a (2*C*N) x 3 matrix
        each pair of rows, from the top and bottom half of the matrix refer to a pair
        of observations taken by two sensors at the same(ish) place and time.
        Within these two halves, there are C submatrices, of shape (Cx3), each one is
        for a particular component of the calibration function.
        
        Z = a (2*C*M) x 3 matrix
        
        C = number of components/parameters the transform requires (e.g. a straight line requires two).
        k = the kernel object
        """
        
        self.jitter = 1e-4
        
        self.k = k
        self.Kzz = k.matrix(Z,Z)+np.eye(Z.shape[0],dtype=np.float32)*self.jitter
        self.Kxx = k.matrix(X,X)+np.eye(X.shape[0],dtype=np.float32)*self.jitter
        self.Kxz = k.matrix(X,Z)
        self.Kzx = tf.transpose(self.Kxz)
        self.KzzinvKzx = tf.linalg.solve(self.Kzz,self.Kzx)
        self.KxzKzzinv = tf.transpose(self.KzzinvKzx)
        self.KxzKzzinvKzx = self.Kxz @ self.KzzinvKzx
        self.C = C
        self.Npairs = int(X.shape[0]/(C*2)) #actual number of observation *pairs*
        self.N = int(X.shape[0]/C)
    def get_qf(self,mu,scale):
        qf_mu = self.KxzKzzinv @ mu
        qf_cov = self.Kxx - self.KxzKzzinvKzx + self.KxzKzzinv @ getcov(scale) @ self.KzzinvKzx
        return qf_mu,qf_cov
    def get_samples(self,mu,scale,num=100):
        """
        Get samples of the function components for every observation pair in X.
        Returns a num x N x (C*2) matrix,
          where num = number of samples
                N = number of observation pairs
                C = number of components
        So for the tensor that is returned, the last dimension consists of the pairs of
        sensors, with each pair being one of the C components.
        """
        qf_mu,qf_cov = self.get_qf(mu,scale)
        batched_mu = tf.transpose(tf.reshape(qf_mu,[2*self.C,self.Npairs]))
        batched_cov = []
        for ni in range(0,self.Npairs*self.C*2,self.Npairs):
            innerbcov = []
            for nj in range(0,self.Npairs*self.C*2,self.Npairs):
                innerbcov.append(tf.linalg.diag_part(qf_cov[ni:(ni+self.Npairs),nj:(nj+self.Npairs)]))
            batched_cov.append(innerbcov)
        samps = tfd.MultivariateNormalFullCovariance(batched_mu,tf.transpose(batched_cov)+tf.eye(2*self.C)*self.jitter).sample(num)
        return samps
    
    def get_samples_one_sensor(self,mu,scale,num=100):
        """
        Get samples of the function components for a sensor.
        Returns a num x N x (C) matrix,
          where num = number of samples
                N = number of observation pairs
                C = number of components
        So for the tensor that is returned, the last dimension consists of the pairs of
        sensors, with each pair being one of the C components.
        """
        
        
        qf_mu,qf_cov = self.get_qf(mu,scale)
        batched_mu = tf.transpose(tf.reshape(qf_mu,[self.C,self.N]))
        batched_cov = []
        for ni in range(0,self.N*self.C,self.N):
            innerbcov = []
            for nj in range(0,self.N*self.C,self.N):
                innerbcov.append(tf.linalg.diag_part(qf_cov[ni:(ni+self.N*2),nj:(nj+self.N)]))
            batched_cov.append(innerbcov)
        samps = tfd.MultivariateNormalFullCovariance(batched_mu,tf.transpose(batched_cov)+tf.eye(self.C)*self.jitter).sample(num)
        return samps
    
def getcov(scale):
    return tf.linalg.band_part(scale, -1, 0) @ tf.transpose(tf.linalg.band_part(scale, -1, 0))

class CalibrationSystem():
    def __init__(self,X,Y,Z,refsensor,N,C,transform_fn,gpflowkernel,likelihoodstd=1.0):
        self.k = Kernel(gpflowkernel)
        self.likelihoodstd = likelihoodstd
        self.X = X
        self.Y = Y
        self.Z = Z
        self.N = N
        self.refsensor = refsensor.astype(np.float32)
        #definition of q(u)
        M = Z.shape[0]
        self.mu = tf.Variable(0.01*tf.random.normal([M,1]))
        self.scale = tf.Variable(1*np.tril(0.1*np.random.randn(M,M)+np.eye(M)),dtype=tf.float32)

        self.jitter = 1e-4
        #parameters for p(u)
        mu_u = tf.zeros([M],dtype=tf.float32)
        cov_u = tf.Variable(self.k.matrix(Z,Z),dtype=tf.float32)
        self.pu = tfd.MultivariateNormalFullCovariance(mu_u,cov_u+np.eye(cov_u.shape[0])*self.jitter)

        self.ref = tf.gather(self.refsensor,tf.transpose(tf.reshape(X[:(2*N),1:2].astype(int),[2,N])))
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=0.05,amsgrad=False)
        self.transform_fn = transform_fn
        self.sm = SparseModel(X,Z,C,self.k)
    def compute(self,its=700,samples=100):
        for it in range(its):
            with tf.GradientTape() as tape:
                qu = tfd.MultivariateNormalTriL(self.mu[:,0],self.scale)
                samps = self.sm.get_samples(self.mu,self.scale,samples)
                #self.samps = samps
                #break
                scaled = (self.transform_fn(samps,self.Y) * (1-self.ref)) + (self.Y * self.ref)
                ell = tf.reduce_mean(tf.reduce_sum(tfd.Normal(0,self.likelihoodstd).log_prob(scaled[:,:,0]-scaled[:,:,1]),1))
                elbo_loss = -ell+tfd.kl_divergence(qu,self.pu)
                gradients = tape.gradient(elbo_loss, [self.mu,self.scale])
                self.optimizer.apply_gradients(zip(gradients, [self.mu, self.scale]))  

