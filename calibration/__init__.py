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
        A Gaussian process Sparse model for performing Variational Inference.
        
        Parameters:
        X = a (2*C*N) x 3 matrix
        each pair of rows, from the top and bottom half of the matrix refer to a pair
        of observations taken by two sensors at the same(ish) place and time.
        Within these two halves, there are C submatrices, of shape (Cx3), each one is
        for a particular component of the calibration function.
        Columns: time, sensor, component
        
        Z = a (2*C*M) x 3 matrix, same structure as for X.
        Columns: time, sensor, component        
        
        C = number of components/parameters the transform requires (e.g. a straight line requires two).
        k = the kernel object
        
        Note the constructor precomputes some matrices for the VI.
        
        This class doesn't hold the variational approximation's mean and covariance,
        but assumes a Gaussian that is used for sampling by calling 'get_samples'
        or 'get_samples_one_sensor'.
        These two methods take mu and scale, which describe the approximation.
        If you set scale to None then it assumes you are not modelling the covariance
        and returns a single 'sample' which is the posterior mean prediction.
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
        if scale is None:
            qf_cov = None
        else:
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
        
        If scale is set to None, then we return a single sample, of the posterior mean
        (i.e. we assume a dirac q(f).
        Returns 1 x N x (C*2) matrix.
        
        """                
        qf_mu,qf_cov = self.get_qf(mu,scale)
        if scale is None:
            return tf.transpose(tf.reshape(qf_mu,[2*self.C,self.Npairs]))[None,:,:]
            
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
        if scale is None:
            return tf.transpose(tf.reshape(qf_mu,[self.C,self.N]))[None,:,:]
            
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
    def __init__(self,X,Y,Z,refsensor,C,transform_fn,gpflowkernel,likemodel='fixed',gpflowkernellike=None,likelihoodstd=1.0,jitter=1e-4,lr=0.02,likelr=None):
        """
        A tool for running the calibration algorithm on a dataset, produces
        estimates of the calibration parameters over time for each sensor.
        
        The input refers to pairs of colocated observations. N = the number of
        these pairs of events.
        
        Parameters
        ----------
        X : An N x 3 matrix. The first column is the time of the observation
        pair, the second and third columns are the index of the sensors.
        Y : An N x 2 matrix. The two columns are the observations from the two
        sensors at the colocation event.
        Z : Either an M x 1 matrix or an M*S x 2 matrix. The former just has
        the inducing input locations specfied in time. The latter also specifies
        their associated sensor in the second column.
        refsensor : a binary, S dimensional vector. 1=reference sensor.
                    (number of sensors inferred from this vector)
        C : number of components required for transform_fn. Scaling will require
        just one. While a 2nd order polynomial will require 3.
        transform_fn : A function of the form: def transform_fn(samps,Y),
           where samps's shape is: [batch (number of samples) 
                                     x number of observations (N) 
                                     x (number of components) (C)].
                 Y's shape is [number of observations (N) x 1].
        gpflowkernel : A one dimensional kernel that describes all components.
          TODO: We should allow different kernels for different components.
                If one wishes to now, then after calling the constructor,
                set self.k with an alternative kernel object.
                
        likemodel : specifies how the likelihood is modelled.
        It can be one of four values:
          - fixed [default, uses the value in likelihoodstd]
          - [not yet] single [optimise a single value [TODO Not Implemented]]!!
          - distribution [uses gpflowkernellike]
          - process [uses gpflowkernellike]                
        likelihoodstd : The standard deviation of the likelihood function,
                which by default computes the difference between observation
                pairs (default=1.0). The self.likelihoodfn could be set to
                a different likelihood.
        jitter : Jitter added to ensure stability (default=1e-4).
        lr, likelr : learning rates.
                             
        TODO what happens if refsensor is integer?
                          if float64 used for others..
        
        """

        #internally we use a different X and Z:
        #we add additional rows to X and Z to account for the components we are
        #modelling. In principle the different components could have different
        #inducing points and kernels, but for simplicity we combine them.
        #These two matrices have three columns, the time, the sensor and the
        #component. They are constructed as below, with the sensor measurement
        #pairs in cosecutive submatrices, which is iterated over C times.
        #Time Sensor Component
        #  1    0    0
        #  2    0    0
        #  1    1    0
        #  2    1    0
        #  1    0    1
        #  2    0    1
        #  1    1    1
        #  2    1    1
        
        self.likemodel = likemodel
        S = len(refsensor)
        self.C = C
        self.Y = Y        
        self.k = Kernel(gpflowkernel)
        
        self.likelihoodstd = likelihoodstd
        self.X = np.c_[np.tile(np.r_[np.c_[X[:,0],X[:,1]],np.c_[X[:,0],X[:,2]]],[self.C,1]),np.repeat(np.arange(self.C),2*len(X))]        
        
        if Z.shape[1]==1:
            Ztemp = np.c_[np.tile(Z,[S,1]),np.repeat(np.arange(S),len(Z))]
        if Z.shape[1]==2:
            Ztemp = Z
        self.Z = np.c_[np.tile(Ztemp,[C,1]),np.repeat(np.arange(self.C),len(Ztemp))]
        
        if likemodel=='distribution' or likemodel=='process':
            assert gpflowkernellike is not None, "You need to specify the kernel to use a distribution or process"
            self.klike = Kernel(gpflowkernellike)
            self.Xlike = np.c_[np.r_[np.c_[X[:,0],X[:,1]],np.c_[X[:,0],X[:,2]]],np.repeat(0,2*len(X))]
            self.Zlike = np.c_[Ztemp,np.repeat(0,len(Ztemp))]
            if likelr is None: likelr = lr * 4 #probably can optimise this a little quicker?
            self.likeoptimizer = tf.keras.optimizers.Adam(learning_rate=likelr,amsgrad=False)
            
        self.N = N = len(X)
        self.refsensor = refsensor.astype(np.float32)
        self.jitter = jitter
        self.transform_fn = transform_fn  
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=lr,amsgrad=False)    
        self.precompute()
        
    def precompute(self):    
        #definition of q(u)
        M = self.Z.shape[0]
        self.mu = tf.Variable(1.0*tf.random.normal([M,1]))
        self.scale = tf.Variable(0.1*np.tril(1.0*np.random.randn(M,M)+1.0*np.eye(M)),dtype=tf.float32)        
        #self.mu = tf.Variable(10.0*tf.random.normal([M,1]))
        #self.scale = tf.Variable(10.0*np.tril(1.0*np.random.randn(M,M)+1.0*np.eye(M)),dtype=tf.float32)
        
        
        
        if self.likemodel=='distribution' or self.likemodel=='process':
            Mlike = self.Zlike.shape[0]
            self.mulike = tf.Variable(0.0001*tf.random.normal([Mlike,1]))
            mu_u = tf.Variable(np.full([Mlike],-12),dtype=tf.float32)
            cov_u = tf.Variable(self.klike.matrix(self.Zlike,self.Zlike),dtype=tf.float32)
            self.pulike = tfd.MultivariateNormalFullCovariance(mu_u,cov_u+np.eye(cov_u.shape[0])*self.jitter)
             #0.8
            self.smlike = SparseModel(self.Xlike,self.Zlike,1,self.k)
        else:
            self.mulike = None
        if self.likemodel=='process':
            self.scalelike = tf.Variable(1e-10*np.eye(Mlike),dtype=tf.float32)
        else:
            self.scalelike = None
        
        #parameters for p(u)
        mu_u = tf.zeros([M],dtype=tf.float32)
        cov_u = tf.Variable(self.k.matrix(self.Z,self.Z),dtype=tf.float32)
        self.pu = tfd.MultivariateNormalFullCovariance(mu_u,cov_u+np.eye(cov_u.shape[0])*self.jitter)
        self.ref = tf.gather(self.refsensor,tf.transpose(tf.reshape(self.X[:(2*self.N),1:2].astype(int),[2,self.N])))
        
        self.sm = SparseModel(self.X,self.Z,self.C,self.k)
        
    def likelihoodfn_nonstationary(self,scaledA,scaledB,varparamA,varparamB):
        return tfd.Normal(0,0.00001+tf.sqrt(tf.exp(varparamA)+tf.exp(varparamB))).log_prob(scaledA-scaledB)    
    
    def likelihoodfn(self,scaledA,scaledB):
        return tfd.Normal(0,self.likelihoodstd).log_prob(scaledA-scaledB)

    #@tf.function
    def run(self,its=None,samples=100,threshold=0.001):
        """ Run the VI optimisation.
        
        its: Number of iterations. Set its to None to automatically stop
        when the ELBO has reduced by less than threshold percent
        (between rolling averages of the last 50 calculations
        and the 50 before that).
        samples: Number of samples for the stochastic sampling of the
        gradient
        threshold: if its is None, this is the percentage change between
        the rolling average, over 50 iterations. Default: 0.001 (0.1%).
        """
        elbo_record = []
        it = 0
        print("Starting Run")
        try:
            while (its is None) or (it<its):
                it+=1
                with tf.GradientTape() as tape:
                    qu = tfd.MultivariateNormalTriL(self.mu[:,0],self.scale)
                    samps = self.sm.get_samples(self.mu,self.scale,samples)
                    scaled = tf.concat([self.transform_fn(samps[:,:,::2],self.Y[:,0:1]),self.transform_fn(samps[:,:,1::2],self.Y[:,1:2])],2)
                    scaled = (scaled * (1-self.ref)) + (self.Y * self.ref)
                    
                    if self.mulike is not None: #if we have non-stationary likelihood variance...
                        qulike = tfd.MultivariateNormalTriL(self.mulike[:,0],self.scalelike)              
                        like = self.smlike.get_samples(self.mulike,self.scalelike,samples)
                        ell = tf.reduce_mean(tf.reduce_sum(self.likelihoodfn_nonstationary(scaled[:,:,0],scaled[:,:,1],like[:,:,0]*(1-self.ref[:,0])-1000*self.ref[:,0],like[:,:,1]*(1-self.ref[:,1])-1000*self.ref[:,1]),1))
                    else: #stationary likelihood variance
                        ell = tf.reduce_mean(tf.reduce_sum(self.likelihoodfn(scaled[:,:,0],scaled[:,:,1]),1))
                    
                    elbo_loss = -ell+tfd.kl_divergence(qu,self.pu)
                    
                    if self.likemodel=='process':
                        assert self.mulike is not None
                        assert self.scalelike is not None
                        elbo_loss += tfd.kl_divergence(qulike,self.pulike)
                    if self.likemodel=='distribution':
                        assert self.mulike is not None
                        elbo_loss -= self.pulike.log_prob(self.mulike[:,0])

                    if it%20==0: print("%d (ELBO=%0.4f)" % (it, elbo_loss))
                    
                    if (self.mulike is None) or (it%50<25): #optimise latent fns
                        gradients = tape.gradient(elbo_loss, [self.mu,self.scale])
                        self.optimizer.apply_gradients(zip(gradients, [self.mu, self.scale]))  
                    else: #this optimises the likelihood...
                        if self.likemodel=='distribution':
                            gradients = tape.gradient(elbo_loss, [self.mulike])
                            self.likeoptimizer.apply_gradients(zip(gradients, [self.mulike]))
                        if self.likemodel=='process':
                            gradients = tape.gradient(elbo_loss, [self.mulike,self.scalelike])
                            self.likeoptimizer.apply_gradients(zip(gradients, [self.mulike,self.scalelike]))

                    elbo_record.append(elbo_loss)
                if its is None:
                    if it>100:
                        oldm = np.median(elbo_record[-100:-50])
                        m = np.median(elbo_record[-50:])
                        if np.abs((oldm-m)/((oldm+m)/2))<threshold:
                            #check that nothing weird's happened!
                            if np.std(elbo_record[-50:])<np.std(elbo_record[-100:-50]):
                                break
        except KeyboardInterrupt:
            pass
        return np.array(elbo_record)
