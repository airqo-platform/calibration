import numpy as np
import tensorflow_probability as tfp

def MAE(x,y):
    """Mean absolute error"""
    return np.mean(np.abs(x-y))

def MSE(x,y):
    """
    Mean Squared Error"""
    return np.mean((x-y)**2)

def NMSE(x,y):
    """Normalised Mean Squared Error
    x = correct
    y = estimate
    see https://math.stackexchange.com/questions/488964/the-definition-of-nmse-normalized-mean-square-error
    """
    return MSE(x,y)/MSE(x,0)

def NLPD(x,y,ystd):
    """(normalised) Negative Log Predictive Density
    
    Definition of Negative Log Predictive Density (NLPD):

    $$L = -\frac{1}{n} \sum_{i=1}^n \log p(y_i=t_i|\mathbf{x}_i)$$

    See http://mlg.eng.cam.ac.uk/pub/pdf/QuiRasSinetal06.pdf, page 13.

    "This loss penalizes both over and under-confident predictions."
    but
    "The NLPD loss favours conservative models, that is models that tend to beunder-confident
    rather than over-confident. This is illustrated in Fig. 7, and canbe deduced from the fact that
    logarithms are being used. An interesting way ofusing the NLPD is to give it relative to the NLPD
    of a predictor that ignoresthe inputs and always predicts the same Gaussian predictive distribution,
    withmean and variance the empirical mean and variance of the training data. Thisrelative NLPD
    translates into a gain of information with respect to the simpleGaussian predictor described."
    """
    return -np.mean(tfp.distributions.Normal(y,ystd).log_prob(x))
    
def compute_test_data(X,Y,trueY,refsensor):
    """
    This method provides three matrices,
         testX, testY, testtrueY
    these are associated with all observations that aren't reference sensors
    """
    testX = np.zeros([0,3])
    testY = np.zeros([0,2])
    testtrueY = np.zeros([0,1])
    C = 1
    for flip in [0,1]:
        #flip sensors
        X = np.c_[X[:,0],X[:,2],X[:,1]]
        Y = np.c_[Y[:,1],Y[:,0]]
        keep = ~np.isin(X[:,1],np.where(refsensor)[0]) #X[:,1]>-100
        testX = np.r_[testX, X[keep,:].copy()]
        testX[:,2] = 0
        testtrueY = np.r_[testtrueY,trueY[keep,None]]
        testY = np.r_[testY,Y[keep,:]]
    return testX, testY, testtrueY
