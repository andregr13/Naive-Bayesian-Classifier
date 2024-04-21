import numpy as np
from scipy.stats import norm

class MyNBClassifier():

    def __init__(self, priors=None, dists=None, attr=None):
        self.priors = None
        self.dists = None
        self.attr = None
    
    # fits model to given training set
    # by best fitting gaussian dist. for each feature of x
    def fit(self, x, y):

        num_class = len(np.unique(y))
        num_feats = x.shape[1]
        
        # initializing class variable
        self.dists = np.empty((num_class, 2, num_feats), dtype=object)
        self.attr = np.arange(num_class)
        self.priors = np.empty(num_class)
        
        # setting prior values for each class
        xs = np.empty((num_class), dtype=object)
        for i in range(num_class):
            xs[i] = x[y == i]
            self.priors[i] = np.count_nonzero(y == i)/y.size
        
        # recording distribution mean and variances 
        # done for each feature to use for predictions
        for j in range(num_class):
            self.dists[j][0] = np.mean(xs[j], axis=0)
            self.dists[j][1] = np.std(xs[j], axis=0) 
        return
    
    # function to estimate bayes theorem posterior 
    # assumes indepence of each feature set
    def bayes(self,test_x,c):
        
        # p(c|test_x) = prior[c] * product( p(x_i | c) ) for all i
        prod = self.priors[c]
        for i in range(test_x.shape[1]):
            prod *= norm.pdf(test_x[:,i], self.dists[c][0][i], self.dists[c][1][i]) 
            
        return prod
    
    # calculates esitmated posterior probability for each class
    # returns the class index with highest probability
    def predict(self,test_x):
        probs = np.empty((self.attr.size,test_x.shape[0]))
        for c in range(self.attr.size):
            probs[c] = self.bayes(test_x, c)
        return np.argmax(probs,axis=0)
