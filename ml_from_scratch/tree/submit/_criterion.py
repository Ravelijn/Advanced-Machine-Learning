import numpy as np


# CLASSIFICATION IMPURITY

    #private function to define the gini impurity
def gini(self,D):
    #initialize the output
    G = 0
    #iterrate through the unique classes
    for c,w in zip(np.unique(D[:,-1]),self.class_weights):
        #compute p for the current c
        p = w*D[D[:,-1]==c].shape[0]/D.shape[0]
        #compute term for the current c
        G += p*(1-p)
    #return gini impurity
    return(G)
    
    #private function to define the shannon entropy
def entropy(self,D):
    #initialize the output
    H = 0
    #iterrate through the unique classes
    for c,w in zip(np.unique(D[:,-1]),self.class_weights):
    #compute p for the current c
        p = w*D[D[:,-1]==c].shape[0]/D.shape[0]
        #compute term for the current c
        H -= p*np.log2(p)
    #return entropy
    return(H)
    
    #protected function to define the impurity
def impurity(self,D):
        #use the selected loss function to calculate the node impurity
    ip = None
    if self.loss == 'gini':
        ip = self.__gini(D)
    elif self.loss == 'entropy':
        ip = self.__entropy(D)
        #return results
    return(ip)
    
    #protected function to compute the value at a leaf node
def leaf_value(self,D):
        return(stats.mode(D[:,-1])[0])
     
    #public function to return model parameters
def get_params(self,deep=False):
    return{'max_depth':self.max_depth,
              'min_samples_split':self.min_samples_split,
              'loss':self.loss,
              'balance_class_weights':self.balance_class_weights}
    
    #train the tree model
def fit(self,Xin,Yin):
        #check if class weights need to be computed?
    if self.balance_class_weights:
        self.class_weights = Yin.shape[0]/(np.unique(Yin).shape[0]*np.bincount(Yin.flatten().astype(int)))
    else:
        self.class_weights = np.ones(np.unique(Yin).shape[0])
        #call the base fit function
    super().fit(Xin, Yin)
    
    
# REGRESSION IMPURITY  
def mse(self, D: np.array) -> float:
    """
    Private function to define the mean squared error
        
    Input:
        D -> data to compute the MSE over
    Output:
        Mean squared error over D
    """
    # compute the mean target for the node
    y_m = np.mean(D[:,-1])
    # compute the mean squared error wrt the mean
    E = np.sum((D[:,-1] - y_m)**2)/D.shape[0]
    # return mse
    return(E)
    
def mae(self, D: np.array) -> float:
    """
    Private function to define the mean absolute error
        
    Input:
        D -> data to compute the MAE over
    Output:
        Mean absolute error over D
    """
    # compute the mean target for the node
    y_m = np.mean(D[:,-1])
    # compute the mean absolute error wrt the mean
    E = np.sum(np.abs(D[:,-1] - y_m))/D.shape[0]
    # return mae
    return(E)
    
def impurity(self, D: np.array) -> float:
    """
    Protected function to define the impurity
        
    Input:
        D -> data to compute the impurity metric over
    Output:
        Impurity metric for D        
    """            
    # use the selected loss function to calculate the node impurity
    ip = None
    if self.loss == 'mse':
        ip = self.__mse(D)
    elif self.loss == 'mae':
        ip = self.__mae(D)
    # return results
    return(ip)
    
def leaf_value(self, D: np.array) -> float:
    """
    Protected function to compute the value at a leaf node
        
    Input:
        D -> data to compute the leaf value
    Output:
        Mean of D           
    """
    return(np.mean(D[:,-1]))   
    