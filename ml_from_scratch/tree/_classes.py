"""
This module gathers tree-based methods
"""

import numpy as np
from scipy import stats



# =============================================================================
# Base decision tree
# =============================================================================

#imports
from abc import ABC,abstractmethod
import numpy as np

#class to control tree node
class Node:
    #initializer
    def __init__(self):
        self.__Bs    = None
        self.__Bf    = None
        self.__left  = None
        self.__right = None
        self.leafv   = None

    #set the split,feature parameters for this node
    def set_params(self,Bs,Bf):
        self.__Bs = Bs
        self.__Bf = Bf
        
    #get the split,feature parameters for this node
    def get_params(self):
        return(self.__Bs,self.__Bf)    
        
    #set the left/right children nodes for this current node
    def set_children(self,left,right):
        self.__left  = left
        self.__right = right
        
    #get the left child node
    def get_left_node(self):
        return(self.__left)
    
    #get the right child node
    def get_right_node(self):
        return(self.__right)
       
#base class to encompass the decision tree algorithm
class DecisionTree(ABC):
    #initializer
    def __init__(self,max_depth=None,min_samples_split=2):
        self.tree              = None
        self.max_depth         = max_depth
        self.min_samples_split = min_samples_split
        
    #protected function to define the impurity
    @abstractmethod
    def _impurity(self,D):
         pass
        
    #protected function to compute the value at a leaf node
    @abstractmethod
    def _leaf_value(self,D):
         pass
        
    #private recursive function to grow the tree during training
    def __grow(self,node,D,level):       
        #are we in a leaf node? let's do some check...
        depth = (self.max_depth is None) or (self.max_depth >= (level+1))
        msamp = (self.min_samples_split <= D.shape[0])
        n_cls = np.unique(D[:,-1]).shape[0] != 1
        
        #not a leaf node
        if depth and msamp and n_cls:
        
            #initialize the function parameters
            ip_node = None
            feature = None
            split   = None
            left_D  = None
            right_D = None
            #determine the possible features on which we can split
            features = np.random.choice([i for i in range(D.shape[1]-1)],size=int(np.sqrt(D.shape[1]-1)),replace=False)
            #iterrate through the possible feature/split combinations
            for f in features:
                for s in np.unique(D[:,f]):
                    #for the current (f,s) combination, split the dataset
                    D_l = D[D[:,f]<=s]
                    D_r = D[D[:,f]>s]
                    #ensure we have non-empty arrays, otherwise treat as a leaf node
                    if D_l.size and D_r.size:
                        #calculate the impurity
                        ip  = (D_l.shape[0]/D.shape[0])*self._impurity(D_l) + (D_r.shape[0]/D.shape[0])*self._impurity(D_r)
                        #now update the impurity and choice of (f,s)
                        if (ip_node is None) or (ip < ip_node):
                            ip_node = ip
                            feature = f
                            split   = s
                            left_D  = D_l
                            right_D = D_r    
            #check if valid parameters were found? If not, treat this as a leaf node & return
            if (split is None) or (feature is None) or (left_D is None) or (right_D is None):
                node.leafv = self._leaf_value(D)
                return
            #set the current node's parameters
            node.set_params(split,feature)
            #declare child nodes
            left_node  = Node()
            right_node = Node()
            node.set_children(left_node,right_node)
            #investigate child nodes
            self.__grow(node.get_left_node(),left_D,level+1)
            self.__grow(node.get_right_node(),right_D,level+1)
                        
        #is a leaf node
        else:
            
            #set the node value & return
            node.leafv = self._leaf_value(D)
            return
     
    #private recursive function to traverse the (trained) tree
    def __traverse(self,node,Xrow):
        #check if we're in a leaf node?
        if node.leafv is None:
            #get parameters at the node
            (s,f) = node.get_params()
            #decide to go left or right?
            if (Xrow[f] <= s):
                return(self.__traverse(node.get_left_node(),Xrow))
            else:
                return(self.__traverse(node.get_right_node(),Xrow))
        else:
            #return the leaf value
            return(node.leafv)
    
    #train the tree model
    def fit(self,Xin,Yin):
        #prepare the input data
        D = np.concatenate((Xin,Yin.reshape(-1,1)),axis=1)
        #set the root node of the tree
        self.tree = Node()
        #build the tree
        self.__grow(self.tree,D,1)
        
    #make predictions from the trained tree
    def predict(self,Xin):
        #iterrate through the rows of Xin
        p = []
        for r in range(Xin.shape[0]):
            p.append(self.__traverse(self.tree,Xin[r,:]))
        #return predictions
        return(np.array(p).flatten())
    
                 
class DecisionTreeClassifier(DecisionTree):
    #initializer
    def __init__(self,max_depth=None,min_samples_split=2,loss='gini',balance_class_weights=False):
        super().__init__(max_depth,min_samples_split)
        self.loss                  = loss   
        self.balance_class_weights = balance_class_weights
        self.class_weights         = None
    
    #private function to define the gini impurity
    def __gini(self,D):
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
    def __entropy(self,D):
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
    def _impurity(self,D):
        #use the selected loss function to calculate the node impurity
        ip = None
        if self.loss == 'gini':
            ip = self.__gini(D)
        elif self.loss == 'entropy':
            ip = self.__entropy(D)
        #return results
        return(ip)
    
    #protected function to compute the value at a leaf node
    def _leaf_value(self,D):
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

                                   
#Decision Tree Regressor
class DecisionTreeRegressor(DecisionTree):
    #initializer
    def __init__(self,max_depth=None,min_samples_split=2,loss='mse'):
        super().__init__(max_depth,min_samples_split)
        self.loss              = loss   
    
    #private function to define the mean squared error
    def __mse(self,D):
        #compute the mean target for the node
        y_m = np.mean(D[:,-1])
        #compute the mean squared error wrt the mean
        E = np.sum((D[:,-1] - y_m)**2)/D.shape[0]
        #return mse
        return(E)
    
    #private function to define the mean absolute error
    def __mae(self,D):
        #compute the mean target for the node
        y_m = np.mean(D[:,-1])
        #compute the mean absolute error wrt the mean
        E = np.sum(np.abs(D[:,-1] - y_m))/D.shape[0]
        #return mae
        return(E)
    
    #protected function to define the impurity
    def _impurity(self,D):
        #use the selected loss function to calculate the node impurity
        ip = None
        if self.loss == 'mse':
            ip = self.__mse(D)
        elif self.loss == 'mae':
            ip = self.__mae(D)
        #return results
        return(ip)
    
    #protected function to compute the value at a leaf node
    def _leaf_value(self,D):
         return(np.mean(D[:,-1]))
     
    #public function to return model parameters
    def get_params(self,deep=False):
        return{'max_depth':self.max_depth,
               'min_samples_split':self.min_samples_split,
               'loss':self.loss}
