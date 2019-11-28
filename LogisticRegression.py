import numpy as np
import math
from sklearn.metrics import roc_auc_score

### Create best weights 
### Create auc roc score 
### Initialization schemes

def sigmoid(z):
    return 1 / (1 + np.exp(-z))


class LogisticClassifier:
    
    def __init__(self, learning_rate=0.1,
                 tol=1e-4, max_iter=1000, 
                 dropout_rate=None,early_stopping=None,intercept = False,
                validation_set=(None,None), verbose = False, seed = 42, last_n_losses = 5):
        
        """
        A custom logistic regression classifier with dropout. 
        
        Uses vanilla gradiant descent (for now).
        
        Parameters:
        -----------
        
            learning_rate (float):
                GD step size. 
            
            tol (float):
                GD converges when difference in loss in subsequent steps is less than tolerance. 
                
            max_iter (int):
                Max number of GD steps. 
                
            dropout_rate (float): 
                The % of features (excluding intercept) that will be randomly dropped at each GD step. 
                Ranges from 0 < dropout_rate < 1
                
            early_stopping (int):
                Stop if the validation loss in the last n steps, is higher on avg. than the loss in 
                the n steps prior. 
                
            validation_set (pd.DataFrame,pd.DataFrame):
                Validation datasets. Required for early stopping. 

            intercept (bool):
                Inidiate whether the model should create an intercept in the design matrix. 

            seed (int):
                For replicability with dropout

            last_n_losses (int):
                Like with early stopping, we modify the convergence check to look at the mean of the last n steps. In general,
                dropout seems to be converging too early, likely cause of an unlucky random sample resulting in minimal decrease 
                training loss. 

                
        Attributes:
        ----------
  
            
        """
        
        # Creating dropout variables
        if dropout_rate is not None and dropout_rate > 0 and dropout_rate <1:
            self.dropout_rate = dropout_rate
            self.dropout = True
            self.seed = seed
            self.last_n_losses = last_n_losses
        else:
            self.dropout = False
            self.dropout_rate = 0 
            

        
        # Parameterizing GD
        self.learning_rate = float(learning_rate)
        self.tol = float(tol)
        self.max_iter = int(max_iter)
        
        # Early stopping 
        if early_stopping is not None:
            if validation_set[0] is not None and validation_set[1] is not None:

                self.early_stopping = int(early_stopping)
                self.X_val, self.y_val = validation_set
                self.optimal_beta = None
                self.best_loss = float('inf')
                self.best_weights = None

            else:
                raise ValueError('Validation Set needed with early stopping')

        else:
            self.X_val, self.y_val = None, None
        
        # Verbose output
        self.verbose = verbose
        
        # Constructing DMatrix. 
        self.add_intercept = intercept
        
        
        self.training_loss_history = []

    def _design_matrix(self, X):
        if self.add_intercept:
            X = np.hstack([ np.ones((X.shape[0], 1)), X])
            
        return X

    def initialize(self, k):
        limit = 1 / math.sqrt(k)
        return np.random.uniform(-limit, limit, (k,))


    def fit_center_scale(self, X):
        self.means = X.mean(axis=0)
        self.standard_error = np.std(X, axis=0)

    #Trying to make it compatible with Sklearn model selection methods 

    # def set_params(self, **params):
    #     self.dropout_rate = params['dropout_rate']
    #     self.tol = params['tol']
    #     self.max_iter = params['max_iter']


    # def get_params(self, deep=True):
    #     para_dic = {
    #         'dropout_rate': self.dropout_rate,
    #         'tol':self.tol,
    #         'max_iter' : self.max_iter,
    #     }
    #     return para_dic

    
    def fit(self, X, y):
        
        self.fit_center_scale(X)

        # Creating Dmatrix
        X = self._design_matrix(X)
        n, k = X.shape
        
        # Convergence & Early stopping 
        previous_loss = -float('inf')
        self.converged = False
        self.stopped_early = False

        # Initializing parameters 
        self.beta = self.initialize(k)
        
        if self.dropout:
            
            # Select dropout features 
            
            all_features = [_ for _ in range(0,X.shape[1])]
            
            for i in range(self.max_iter):

                
                # Dropout features are calculated at each GD step. 
                # Keep rate = 1 - dropout_rate
                np.random.seed(self.seed)
                random_set = np.random.choice(all_features, size = int(round((X.shape[1])*(1-self.dropout_rate),0)),replace=False )
                
    
                
                y_hat = sigmoid(X[:,random_set] @ self.beta[random_set])
                self.loss = np.mean(-y * np.log(y_hat) - (1-y) * np.log(1-y_hat))
                
                

                # early stopping
                if self.val_loss():
                    self.stopped_early = True
                    break 
                    
                #convergence check
                if i > self.last_n_losses:
                    if abs(np.mean(self.training_loss_history[-self.last_n_losses:]) - self.loss) < self.tol:
                        self.converged = True
                        self.training_loss_history.append(self.loss)
                        break
                    else:
                        previous_loss = self.loss
                        self.training_loss_history.append(self.loss)
                else:
                    self.training_loss_history.append(self.loss)
                    previous_loss = self.loss

                # GD with dropout. 
                residuals = (y_hat - y).values.reshape( (n, 1) )
                gradient = (X[:,random_set] * residuals).mean(axis=0)
                self.beta[random_set] -= self.learning_rate * gradient
                
                
                if self.verbose:
                    if i % 10 == 0:
                        if self.X_val is not None:
                            print(f'round {i}, train_loss: {self.training_loss_history[-1]}, val_loss: {self.val_loss_history[-1]} ')
                        else:
                            print(f'round {i}, train_loss: {self.training_loss_history[-1]}')
                
                
            
        else:
            
            for i in range(self.max_iter):

                
                
                y_hat = sigmoid(X @ self.beta)
                self.loss = np.mean(-y * np.log(y_hat) - (1-y) * np.log(1-y_hat))
                self.training_loss_history.append(self.loss)
                
                # early stopping
                if self.val_loss():
                    self.stopped_early = True
                    break 

                # convergence check
                if abs(previous_loss - self.loss) < self.tol:
                    self.converged = True
                    break
                else:
                    previous_loss = self.loss

                # GD without dropout. 
                residuals = (y_hat - y).values.reshape( (n, 1) )
                gradient = (X * residuals).mean(axis=0)
                self.beta -= self.learning_rate * gradient
                
                if self.verbose:
                    if i % 1 == 0:
                        if self.X_val is not None:
                            print(f'round {i}, train_loss: {self.training_loss_history[-1]}, val_loss: {self.val_loss_history[-1]} ')
                        else:
                            print(f'round {i}, train_loss: {self.training_loss_history[-1]}')

        self.iterations = i+1
        
    def predict_proba(self, X):
        
        # Add intercept to prediction data 
        X = self._design_matrix(X)
        
        if self.dropout:
            
            # When dropout, then we reduce the betas in order to dampen the sigmoid input slightly, since during training 
            # Fewer features were present at each training step. 
            
            w = self.beta * (1-self.dropout_rate) 
            #return sigmoid(X @ w)
            return sigmoid(X @ self.beta)
        else:
            return sigmoid(X @ self.beta)
        
    def predict(self, X):
        return (self.predict_proba(X) > 0.5).astype(int)
    
    def val_loss(self):
        
        if self.X_val is None:
            return False
        
        # Create Validation loss & AUC history vector 
        if not hasattr(self, 'val_loss_history'):
            self.val_loss_history = []

        if not hasattr(self, 'auc_history'):
            self.auc_history = []

        # Validation Loss 
        val_y_hat = self.predict_proba(self.X_val)
        loss = np.mean(-self.y_val * np.log(val_y_hat) - \
                   (1-self.y_val) * np.log(1-val_y_hat))
        self.val_loss_history.append(loss)


        # Keep a copy of the weights associated with the lowest validation loss. 
        if len(self.val_loss_history) > 0:
            if self.val_loss_history[-1] < self.best_loss:
                self.best_weights = self.beta
                self.best_loss = self.val_loss_history[-1]
        else:
            self.best_weights = self.beta
            self.best_loss = self.val_loss_history[-1]


        # ROC AUC scores           
        auc = roc_auc_score(self.y_val, val_y_hat)
        self.auc_history.append(auc)

        # Check if GD needs to be early stopped. 
        t = self.early_stopping
        if t and len(self.val_loss_history) > t * 2:
            recent_best = min(self.val_loss_history[-t:])
            previous_best = min(self.val_loss_history[:-t])
            if recent_best > previous_best:
                self.beta = self.best_weights
                return True
            
        return False