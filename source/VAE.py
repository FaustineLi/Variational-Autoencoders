import numpy as np

class VAE(object):

    def __init__(self, input_dim, output_dim, params):

        # initialize size of VAE 
        self.layers = input_dim + [2] + output_dim[::-1]

        # intialize weights
        self.weights = {}
        for i in range(len(self.layers)-1):
            self.weights[i] = np.random.uniform(-0.1, 0.1, (self.layers[i], self.layers[i+1]))

        # set params
        self.alpha = params['alpha']
        #self.activation = params['activation']
        self.loss = params['loss']

    def fit(self, train_data):
        '''fits the NN model'''
        pass

    def predict(self, new_data):
        '''predicts on a trained VAE model'''
        pass

    def backprop(self):
        '''back-propagation algorithm'''
        pass
    
    def activation(self, x):
        return 1 / (1 + np.exp(-x))
    
    def derivative(self, x):
        pass
            
    def feedforward(self, train_data):
        '''feedforward update step'''
        
        layer_input = {}
        layer_activation = {}
        
        layer_input[0] = train_data @ self.weights[0]
        layer_activation[0] = self.activation(layer_input[0])
            
        for i in range(1, len(self.layers) - 1):
            layer_input[i] = layer_input[i-1] @ self.weights[i]
            layer_activation[i] = self.activation(layer_input[i])
                
        return layer_activation[len(self.layers) - 2]