import numpy as np

class VAE(object):

    def __init__(self, input_dim, output_dim, params):

        # initialize size of VAE 
        self.encoder_layer_sizes = input_dim + [2]
        self.decoder_layer_sizes = [2] + output_dim[::-1] 
        self.total_layer_sizes   = input_dim + [2] + output_dim[::-1]
       
        self.number_encoder_layers = len(self.encoder_layer_sizes) - 1
        self.number_decoder_layers = len(self.decoder_layer_sizes) - 1
        self.number_total_layers   = len(self.total_layer_sizes) - 1

        # intialize weights
        self.encoder_weights = {}
        for i in range(self.number_encoder_layers):
            self.encoder_weights[i] = np.random.uniform(-0.1, 0.1, 
                                                        (self.encoder_layer_sizes[i], 
                                                         self.encoder_layer_sizes[i+1])) 
        self.decoder_weights = {}
        for i in range(self.number_decoder_layers):
            self.decoder_weights[i] = np.random.uniform(-0.1, 0.1, 
                                                        (self.decoder_layer_sizes[i],
                                                         self.decoder_layer_sizes[i+1]))
        # set params
        self.alpha = params['alpha']
        #self.activation = params['activation']
        #self.loss = params['loss']
        self.max_iter = params['max_iter']

    def train(self, train_data):
        '''train the VAE model'''
        yhat = np.zeros_like(train_data)
        step = self.loss(train_data, yhat)
        count = 0
        while step > 1e-3 and count < self.max_iter:        
            # feed forward network
            yhat = self.feedforward(train_data)
            
            # backpropogate errors
            grad_encoder, grad_decoder = self.backprop(train_data, yhat)
        
            # update weights with gradient descent
            for i in range(self.number_decoder_layers):
                self.decoder_weights[i] -= self.alpha * grad_decoder[i]
                
            for i in range(self.number_encoder_layers):
                self.encoder_weights[i] -= self.alpha * grad_encoder[i]
                
            step = step - self.loss(train_data, yhat) 
            count += 1
            
        return yhat

    def predict_(self):
        '''predicts on a trained VAE model'''        
        
        # sample from latent variable space
        
        # feedforward on decoder
        
        pass

    def backprop(self, y, yhat):
        '''back-propagation algorithm'''
        # initialize 
        grad_decoder = {}
        grad_encoder = {}
        
        # backpropogate error through decoder layers
        delta = - self.grad_loss(y, yhat) * self.grad_activation(self.decoder_input[1])
        grad_decoder[1] = self.decoder_activation[0].T @ delta        
        
        delta = delta @ self.decoder_weights[1].T * self.grad_activation(self.decoder_input[0])
        grad_decoder[0] = self.encoder_activation[1].T @ delta 
        
        # backpropogate errors through encoder layers
        delta = delta @ self.decoder_weights[0].T * self.grad_activation(self.encoder_input[1])
        grad_encoder[1] = self.encoder_activation[0].T @ delta
        
        delta = delta @ self.encoder_weights[1].T * self.grad_activation(self.encoder_input[0])
        grad_encoder[0] = y.T @ delta
        
        return grad_encoder, grad_decoder
    
    def activation(self, x):
        '''activation function'''
        return 1 / (1 + np.exp(-x))
    
    def grad_activation(self, x):
        '''derivative of the activation function'''
        return x * (1 - x)
    
    def loss(self, x, y):
        '''loss function'''
        return 0.5 * np.sum(x - y) ** 2
    
    def grad_loss(self, x, y):
        '''gradient of loss function'''
        return x - y   
            
    def feedforward(self, train_data):
        '''feedforward update step'''
        # initialize storage for activations
        self.encoder_input = {}
        self.encoder_activation = {}
        self.decoder_input = {}
        self.decoder_activation = {}
        
        self.encoder_input[0]      = train_data @ self.encoder_weights[0]
        self.encoder_activation[0] = self.activation(self.encoder_input[0])
            
        # feedforward update on encoder network
        for i in range(1, self.number_encoder_layers):
            self.encoder_input[i] = self.encoder_input[i-1] @ self.encoder_weights[i]
            self.encoder_activation[i] = self.activation(self.encoder_input[i])
        
        # store output as encoded latent variable parameters
        self.mu = self.encoder_activation[i][0]
        self.sigma = self.encoder_activation[i][1]
        
        # sample latent variable using reparameterization trick
        
        # feedforward update on the decoder network
        self.decoder_input[0]      = self.encoder_activation[i] @ self.decoder_weights[0]
        self.decoder_activation[0] = self.activation(self.decoder_input[0])
        
        for i in range(1, self.number_decoder_layers):
            self.decoder_input[i] = self.decoder_input[i-1] @ self.decoder_weights[i]
            self.decoder_activation[i] = self.activation(self.decoder_input[i])
            
        # output activation
        self.output = self.decoder_activation[i]

        return self.output