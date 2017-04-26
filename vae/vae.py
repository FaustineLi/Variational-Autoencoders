class vae(object):

    def __init__(self, input_dim, output_dim, params, latent_dim = 2):
        '''intializes weights matrix and parameters'''

        # initialize size of VAE
        self.latent_dim = latent_dim
        self.encoder_layer_sizes = input_dim + [2]
        self.decoder_layer_sizes = [latent_dim] + output_dim[::-1] 
        self.total_layer_sizes   = input_dim + [2, latent_dim] + output_dim[::-1]
       
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
        self.max_iter = params['max_iter']
        self.activation = params['activation']
        self.grad_activation = params['grad_act']
        self.reconst_loss = params['loss']
        self.reconst_grad = params['grad_loss']
        self.mode = params['mode']
        
        if self.mode is 'autoencoder':
            self.loss = self.reconst_loss
            self.grad_loss = self.reconst_grad
        else:
            self.loss = (lambda y, yhat: self.reconst_loss(y, yhat) + self.KLD_loss())
            self.grad_loss = (lambda y, yhat: self.reconst_grad(y, yhat) + self.KLD_loss())
    
    def KLD_loss(self):
        '''Kullback–Leibler divergence loss'''
        return - 0.5 * np.mean(1 + self.sigma - self.mu**2 - np.exp(self.sigma), axis=-1)
        
    def backprop(self, X, y, yhat):
        '''back-propagation algorithm'''
        # initialize 
        grad_decoder = {}
        grad_encoder = {}
    
        # backpropogate error through decoder layers
        rev_range = np.arange(self.number_decoder_layers)[::-1]
        n = rev_range[0]
        
        if n == 0:
            delta = - self.grad_loss(y, yhat) * self.grad_activation(self.decoder_input[0])
            grad_decoder[0] = self.encoder_activation[0].T @ delta
        else:
            delta = - self.grad_loss(y, yhat) * self.grad_activation(self.decoder_input[n])
            grad_decoder[n] = self.decoder_activation[n-1].T @ delta
            
            for i in rev_range[1:-1]:
                delta = delta @ self.decoder_weights[i+1].T * self.grad_activation(self.decoder_input[i])
                grad_decoder[i] = self.decoder_activation[i-1].T @ delta 
            
            delta = delta @ self.decoder_weights[1].T * self.grad_activation(self.decoder_input[0])
            grad_decoder[0] = self.encoder_activation[1].T @ delta

        # backpropogate errors through encoder layers
        rev_range = np.arange(self.number_encoder_layers)[::-1]
        n = rev_range[0]
        
        if n != 0:
            delta = delta @ self.decoder_weights[0].T * self.grad_activation(self.encoder_input[n])
            grad_encoder[n] = self.encoder_activation[0].T @ delta
        
            for i in rev_range[1:-1]:
                delta = delta @ self.encoder_weights[i+1].T * self.grad_activation(self.encoder_input[i])
                grad_encoder[i] = self.encoder_activation[i-1].T @ delta
                
            delta = delta @ self.encoder_weights[1].T * self.grad_activation(self.encoder_input[0])
            grad_encoder[0] = X.T @ delta
            
        else:
            delta = delta @ self.decoder_weights[0].T * self.grad_activation(self.encoder_input[0])
            grad_encoder[0] = X.T @ delta
    
        return grad_encoder, grad_decoder 
            
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
        self.mu = self.encoder_activation[self.number_encoder_layers - 1][:,1]
        self.sigma = self.encoder_activation[self.number_encoder_layers - 1][:,0]
        
        # sample latent variable using reparameterization trick
        if self.mode is 'vae':
            self.epsilon = np.random.normal(0, 1, (self.mu.size, self.latent_dim))
            self.z = self.mu + self.sigma * self.epsilon
        else:
            self.z = self.encoder_activation[self.number_encoder_layers - 1]
    
        # feedforward update on the decoder network
        self.decoder_input[0]      = self.z @ self.decoder_weights[0]
        self.decoder_activation[0] = self.activation(self.decoder_input[0])
        
        for i in range(1, self.number_decoder_layers):
            self.decoder_input[i] = self.decoder_input[i-1] @ self.decoder_weights[i]
            self.decoder_activation[i] = self.activation(self.decoder_input[i])

        return self.decoder_activation[self.number_decoder_layers - 1]
    
    
    def train(self, X, y):
        '''trains the VAE model'''
        for i in range(self.max_iter):
            yhat = self.feedforward(X)
            grad_encoder, grad_decoder = self.backprop(X, y, yhat)
            for i in range(self.number_decoder_layers):
                self.decoder_weights[i] -= self.alpha * grad_decoder[i]
        
            for j in range(self.number_encoder_layers):
                self.encoder_weights[j] -= self.alpha * grad_encoder[j]
                change = grad_encoder[j]
                
        return None

    def predict(self, X):
        '''predicts on a trained VAE model'''        
        return self.feedforward(X)
    
    def generate(self, z):
        '''generates new images from a trained VAE model'''        
        # feedforward on decoder
        self.gen_input = {}
        self.gen_activation = {}
        self.gen_input[0]     = z.T @ self.decoder_weights[0]
        self.gen_activation[0] = self.activation(self.gen_input[0])
        
        for i in range(1, self.number_decoder_layers):
            self.gen_input[i] = self.gen_input[i-1] @ self.decoder_weights[i]
            self.gen_activation[i] = self.activation(self.gen_input[i])

        return self.gen_activation[self.number_decoder_layers - 1]