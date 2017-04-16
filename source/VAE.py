class VAE(object):

    def __init__(self, input_dim, output_dim, params):

        # initialize size of VAE 
        self.encoder_layer_sizes = input_dim + [2]
        self.decoder_layer_sizes = [2] + output_dim[:-1] 
        self.total_layer_sizes   = input_dim + [2] + output_dim[::-1]
       
        self.number_encoder_layers = len(self.encoder_layer_sizes) - 1
        self.number_decoder_layers = len(self.decoder_layer_sizes) - 1
        self.number_total_layers   = len(self.total_layer_sizes) - 1

        # intialize weights
        self.weights = {}
        for i in range(self.number_total_layers):
            self.weights[i] = np.random.uniform(-0.1, 0.1, (self.total_layer_sizes[i], 
                                                            self.total_layer_sizes[i+1]))

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
        '''activation function'''
        return 1 / (1 + np.exp(-x))
    
    def derivative(self, x):
        '''derivative of the activation function'''
        pass
            
    def feedforward(self, train_data):
        '''feedforward update step'''
        # initialize storage for activations
        self.layer_input = {}
        self.layer_activation = {}
        
        self.layer_input[0] = train_data @ self.weights[0]
        self.layer_activation[0] = self.activation(self.layer_input[0])
            
        # feedforward update on encoder network
        for i in range(1, self.number_encoder_layers):
            self.layer_input[i] = self.layer_input[i-1] @ self.weights[i]
            self.layer_activation[i] = self.activation(self.layer_input[i])
        
        # store output as encoded latent variable parameters
        self.mu = self.layer_activation[self.number_encoder_layers-1][0]
        self.sigma = self.layer_activation[self.number_encoder_layers-1][1]
        
        # sample latent variable using reparameterization trick
        
        # feedforward update on the decoder network 
        for i in range(self.number_encoder_layers, self.number_total_layers):
            self.layer_input[i] = self.layer_input[i-1] @ self.weights[i]
            self.layer_activation[i] = self.activation(self.layer_input[i])
            
        # output activation
        self.output = self.layer_activation[self.number_total_layers - 1]

        return self.output