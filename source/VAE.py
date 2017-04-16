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