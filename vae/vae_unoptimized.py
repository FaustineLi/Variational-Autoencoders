import math
import random

class vae_unoptimized(object):

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
            W = list()
            for h in range(self.encoder_layer_sizes[i]):
                row = list()
                for k in range(self.encoder_layer_sizes[i+1]):
                    row.append(random.uniform(-0.1, 0.1))
                W.append(row)
            self.encoder_weights[i] = W
            
        self.decoder_weights = {}
        for i in range(self.number_decoder_layers):
            W = list()
            for h in range(self.decoder_layer_sizes[i]):
                row = list()
                for k in range(self.decoder_layer_sizes[i+1]):
                    row.append(random.uniform(-0.1, 0.1))
                W.append(row)
            self.decoder_weights[i] = W
            
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
            self.grad_loss = self.reconst_grad
    
    def KLD_loss(self):
        '''Kullback–Leibler divergence loss'''
        return - 0.5 * sum(1 + self.sigma - self.mu**2 - math.exp(self.sigma), axis=-1)
    
    def KLD_grad(self):
        '''Kullback–Leibler divergence loss'''
        grad_sigma = list()
        for i, r in enumerate(self.sigma):
            grad_sigma.append(-0.5 * math.exp(self.sigma[i]) - 0.5)
        
        grad_mu = list()
        for i, r in enumerate(self.mu):
            grad_mu.append(-1 * self.mu[i])
            
        return [grad_sigma, grad_mu]
    
    def matmult(self, A, B):
        '''matrix multiplication'''
        mat = list()
        for i, row in enumerate(A):
            r = list()
            for j in range(len(B[0])):
                s = 0
                for k, column in enumerate(row):
                    s += A[i][k] * B[k][j]
                r.append(s)
            mat.append(r)
        return mat
        
    def scalar_mult(self, a, B):
        '''scalar muliplication'''
        mat = list()
        for row in B:
            r = list()
            for column in row:
                r.append(a * column)
            mat.append(r)
        return mat
        
    def multiply(self, A, B):
        '''elementwise multiplication'''
        mat = list()
        for i, row in enumerate(A):
            r = list()
            for j, column in enumerate(row):
                r.append(A[i][j] * B[i][j])
            mat.append(r)
        return mat
    
    def vec_mat_mult(self, v, A):
        mat = list()
        for i, row in enumerate(A):
            r = list()
            for j, column in enumerate(row):
                r.append(A[i][j] * v[i])
            mat.append(r)
        return mat 
    
    def t(self, A):
        '''transpose'''
        return [list(x) for x in zip(*A)]
    
    def mat_sub(self, A, B):
        '''elementwise subtraction'''
        mat = list()
        for i, row in enumerate(A):
            r = list()
            for j, column in enumerate(row):
                r.append(A[i][j] - B[i][j])
            mat.append(r)
        return mat
    
    def mat_add(self, A, B):
        '''elementwise addition'''
        mat = list()
        for i, row in enumerate(A):
            r = list()
            for j, column in enumerate(row):
                r.append(A[i][j] + B[i][j])
            mat.append(r)
        return mat
    
    def vec_mat_add(self, v, A):
        mat = list()
        for i, row in enumerate(A):
            r = list()
            for j, column in enumerate(row):
                r.append(A[i][j] + v[i])
            mat.append(r)
        return mat
        
    def backprop(self, X, y, yhat):
        '''back-propagation algorithm'''
        # initialize 
        grad_decoder = {}
        grad_encoder = {}
    
        # backpropogate error through decoder layers
        rev_range = list(range(self.number_decoder_layers))[::-1]
        n = rev_range[0]
        
        if n == 0:
            delta = self.multiply(self.grad_loss(y, yhat), self.grad_activation(self.decoder_input[0])) 
        else:
            delta = self.multiply(self.grad_loss(y, yhat), self.grad_activation(self.decoder_input[n]))
            grad_decoder[n] = self.matmult(self.t(self.decoder_activation[n-1]), delta)
            
            for i in rev_range[1:-1]:
                delta = self.multiply(self.matmult(delta, self.t(self.decoder_weights[i+1])),
                                      self.grad_activation(self.decoder_input[i]))
                grad_decoder[i] = self.matmult(self.t(self.decoder_activation[i-1]), delta)
            
            delta = self.multiply(self.matmult(delta, self.t(self.decoder_weights[1])), 
                                  self.grad_activation(self.decoder_input[0]))
            grad_decoder[0] = self.matmult(self.t(self.encoder_activation[1]), delta)

        # backpropogate errors through encoder layers
        rev_range = list(range(self.number_encoder_layers))[::-1]
        n = rev_range[0]
        
        if n == 0:
            delta = self.multiply(self.matmult(delta, self.t(self.decoder_weights[0])), 
                                  self.grad_activation(self.encoder_input[0]))
            if self.mode is 'vae':
                delta_kld = self.matmult(self.t(self.KLD_grad()), self.grad_activation(self.encoder_input[n]))
                delta = self.mat_add(delta, delta_kld)
                    
            grad_encoder[0] = self.matmult(t(X), delta)

        else:
            delta = self.multiply(self.matmult(delta, self.t(self.decoder_weights[0])), 
                                  self.grad_activation(self.encoder_input[n]))
            if self.mode is 'vae':
                delta_kld = self.matmult(self.t(self.KLD_grad()), self.grad_activation(self.encoder_input[n]))
                delta = self.mat_add(delta, delta_kld)
                    
            grad_encoder[n] = self.matmult(self.t(self.encoder_activation[0]), delta)
        
            for i in rev_range[1:-1]:
                delta = self.multiply(self.matmult(delta, self.t(self.encoder_weights[i+1])),
                                      self.grad_activation(self.encoder_input[i]))
                grad_encoder[i] = self.matmult(self.t(self.encoder_activation[i-1]), delta)
                
            delta = self.multiply(self.matmult(delta, self.t(self.encoder_weights[1])), 
                                  self.grad_activation(self.encoder_input[0]))
            grad_encoder[0] = self.matmult(self.t(X), delta)
    
        return grad_encoder, grad_decoder 
            
    def feedforward(self, train_data):
        '''feedforward update step'''
        
        # initialize storage for activations
        self.encoder_input = {}
        self.encoder_activation = {}
        self.decoder_input = {}
        self.decoder_activation = {}
        
        self.encoder_input[0]      = self.matmult(train_data, self.encoder_weights[0])
        self.encoder_activation[0] = self.activation(self.encoder_input[0])
            
        # feedforward update on encoder network
        for i in range(1, self.number_encoder_layers):
            self.encoder_input[i] = self.matmult(self.encoder_input[i-1], self.encoder_weights[i])
            self.encoder_activation[i] = self.activation(self.encoder_input[i])
        
        # store output as encoded latent variable parameters
        self.mu = self.t(self.encoder_activation[self.number_encoder_layers - 1])[1]
        self.sigma = self.t(self.encoder_activation[self.number_encoder_layers - 1])[0]
        
        # sample latent variable using reparameterization trick
        epsilon = list()
        if self.mode is 'vae': 
            for i,_ in enumerate(self.mu):
                r = list()
                for j in range(self.latent_dim):
                    r.append(random.normalvariate(0,1))
                epsilon.append(r)
        
            self.z = self.vec_mat_add(self.mu, self.vec_mat_mult(self.sigma, epsilon))
        else:
            self.z = self.encoder_activation[self.number_encoder_layers - 1]
    
        # feedforward update on the decoder network
        self.decoder_input[0]      = self.matmult(self.z, self.decoder_weights[0])
        self.decoder_activation[0] = self.activation(self.decoder_input[0])
        
        for i in range(1, self.number_decoder_layers):
            self.decoder_input[i] = self.matmult(self.decoder_input[i-1], self.decoder_weights[i])
            self.decoder_activation[i] = self.activation(self.decoder_input[i])

        return self.decoder_activation[self.number_decoder_layers - 1]
    
    def train(self, X, y):
        '''trains the VAE model'''
        for i in range(self.max_iter):
            yhat = self.feedforward(X)
            grad_encoder, grad_decoder = self.backprop(X, y, yhat)
            
            for i in range(self.number_decoder_layers):
                self.decoder_weights[i] = self.mat_sub(self.decoder_weights[i], 
                                                       self.scalar_mult(self.alpha, grad_decoder[i]))
        
            for j in range(self.number_encoder_layers):
                self.encoder_weights[j] = self.mat_sub(self.encoder_weights[j],
                                                       self.scalar_mult(self.alpha, grad_encoder[j]))
        return None

    def predict(self, X):
        '''predicts on a trained VAE model'''        
        return self.feedforward(X)
    
    def generate(self, z):
        '''generates new images from a trained VAE model'''        
        # feedforward on decoder
        self.gen_input = {}
        self.gen_activation = {}
        self.gen_input[0]     = self.matmul(z.T, self.decoder_weights[0])
        self.gen_activation[0] = self.activation(self.gen_input[0])
        
        for i in range(1, self.number_decoder_layers):
            self.gen_input[i] = self.matmul(self.gen_input[i-1], self.decoder_weights[i])
            self.gen_activation[i] = self.activation(self.gen_input[i])

        return self.gen_activation[self.number_decoder_layers - 1]