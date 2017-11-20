import numpy as np
import Network

class VAE:
    '''Variational Autoencoder'''

    def __init__(self, dimensions, latent_dim, params):
        self.encoder = Network(dimensions[0] + [2], params)
        self.decoder = Network([latent_dim] + dimensions[1], params)

    def _forwardstep(self, X):
        # encoder learns parameters
        latent = self.encoder._feedforward(X)
        self.mu = latent[:,0]
        self.log_sigma = latent[:,1]

        # sample from gaussian with learned parameters
        epsilon = np.random.normal(0, 1, size=(X.shape[0], latent_dim))
        z_sample = self.mu + np.sqrt(np.exp(self.log_sigma)) * epsilon

        # pass sampled vector through to decoder
        X_hat = self.decoder._feedforward(z_sample)
        
        return X_hat


    def _backwardstep(self, X, X_hat):
        pass

    def learn(self, X):
        pass

    def predict(self, X):
        pass

    def generate(self):
        pass