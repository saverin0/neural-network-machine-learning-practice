import numpy as np
import os
import joblib


class Perceptron:
    def __init__(self, eta: float=None, epochs: int=None):
        # Initialize weights with small random values (3 for 2 features + bias)
        self.weights = np.random.randn(3) * 1e-4 
        training = (eta is not None) and (epochs is not None)
        if training:
            print(f"initial weights before training: \n{self.weights}\n")
        self.eta = eta      # Learning rate
        self.epochs = epochs  # Number of training epochs
    
    def _z_outcome(self, inputs, weights):
        # Compute the linear combination of inputs and weights
        return np.dot(inputs, weights)
    
    def activation_function(self, z):
        # Apply step function: returns 1 if z > 0, else 0
        return np.where(z > 0, 1, 0)
    
    def fit(self, X, y):
        # Store training data
        self.X = X
        self.y = y
        
        # Add bias term to input features (last column is -1 for bias)
        X_with_bias = np.c_[self.X, -np.ones((len(self.X), 1))]
        print(f"X with bias: \n{X_with_bias}")
        
        # Training loop for specified number of epochs
        for epoch in range(self.epochs):
            print("--"*10)
            print(f"for epoch >> {epoch}")
            print("--"*10)
            
            # Forward pass: compute predictions
            z = self._z_outcome(X_with_bias, self.weights)
            y_hat = self.activation_function(z)
            print(f"predicted value after forward pass: \n{y_hat}")
            
            # Compute error (difference between actual and predicted)
            self.error = self.y - y_hat
            print(f"error: \n{self.error}")
            
            # Update weights using the perceptron learning rule
            self.weights = self.weights + self.eta * np.dot(X_with_bias.T, self.error)
            print(f"updated weights after epoch: {epoch + 1}/{self.epochs}: \n{self.weights}")
            print("##"*10)
            
    def predict(self, X):
        # Add bias term to input features for prediction
        X_with_bias = np.c_[X, -np.ones((len(X), 1))]
        z = self._z_outcome(X_with_bias, self.weights)
        return self.activation_function(z)
    
    def total_loss(self):
        # Calculate and print the sum of errors (total loss)
        total_loss = np.sum(self.error)
        print(f"\ntotal loss: {total_loss}\n")
        return total_loss
    
    def _create_dir_return_path(self, model_dir, filename):
        # Create directory if it doesn't exist and return full file path
        os.makedirs(model_dir, exist_ok=True)
        return os.path.join(model_dir, filename)
    
    def save(self, filename, model_dir=None):
        # Save the trained model to disk using joblib
        if model_dir is not None:
            model_file_path = self._create_dir_return_path(model_dir, filename)
            joblib.dump(self, model_file_path)
        else:
            model_file_path = self._create_dir_return_path("model", filename)
            joblib.dump(self, model_file_path)
    
    def load(self, filepath):
        # Load a saved model from disk
        return joblib.load(filepath)