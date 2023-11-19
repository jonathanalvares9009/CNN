import numpy as np

class Fully_Connected:
    """
    Fully Connected Layer which is simply a Multi-Layer Perceptron (MLP) that is 
    similar to an ideal neural network architecture, the Dense Layer is like the 
    brain of CNN which makes decisions based on the features extracted from the convolution, 
    and pooling layers.
    """

    def __init__(self, input_size, output_size):
        self.input_size = input_size # Size of the inputs coming
        self.output_size = output_size # Size of the output producing
        self.weights = np.random.randn(output_size, self.input_size)
        self.biases = np.random.rand(output_size, 1)


    # The Softmax function creates a probability distribution by exponentiating each element 
    # of the input vector and normalizing them through the sum of all exponentiated values. 
    # This normalization ensures that the resulting probabilities sum up to 1, 
    # representing a valid probability distribution across the classes.
    def softmax(self, z):
        # Shift the input values to avoid numerical instability
        shifted_z = z - np.max(z)
        exp_values = np.exp(shifted_z)
        sum_exp_values = np.sum(exp_values, axis=0)
        log_sum_exp = np.log(sum_exp_values)

        # Compute the softmax probabilities
        probabilities = exp_values / sum_exp_values

        return probabilities
    

    # One more thing we need is the derivative of softmax because when performing backpropagation, 
    # we need to calculate the gradients of loss with respect to outputs and there we need the 
    # derivative of softmax
    def softmax_derivative(self, s):
        return np.diagflat(s) - np.dot(s, s.T)
    
    def forward(self, input_data):
        self.input_data = input_data
        # Flattening the inputs from the previous layer into a vector
        flattened_input = input_data.flatten().reshape(1, -1)
        self.z = np.dot(self.weights, flattened_input.T) + self.biases

        # Applying Softmax
        self.output = self.softmax(self.z)
        return self.output
    
    def backward(self, dL_dout, lr):
        # Calculate the gradient of the loss with respect to the pre-activation (z)
        dL_dy = np.dot(self.softmax_derivative(self.output), dL_dout)
        # Calculate the gradient of the loss with respect to the weights (dw)
        dL_dw = np.dot(dL_dy, self.input_data.flatten().reshape(1, -1))

        # Calculate the gradient of the loss with respect to the biases (db)
        dL_db = dL_dy

        # Calculate the gradient of the loss with respect to the input data (dL_dinput)
        dL_dinput = np.dot(self.weights.T, dL_dy)
        dL_dinput = dL_dinput.reshape(self.input_data.shape)

        # Update the weights and biases based on the learning rate and gradients
        self.weights -= lr * dL_dw
        self.biases -= lr * dL_db

        # Return the gradient of the loss with respect to the input data
        return dL_dinput
