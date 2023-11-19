import numpy as np

class MaxPool:
    """
    MaxPooling is a simple kind of pooling function that selectively passes the maximum values 
    from each window to the subsequent layer.
    """

    def __init__(self, pool_size):
        self.pool_size = pool_size


    # During the forward pass of Max Pooling, the input is taken from the output 
    # of the convolution operation. It is essential to consider the number of channels 
    # in which the output is generated, as convolving with multiple filters can produce 
    # multiple feature maps.
    def forward(self, input_data):

        self.input_data = input_data
        self.num_channels, self.input_height, self.input_width = input_data.shape
        self.output_height = self.input_height // self.pool_size
        self.output_width = self.input_width // self.pool_size

        # Determining the output shape
        self.output = np.zeros((self.num_channels, self.output_height, self.output_width))

        # Iterating over different channels
        for c in range(self.num_channels):
            # Looping through the height
            for i in range(self.output_height):
                # looping through the width
                for j in range(self.output_width):

                    # Starting postition
                    start_i = i * self.pool_size
                    start_j = j * self.pool_size

                    # Ending Position
                    end_i = start_i + self.pool_size
                    end_j = start_j + self.pool_size

                    # Creating a patch from the input data
                    patch = input_data[c, start_i:end_i, start_j:end_j]

                    #Finding the maximum value from each patch/window
                    self.output[c, i, j] = np.max(patch)

        return self.output


    # We transmit the maximum gradients obtained from the previous layer directly to the 
    # corresponding locations in the next layer. This process ensures that the maximum 
    # gradient values flow through the MaxPooling layer and continue propagating through the network. 
    def backward(self, dL_dout, lr):
        dL_dinput = np.zeros_like(self.input_data)

        for c in range(self.num_channels):
            for i in range(self.output_height):
                for j in range(self.output_width):
                    start_i = i * self.pool_size
                    start_j = j * self.pool_size

                    end_i = start_i + self.pool_size
                    end_j = start_j + self.pool_size
                    patch = self.input_data[c, start_i:end_i, start_j:end_j]

                    mask = patch == np.max(patch)

                    dL_dinput[c,start_i:end_i, start_j:end_j] = dL_dout[c, i, j] * mask

        return dL_dinput
