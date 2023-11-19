

from convolution import Convolution
from fully_connected import Fully_Connected
from maxpool import MaxPool


def cross_entropy_loss(predictions, targets):

    num_samples = 10

    # Avoid numerical instability by adding a small epsilon value
    epsilon = 1e-7
    predictions = np.clip(predictions, epsilon, 1 - epsilon)
    loss = -np.sum(targets * np.log(predictions)) / num_samples
    return loss

def cross_entropy_loss_gradient(actual_labels, predicted_probs):
    num_samples = actual_labels.shape[0]
    gradient = -actual_labels / (predicted_probs + 1e-7) / num_samples

    return gradient


def train_network(X, y, conv, pool, full, lr=0.01, epochs=200):
    for epoch in range(epochs):
        total_loss = 0.0
        correct_predictions = 0

        for i in range(len(X)):
            # Forward pass
            conv_out = conv.forward(X[i])
            pool_out = pool.forward(conv_out)
            full_out = full.forward(pool_out)
            loss = cross_entropy_loss(full_out.flatten(), y[i])
            total_loss += loss

            # Converting to One-Hot encoding
            one_hot_pred = np.zeros_like(full_out)
            one_hot_pred[np.argmax(full_out)] = 1
            one_hot_pred = one_hot_pred.flatten()

            num_pred = np.argmax(one_hot_pred)
            num_y = np.argmax(y[i])

            if num_pred == num_y:
                correct_predictions += 1
            # Backward pass
            gradient = cross_entropy_loss_gradient(y[i], full_out.flatten()).reshape((-1, 1))
            full_back = full.backward(gradient, lr)
            pool_back = pool.backward(full_back, lr)
            conv_back = conv.backward(pool_back, lr)

        # Print epoch statistics
        average_loss = total_loss / len(X)
        accuracy = correct_predictions / len(X_train) * 100.0
        print(f"Epoch {epoch + 1}/{epochs} - Loss: {average_loss:.4f} - Accuracy: {accuracy:.2f}%")

def predict(input_sample, conv, pool, full):
    # Forward pass through Convolution and pooling
    conv_out = conv.forward(input_sample)
    pool_out = pool.forward(conv_out)
    # Flattening
    flattened_output = pool_out.flatten()
    # Forward pass through fully connected layer
    predictions = full.forward(flattened_output)
    return predictions


import tensorflow.keras as keras
import numpy as np

# Load the Fashion MNIST dataset
(train_images, train_labels), (test_images, test_labels) = keras.datasets.fashion_mnist.load_data()

X_train = train_images[:5000] / 255.0
y_train = train_labels[:5000]

X_test = train_images[5000:10000] / 255.0
y_test = train_labels[5000:10000]

X_train.shape


from keras.utils import to_categorical

y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

y_test[0]

np.array([0., 0., 0., 0., 1., 0., 0., 0., 0., 0.], dtype=np.float32)

conv = Convolution(X_train[0].shape, 6, 1)
pool = MaxPool(2)
full = Fully_Connected(121, 10)

train_network(X_train, y_train, conv, pool, full)

predictions = []

for data in X_test:
    pred = predict(data, conv, pool, full)
    one_hot_pred = np.zeros_like(pred)
    one_hot_pred[np.argmax(pred)] = 1
    predictions.append(one_hot_pred.flatten())

predictions = np.array(predictions)


from sklearn.metrics import accuracy_score

accuracy_score(predictions, y_test)
