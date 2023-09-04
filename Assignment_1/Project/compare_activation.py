import math
import time
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score, top_k_accuracy_score

# Obtain the start time
start = time.time()


class Activation(object):
    """
    Create an activation class
    For each time, we can initialize an activation function object with one specific function
    For example: f = Activation("relu")  means we create a ReLU activation function.

    Define the two activation functions: ReLU and GeLU
    With their derivative functions
    """

    # ReLU activation
    def __relu(self, x):
        return np.maximum(0, x)

    def __relu_derive(self, a):
        # a = relu(x)
        return (a > 0) * 1

    # GELU activation
    def __gelu(self, x):
        return 0.5 * x * (1 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * np.power(x, 3))))

    def __gelu_derive(self, a):
        # a = gelu(x)
        return ((np.tanh((np.sqrt(2) * (0.044715 * a ** 3 + a)) / np.sqrt(np.pi)) + ((np.sqrt(2) * a * (
                0.134145 * a ** 2 + 1) * ((1 / np.cosh(
            (np.sqrt(2) * (0.044715 * a ** 3 + a)) / np.sqrt(np.pi))) ** 2)) / np.sqrt(np.pi) + 1))) / 2

    def __init__(self, activation='relu'):
        if activation == 'relu':
            self.f = self.__relu
            self.f_deriv = self.__relu_derive
        elif activation == 'gelu':
            self.f = self.__gelu
            self.f_deriv = self.__gelu_derive


class HiddenLayer(object):
    def __init__(self, n_in, n_out, activation_last_layer='relu', activation='relu'):
        """
        Define the hidden layer for the mlp. For example, h1 = HiddenLayer(10, 5, activation="relu")
        Means we create a layer with 10 dimension input and 5 dimension output, and using tanh activation function.
        Make sure the input size of hidden layer should be matched with the output size of the previous layer!

        Typical hidden layer of an MLP: units are fully-connected and have
        ReLU activation function, batch normalization and dropout.
        Weight matrix W is of shape (n_in,n_out) and the bias vector b is of shape (n_out,).

        NOTE : The nonlinear used here is ReLU

        Hidden unit activation is given by: ReLU(dot(input,W) + b)

        :type n_in: int
        :param n_in: dimensionality of input

        :type n_out: int
        :param n_out: number of hidden units

        :type activation: string
        :param activation: non-linearity to be applied in the hidden layer
        """
        self.input = None
        self.output = None
        if activation:
            self.activation = Activation(activation).f

        # Activation derivative of last layer
        self.activation_deriv = None
        if activation_last_layer:
            self.activation_deriv = Activation(activation_last_layer).f_deriv

        # Randomly assign small values. Because the ReLU and GeLU are both non-linear activation
        # So, we can use the HE initialization for them.
        self.W = np.random.uniform(low=-np.sqrt(2 / n_out), high=np.sqrt(2 / n_out), size=(n_in, n_out))
        self.b = np.zeros(n_out, )

        # We set he size of weight gradation as the size of weight
        self.grad_W = np.zeros(self.W.shape)
        self.grad_b = np.zeros(self.b.shape)

        # Dropout parameter for:
        # dropout: whether to use dropout. mask: dropout matrix. output_layer: whether the layer is the output layer.
        self.dropout = None
        self.mask = None
        self.output_layer = False

        # The parameter for the Batch normalization, including:
        # Whether to use the batch normalization. The mean and variance in testing process
        # The gamma and bete with their update parameters.
        self.batch_norm = None
        self.running_mean = None
        self.running_var = None
        self.std = None
        self.xn = None
        self.xc = None
        self.batch_size = None
        self.gamma = np.ones(n_out, )
        self.beta = np.zeros(n_out, )
        self.grad_gamma = np.zeros(self.gamma.shape)
        self.grad_beta = np.zeros(self.beta.shape)

        # The parameter for Momentum batch normalization
        # Indicates the current velocity of the weight and bias momentum
        self.grad_W_V = np.zeros(self.W.shape)
        self.grad_b_V = np.zeros(self.b.shape)
        self.grad_gamma_V = np.zeros(self.gamma.shape)
        self.grad_beta_V = np.zeros(self.beta.shape)

        # Adam in batch normalization
        # The 1st moment vector
        self.grad_W_mt = np.zeros(self.W.shape)
        self.grad_b_mt = np.zeros(self.b.shape)
        self.grad_gamma_mt = np.zeros(self.gamma.shape)
        self.grad_beta_mt = np.zeros(self.beta.shape)
        # The 2nd moment vector (squared)
        self.grad_W_vt = np.zeros(self.W.shape)
        self.grad_b_vt = np.zeros(self.b.shape)
        self.grad_gamma_vt = np.zeros(self.gamma.shape)
        self.grad_beta_vt = np.zeros(self.beta.shape)

    def forward(self, input, test_flg=False, dropout_rate=0.5):
        """
        The forward progress for the hidden layer, including the dropout and batch normalization
        Calculate the output of the output layer
        Can determine whether to use the test mode

        :type input: numpy.array
        :param input: a symbolic tensor of shape (batch_size, n_in)

        :type test_flg: boolean
        :param test_flg: whether is the test progress

        :type dropout_rate: float
        :param dropout_rate: probability for dropout

        :return: output of shape (batch_size, n_out) for this hidden layer
        """
        # Calculate the perception
        lin_output = np.dot(input, self.W) + self.b

        # For the forward progress, use dropout first and then batch normalization
        # The dropout progress
        if self.dropout == True and self.output_layer == False:
            lin_output = self.dropout_forward(lin_output, dropout_rate=dropout_rate, test_flg=test_flg)

        # The batch normalization progress
        if self.batch_norm:
            lin_output = self.batch_normalization_forward(lin_output, test_flg=test_flg)

        # Use the activation function
        self.output = (
            lin_output if self.activation is None
            else self.activation(lin_output)
        )
        self.input = input
        return self.output

    def backward(self, delta, output_layer=False):
        """
        The backward progress for the hidden layer, including the dropout and batch normalization
        Calculate the delta for the derivative used for calculating loss function

        :type delta: numpy.array
        :param delta: a symbolic tensor of shape (batch_size, n_out)

        :type output_layer: boolean
        :param output_layer: whether is the output layer of the network or not

        :return: new delta of shape (batch_size, n_in)
        """
        self.output_layer = output_layer
        # For the backward progress, consider batch normalization first and then dropout
        # The batch normalization backward progress
        if self.batch_norm:
            delta = self.batch_normalization_backward(delta)

        # The dropout backward progress
        if self.dropout == True and self.output_layer == False:
            delta = self.dropout_backward(delta)

        # Calculate the gradation of weight and bias for update
        self.grad_W = np.atleast_2d(self.input).T.dot(np.atleast_2d(delta))
        self.grad_b = np.sum(delta, axis=0)

        # The activation derivative progress
        if self.activation_deriv:
            delta = delta.dot(self.W.T) * self.activation_deriv(self.input)
        return delta

    def batch_normalization_forward(self, x, test_flg=False):
        """
        The batch normalization forward progress, including the train mode and the test mode
        Used for normalization for the input
        Accept the output of the perception and return a new output

        :type x: numpy.array
        :param x: a symbolic tensor of shape (batch_size, n_out)

        :type test_flg: boolean
        :param test_flg: determine the test mode or train mode

        :return: output of batch normalization in shape (batch_size, n_out)
        """
        # Initialize the mean and variance of the input with the shape of (,n_in)
        if self.running_mean is None:
            # self.input_shape = x.shape
            N, D = x.shape
            self.running_mean = np.zeros(D)
            self.running_var = np.zeros(D)

        # If it is the train progress
        if test_flg is False:
            # Calculate the mean
            mean = np.mean(x, axis=0)
            # Subtract the mean of each training example
            xc = x - mean
            # Calculate the variance
            var = np.mean(xc ** 2, axis=0)
            # Add epsilon for numerical stability, then sqrt
            std = np.sqrt(var + 10e-7)
            # Execute normalization
            xn = xc / std

            # Store the parameters
            self.batch_size = x.shape[0]
            self.xc = xc
            self.xn = xn
            self.std = std
            # Update the mean and variance after each progress
            self.running_mean = 0.9 * self.running_mean + (1 - 0.9) * mean
            self.running_var = 0.9 * self.running_var + (1 - 0.9) * var
        else:
            # In the test progress of batch normalization, only two steps
            # Subtract the mean of each test example
            xc = x - self.running_mean
            # Execute normalization
            xn = xc / (np.sqrt(self.running_var + 10e-7))

        # The transformation step
        out = self.gamma * xn + self.beta
        # return out.reshape(*self.input_shape)
        return out

    def batch_normalization_backward(self, dout):
        """
        The backward progress for the batch normalization.
        Accept the delta and return a new

        :type dout: numpy.array
        :param dout: a symbolic tensor of shape (batch_size, n_out)

        :return: new delta of shape (batch_size, n_out)
        """
        # Calculate the derivative of the beta
        dbeta = np.sum(dout, axis=0)
        # Calculate the derivative of the gamma
        dgamma = np.sum(dout * self.xn, axis=0)
        # Derivative of the batch normalization forward progress
        dxn = self.gamma * dout
        dxc = dxn / self.std
        dstd = -np.sum((dxn * self.xc) / (self.std * self.std), axis=0)
        dvar = 0.5 * dstd / self.std
        dxc += (2.0 / self.batch_size) * self.xc * dvar
        dmu = np.sum(dxc, axis=0)
        # Calculate the new delta
        dx = dxc - dmu / self.batch_size

        # Update the parameters
        self.grad_gamma = dgamma
        self.grad_beta = dbeta
        # return dx.reshape(*self.input_shape)
        return dx

    def dropout_forward(self, x, dropout_rate=0.5, test_flg=False):
        """
        The dropout forward progress, dropout the several nodes in the neural network of each hidden layer

        :type x: numpy.array
        :param x: a symbolic tensor of shape (batch_size, n_out)

        :type dropout_rate: float
        :param dropout_rate: rate for the dropout

        :type test_flg: boolean
        :param test_flg: determine the test mode or train mode

        :return: output of dropout in shape of (batch_size, n_out)
        """
        # If it is the train progress
        if not test_flg:
            # Create a matrix in binary (0 and 1) according to the dropout rate and rescale
            self.mask = np.random.binomial(1, 1 - dropout_rate, x.shape) / (1 - dropout_rate)
            return x * self.mask
        else:
            return x

    def dropout_backward(self, dout):
        """
        The dropout backward progress

        :type dout: numpy.array
        :param dout: a symbolic tensor of shape (batch_size, n_out)

        :return: new delta of shape (batch_size, n_out)
        """
        if self.mask is None:
            return dout
        else:
            return dout * self.mask


class MLP:
    def __init__(self, layers, activation=[None, 'relu', 'relu'], dropout=True, batch_norm=True):
        """
        For initialization, the code will create all layers automatically based on the provided parameters.
        The Multi-Layer perception, also known as the Artificial Neural Network
        Basic including the input layer, hidden layer and the output layer
        In this MLP, we add the batch normalization, softmax with cross entropy and activation layer
        User can choose the two types od optimizer for the parameter update: momentum in SGD and Adam
        The data training is in form of the mini-batch

        :type layers: list
        :param layers: a list containing the number of units in each layer.
        Should be at least two values

        :type activation: list
        :param activation: activation function to be used. Can be "logistic" or "tanh"

        :type dropout: boolean
        :param dropout: whether to use the dropout or not

        :type batch_norm: boolean
        :param batch_norm: whether to use the batch normalization or not
        """
        # Initialize layers
        self.layers = []
        self.params = []

        # Append the hidden layer and activation layer to the initialization
        self.activation = activation
        for i in range(len(layers) - 1):
            self.layers.append(HiddenLayer(layers[i], layers[i + 1], activation[i], activation[i + 1]))

        # Set the last layer as the output layer
        self.layers[-1].output_layer = True

        # Set the attributes of whether to use dropout and batch normalization or not
        for layer in self.layers:
            layer.dropout = dropout
            layer.batch_norm = batch_norm

    def forward(self, input, dropout_rate=0.5, test_flg=False):
        """
        Forward progress: pass the information through the layers and out the results of final output layer

        :type input: numpy.array
        :param input: a symbolic tensor of shape (batch_size, n_in)

        :type dropout_rate: float
        :param dropout_rate: probability for dropout

        :type test_flg: boolean
        :param test_flg: whether is the test progress

        :return: output of shape (batch_size, n_out) for this hidden layer
        """
        for layer in self.layers:
            output = layer.forward(input, dropout_rate=dropout_rate, test_flg=test_flg)
            input = output
        return output

    def crossEntropy_softmax(self, y_hat, y):
        """
        The softmax with cross entropy layer accepting two parameters: y_predict and y_target
        Combine the softmax with the cross entropy to calculate the loss

        :type y_hat: numpy.array
        :param y_hat: predict value of the input data

        :type y: numpy.array
        :param y: target value of the input data

        :return: loss of the output
        """
        # Calculate the softmax
        exps = np.exp(y_hat - np.max(y_hat, axis=1, keepdims=True))
        out = exps / np.sum(exps, axis=1, keepdims=True)
        batch_size = y.shape[0]
        # Calculate the cross entropy loss
        return -np.sum(np.log(out[np.arange(batch_size), y.reshape(-1)] + 1e-7)) / batch_size

    def delta_crossEntropy_softmax(self, y_hat, y):
        """
        The softmax with cross entropy layer accepting two parameters: y_predict and y_target
        Combine the softmax with the cross entropy to calculate the delta

        :type y_hat: numpy.array
        :param y_hat: predict value of the input data

        :type y: numpy.array
        :param y: target value of the input data

        :return: the delta of the output
        """
        # Calculate the softmax
        exps = np.exp(y_hat - np.max(y_hat, axis=1, keepdims=True))
        out = exps / np.sum(exps, axis=1, keepdims=True)
        batch_size = y.shape[0]
        # Calculate the cross entropy delta
        # Because the shape of y is (batch_size,1), we need to transfer it to the (batch_size,)
        out[np.arange(batch_size), y.reshape(-1)] -= 1
        return out / batch_size

    def criterion_CE(self, y, y_hat):
        """
        Define the objection/loss function, we use Cross-Entropy as the loss
        Using the cross entropy function to calculate the loss and delta
        Including the derivative of the activation function

        :type y: numpy.array
        :param y: target value of the input data

        :type y_hat: numpy.array
        :param y_hat: predict value of the input data

        :return: loss and delta
        """
        activation_deriv = Activation(self.activation[-1]).f_deriv
        # Cross Entropy delta and loss
        error = self.delta_crossEntropy_softmax(y_hat, y)
        loss = self.crossEntropy_softmax(y_hat, y)
        delta = error * activation_deriv(y_hat)
        return loss, delta

    def backward(self, delta):
        """
        Backward progress: pass the delta and update the parameter for next forward progress

        :type delta: numpy.array
        :param delta: a symbolic tensor of shape (batch_size, n_out)
        """
        # Set the last layer as the output layer
        delta = self.layers[-1].backward(delta, output_layer=True)
        for layer in reversed(self.layers[:-1]):
            delta = layer.backward(delta)

    def update(self, lr, optimizer=None, weight_decay_lambda=1, momentum=0.9, iter=100, rho1=0.9, rho2=0.999,
               epsilon=1e-8):
        """
        Update the network weights after backward.
        The update for the weight and bias, lr decide the learning rate
        of the weight update. None, Momentum and Adam can be selected for
        the optimizer.

        :type lr: float
        :param lr: learning rate for the update

        :type optimizer: str
        :param optimizer: None, momentum and Adam, select one of the optimizer

        :type weight_decay_lambda: float
        :param weight_decay_lambda: constant value of the weight decay, Normally is 0.9

        :type momentum: float
        :param momentum: constant value of the momentum in SGD. Normally is 0.9

        :type iter: int
        :param iter: epoch of the training process

        :type rho1: float
        :param rho1: exponential decay rate for the first moment estimates. Normally is 0.9

        :type rho2: float
        :param rho2: exponential decay rate for the second-moment estimates. Normally is 0.999

        :type epsilon: float
        :param epsilon: constant value for numerical stability. Normally is 1e-8
        """
        for layer in self.layers:
            # Normal mode with no optimizer
            if optimizer is None:
                # update the weight with the weight decay
                # According to the lecture 4: 𝜃 = (1−𝜂𝛼)𝜃 - 𝜂𝛻(𝜃)
                layer.W = (1 - lr * weight_decay_lambda) * layer.W - lr * layer.grad_W
                layer.b -= lr * layer.grad_b

                # for batch normalization parameter with the weight decay
                layer.gamma = (1 - lr * weight_decay_lambda) * layer.gamma - lr * layer.grad_gamma
                layer.beta -= lr * layer.grad_beta

            # Momentum with SGD
            elif optimizer.lower() == 'momentum':
                # update the weight in momentum with the weight decay
                # According to the lecture 3
                # v = momentum * v + learning_rate * gradient
                # w = w - v
                layer.grad_W_V = momentum * layer.grad_W_V + lr * layer.grad_W
                layer.grad_b_V = momentum * layer.grad_b_V + lr * layer.grad_b
                layer.W = (1 - lr * weight_decay_lambda) * layer.W - layer.grad_W_V
                layer.b = layer.b - layer.grad_b_V

                # for batch normalization parameter with momentum and weight decay
                layer.grad_gamma_V = momentum * layer.grad_gamma_V + lr * layer.grad_gamma
                layer.grad_beta_V = momentum * layer.grad_beta_V + lr * layer.grad_beta
                layer.gamma = (1 - lr * weight_decay_lambda) * layer.gamma - layer.grad_gamma_V
                layer.beta = layer.beta - layer.grad_beta_V

            # Adam
            elif optimizer.lower() == 'adam':
                # update the weight in Adam with the weight decay
                # According to the lecture 3
                # mt = beta1 * mt + (1-beta1) * gradient
                # vt = beta2 * vt + (1-beta2) * (gradient**2)
                layer.grad_W_mt = rho1 * layer.grad_W_mt + (1 - rho1) * layer.grad_W
                layer.grad_W_vt = rho2 * layer.grad_W_vt + (1 - rho2) * (layer.grad_W ** 2)
                layer.grad_b_mt = rho1 * layer.grad_b_mt + (1 - rho1) * layer.grad_b
                layer.grad_b_vt = rho2 * layer.grad_b_vt + (1 - rho2) * (layer.grad_b ** 2)
                # mt_vector = mt / (1-beta1**iter)
                # vt_vector = vt / (1-beta2**iter)
                w_mt_vector = layer.grad_W_mt / (1 - rho1 ** iter)
                w_vt_vector = layer.grad_W_vt / (1 - rho2 ** iter)
                b_mt_vector = layer.grad_b_mt / (1 - rho1 ** iter)
                b_vt_vector = layer.grad_b_vt / (1 - rho2 ** iter)
                # w = w - lr * mt_vector / (np.sqrt(vt_vector + epsilon))
                layer.W = (1 - lr * weight_decay_lambda) * layer.W - lr * w_mt_vector / (np.sqrt(w_vt_vector + epsilon))
                layer.b = layer.b - lr * b_mt_vector / (np.sqrt(b_vt_vector + epsilon))

                # for batch normalization parameter with Adam and weight decay
                layer.grad_gamma_mt = rho1 * layer.grad_gamma_mt + (1 - rho1) * layer.grad_gamma
                layer.grad_gamma_vt = rho2 * layer.grad_gamma_vt + (1 - rho2) * (layer.grad_gamma ** 2)
                layer.grad_beta_mt = rho1 * layer.grad_beta_mt + (1 - rho1) * layer.grad_beta
                layer.grad_beta_vt = rho2 * layer.grad_beta_vt + (1 - rho2) * (layer.grad_beta ** 2)
                gamma_mt_vector = layer.grad_gamma_mt / (1 - rho1 ** iter)
                gamma_vt_vector = layer.grad_gamma_vt / (1 - rho2 ** iter)
                beta_mt_vector = layer.grad_beta_mt / (1 - rho1 ** iter)
                beta_vt_vector = layer.grad_beta_vt / (1 - rho2 ** iter)
                layer.gamma = (1 - lr * weight_decay_lambda) * layer.gamma - lr * gamma_mt_vector / (
                    np.sqrt(gamma_vt_vector + epsilon))
                layer.beta = layer.beta - lr * beta_mt_vector / (np.sqrt(beta_vt_vector + epsilon))

    def fit(self, X, y, learning_rate=0.1, epochs=100, mini_batch_size=128, optimizer=None, weight_decay_lambda=1,
            momentum=0.9, rho1=0.9,
            rho2=0.999, epsilon=1e-8, dropout_rate=0.5):
        """
        Online Learning
        Define the training function
        It will return all losses within the whole training process.

        :type X: numpy.array
        :param X: data of the training dataset

        :type y: numpy.array
        :param y: label of the training dataset

        :type learning_rate: float
        :param learning_rate: learning rate for the update

        :type epochs: int
        :param epochs: epoch of the training process

        :type mini_batch_size: int
        :param mini_batch_size: size of each batch

        :type optimizer: str
        :param optimizer: None, momentum and Adam, select one of the optimizer

        :type weight_decay_lambda: float
        :param weight_decay_lambda: constant value of the weight decay, Normally is 0.9

        :type momentum: float
        :param momentum: constant value of the momentum in SGD. Normally is 0.9

        :type rho1: float
        :param rho1: exponential decay rate for the first moment estimates. Normally is 0.9

        :type rho2: float
        :param rho2: exponential decay rate for the second-moment estimates. Normally is 0.999

        :type epsilon: float
        :param epsilon: constant value for numerical stability. Normally is 1e-8

        :type dropout_rate: float
        :param dropout_rate: rate for the dropout

        :return: loss, accuracy of the training set, accuracy of the validation set
        """
        X = np.array(X)
        y = np.array(y)
        # Initialize the loss, accuracy of the training set, accuracy of the validation set for all epoch
        to_return = np.zeros(epochs)
        train_accuracy = np.zeros(epochs)
        val_accuracy = np.zeros(epochs)

        for k in range(epochs):
            # Partition of the training dataset
            mini_batch_X, mini_batch_Y = self.mini_batch(X, y, mini_batch_size)
            loss = np.zeros(mini_batch_X.shape[0])

            for it in range(mini_batch_X.shape[0]):
                # Forward pass
                y_hat = self.forward(mini_batch_X[it], dropout_rate)
                # Loss function and backward pass
                loss[it], delta = self.criterion_CE(mini_batch_Y[it], y_hat)
                self.backward(delta)
                # Update
                self.update(lr=learning_rate, optimizer=optimizer, weight_decay_lambda=weight_decay_lambda,
                            momentum=momentum, iter=epochs, rho1=rho1, rho2=rho2, epsilon=epsilon)

            # Store the loss for one epoch
            to_return[k] = np.mean(loss)

            # Predict the accuracy of the training and validation data for one epoch
            train_pre = nn_relu.predict(X_train)
            train_acc = accuracy_score(y_train, train_pre)
            val_pre = nn_relu.predict(X_val)
            val_acc = accuracy_score(y_val, val_pre)

            # Store the accuracy for one epoch and print
            val_accuracy[k] = val_acc
            train_accuracy[k] = train_acc
            print(
                'epoch {0} loss: {1:.6f} train_accuracy: {2:.6f} validate_accuracy: {3:.6f}'.format(k,
                                                                                                    float(to_return[k]),
                                                                                                    train_acc, val_acc))
        return to_return, train_accuracy, val_accuracy

    def mini_batch(self, X, Y, batch_size=64):
        """
        Define the function for the mini batch training
        Divide the training dataset to the several batches
        Each batch has the size of the batch_size
        If the last batch does not have the enough data, then full it to the batch size.

        :type X: numpy.array
        :param X: data of the training dataset

        :type Y: numpy.array
        :param Y: label of the training dataset

        :type batch_size: int
        :param batch_size: size of each batch

        :return: mini batch of the features and label in the type of numpy array
        """
        # Initialize the random seed
        np.random.seed(2023)
        mini_batch_X = []
        mini_batch_Y = []
        # Get the number of training example
        m = X.shape[0]

        # Shuffle (X, Y), m is the num of instances of data set
        permutation = np.random.permutation(m)
        shuffled_X = X[permutation, :]
        shuffled_Y = Y[permutation, :].reshape((m, 1))

        # Split the training data into batch_size=64
        # Round down
        num_batches = math.floor(m / batch_size)
        # Partition
        for i in range(0, num_batches):
            batch_X = shuffled_X[i * batch_size:(i + 1) * batch_size, :]
            batch_Y = shuffled_Y[i * batch_size:(i + 1) * batch_size, :]
            mini_batch_X.append(batch_X)
            mini_batch_Y.append(batch_Y)

        # If there are remaining training examples, add them into the last batch
        if m % batch_size != 0:
            batch_X = shuffled_X[num_batches * batch_size:, :]
            batch_Y = shuffled_Y[num_batches * batch_size:, :]
            # Calculate the minus between the number of data in the last batch and the batch size
            # Then Calculate the number required to full of the batch
            add_X = shuffled_X[m - (num_batches * batch_size) - batch_size:, :]
            add_Y = shuffled_Y[m - (num_batches * batch_size) - batch_size:, :]
            # Combine the rest data and the additional data to an integral batch
            batch_X = np.vstack((batch_X, add_X))
            batch_Y = np.vstack((batch_Y, add_Y))
            mini_batch_X.append(batch_X)
            mini_batch_Y.append(batch_Y)

        # Return the mini batch of the features and label
        return np.array(mini_batch_X), np.array(mini_batch_Y)

    def predict(self, x):
        """
        Define the prediction function
        We can use predict function to predict the results of new data, by using the well-trained network.

        :type x: numpy.array
        :param x: data of the dataset

        :return: predict label of the input dataset
        """
        x = np.array(x)
        pred = np.zeros(x.shape[0])
        # Put all data to the forward progress
        output = nn_relu.forward(x[:, :], test_flg=True)
        # Softmax progress
        exps = np.exp(output - np.max(output, axis=1, keepdims=True))
        output = exps / np.sum(exps, axis=1, keepdims=True)
        # Select the max one as the label
        pred = np.argmax(output, axis=1)
        return pred



class HiddenLayer2(object):
    def __init__(self, n_in, n_out, activation_last_layer='relu', activation='relu'):
        """
        Define the hidden layer for the mlp. For example, h1 = HiddenLayer(10, 5, activation="relu")
        Means we create a layer with 10 dimension input and 5 dimension output, and using tanh activation function.
        Make sure the input size of hidden layer should be matched with the output size of the previous layer!

        Typical hidden layer of an MLP: units are fully-connected and have
        ReLU activation function, batch normalization and dropout.
        Weight matrix W is of shape (n_in,n_out) and the bias vector b is of shape (n_out,).

        NOTE : The nonlinear used here is ReLU

        Hidden unit activation is given by: ReLU(dot(input,W) + b)

        :type n_in: int
        :param n_in: dimensionality of input

        :type n_out: int
        :param n_out: number of hidden units

        :type activation: string
        :param activation: non-linearity to be applied in the hidden layer
        """
        self.input = None
        self.output = None
        if activation:
            self.activation = Activation(activation).f

        # Activation derivative of last layer
        self.activation_deriv = None
        if activation_last_layer:
            self.activation_deriv = Activation(activation_last_layer).f_deriv

        # Randomly assign small values. Because the ReLU and GeLU are both non-linear activation
        # So, we can use the HE initialization for them.
        self.W = np.random.uniform(low=-np.sqrt(2 / n_out), high=np.sqrt(2 / n_out), size=(n_in, n_out))
        self.b = np.zeros(n_out, )

        # We set he size of weight gradation as the size of weight
        self.grad_W = np.zeros(self.W.shape)
        self.grad_b = np.zeros(self.b.shape)

        # Dropout parameter for:
        # dropout: whether to use dropout. mask: dropout matrix. output_layer: whether the layer is the output layer.
        self.dropout = None
        self.mask = None
        self.output_layer = False

        # The parameter for the Batch normalization, including:
        # Whether to use the batch normalization. The mean and variance in testing process
        # The gamma and bete with their update parameters.
        self.batch_norm = None
        self.running_mean = None
        self.running_var = None
        self.std = None
        self.xn = None
        self.xc = None
        self.batch_size = None
        self.gamma = np.ones(n_out, )
        self.beta = np.zeros(n_out, )
        self.grad_gamma = np.zeros(self.gamma.shape)
        self.grad_beta = np.zeros(self.beta.shape)

        # The parameter for Momentum batch normalization
        # Indicates the current velocity of the weight and bias momentum
        self.grad_W_V = np.zeros(self.W.shape)
        self.grad_b_V = np.zeros(self.b.shape)
        self.grad_gamma_V = np.zeros(self.gamma.shape)
        self.grad_beta_V = np.zeros(self.beta.shape)

        # Adam in batch normalization
        # The 1st moment vector
        self.grad_W_mt = np.zeros(self.W.shape)
        self.grad_b_mt = np.zeros(self.b.shape)
        self.grad_gamma_mt = np.zeros(self.gamma.shape)
        self.grad_beta_mt = np.zeros(self.beta.shape)
        # The 2nd moment vector (squared)
        self.grad_W_vt = np.zeros(self.W.shape)
        self.grad_b_vt = np.zeros(self.b.shape)
        self.grad_gamma_vt = np.zeros(self.gamma.shape)
        self.grad_beta_vt = np.zeros(self.beta.shape)

    def forward(self, input, test_flg=False, dropout_rate=0.5):
        """
        The forward progress for the hidden layer, including the dropout and batch normalization
        Calculate the output of the output layer
        Can determine whether to use the test mode

        :type input: numpy.array
        :param input: a symbolic tensor of shape (batch_size, n_in)

        :type test_flg: boolean
        :param test_flg: whether is the test progress

        :type dropout_rate: float
        :param dropout_rate: probability for dropout

        :return: output of shape (batch_size, n_out) for this hidden layer
        """
        # Calculate the perception
        lin_output = np.dot(input, self.W) + self.b

        # For the forward progress, use dropout first and then batch normalization
        # The dropout progress
        if self.dropout == True and self.output_layer == False:
            lin_output = self.dropout_forward(lin_output, dropout_rate=dropout_rate, test_flg=test_flg)

        # The batch normalization progress
        if self.batch_norm:
            lin_output = self.batch_normalization_forward(lin_output, test_flg=test_flg)

        # Use the activation function
        self.output = (
            lin_output if self.activation is None
            else self.activation(lin_output)
        )
        self.input = input
        return self.output

    def backward(self, delta, output_layer=False):
        """
        The backward progress for the hidden layer, including the dropout and batch normalization
        Calculate the delta for the derivative used for calculating loss function

        :type delta: numpy.array
        :param delta: a symbolic tensor of shape (batch_size, n_out)

        :type output_layer: boolean
        :param output_layer: whether is the output layer of the network or not

        :return: new delta of shape (batch_size, n_in)
        """
        self.output_layer = output_layer
        # For the backward progress, consider batch normalization first and then dropout
        # The batch normalization backward progress
        if self.batch_norm:
            delta = self.batch_normalization_backward(delta)

        # The dropout backward progress
        if self.dropout == True and self.output_layer == False:
            delta = self.dropout_backward(delta)

        # Calculate the gradation of weight and bias for update
        self.grad_W = np.atleast_2d(self.input).T.dot(np.atleast_2d(delta))
        self.grad_b = np.sum(delta, axis=0)

        # The activation derivative progress
        if self.activation_deriv:
            delta = delta.dot(self.W.T) * self.activation_deriv(self.input)
        return delta

    def batch_normalization_forward(self, x, test_flg=False):
        """
        The batch normalization forward progress, including the train mode and the test mode
        Used for normalization for the input
        Accept the output of the perception and return a new output

        :type x: numpy.array
        :param x: a symbolic tensor of shape (batch_size, n_out)

        :type test_flg: boolean
        :param test_flg: determine the test mode or train mode

        :return: output of batch normalization in shape (batch_size, n_out)
        """
        # Initialize the mean and variance of the input with the shape of (,n_in)
        if self.running_mean is None:
            # self.input_shape = x.shape
            N, D = x.shape
            self.running_mean = np.zeros(D)
            self.running_var = np.zeros(D)

        # If it is the train progress
        if test_flg is False:
            # Calculate the mean
            mean = np.mean(x, axis=0)
            # Subtract the mean of each training example
            xc = x - mean
            # Calculate the variance
            var = np.mean(xc ** 2, axis=0)
            # Add epsilon for numerical stability, then sqrt
            std = np.sqrt(var + 10e-7)
            # Execute normalization
            xn = xc / std

            # Store the parameters
            self.batch_size = x.shape[0]
            self.xc = xc
            self.xn = xn
            self.std = std
            # Update the mean and variance after each progress
            self.running_mean = 0.9 * self.running_mean + (1 - 0.9) * mean
            self.running_var = 0.9 * self.running_var + (1 - 0.9) * var
        else:
            # In the test progress of batch normalization, only two steps
            # Subtract the mean of each test example
            xc = x - self.running_mean
            # Execute normalization
            xn = xc / (np.sqrt(self.running_var + 10e-7))

        # The transformation step
        out = self.gamma * xn + self.beta
        # return out.reshape(*self.input_shape)
        return out

    def batch_normalization_backward(self, dout):
        """
        The backward progress for the batch normalization.
        Accept the delta and return a new

        :type dout: numpy.array
        :param dout: a symbolic tensor of shape (batch_size, n_out)

        :return: new delta of shape (batch_size, n_out)
        """
        # Calculate the derivative of the beta
        dbeta = np.sum(dout, axis=0)
        # Calculate the derivative of the gamma
        dgamma = np.sum(dout * self.xn, axis=0)
        # Derivative of the batch normalization forward progress
        dxn = self.gamma * dout
        dxc = dxn / self.std
        dstd = -np.sum((dxn * self.xc) / (self.std * self.std), axis=0)
        dvar = 0.5 * dstd / self.std
        dxc += (2.0 / self.batch_size) * self.xc * dvar
        dmu = np.sum(dxc, axis=0)
        # Calculate the new delta
        dx = dxc - dmu / self.batch_size

        # Update the parameters
        self.grad_gamma = dgamma
        self.grad_beta = dbeta
        # return dx.reshape(*self.input_shape)
        return dx

    def dropout_forward(self, x, dropout_rate=0.5, test_flg=False):
        """
        The dropout forward progress, dropout the several nodes in the neural network of each hidden layer

        :type x: numpy.array
        :param x: a symbolic tensor of shape (batch_size, n_out)

        :type dropout_rate: float
        :param dropout_rate: rate for the dropout

        :type test_flg: boolean
        :param test_flg: determine the test mode or train mode

        :return: output of dropout in shape of (batch_size, n_out)
        """
        # If it is the train progress
        if not test_flg:
            # Create a matrix in binary (0 and 1) according to the dropout rate and rescale
            self.mask = np.random.binomial(1, 1 - dropout_rate, x.shape) / (1 - dropout_rate)
            return x * self.mask
        else:
            return x

    def dropout_backward(self, dout):
        """
        The dropout backward progress

        :type dout: numpy.array
        :param dout: a symbolic tensor of shape (batch_size, n_out)

        :return: new delta of shape (batch_size, n_out)
        """
        if self.mask is None:
            return dout
        else:
            return dout * self.mask


class MLP2:
    def __init__(self, layers, activation=[None, 'relu', 'relu'], dropout=True, batch_norm=True):
        """
        For initialization, the code will create all layers automatically based on the provided parameters.
        The Multi-Layer perception, also known as the Artificial Neural Network
        Basic including the input layer, hidden layer and the output layer
        In this MLP, we add the batch normalization, softmax with cross entropy and activation layer
        User can choose the two types od optimizer for the parameter update: momentum in SGD and Adam
        The data training is in form of the mini-batch

        :type layers: list
        :param layers: a list containing the number of units in each layer.
        Should be at least two values

        :type activation: list
        :param activation: activation function to be used. Can be "logistic" or "tanh"

        :type dropout: boolean
        :param dropout: whether to use the dropout or not

        :type batch_norm: boolean
        :param batch_norm: whether to use the batch normalization or not
        """
        # Initialize layers
        self.layers = []
        self.params = []

        # Append the hidden layer and activation layer to the initialization
        self.activation = activation
        for i in range(len(layers) - 1):
            self.layers.append(HiddenLayer2(layers[i], layers[i + 1], activation[i], activation[i + 1]))

        # Set the last layer as the output layer
        self.layers[-1].output_layer = True

        # Set the attributes of whether to use dropout and batch normalization or not
        for layer in self.layers:
            layer.dropout = dropout
            layer.batch_norm = batch_norm

    def forward(self, input, dropout_rate=0.5, test_flg=False):
        """
        Forward progress: pass the information through the layers and out the results of final output layer

        :type input: numpy.array
        :param input: a symbolic tensor of shape (batch_size, n_in)

        :type dropout_rate: float
        :param dropout_rate: probability for dropout

        :type test_flg: boolean
        :param test_flg: whether is the test progress

        :return: output of shape (batch_size, n_out) for this hidden layer
        """
        for layer in self.layers:
            output = layer.forward(input, dropout_rate=dropout_rate, test_flg=test_flg)
            input = output
        return output

    def crossEntropy_softmax(self, y_hat, y):
        """
        The softmax with cross entropy layer accepting two parameters: y_predict and y_target
        Combine the softmax with the cross entropy to calculate the loss

        :type y_hat: numpy.array
        :param y_hat: predict value of the input data

        :type y: numpy.array
        :param y: target value of the input data

        :return: loss of the output
        """
        # Calculate the softmax
        exps = np.exp(y_hat - np.max(y_hat, axis=1, keepdims=True))
        out = exps / np.sum(exps, axis=1, keepdims=True)
        batch_size = y.shape[0]
        # Calculate the cross entropy loss
        return -np.sum(np.log(out[np.arange(batch_size), y.reshape(-1)] + 1e-7)) / batch_size

    def delta_crossEntropy_softmax(self, y_hat, y):
        """
        The softmax with cross entropy layer accepting two parameters: y_predict and y_target
        Combine the softmax with the cross entropy to calculate the delta

        :type y_hat: numpy.array
        :param y_hat: predict value of the input data

        :type y: numpy.array
        :param y: target value of the input data

        :return: the delta of the output
        """
        # Calculate the softmax
        exps = np.exp(y_hat - np.max(y_hat, axis=1, keepdims=True))
        out = exps / np.sum(exps, axis=1, keepdims=True)
        batch_size = y.shape[0]
        # Calculate the cross entropy delta
        # Because the shape of y is (batch_size,1), we need to transfer it to the (batch_size,)
        out[np.arange(batch_size), y.reshape(-1)] -= 1
        return out / batch_size

    def criterion_CE(self, y, y_hat):
        """
        Define the objection/loss function, we use Cross-Entropy as the loss
        Using the cross entropy function to calculate the loss and delta
        Including the derivative of the activation function

        :type y: numpy.array
        :param y: target value of the input data

        :type y_hat: numpy.array
        :param y_hat: predict value of the input data

        :return: loss and delta
        """
        activation_deriv = Activation(self.activation[-1]).f_deriv
        # Cross Entropy delta and loss
        error = self.delta_crossEntropy_softmax(y_hat, y)
        loss = self.crossEntropy_softmax(y_hat, y)
        delta = error * activation_deriv(y_hat)
        return loss, delta

    def backward(self, delta):
        """
        Backward progress: pass the delta and update the parameter for next forward progress

        :type delta: numpy.array
        :param delta: a symbolic tensor of shape (batch_size, n_out)
        """
        # Set the last layer as the output layer
        delta = self.layers[-1].backward(delta, output_layer=True)
        for layer in reversed(self.layers[:-1]):
            delta = layer.backward(delta)

    def update(self, lr, optimizer=None, weight_decay_lambda=1, momentum=0.9, iter=100, rho1=0.9, rho2=0.999,
               epsilon=1e-8):
        """
        Update the network weights after backward.
        The update for the weight and bias, lr decide the learning rate
        of the weight update. None, Momentum and Adam can be selected for
        the optimizer.

        :type lr: float
        :param lr: learning rate for the update

        :type optimizer: str
        :param optimizer: None, momentum and Adam, select one of the optimizer

        :type weight_decay_lambda: float
        :param weight_decay_lambda: constant value of the weight decay, Normally is 0.9

        :type momentum: float
        :param momentum: constant value of the momentum in SGD. Normally is 0.9

        :type iter: int
        :param iter: epoch of the training process

        :type rho1: float
        :param rho1: exponential decay rate for the first moment estimates. Normally is 0.9

        :type rho2: float
        :param rho2: exponential decay rate for the second-moment estimates. Normally is 0.999

        :type epsilon: float
        :param epsilon: constant value for numerical stability. Normally is 1e-8
        """
        for layer in self.layers:
            # Normal mode with no optimizer
            if optimizer is None:
                # update the weight with the weight decay
                # According to the lecture 4: 𝜃 = (1−𝜂𝛼)𝜃 - 𝜂𝛻(𝜃)
                layer.W = (1 - lr * weight_decay_lambda) * layer.W - lr * layer.grad_W
                layer.b -= lr * layer.grad_b

                # for batch normalization parameter with the weight decay
                layer.gamma = (1 - lr * weight_decay_lambda) * layer.gamma - lr * layer.grad_gamma
                layer.beta -= lr * layer.grad_beta

            # Momentum with SGD
            elif optimizer.lower() == 'momentum':
                # update the weight in momentum with the weight decay
                # According to the lecture 3
                # v = momentum * v + learning_rate * gradient
                # w = w - v
                layer.grad_W_V = momentum * layer.grad_W_V + lr * layer.grad_W
                layer.grad_b_V = momentum * layer.grad_b_V + lr * layer.grad_b
                layer.W = (1 - lr * weight_decay_lambda) * layer.W - layer.grad_W_V
                layer.b = layer.b - layer.grad_b_V

                # for batch normalization parameter with momentum and weight decay
                layer.grad_gamma_V = momentum * layer.grad_gamma_V + lr * layer.grad_gamma
                layer.grad_beta_V = momentum * layer.grad_beta_V + lr * layer.grad_beta
                layer.gamma = (1 - lr * weight_decay_lambda) * layer.gamma - layer.grad_gamma_V
                layer.beta = layer.beta - layer.grad_beta_V

            # Adam
            elif optimizer.lower() == 'adam':
                # update the weight in Adam with the weight decay
                # According to the lecture 3
                # mt = beta1 * mt + (1-beta1) * gradient
                # vt = beta2 * vt + (1-beta2) * (gradient**2)
                layer.grad_W_mt = rho1 * layer.grad_W_mt + (1 - rho1) * layer.grad_W
                layer.grad_W_vt = rho2 * layer.grad_W_vt + (1 - rho2) * (layer.grad_W ** 2)
                layer.grad_b_mt = rho1 * layer.grad_b_mt + (1 - rho1) * layer.grad_b
                layer.grad_b_vt = rho2 * layer.grad_b_vt + (1 - rho2) * (layer.grad_b ** 2)
                # mt_vector = mt / (1-beta1**iter)
                # vt_vector = vt / (1-beta2**iter)
                w_mt_vector = layer.grad_W_mt / (1 - rho1 ** iter)
                w_vt_vector = layer.grad_W_vt / (1 - rho2 ** iter)
                b_mt_vector = layer.grad_b_mt / (1 - rho1 ** iter)
                b_vt_vector = layer.grad_b_vt / (1 - rho2 ** iter)
                # w = w - lr * mt_vector / (np.sqrt(vt_vector + epsilon))
                layer.W = (1 - lr * weight_decay_lambda) * layer.W - lr * w_mt_vector / (np.sqrt(w_vt_vector + epsilon))
                layer.b = layer.b - lr * b_mt_vector / (np.sqrt(b_vt_vector + epsilon))

                # for batch normalization parameter with Adam and weight decay
                layer.grad_gamma_mt = rho1 * layer.grad_gamma_mt + (1 - rho1) * layer.grad_gamma
                layer.grad_gamma_vt = rho2 * layer.grad_gamma_vt + (1 - rho2) * (layer.grad_gamma ** 2)
                layer.grad_beta_mt = rho1 * layer.grad_beta_mt + (1 - rho1) * layer.grad_beta
                layer.grad_beta_vt = rho2 * layer.grad_beta_vt + (1 - rho2) * (layer.grad_beta ** 2)
                gamma_mt_vector = layer.grad_gamma_mt / (1 - rho1 ** iter)
                gamma_vt_vector = layer.grad_gamma_vt / (1 - rho2 ** iter)
                beta_mt_vector = layer.grad_beta_mt / (1 - rho1 ** iter)
                beta_vt_vector = layer.grad_beta_vt / (1 - rho2 ** iter)
                layer.gamma = (1 - lr * weight_decay_lambda) * layer.gamma - lr * gamma_mt_vector / (
                    np.sqrt(gamma_vt_vector + epsilon))
                layer.beta = layer.beta - lr * beta_mt_vector / (np.sqrt(beta_vt_vector + epsilon))

    def fit(self, X, y, learning_rate=0.1, epochs=100, mini_batch_size=128, optimizer=None, weight_decay_lambda=1,
            momentum=0.9, rho1=0.9,
            rho2=0.999, epsilon=1e-8, dropout_rate=0.5):
        """
        Online Learning
        Define the training function
        It will return all losses within the whole training process.

        :type X: numpy.array
        :param X: data of the training dataset

        :type y: numpy.array
        :param y: label of the training dataset

        :type learning_rate: float
        :param learning_rate: learning rate for the update

        :type epochs: int
        :param epochs: epoch of the training process

        :type mini_batch_size: int
        :param mini_batch_size: size of each batch

        :type optimizer: str
        :param optimizer: None, momentum and Adam, select one of the optimizer

        :type weight_decay_lambda: float
        :param weight_decay_lambda: constant value of the weight decay, Normally is 0.9

        :type momentum: float
        :param momentum: constant value of the momentum in SGD. Normally is 0.9

        :type rho1: float
        :param rho1: exponential decay rate for the first moment estimates. Normally is 0.9

        :type rho2: float
        :param rho2: exponential decay rate for the second-moment estimates. Normally is 0.999

        :type epsilon: float
        :param epsilon: constant value for numerical stability. Normally is 1e-8

        :type dropout_rate: float
        :param dropout_rate: rate for the dropout

        :return: loss, accuracy of the training set, accuracy of the validation set
        """
        X = np.array(X)
        y = np.array(y)
        # Initialize the loss, accuracy of the training set, accuracy of the validation set for all epoch
        to_return = np.zeros(epochs)
        train_accuracy = np.zeros(epochs)
        val_accuracy = np.zeros(epochs)

        for k in range(epochs):
            # Partition of the training dataset
            mini_batch_X, mini_batch_Y = self.mini_batch(X, y, mini_batch_size)
            loss = np.zeros(mini_batch_X.shape[0])

            for it in range(mini_batch_X.shape[0]):
                # Forward pass
                y_hat = self.forward(mini_batch_X[it], dropout_rate)
                # Loss function and backward pass
                loss[it], delta = self.criterion_CE(mini_batch_Y[it], y_hat)
                self.backward(delta)
                # Update
                self.update(lr=learning_rate, optimizer=optimizer, weight_decay_lambda=weight_decay_lambda,
                            momentum=momentum, iter=epochs, rho1=rho1, rho2=rho2, epsilon=epsilon)

            # Store the loss for one epoch
            to_return[k] = np.mean(loss)

            # Predict the accuracy of the training and validation data for one epoch
            train_pre = nn_gelu.predict(X_train)
            train_acc = accuracy_score(y_train, train_pre)
            val_pre = nn_gelu.predict(X_val)
            val_acc = accuracy_score(y_val, val_pre)

            # Store the accuracy for one epoch and print
            val_accuracy[k] = val_acc
            train_accuracy[k] = train_acc
            print(
                'epoch {0} loss: {1:.6f} train_accuracy: {2:.6f} validate_accuracy: {3:.6f}'.format(k,
                                                                                                    float(to_return[k]),
                                                                                                    train_acc, val_acc))
        return to_return, train_accuracy, val_accuracy

    def mini_batch(self, X, Y, batch_size=64):
        """
        Define the function for the mini batch training
        Divide the training dataset to the several batches
        Each batch has the size of the batch_size
        If the last batch does not have the enough data, then full it to the batch size.

        :type X: numpy.array
        :param X: data of the training dataset

        :type Y: numpy.array
        :param Y: label of the training dataset

        :type batch_size: int
        :param batch_size: size of each batch

        :return: mini batch of the features and label in the type of numpy array
        """
        # Initialize the random seed
        np.random.seed(2023)
        mini_batch_X = []
        mini_batch_Y = []
        # Get the number of training example
        m = X.shape[0]

        # Shuffle (X, Y), m is the num of instances of data set
        permutation = np.random.permutation(m)
        shuffled_X = X[permutation, :]
        shuffled_Y = Y[permutation, :].reshape((m, 1))

        # Split the training data into batch_size=64
        # Round down
        num_batches = math.floor(m / batch_size)
        # Partition
        for i in range(0, num_batches):
            batch_X = shuffled_X[i * batch_size:(i + 1) * batch_size, :]
            batch_Y = shuffled_Y[i * batch_size:(i + 1) * batch_size, :]
            mini_batch_X.append(batch_X)
            mini_batch_Y.append(batch_Y)

        # If there are remaining training examples, add them into the last batch
        if m % batch_size != 0:
            batch_X = shuffled_X[num_batches * batch_size:, :]
            batch_Y = shuffled_Y[num_batches * batch_size:, :]
            # Calculate the minus between the number of data in the last batch and the batch size
            # Then Calculate the number required to full of the batch
            add_X = shuffled_X[m - (num_batches * batch_size) - batch_size:, :]
            add_Y = shuffled_Y[m - (num_batches * batch_size) - batch_size:, :]
            # Combine the rest data and the additional data to an integral batch
            batch_X = np.vstack((batch_X, add_X))
            batch_Y = np.vstack((batch_Y, add_Y))
            mini_batch_X.append(batch_X)
            mini_batch_Y.append(batch_Y)

        # Return the mini batch of the features and label
        return np.array(mini_batch_X), np.array(mini_batch_Y)

    def predict(self, x):
        """
        Define the prediction function
        We can use predict function to predict the results of new data, by using the well-trained network.

        :type x: numpy.array
        :param x: data of the dataset

        :return: predict label of the input dataset
        """
        x = np.array(x)
        pred = np.zeros(x.shape[0])
        # Put all data to the forward progress
        output = nn_gelu.forward(x[:, :], test_flg=True)
        # Softmax progress
        exps = np.exp(output - np.max(output, axis=1, keepdims=True))
        output = exps / np.sum(exps, axis=1, keepdims=True)
        # Select the max one as the label
        pred = np.argmax(output, axis=1)
        return pred


# Load the dataset
X = np.load("../Assignment1-Dataset/train_data.npy")
y = np.load("../Assignment1-Dataset/train_label.npy")
X_test = np.load("../Assignment1-Dataset/test_data.npy")
y_test = np.load("../Assignment1-Dataset/test_label.npy")

# Split the dataset to the training set and validation set, the proportion of is 8:1:1
X_train = X[:40000]
y_train = y[:40000]
X_val = X[40000:]
y_val = y[40000:]

# Try different MLP models, dropout, batch normalization
# You can set different nodes of hidden layers, activations, and options for dropout, batch normalization
# But notice that the number of the hidden layers must equal to the activation layers
# activation layer: "relu", "gelu"
# dropout and batch_norm: True, False
nn_relu = MLP([128, 1280, 640, 320, 120, 32, 10], [None, 'relu', 'relu', 'relu', 'relu', 'relu', 'relu'],
         dropout=True,
         batch_norm=True)
loss_relu, train_acc_relu, validata_acc_relu = nn_relu.fit(X_train, y_train,
                                        learning_rate=0.001, epochs=100,
                                        mini_batch_size=64,
                                        optimizer="momentum",
                                        weight_decay_lambda=0.001,
                                        momentum=0.9, rho1=0.9,
                                        rho2=0.999, dropout_rate=0.4)

nn_gelu = MLP2([128, 1280, 640, 320, 120, 32, 10], [None, 'gelu', 'gelu', 'gelu', 'gelu', 'gelu', 'gelu'],
         dropout=True,
         batch_norm=True)
loss_gelu, train_acc_gelu, validata_acc_gelu = nn_gelu.fit(X_train, y_train, learning_rate=0.001, epochs=100, mini_batch_size=64,
                                       optimizer="momentum", weight_decay_lambda=0.001, momentum=0.9, rho1=0.9,
                                       rho2=0.999, dropout_rate=0.4)

# Plot the loss figure
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.plot(loss_relu, label='loss_ReLU')
plt.plot(loss_gelu, label='loss_GELU')
plt.legend()
plt.show()

# Plot the accuracy figure
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.plot(train_acc_relu, label='train_acc_ReLU')
plt.plot(validata_acc_relu, label='val_acc_ReLU')
plt.plot(train_acc_gelu, label='train_acc_GELU')
plt.plot(validata_acc_gelu, label='val_acc_GELU')
plt.legend()
plt.show()

# Calculate the accuracy of the test dataset
_pre = nn_relu.predict(X_test)
acc = accuracy_score(y_test, _pre)
f1 = f1_score(y_test, _pre, average='macro')
recall = recall_score(y_test, _pre, average='macro')
precision = precision_score(y_test, _pre, average='macro')
print('Test-accuracy-relu: {0:.4f}, F1_score: {1:.4f}, Recall: {2:.4f}, Precision: {3:.4f}'.format(acc, f1, recall,
                                                                                              precision))

# Calculate the accuracy of the test dataset
_pre128 = nn_gelu.predict(X_test)
acc_128 = accuracy_score(y_test, _pre128)
f1_128 = f1_score(y_test, _pre128, average='macro')
recall_128 = recall_score(y_test, _pre128, average='macro')
precision_128 = precision_score(y_test, _pre128, average='macro')
print('Test-accuracy-gelu: {0:.4f}, F1_score: {1:.4f}, Recall: {2:.4f}, Precision: {3:.4f}'.format(acc_128,
                                                                                                         f1_128,
                                                                                                         recall_128,
                                                                                                         precision_128))

# Calculate the code run time
end = time.time()
print("Code running Time: {:.3f} min".format((end - start) / 60))
