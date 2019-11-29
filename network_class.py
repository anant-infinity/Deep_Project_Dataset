
#### Libraries
# Standard library
import random
import csv
# Third-party libraries
import numpy as np
import matplotlib.pyplot as plt

class Network(object):

    def __init__(self, sizes, act_func_flag, cost_func_flag, plot_flag, log_flag, momentum_flag):
        self.num_layers = len(sizes)
        self.sizes = sizes
        self.biases = [np.random.randn(y, 1) for y in sizes[1:]]
        self.weights = [np.random.randn(y, x)
                        for x, y in zip(sizes[:-1], sizes[1:])]
        self.accuracy_array = []
        self.epochs_array = []
        self.act_func_flag = act_func_flag
        self.cost_func_flag = cost_func_flag
        self.plot_flag = plot_flag
        self.log_flag = log_flag
        self.momentum_flag = momentum_flag

    def feedforward(self, a):

        for b, w in zip(self.biases, self.weights):
            a = self.act_func(np.dot(w, a)+b)
        return a

    def SGD(self, training_data, epochs, mini_batch_size, eta, gamma, beta_1, beta_2,
            test_data=None):

        if test_data: n_test = len(test_data)
        n = len(training_data)
        self.epochs_array = np.arange(epochs)
        for j in range(epochs):
            t = 1
            random.shuffle(training_data)
            mini_batches = [
                training_data[k:k+mini_batch_size]
                for k in range(0, n, mini_batch_size)]
            for mini_batch in mini_batches:
                self.update_mini_batch(mini_batch, eta, gamma, beta_1, beta_2, t)
                t += 1
            if test_data:
                k = self.evaluate(test_data, j)
                print("Epoch {0}: {1} / {2}".format(
                    j, k, n_test))
                self.accuracy_array.append(k/n_test)
            else:
                print("Epoch {0} complete".format(j))
        if self.plot_flag == 1:
            self.plot()

    def update_mini_batch(self, mini_batch, eta, gamma, beta_1, beta_2, t):

        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        update_minus1_b = [np.zeros(b.shape) for b in self.biases]
        update_minus1_w = [np.zeros(w.shape) for w in self.weights]

        epsilon = np.power(0.1, 8)

        v_t_b = [np.zeros(b.shape) for b in self.biases]
        v_t_w = [np.zeros(w.shape) for w in self.weights]

        m_t_b = [np.zeros(b.shape) for b in self.biases]
        m_t_w = [np.zeros(w.shape) for w in self.weights]

        update_minus1_vtb = [np.zeros(b.shape) for b in self.biases]
        update_minus1_vtw = [np.zeros(w.shape) for w in self.weights]

        update_minus1_mtb = [np.zeros(b.shape) for b in self.biases]
        update_minus1_mtw = [np.zeros(w.shape) for w in self.weights]



        for x, y in mini_batch:

            # For using nestorov gradient descent
            if(self.momentum_flag == 2):
                self.weights = [w - (gamma/len(mini_batch))*u_m1_w
                        for w, u_m1_w in zip(self.weights, update_minus1_w)]

                self.biases = [b - (gamma/len(mini_batch))*u_m1_b
                       for b, u_m1_b in zip(self.biases, update_minus1_b)]

            delta_nabla_b, delta_nabla_w = self.backprop(x, y)


            update_minus1_b = [nb for nb in nabla_b]
            nabla_b = [nb+dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]

            update_minus1_w = [nw for nw in nabla_w]
            nabla_w = [nw+dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]

        # For using momentum and nestorov gradient descent
        if(self.momentum_flag == 1):
            # minus gamma times update t-1
            self.weights = [w-(eta/len(mini_batch))*nw - (gamma/len(mini_batch))*u_m1_w
                        for w, nw, u_m1_w in zip(self.weights, nabla_w, update_minus1_w)]

            self.biases = [b-(eta/len(mini_batch))*nb - (gamma/len(mini_batch))*u_m1_b
                       for b, nb, u_m1_b in zip(self.biases, nabla_b, update_minus1_b)]

        elif(self.momentum_flag == 2):
            self.weights = [w-(eta/len(mini_batch))*nw
                        for w, nw in zip(self.weights, nabla_w)]

            self.biases = [b-(eta/len(mini_batch))*nb
                       for b, nb in zip(self.biases, nabla_b)]

        # ADAM = annealing + momentum
        elif(self.momentum_flag == 3):

            # TODO correct this , beta_1 ^^ t
            update_minus1_vtw = [nw for nw in v_t_w]
            v_t_w = [np.multiply(np.multiply(n_t_w1, beta_2) + np.multiply(np.multiply(nw,nw),(1-beta_2)),
                                 1/(1-np.power(beta_2,t))) for n_t_w1, nw in zip( update_minus1_vtw, nabla_w)]
            update_minus1_vtb = [nb for nb in v_t_b]
            v_t_b = [np.multiply(np.multiply(n_t_b,beta_2) + np.multiply(np.multiply(nb,nb),(1-beta_2)),
                                 1 / (1 - np.power(beta_2, t))) for n_t_b, nb in zip(update_minus1_vtb, nabla_b)]

            update_minus1_mtw = [nw for nw in m_t_w]
            m_t_w = [np.multiply(np.multiply(m_t_w, beta_1) + np.multiply(nw,(1-beta_1)),
                                 1/(1-np.power(beta_1,t)))  for m_t_w, nw in zip(update_minus1_mtw, nabla_w)]

            update_minus1_mtb = [nb for nb in m_t_b]
            m_t_b = [np.multiply(np.multiply(m_t_b, beta_1) + np.multiply(nb, (1 - beta_1)),
                                 1 / (1 - np.power(beta_1, t))) for m_t_b, nb in zip(update_minus1_mtb, nabla_b)]

            self.weights = [w - (np.multiply(((eta/(np.sqrt(np.add(n1_t_w,epsilon))))/len(mini_batch)), m1_t_w))
                            for w, n1_t_w, m1_t_w in zip(self.weights, v_t_w, m_t_w)]

            self.biases = [b - (np.multiply(((eta/(np.sqrt(np.add(v1_t_b,epsilon))))/len(mini_batch)),m1_t_b))
                           for b, v1_t_b, m1_t_b in zip(self.biases, v_t_b, m_t_b)]

            # For using standard gradient descent
        elif(self.momentum_flag == 0):
            self.weights = [w - (eta / len(mini_batch)) * nw
                            for w, nw in zip(self.weights, nabla_w)]
            self.biases = [b - (eta / len(mini_batch)) * nb
                           for b, nb in zip(self.biases, nabla_b)]




    def backprop(self, x, y):

        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        # feedforward
        activation = x
        activations = [x] # list to store all the activations, layer by layer
        zs = [] # list to store all the z vectors, layer by layer
        for b, w in zip(self.biases, self.weights):
            z = np.dot(w, activation)+b
            zs.append(z)
            activation = self.act_func(z)
            activations.append(activation)
        # backward pass
        if(self.cost_func_flag == 1):
            delta = self.cost_derivative(activations[-1], zs[-1], y)
        else:
            delta = self.cost_derivative(activations[-1], zs[-1], y) * \
                    self.act_func_prime(zs[-1])

        nabla_b[-1] = delta
        nabla_w[-1] = np.dot(delta, activations[-2].transpose())

        for l in range(2, self.num_layers):
            z = zs[-l]
            sp = self.act_func_prime(z)
            delta = np.dot(self.weights[-l+1].transpose(), delta) * sp
            nabla_b[-l] = delta
            nabla_w[-l] = np.dot(delta, activations[-l-1].transpose())

        return (nabla_b, nabla_w)

    def evaluate(self, test_data, epoch_num):
        test_results = [(np.argmax(self.feedforward(x)), y)
                        for (x, y) in test_data]

        if (self.log_flag == 1):
            with open('LOG.csv', 'a+') as out:
                csv_out = csv.writer(out)
                if epoch_num == 0:
                    csv_out.writerow(['Model Output', 'Label'])
                csv_out.writerow(['Epoch Number', epoch_num])
                for row in test_results:
                    csv_out.writerow(row)

        return sum(int(x == y) for (x, y) in test_results)

    def plot(self):
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.plot(self.epochs_array, self.accuracy_array)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Accuracy')
        plt.show()

    def cost_derivative(self, output_activations, input_activations, y):
        if(self.cost_func_flag == 1):
            return (output_activations-y)
        # Cross Entropy with Softmax
        elif (self.cost_func_flag == 2):
            exps = np.exp(input_activations)
            exps = exps / np.sum(exps)
            return (exps-y)

    # Activation Function - Sigmoid or tanh
    def act_func(self, z):
        if (self.act_func_flag==1):
            return 1.0/(1.0+np.exp(-z))
        else:
            return np.tanh(z)

    # Activation Function derivative
    def act_func_prime(self, z):
        if(self.act_func_flag == 1):
            return self.act_func(z)*(1-self.act_func(z))
        else:
            return 1-(np.tanh(z))**2

