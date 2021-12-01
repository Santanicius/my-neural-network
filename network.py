
import pandas as pd
import math
import numpy as np
from random import seed
from random import randrange
from random import random


class Network:

    def __init__(self, learning_rate=0.5, epoch=100, n_hiddens=1, error_rate=0.5, output_mode='Logistics'):
        seed(1)
        self.learning_rate = learning_rate
        self.epoch = epoch
        self.n_inputs = 0
        self.n_outputs = 0
        self.n_hiddens = n_hiddens
        self.listErrors = []
        self.network = []
        self.input = []
        self.dic_encode = dict()
        self.error_rate = error_rate
        self.out_mode = output_mode
        
    def geometric_mean(self, inputs, outputs):
        return math.floor(math.sqrt((float(inputs * outputs))))

    def initialize(self, dataf):
        for col in dataf.drop(columns="classe", axis=1):
            dataf[str(col)] = dataf[str(col)].astype(float)
        self.input = dataf.values.tolist()

        self.n_inputs = len(self.input[0]) - 1

        last = len(self.input[0]) - 1

        classlist = [row[last] for row in self.input]
        self.n_outputs = len(set(classlist))
        self.dic_encode = dict()
        for i, value in enumerate(set(classlist)):
            self.dic_encode[value] = i

        for row in self.input:
            row[last] = self.dic_encode[row[last]]
        if self.n_hiddens == 0 or self.n_hiddens == 1:
            self.n_hiddens = self.geometric_mean(self.n_inputs, self.n_outputs)
        self.shuffle()
        self.normalization()

    def shuffle(self):
        temp = list(self.input)
        self.input = list()
        while len(temp) > 0:
            index = randrange(len(temp))
            self.input.append(temp.pop(index))

    def normalization(self):
        state = [[min(col), max(col)] for col in zip(*self.input)]
        for row in self.input:
            for i in range(len(row) - 1):
                row[i] = (row[i] - state[i][0]) / (state[i][1] - state[i][0])

    def init_network(self):
        self.network = list()
        hidden = [{'W': [random() for _ in range(self.n_inputs + 1)]} for _ in range(self.n_hiddens)]
        self.network.append(hidden)
        output = [{'W': [random() for _ in range(self.n_hiddens + 1)]} for _ in range(self.n_outputs)]
        self.network.append(output)

    def activate(self, weight, inputs):
        act = weight[-1]
        for i in range(len(weight) - 1):
            act += weight[i] * inputs[i]
        return act

    def output_function(self, value):
        if self.out_mode == 'Linear':
            return value / 10.0
        elif self.out_mode == 'Logistica':
            val = 0.00
            try:
                val = (1.0 + math.exp(-value))
            except OverflowError:
                val = float('inf')
            return 1.0 / val
        else:
            return math.tanh(value)

    def gradient(self, value):
        if self.out_mode == 'Linear':
            return 1.0 / 10.0
        elif self.out_mode == 'Logistica':
            x = self.output_function(value)
            return x * (1.0 - x)
        else:
            return 1.0 - (self.output_function(value) ** 2)

    def forward_propagate(self, row):
        temp_inputs = row
        for layer in self.network:
            news = list()
            for neuron in layer:
                act = self.activate(neuron['W'], temp_inputs)
                neuron['OUTPUT'] = self.output_function(act)
                news.append(neuron['OUTPUT'])
            temp_inputs = news
        return temp_inputs

    def backward_propagate_error(self, rec):
        for i in reversed(range(len(self.network))):
            layer = self.network[i]
            errors = list()
            if i != len(self.network) - 1:
                for j in range(len(layer)):
                    error = 0.0
                    for neuron in self.network[i + 1]:
                        error += (neuron['W'][j] * neuron['DELTA'])
                    errors.append(error)
            else:
                for j in range(len(layer)):
                    neuron = layer[j]
                    errors.append(rec[j] - neuron['OUTPUT'])
            for j in range(len(layer)):
                neuron = layer[j]
                neuron['DELTA'] = errors[j] * self.gradient(neuron['OUTPUT'])

    def update_weights(self, row):
        for i in range(len(self.network)):
            inputs = row[:-1]
            if i != 0:
                inputs = [neuron['OUTPUT'] for neuron in self.network[i - 1]]
            for neuron in self.network[i]:
                for j in range(len(inputs)):
                    neuron['W'][j] += self.learning_rate * neuron['DELTA'] * inputs[j]
                neuron['W'][-1] += self.learning_rate * neuron['DELTA']

    def train(self, dataf):
        self.initialize(dataf)
        self.init_network()
        error = 100
        epochs = 0
        while (self.error_rate < error) & (self.epoch > epochs):
            error = 0
            for row in self.input:
                outs = self.forward_propagate(row)
                expected = np.zeros(self.n_outputs)
                expected[row[-1]] = 1
                error += sum([(expected[i] - outs[i]) ** 2 for i in range(len(expected))])
                self.backward_propagate_error(expected)
                self.update_weights(row)
            if epochs % 100 == 0:
                print("> epoch={:.4f}, error={:.4f}".format(epochs, error/len(self.input)))
            self.listErrors.append(error)
            epochs += 1

    def prediction(self, row):
        outputs = self.forward_propagate(row)
        return outputs.index(max(outputs))

    def accuracy(self, facts, predicteds, dist):
        cont = 0
        for i in range(len(facts)):
            if facts[i] == predicteds[i]:
                cont += 1
        accuracy = cont / float(len(facts)) * 100.0

        confusion_matrix = [[0 for _ in range(len(dist))] for _ in range(len(dist))]
        for i in range(len(facts)):
            confusion_matrix[int(facts[i])][int(predicteds[i])] += 1
        dataF = pd.DataFrame(data=confusion_matrix, columns=list(self.dic_encode), index=list(self.dic_encode))
        return (accuracy, dataF)

    def test(self, dataf):
        list_encode = list(self.dic_encode)
        self.initialize(dataf)
        predicted = list()
        for row in self.input:
            predicted.append(self.prediction(row))
        fact = [row[-1] for row in self.input]
        return self.accuracy(fact, predicted, list_encode)
    


# mlp = Network(learning_rate=0.02, error_rate=0.0001, output_mode='Linear', n_hiddens=1, epoch=2000)
# mlp.train(pd.read_csv("dataset/base_treinamento.csv"))
# acc, df = mlp.test(pd.read_csv("dataset/base_teste.csv"))

# print("Treinamento Concluido")
# print("\n\nAcurárcia = ",acc,"\nMatriz de Confusão:\n", df)




