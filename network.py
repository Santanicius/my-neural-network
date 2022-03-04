import pandas as pd
import math
import numpy as np
from random import seed
from random import randrange
from random import random
from sklearn.model_selection import train_test_split

class Network:

    def __init__(self, learning_rate=0.5, epoch=100, n_hiddens=1, error_rate=0.5, output_mode='Hyperbolic'):
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
        dataf = dataf.apply(pd.to_numeric, errors='coerce')
        
                
        # Fanzendo o shuffle
        dataf = dataf.sample(frac=1)
        
        # Tranformando o dataframe em lista
        self.input = dataf.values.tolist()

        self.n_inputs = len(self.input[0]) - 1
        
        # Pegando a lista de classe ordenada
        classlist = sorted(list(dataf['class'].unique()))
        
        # Criando o dicionario de classificação
        self.n_outputs = len(classlist)
        self.dic_encode = dict()
        
        for i, value in enumerate(classlist):
            self.dic_encode[value] = i
            
        for row in self.input:
            row[self.n_inputs] = self.dic_encode[row[self.n_inputs]]
            
        
        if self.n_hiddens == 0 or self.n_hiddens == 1:
            self.n_hiddens = math.floor((self.n_inputs + self.n_outputs) / 2)
            
        # self.shuffle()
        self.normalization()

    def shuffle(self):
        # Fazendo uma copia da lista de entradas
        temp = list(self.input)
        self.input = list()
        while len(temp) > 0:
            index = randrange(len(temp))
            self.input.append(temp.pop(index))

    def normalization(self):
        state = [[min(col), max(col)] for col in zip(*self.input)]
        for row in self.input:
            for i in range(len(row) - 1):
                if (state[i][1] - state[i][0]) != 0:
                    row[i] = (row[i] - state[i][0]) / (state[i][1] - state[i][0])
                
                
    # Arquitetura da Rede
    def init_network(self):
        self.network = list()
        # +1 por conta Com o Bias
        hidden = [{'W': [random() for _ in range(self.n_inputs+1)]} for _ in range(self.n_hiddens)]
        self.network.append(hidden)
        
        hidden2 = [{'W': [random() for _ in range(self.n_hiddens)]} for _ in range(self.n_hiddens)]
        self.network.append(hidden2)
        
        output = [{'W': [random() for _ in range(self.n_hiddens)]} for _ in range(self.n_outputs)]
        self.network.append(output)

    def activate(self, weight, inputs):
        act = np.dot(weight, inputs)
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
            return (1.0 - (math.tanh(value) ** 2))

    def forward_propagate(self, row):
        temp_inputs = list(row)
        
        # Fazemos a classe virar o bias
        temp_inputs[-1] = 1
        for layer in self.network:
            news = list()
            for neuron in layer:
                # Poderia realizar a multiplicação de matriz no lugar
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
            error_per_batch = 0
            i = 1
            for row in self.input:
                outs = self.forward_propagate(row)
                expected = np.zeros(self.n_outputs)
                expected[row[-1]] = 1
                if i % 32 == 0:
                    error += error_per_batch/32
                    error_per_batch = 0
                    self.backward_propagate_error(expected)
                    self.update_weights(row)
                else:
                    error_per_batch += sum([(expected[i] - outs[i]) ** 2 for i in range(len(expected))])
                i = i + 1
            print("> epoch={:.4f}, error={:.4f}".format(epochs, float(error/len(self.input))))
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

    def show_att(self):
      print("{ ",
                self.learning_rate, ",\n",
                self.n_inputs, ",\n",
                self.n_outputs,",\n",
                self.listErrors,",\n",
                self.network,",\n",
                self.input,",\n",
                self.dic_encode,",\n",
                self.error_rate,",\n",
                self.out_mode,
            "\n}"
            )

data = pd.read_csv("dataset/mnist_784.csv")
data20, _ = train_test_split(data, train_size=0.2)
train, test = train_test_split(data20, train_size=0.8)

mlp = Network(learning_rate=0.002, error_rate=0.01, output_mode='tanh', n_hiddens=20, epoch=100)
mlp.show_att()
print("Iniciando o treinamento :3")

mlp.train(train)
acc, mat_conf = mlp.test(test)


print("Treinamento Concluido")
print(f"\n\nAcurácia = {acc} \nMatriz de Confusão:\n{mat_conf}")