import math
import numpy as np
import pandas as pd
from random import random, randrange
from random import seed

class Network:
    MODE_LINEAR = 0
    MODE_LOGISTIC = 1
    MODE_HYPERBOLIC = 2

    def __init__(self, n_inputs = 1, n_outputs = 1, n_hiddens = 0, mode=MODE_LINEAR, learning_rate=0.1, error=0.3,epochs=100):
        seed(1)
        self.error = error
        self.epochs = epochs
        self.n_inputs = n_inputs
        self.n_outputs = n_outputs
        self.n_hiddens = n_hiddens
        self.learning_rate = learning_rate
        self.input = list()
        self.errorArray = list()
        self.dic_encode = dict()
        self.mode = mode
        self.listErrors = list()

    def propag(self, net):
        if self.mode == self.MODE_LINEAR:
            return net/10.0
        if self.mode == self.MODE_LOGISTIC:
            return 1.0 / (1.0 + np.exp(-net))
        if self.mode == self.MODE_HYPERBOLIC:
            return (1.0 - np.exp(-2*net)) / (1.0 + np.exp(-2*net))
    
    def derivative_function(self, v):  
        if self.mode == self.MODE_LINEAR:
            return 0.1
        if self.mode == self.MODE_LOGISTIC:
            return self.propag(v) * (1.0-self.propag(v))
        if self.mode == self.MODE_HYPERBOLIC:
            return 1.0 - self.propag(v)**2

    def shuffle(self):
        input_list = list(self.input)
        self.input = list()
        while len(input_list) > 0:
            index = randrange(len(input_list))
            self.input.append(input_list.pop(index))

    def initialize(self, data):
        # print("Df init => ", data.drop(data.columns[-1], axis=1))
        # print("columns => ", data.columns)
        dframe = data.drop(columns="classe", axis=1)
        for column in dframe:
            data[str(column)] = data[str(column)].astype(float)
        self.input = data.values.tolist()
        self.n_inputs = len(self.input[0]) - 1
    
        lastElem = len(self.input[0])-1
        #class column
        class_list = [row[lastElem] for row in self.input]
        self.n_outputs = len(set(class_list))
        
        # Definindo o numero de camadas se nn passado por parâmetro do construtor
        if self.n_hiddens == 0:
            self.n_hiddens = math.floor(math.sqrt((float(self.n_inputs * self.n_outputs))))
        
        self.hiddenLayer = [{'W': [random() for _ in range(self.n_inputs + 1)]} for _ in range(self.n_hiddens)]
        self.outputLayer = [{'W': [random() for _ in range(self.n_hiddens + 1)]} for _ in range(self.n_outputs)]
        
        # Encoding
        self.dic_encode = dict()
        for i, value in enumerate(set(class_list)):
            self.dic_encode[value] = i
        print("Enconde => ", self.dic_encode)
        
        for row in self.input:
            row[lastElem] = self.dic_encode[row[lastElem]]
    
        self.shuffle()
        self.normalize()
    
    def forward_propagate(self, row_data):
        network = [self.hiddenLayer, self.outputLayer]
        inputs = row_data
        for layer in network:
            next_inputs = list()
            for neuron in layer:
                # Activation
                weights = neuron['W']
                net = weights[-1] # initialize with bias
                for i in range(len(weights) - 1):
                    net += weights[i] * inputs[i]
                neuron['O'] = self.propag(net)
                next_inputs.append(neuron['O'])
            inputs = next_inputs
        return inputs
    
    # Backpropagate error and store in neurons
    def backward_propagate_error(self, expected):
        #delta calculate for output  layer
        for index in range(len(self.outputLayer)):
            neuron = self.outputLayer[index]
            error = expected[index] - neuron['O']
            neuron['delta'] = error * self.derivative_function(neuron['O'])

        #delta calculate for hidden layer
        for index in range(len(self.hiddenLayer)):
            error = 0.0
            for neuron in self.outputLayer:
                error += (neuron['W'][index] * neuron['delta'])
            neuron = self.hiddenLayer[index]
            neuron['delta'] = error * self.derivative_function(neuron['O'])

    def update_weights(self, data_row):
        inputs = data_row[:-1]
        # Camada oculta
        for neuron in self.hiddenLayer:
            for j in range(len(inputs)):
                neuron['W'][j] += self.learning_rate * neuron['delta'] * inputs[j]
            neuron['W'][-1] += self.learning_rate * neuron['delta']
        
        inputs = [neuron['O'] for neuron in self.hiddenLayer]
        for neuron in self.outputLayer:
            for j in range(len(inputs)):
                neuron['W'][j] += self.learning_rate * neuron['delta'] * inputs[j]
            neuron['W'][-1] += self.learning_rate * neuron['delta']

    def train(self, data):
        self.initialize(data)
        sum_error = 100
        epoch = 0
        while (epoch < self.epochs) and (self.error < sum_error):
            sum_error = 0
            for row in self.input:
                outputs = self.forward_propagate(row)
                expected = np.zeros(self.n_outputs)
                expected[row[len(row)-1]] = 1
                sum_error += sum([(expected[i]-outputs[i])**2 for i in range(len(expected))])
                self.backward_propagate_error(expected)
                self.update_weights(row)
            self.listErrors.append(sum_error)
            if epoch % 100 == 0:
                print("> epoch={:.4f}, error={:.4f}".format(epoch, sum_error/len(self.input)))
            epoch += 1
        return True
    
    def predict(self, row):
        outputs = self.forward_propagate(row)
        return outputs.index(max(outputs))
    
    def test(self, data):
        # Lista do encode 
        dist = list(self.dic_encode)
        self.input = None
        self.initialize(data)
        
        pred = list()
        for row in self.input:
            pred.append(self.predict(row))
        fact = [row[-1] for row in self.input]
        return self.accuracy(fact, pred, dist)
    
    def accuracy(self, facts, predicteds, dist):
        res = 0
        for i in range(len(facts)):
            if facts[i] == predicteds[i]:
                res += 1
        acc = res / float(len(facts)) * 100.0

        matrix = [[0 for _ in range(len(dist))] for _ in range(len(dist))]
        for i in range(len(facts)):
            matrix[int(facts[i])][int(predicteds[i])] += 1
        df = pd.DataFrame(data=matrix, columns=list(self.dic_encode), index=list(self.dic_encode))
        return (acc, df)
    
    def normalize(self):

        states = [[min(col), max(col)] for col in zip(*self.input)]
        for row in self.input:
            for index in range(len(row) - 1):
                row[index] = (row[index] - states[index][0]) / (states[index][1] - states[index][0])


dataframe_train = pd.read_csv("dataset/base_treinamento.csv")
dataframe_test = pd.read_csv("dataset/base_teste.csv")
input_size = len(dataframe_train.columns) - 1

columns_train = dataframe_train.columns[:input_size]
columns_test = dataframe_test.columns[:input_size]

print("Dataframe Treinamento =>\n", dataframe_train)
print("\n\nDataframe Teste =>\n", dataframe_test)

mlp = Network(epochs=2000, error=0.0001, mode=0, n_hiddens=0, learning_rate=0.02)
res = mlp.train(data=dataframe_train)
print("Treinamento Concluido: ", res)

errors = mlp.listErrors
print("\n\nLista de Erros: \n", errors)
acc, confusion_mat = mlp.test(dataframe_test)
print("\n\nAcurárcia = ",acc,"\nMatriz de Confusão:\n", np.matrix(confusion_mat))




