import numpy as np

def sigmoid(x):
  return 1/(1+np.exp(-x))

class Layer:

    def __init__(self, index, in_size, out_size, mean=0, sd=-1):
        if sd < 0:
            sd = np.sqrt(2/(in_size+out_size))
        self.id = index
        self.__in_size = in_size
        self.__out_size = out_size
        self.__matrix = np.random.normal(mean,sd,(in_size, out_size))
        self.__bias = np.random.normal(mean,sd,(1, out_size))


    def getOutputSize(self):
        return self.__out_size


    def multiply(self, input):   
        self.__input = input
        self.__output = sigmoid(np.matmul(input, self.__matrix) + self.__bias)
        return self.__output


    def getOutput(self):
        return self.__output


    def calcDelta(self, nextLayer, target):
        y = self.__output

        if nextLayer == None:
            d = target
            self.__delta = y*(1-y)*(d-y)

        else:
            w = nextLayer.__matrix
            temp = np.matmul(nextLayer.__delta, w.T)
            self.__delta = y*(1-y)*temp


    def update(self, l_rate):
        self.__matrix = self.__matrix + l_rate*(np.matmul(self.__input.T, self.__delta))
        self.__bias = self.__bias + l_rate*(np.matmul(np.ones((1, len(self.__input))), self.__delta))
        