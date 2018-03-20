import layer as L
import numpy as np

class Model:

    def __init__(self, in_size, l_rate, batch_size, n_epoch):
        self.__l_rate = l_rate
        self.__batch_size = batch_size
        self.__n_epoch = n_epoch
        self.__layers = []
        self.__in_size = in_size


    # def prevSize(self):
    #     x = self.__in_size
    #     if len(self.__layers) > 0:
    #         x = self.__layers[-1].getOutputSize()
    #     return x

    def addLayer(self, size):
        x = self.__in_size
        if len(self.__layers) > 0:
            x = self.__layers[-1].getOutputSize()
        print(x,size)
        self.__layers.append(L.Layer(len(self.__layers)+1, x, size))

    # def addLayer(self, size, mean,sd):
    #     self.__layers.append(L.Layer(len(self.__layers)+1, prevSize(), size, mean, sd))


    def train(self, input, target):
        lis = self.__layers
        l = self.__l_rate
        b = self.__batch_size
        n = len(input)

        for i in range(self.__n_epoch):
            ind = 0
            error = 0
            for ind in range(0,n,b):
                inp = input[ind:ind+b]
                tar = target[ind:ind+b]

                for layer in lis:
                    inp = layer.multiply(inp)

                lis[-1].calcDelta(None, tar)
                for j in range(len(lis)-1,0,-1):
                    lis[j-1].calcDelta(lis[j],tar)

                for j in range(len(lis)-1,-1,-1):
                    lis[j].update(l)

                tar = lis[-1].getOutput() - tar
                error += np.sum(0.5*tar*tar)/n

            print("epoch_count:", i+1, "  avg_error:",error)


    def test(self, input):
        for layer in self.__layers:
            input = layer.multiply(input)
        out = self.__layers[-1].getOutput()
        row = out[0]
        return max([(row[i],i) for i in range(0,len(row))])[1]

