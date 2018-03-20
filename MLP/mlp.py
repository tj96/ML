import load_data as ld
import test
import sys
import model

def getParameters():
    l=0.1
    b=10
    e=100
    # print("Enter batch size (1 for online learning, 0 for full-batch):")
    # b= int(input())
    # if b == 0:
    #     b = 99999999

    # print("Enter learning rate:")
    # l = float(input())

    # print("Enter Max epochs:")
    # e = int(input())

    return l,b,e

if __name__ == '__main__':
    l,b,e = getParameters()
    train_x, train_y, test_x, test_y = ld.load_data(sys.argv[1])
    c = max(max(train_y),max(test_y)) + 1
    lis = []
    for i in range(len(train_y)):
        temp = [0.0]*c
        temp[train_y[i]] = 1.0
        lis.append(temp)
    train_y = lis

    M = model.Model(len(train_x[0]),l,b,e)
    M.addLayer(4)
    M.addLayer(c)

    M.train(train_x,train_y)
    output = []
    for x,y in zip(test_x,test_y):
        output.append((y, M.test(x)))

    # print(output)
    print(test.accuracy(output))
