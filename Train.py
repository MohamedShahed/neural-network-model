import numpy as np


def sigmoid(s):
    return 1/(1+np.exp(-s))
     
def sigmoidPrime(s):
    return s*(1-s)
    
        
def readFile(path):
    file=open(path, "r")
    line1=file.readline()
    arr1=[int(x) for x in line1.split()]
    numberOfRow=file.readline()
    matrix=[]
    for i in range(int(numberOfRow)):
        line=file.readline()
        arr2=[float(x) for x in line.split()]
        matrix.append(arr2)
    matrix=np.array(matrix, dtype=float)
    X=np.array(matrix[:,0:8], dtype=float)
    Y=matrix[:,8]
    Y=np.matrix(Y)
    Y=Y.T
    return X, Y, arr1


class Neural_Network():
    def __init__(self, M, L, N):
        self.inputSize=M
        self.outputSize=N
        self.hiddenSize=L
        

        
        self.W1=np.random.randn(self.inputSize, self.hiddenSize)
        self.W2=np.random.randn(self.hiddenSize, self.outputSize)
    
    
    def forward(self, X):
        self.z=np.dot(X, self.W1)
        self.z2=sigmoid(self.z)
        self.z3=np.dot(self.z2, self.W2)
        o =sigmoid(self.z3)
        self.output=o
        return o
    
    def backward(self, X, Y, o):
        self.o_error=Y - o
        self.o_delta=np.multiply(self.o_error,sigmoidPrime(o))
        
        self.z2_error=self.o_delta.dot(self.W2.T)
        self.z2_delta=np.multiply(self.z2_error,sigmoidPrime(self.z2))
        
        self.W1+=X.T.dot(self.z2_delta)*0.01
        self.W2+=self.z2.T.dot(self.o_delta)*0.01
    
    def train(self, X, Y):
        o=self.forward(X)
        self.backward(X, Y, o)
        
    def saveWeights(self):
        np.savetxt("W1.txt", self.W1, fmt="%s")
        np.savetxt("W2.txt", self.W2, fmt="%s")

    def MSE(self, X):
        O=self.forward(X)
        mse=np.mean(np.square(O-Y))
        print("the MSE is : " +str(mse))
        



X, Y, arr1=readFile("dataset.txt")

''' scalling '''
X=X/np.amax(X, axis=0)
Y=Y/np.amax(Y, axis=0)

NN=Neural_Network(arr1[0], arr1[1], arr1[2])


for i in range(50000):
    NN.train(X, Y)
    
NN.MSE(X)
NN.saveWeights()