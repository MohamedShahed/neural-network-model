import numpy as np 

def readWights(path1, path2):
    file1=open(path1, "r")
    file2=open(path2, "r")
    W1=file1.read()
    W2=file2.read()
    
    W1=np.matrix(W1)
    W1=np.array(W1, dtype=float)
    W1=np.array(W1.flatten().reshape((8,10)))
    
    W2=np.matrix(W2)
    W2=np.array(W2, dtype=float)
    W2=np.array(W2.flatten().reshape((10,1)))
    
    return W1, W2

    
#f1 = open(r"W1.txt","r")
#f2 = open(r"W2.txt","r")
#
#W1=f1.read()
#W2=f2.read()
#
#W1=np.matrix(W1)
#W1=np.array(W1, dtype=float)
#W1=np.array(W1.flatten().reshape((8,10)))
#
#W2=np.matrix(W2)
#W2=np.array(W2, dtype=float)
#W2=np.array(W2.flatten().reshape((10,1)))

W1, W2=readWights("W1.txt", "W2.txt")

def sigmoid(s):
    return 1/(1+np.exp(-s))
     
def sigmoidPrime(s):
    return s*(1-s)
   
        
def readFile(path):
    file=open(path, "r")
    file.readline()
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
    return X, Y


class Neural_Network():
    def forward(self, X):
        self.z=np.dot(X, W1)
        self.z2=sigmoid(self.z)
        self.z3=np.dot(self.z2, W2)
        o =sigmoid(self.z3)
        return o
    

    def MSE(self, X):
        O=self.forward(X)
        mse=np.mean(np.square(O-Y))
        print("the MSE is : " +str(mse))



X, Y=readFile("dataset.txt")


''' scalling '''
X=X/np.amax(X, axis=0)
Y=Y/np.amax(Y, axis=0)


NN=Neural_Network()
NN.MSE(X)
