import numpy as np 

f1 = open(r"W1.txt","r")
f2 = open(r"W2.txt","r")

W1=f1.read()
W2=f2.read()

W1=np.matrix(W1)
W1=np.array(W1, dtype=float)
W1=np.array(W1.flatten().reshape((8,10)))

W2=np.matrix(W2)
W2=np.array(W2, dtype=float)
W2=np.array(W2.flatten().reshape((10,1)))



def sigmoid(s):
    return 1/(1+np.exp(-s))
     
def sigmoidPrime(s):
    return s*(1-s)
    

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




f = open(r"train.txt","r")
line1=f.readline()
arr1=[int(x) for x in line1.split()]

numberOfRow=f.readline()
matrix=[]
for i in range(int(numberOfRow)):
    line=f.readline()
    arr2=[float(x) for x in line.split()]
    matrix.append(arr2)
matrix=np.array(matrix, dtype=float)


X=np.array(matrix[:,0:8], dtype=float)
Y=matrix[:,8]

Y=np.matrix(Y)
Y=Y.T


''' scalling '''
X=X/np.amax(X, axis=0)
Y=Y/np.amax(Y, axis=0)


NN=Neural_Network()
NN.MSE(X)
