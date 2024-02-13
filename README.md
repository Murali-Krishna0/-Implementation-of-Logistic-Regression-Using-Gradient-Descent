# Implementation-of-Logistic-Regression-Using-Gradient-Descent

## AIM:
To write a program to implement the the Logistic Regression Using Gradient Descent.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import the required libraries.
2. Load the dataset.
3. Define X and Y array.
4. Define a function for costFunction,cost and gradient.
5. Define a function to plot the decision boundary. 6.Define a function to predict the 
   Regression value.
   
## Program:
Program to implement the the Logistic Regression Using Gradient Descent.
Developed by:Murali Krishna S 
RegisterNumber:212223230129  
```
import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize

data=np.loadtxt("ex2data1.txt",delimiter=',')
X=data[:,[0,1]]
y=data[:,2]

X[:5]

y[:5]

plt.figure()
plt.scatter(X[y==1][:,0],X[y==1][:,1],label="Admitted")
plt.scatter(X[y==0][:,0],X[y==0][:,1],label="Not Admitted")
plt.xlabel("Exam 1 score")
plt.ylabel("Exam 2 score")
plt.legend()
plt.show()

def sigmoid(z):
    return 1/(1+np.exp(-z))

plt.plot()
X_plot=np.linspace(-10,10,100)
plt.plot(X_plot,sigmoid(X_plot))
plt.show()

def costFunction (theta,X,y):
    h=sigmoid(np.dot(X,theta))
    J=-(np.dot(y,np.log(h))+np.dot(1-y,np.log(1-h)))/X.shape[0]
    grad=np.dot(X.T,h-y)/X.shape[0]
    return J,grad

X_train=np.hstack((np.ones((X.shape[0],1)),X))
theta=np.array([0,0,0])
J,grad=costFunction(theta,X_train,y)
print(J)
print(grad)

X_train=np.hstack((np.ones((X.shape[0],1)),X))
theta=np.array([-24,0.2,0.2])
J,grad=costFunction(theta,X_train,y)
print(J)
print(grad)

def cost (theta,X,y):
    h=sigmoid(np.dot(X,theta))
    J=-(np.dot(y,np.log(h))+np.dot(1-y,np.log(1-h)))/X.shape[0]
    return J

def gradient (theta,X,y):
    h=sigmoid(np.dot(X,theta))
    grad=np.dot(X.T,h-y)/X.shape[0]
    return grad

X_train=np.hstack((np.ones((X.shape[0],1)),X))
theta=np.array([0,0,0])
res=optimize.minimize(fun=cost,x0=theta,args=(X_train,y),method='Newton-CG',jac=gradient)
print(res.fun)
print(res.x)

def plotDecisionBoundary(theta,X,y):
    x_min,x_max=X[:,0].min()-1,X[:,0].max()+1
    y_min,y_max=X[:,1].min()-1,X[:,1].max()+1
    xx,yy=np.meshgrid(np.arange(x_min,x_max,0.1),np.arange(y_min,y_max,0.1))
    X_plot=np.c_[xx.ravel(),yy.ravel()]
    X_plot=np.hstack((np.ones((X_plot.shape[0],1)),X_plot))
    y_plot=np.dot(X_plot,theta).reshape(xx.shape)
    
    plt.figure()
    plt.scatter(X[y==1][:,0],X[y==1][:,1],label="Admitted")
    plt.scatter(X[y==0][:,0],X[y==0][:,1],label="Not Admitted")
    plt.contour(xx,yy,y_plot,levels=[0])
    plt.xlabel("Exam 1 score")
    plt.ylabel("Exam 2 score")
    plt.legend()
    plt.show()


plotDecisionBoundary(res.x,X,y)

prob=sigmoid(np.dot(np.array([1,45,85]),res.x))
print(prob)

def predict(theta,X):
    X_train =np.hstack((np.ones((X.shape[0],1)),X))
    prob=sigmoid(np.dot(X_train,theta))
    return (prob>=0.5).astype(int)
np.mean(predict(res.x,X)==y)
```

## Output:
![image](https://github.com/Murali-Krishna0/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/149054535/a967f6f8-2fd8-4aa7-8bec-deb981ac9855)
![image](https://github.com/Murali-Krishna0/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/149054535/89759088-5f4a-4ad4-97f8-6e31750bc998)
![image](https://github.com/Murali-Krishna0/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/149054535/1c5d0560-5907-49fc-874e-b623f5e050b0)
![image](https://github.com/Murali-Krishna0/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/149054535/f73023fa-5e14-4efe-bf76-6a46979e656a)
![image](https://github.com/Murali-Krishna0/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/149054535/83c89776-fd92-4d03-84e0-95fd66b99097)
![image](https://github.com/Murali-Krishna0/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/149054535/2355592a-2821-48c9-b382-58b67cfbac79)
![image](https://github.com/Murali-Krishna0/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/149054535/65be8c9e-9a22-4816-a17c-572e98d41054)
![image](https://github.com/Murali-Krishna0/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/149054535/46060c1a-c427-4613-adad-76ecbd5bbc8c)




## Result:
Thus the program to implement the the Logistic Regression Using Gradient Descent is written and verified using python programming.

