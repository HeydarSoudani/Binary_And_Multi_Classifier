import numpy as np
from functions import sigmoid


def loss(w, x, y, coefficient=1):
  m = y.shape[0]
  y_hat = sigmoid(coefficient*np.matmul(x, w.T))
  return -(1/m)*(np.dot(y, np.log(y_hat)) + np.dot(1-y, np.log(1-y_hat)))

def grad_loss(w, x, y, coefficient=1):
  m = y.shape[0]
  y_hat = sigmoid(coefficient*np.matmul(x, w.T))
  return -(1/m)*coefficient*np.matmul(x.T, (y-y_hat))

def hessian_loss(w, x, y, coefficient=1):
  m = y.shape[0]
  y_hat = sigmoid(coefficient*np.matmul(x, w.T))
  return (1/m)*np.power(coefficient, 2)*(y_hat * (1 - y_hat)) * x.T @ x
  # return (1/m)*np.power(coefficient, 2)*np.matmul(x.T,x)*np.matmul(y_hat.T, (1-y_hat))

def landa_2_loss(w, x, y, coefficient=1):
  return np.dot(np.dot(grad_loss(w,x, y, coefficient).T, np.linalg.inv(hessian_loss(w, x, y, coefficient))), grad_loss(w,x, y, coefficient))

def point_likelihood(w, xi, yi, coefficient=1):
  y_hat = sigmoid(coefficient*np.dot(xi, w))
  return (y_hat**yi)*((1-y_hat)**(1-yi))
