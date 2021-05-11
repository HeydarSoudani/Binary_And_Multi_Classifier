import numpy as np
import matplotlib.pyplot as plt

from functions import sigmoid
from SOM_problem import loss, grad_loss 

class gd():
  def __init__(self, x, y, w_init=np.array([0., 0.]), lr=0, iteration=10, alpha=0.5, beta=0.8, ps_coeff=1):
    self.x = x
    self.y = y
    self.w = np.copy(w_init)

    self.lr = lr
    self.iteration = iteration
    self.alpha = alpha
    self.beta = beta
    self.ps_coeff = ps_coeff

    self.loss_list = list()
  
  def run(self):
    # Append loss in loss_list 
    self.loss_list.append(loss(self.w, self.x, self.y, coefficient=self.ps_coeff))

    for _ in range(self.iteration):
      # Calculate gradient_loss
      g_loss = grad_loss(self.w, self.x, self.y, coefficient=self.ps_coeff)

      # update w with fixed step size
      if self.lr != 0:
        self.w -= self.lr*g_loss
      
      # Find tk with backtracking line search
      else:
        t = 1
        while True:
          grad_loss_norm = np.linalg.norm(g_loss, ord=2)
          if loss(self.w-t*g_loss, self.x, self.y, coefficient=self.ps_coeff) <= loss(self.w, self.x, self.y, coefficient=self.ps_coeff) - self.alpha*t*grad_loss_norm:
            break
          t *= self.beta
        
        # update w
        self.w -= t*g_loss

      # Append loss in loss_list 
      self.loss_list.append(loss(self.w, self.x, self.y, coefficient=self.ps_coeff))
      

  def predict(self, x_test):
    y_pred = sigmoid(self.ps_coeff*np.matmul(x_test, self.w.T))
    y_pred[np.where(y_pred >= 1/2)] = 1
    y_pred[np.where(y_pred < 1/2)] = 0
    return y_pred


  def accuracy(self, y_act, y_pred):
    return (len(np.where(y_act == y_pred)[0])/y_act.shape[0])*100
  
  def loss_changes_diagram(self):
    plt.plot([i for i in range(self.iteration+1)], self.loss_list, label='ps coeff: '+str(self.ps_coeff))

