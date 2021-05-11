import numpy as np
import matplotlib.pyplot as plt

from functions import sigmoid
from SOM_problem import loss, grad_loss, hessian_loss, landa_2_loss


class newton():
  def __init__(self, x, y, w_init=np.array([0., 0.]), iteration=10, tolerance=10e-4, alpha=0.5, beta=0.8, ps_coeff=1):
    self.x = x
    self.y = y
    self.w = np.copy(w_init)

    self.iteration = iteration
    self.tolerance = tolerance
    self.alpha = alpha
    self.beta = beta
    self.ps_coeff = ps_coeff

    self.loss_list = list()
  
  def run(self):
    # Append loss in loss_list 
    self.loss_list.append(loss(self.w, self.x, self.y, coefficient=self.ps_coeff))

    for k in range(self.iteration):
      # Calculate gradient_loss & hessian_loss
      h_loss = hessian_loss(self.w, self.x, self.y, coefficient=self.ps_coeff)
      g_loss = grad_loss(self.w, self.x, self.y, coefficient=self.ps_coeff)

      # Compute the Newton step and decrement
      landa_2 = landa_2_loss(self.w, self.x, self.y, coefficient=self.ps_coeff)
      delta_w = (-1) * np.matmul(np.linalg.inv(h_loss), g_loss)

      # Line search: Choosing step size t by backtracking line search
      t = 1
      v = (-1) * np.dot(np.linalg.inv(h_loss), g_loss)
      while loss(self.w + t*v ,self.x, self.y, coefficient=self.ps_coeff) > loss(self.w, self.x, self.y, coefficient=self.ps_coeff) + self.alpha*t*np.dot(g_loss.T, v):
        t *= self.beta
      
      # Update x
      self.w += t*delta_w

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
    plt.ylabel('Loss')
    plt.xlabel('Iteration')
    plt.plot([i for i in range(self.iteration+1)], self.loss_list, label='ps coeff: '+str(self.ps_coeff))