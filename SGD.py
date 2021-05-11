import numpy as np
import matplotlib.pyplot as plt

from VRM_problem import loss, grad_loss
from functions import linearPredict, softmax, multinomialLogReg

class sgd():
  def __init__(self, train_data, test_data, w, b, batch_size=1, lr=0.1, t=1000, gamma=1e-4):
    self.train_data = train_data
    self.test_data = test_data
    self.w = np.copy(w)
    self.b = b

    self.batch_size = batch_size
    self.lr = lr
    self.t = t
    self.gamma = gamma # Coeff of loss function regularization term

    self.M = self.train_data.shape[0]
    self.class_num = w.shape[1]

    self.loss_list = list()
    self.accuracy_list = list()
    self.variance_list = list()
  
  def run(self):
    
    for i in range(self.t+1):

      np.random.shuffle(self.train_data)
      train_data = self.train_data

      # Initialize Variance variables
      var_w = np.zeros(self.w.shape)
      sum_w = np.zeros(self.w.shape)
      sumsq_w = np.zeros(self.w.shape)
      var_b = np.zeros(self.b.shape)
      sum_b = np.zeros(self.b.shape)
      sumsq_b = np.zeros(self.b.shape)

      n = int(self.M/self.batch_size)
      for j in range(n):
        batch_data = train_data[j:j+self.batch_size]
        x_batch = batch_data[:, 0:-1]
        y_batch = batch_data[:, -1]

        probabilities, _ = multinomialLogReg(x_batch, self.w, self.b)
        g_loss_wi, g_loss_bi = grad_loss(x_batch, y_batch, probabilities, self.w, self.b, gamma=self.gamma)

        # Update w, b
        self.w -= self.lr*g_loss_wi
        self.b -= self.lr*g_loss_bi

        # Updata w & b variance variables
        sum_w += self.w
        sumsq_w += np.multiply(self.w, self.w)
        sum_b += self.b
        sumsq_b += np.multiply(self.b, self.b)
      
      var_w = (sumsq_w - ((np.multiply(sum_w, sum_w)/n))) / (n-1)
      var_b = (sumsq_b - ((np.multiply(sum_b, sum_b)/n))) / (n-1)

      ### Print result after some steps
      # Calculate loss on training data
      x_train = train_data[:, 0:-1]
      y_train = train_data[:, -1]
      probabilities, _ = multinomialLogReg(x_train, self.w, self.b)
      loss_value = loss(probabilities, y_train, self.w, self.b, gamma=self.gamma)
      self.loss_list.append(loss_value)

      # Calculate accuracy on evaluation data
      x_test = self.test_data[:, 0:-1]
      y_test = self.test_data[:, -1]
      _, predictions = self.predict(x_test)
      acc = self.accuracy(y_test, predictions)
      self.accuracy_list.append(acc)

      # Calculate steps's variance
      variance_value = np.sum(var_w) +  np.sum(var_b)
      self.variance_list.append(variance_value)

      # Print result
      print("Iteration: {}, Loss: {:.2f}, Accuracy: {:.2f}, Variance: {:.4f}".format(i, loss_value, acc, variance_value))
        
        
  def predict(self, x_test):
    probabilities, predictions = multinomialLogReg(x_test, self.w, self.b)
    return probabilities, predictions

  def accuracy(self, y_act, y_pred):
    correctPred = 0
    for i in range(len(y_pred)):
      if y_pred[i] == y_act[i]:
        correctPred += 1
    accuracy = correctPred/len(y_pred)*100
    return accuracy

  def plot_loss(self, ax, label):
    iter_num = self.t + 1
    return ax.plot([i for i in range(iter_num)], self.loss_list, label=label)

  def plot_eval_data_accuracy(self, ax, label):
    iter_num = self.t + 1
    return ax.plot([i for i in range(iter_num)], self.accuracy_list, label=label)
  
  def plot_steps_variance(self, ax, label):
    iter_num = self.t + 1
    return ax.plot([i for i in range(iter_num)], self.variance_list, label=label)

