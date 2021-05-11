import numpy as np
import matplotlib.pyplot as plt

from VRM_problem import loss, grad_loss
from functions import linearPredict, softmax, multinomialLogReg

class svrg():
  def __init__(self, train_data, test_data, w, b, m=1000, lr=0.1, s =10, option='1', gamma=1e-4):
    self.train_data = train_data
    self.test_data = test_data
    self.w = np.copy(w)
    self.b = b

    self.m = m # iteration number for Inner loop
    self.s = s # iteration number for Outer loop
    self.lr = lr # learning rate for update weights
    self.gamma = gamma # Coeff of loss function regularization term
    self.option = option
    

    self.n_classes = self.w.shape[0]
    self.n_samples = self.train_data.shape[0]

    self.loss_list = list()
    self.accuracy_list = list()
    self.variance_list = list()
  
  def run(self):
    # Initial a version of estimated w
    w_tilda = self.w
    b_tilda = self.b

    x_train = self.train_data[:, 0:-1]
    y_train = self.train_data[:, -1]

    # Outer iteration
    for i in range(self.s):
      
      if self.option == '2':
        w_list = np.array([])
        b_list = np.array([])
      
      # Initialize Variance variables
      var_w = np.zeros(self.w.shape)
      sum_w = np.zeros(self.w.shape)
      sumsq_w = np.zeros(self.w.shape)
      var_b = np.zeros(self.b.shape)
      sum_b = np.zeros(self.b.shape)
      sumsq_b = np.zeros(self.b.shape)
      
      # Maintain the average gradient
      probabilities, _ = multinomialLogReg(x_train, w_tilda, b_tilda)
      grad_loss_list = [grad_loss(x_train[i:i+1], y_train[i:i+1], probabilities[i:i+1], w_tilda, b_tilda, gamma=self.gamma) for i in range(self.n_samples)]
      g_loss_w_tilda_list = np.array([i[0] for i in grad_loss_list])
      g_loss_b_tilda_list = np.array([i[1] for i in grad_loss_list])

      mu_tilda_w = np.mean(g_loss_w_tilda_list, axis=0) 
      mu_tilda_b = np.mean(g_loss_b_tilda_list, axis=0) 

      # Inner iteration
      for t in range(self.m):
        i_t = np.random.randint(self.n_samples)
        x_data = x_train[i_t:i_t+1]
        y_data = y_train[i_t:i_t+1]
        
        probabilities, _ = multinomialLogReg(x_data, self.w, self.b)
        g_loss_wi, g_loss_bi = grad_loss(x_data, y_data, probabilities, self.w, self.b, gamma=self.gamma)

        # Update w &
        self.w -= self.lr * (g_loss_wi - g_loss_w_tilda_list[i_t] + mu_tilda_w)
        self.b -= self.lr * (g_loss_bi - g_loss_b_tilda_list[i_t] + mu_tilda_b)

        if self.option == '2':
          w_list = np.append(w_list, self.w)
          b_list = np.append(b_list, self.b)
        
        # Update variance variables
        sum_w += self.w
        sumsq_w += np.multiply(self.w, self.w)
        sum_b += self.b
        sumsq_b += np.multiply(self.b, self.b)
      
      var_w = (sumsq_w - ((np.multiply(sum_w, sum_w)/self.m))) / (self.m - 1)
      var_b = (sumsq_b - ((np.multiply(sum_b, sum_b)/self.m))) / (self.m - 1)
      
      # Update w_tilda
      if self.option == '2':
        t = np.ceil(np.random.uniform(0, self.m)) # Select t randomly
        w_tilda = w_list[t]
        b_tilda = b_list[t]
      else:
        w_tilda = self.w
        b_tilda = self.b


      ### Print result after some steps
      # Calculate loss on training data
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
      print("Iteration: {}, Loss: {:.2f}, Accuracy: {:.2f}, Variance: {:.4f}.".format(i, loss_value, acc, variance_value))

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
    return ax.plot([i for i in range(self.s)], self.loss_list, label=label)

  def plot_eval_data_accuracy(self, ax, label):
    return ax.plot([i for i in range(self.s)], self.accuracy_list, label=label)
  
  def plot_steps_variance(self, ax, label):
    return ax.plot([i for i in range(self.s)], self.variance_list, label=label)
    
