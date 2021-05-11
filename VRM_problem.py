import numpy as np
from functions import softmax


def loss(probabilities, target, w, b, gamma=1e-4):  
  n_samples = probabilities.shape[0]
  CELoss = 0
  for sample, i in zip(probabilities, target):
      CELoss += -np.log(sample[int(i)])
  CELoss /= n_samples
  CELoss += gamma*(np.linalg.norm(w)**2 + np.linalg.norm(b)**2)
  return CELoss


def grad_loss(features, target, probabilities, weight, biases, gamma=0.01):
  target = target.astype(int)
  probabilities[np.arange(features.shape[0]),target] -= 1 # Substract 1 from the scores of the correct outcome
  
  grad_weight = probabilities.T.dot(features) + gamma*weight # gradient of loss w.r.t. weights
  grad_biases = np.sum(probabilities, axis = 0).reshape(-1,1) + gamma*biases # gradient of loss w.r.t. biases

  return grad_weight, grad_biases

