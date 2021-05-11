import numpy as np

def sigmoid(x):
  return 1/(1+np.exp(-x))

def linearPredict(features, weights, biases):
  n_samples = features.shape[0]
  n_classes = weights.shape[0]

  # creating empty(garbage value) array for each feature set
  logitScores = np.array([np.empty([n_classes]) for i in range(n_samples)])
  for i in range(n_samples):
    # calculates logit score for each feature set then flattens the logit vector 
    logitScores[i] = (weights.dot(features[i].reshape(-1,1)) + biases).reshape(-1)
  
  return logitScores


def softmax(z):
  n_samples = z.shape[0]
  n_classes = z.shape[1]

  # Creating empty (garbage value) array for each feature set
  probabilities = np.array([np.empty([n_classes]) for i in range(n_samples)])
  for i in range(n_samples):
    exp = np.exp(z[i]) # exponentiates each element of the logit array
    sumOfArr = np.sum(exp) # adds up all the values in the exponentiated array
    probabilities[i] = exp/sumOfArr # logit scores to probability values
  
  return probabilities


"""Performs logistic regression on a given feature set."""
def multinomialLogReg(features, weights, biases):
  logitScores = linearPredict(features, weights, biases)
  probabilities = softmax(logitScores)
  predictions = np.array([np.argmax(i) for i in probabilities])

  return probabilities, predictions