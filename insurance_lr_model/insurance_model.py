#!/usr/bin/env python

import pandas as pd
import numpy as np
from data_loader import load_simulated_data, load_insurance_data


def mse(Y, Yhat):
    """
    Calculates mean squared error between ground-truth Y and predicted Y

    """
    return np.mean((Y-Yhat)**2)

def rsquare(Y, Yhat):
    """
    Implementation of the R squared metric based on ground-truth Y and predicted Y
    """
    
    Y = np.asarray(Y)
    Ymean = np.mean(Y)
    Yhat = np.asarray(Yhat)
    
    residual = np.sum((Y-Yhat)**2)
    total = np.sum((Y-Ymean)**2)
    
    return 1-(residual/total)


class LinearRegression:
    """
    Class for linear regression
    """
    
    def __init__(self, learning_rate=0.1):
        """
        Constructor for the class. Learning rate is
        any positive number controlling step size of gradient descent.
        """

        self.learning_rate = learning_rate
        self.theta = None # theta is initialized once we fit the model
    
    def _calculate_gradient(self, Xmat, Y, theta_p, h=1e-5):
        """
        Helper function for computing the gradient at a point theta_p.
        """

        # get dimensions of the matrix
        n, d = Xmat.shape

        grad_vec = np.zeros(d)
        
        # using partial derivative with respect to theta_p[i]
        # (L(theta_p + h) - L(theta_p - h))/2h
        
        for i in range(d):
            plus_h = theta_p.copy()
            minus_h = theta_p.copy()
            
            plus_h[i] += h
            minus_h[i] -=h
            
            loss_plus_h =  np.sum((Xmat.dot(plus_h) - Y)**2)/(2*n)
            loss_minus_h =  np.sum((Xmat.dot(minus_h) - Y)**2)/(2*n)
            
            grad_vec[i] = (loss_plus_h - loss_minus_h)/(2*h)

        return grad_vec

    def fit(self, Xmat, Y, max_iterations=1000, tolerance=1e-6, verbose=False):
        """
        Fit a linear regression model using training data Xmat and Y.
        """

        # get dimensions of the matrix
        n, d = Xmat.shape        
        
        # initialize the first theta and theta new randomly
        theta = np.random.uniform(-5, 5, d)
        theta_new = np.random.uniform(-5, 5, d)
        iteration = 0
        
        # testing dif alphas
        alpha = .5

        # performs gradient descent until "convergence"
        while iteration < max_iterations:
            gradient = self._calculate_gradient(Xmat, Y, theta_new, h=1e-5)
            theta = theta_new.copy()
            # gradient update rule
            theta_new = theta - (alpha*gradient)
            
            mad = np.mean(np.abs(theta_new - theta))
            
            if mad < tolerance:
                if verbose:
                    print("Mean absolute difference is less than the tolerance argument")
                break

            iteration += 1
            
        # set the theta attribute of the model to the final value from gradient descent
        self.theta = theta_new.copy()

def main():
    """
    Do not edit this function. This function is used for grading purposes only.
    """

    np.random.seed(0)

    # simulated data
    Xmat, Y, feature_names = load_simulated_data()
    model = LinearRegression()
    model.fit(Xmat, Y)
    Yhat = Xmat @ model.theta
    theta = [float(round(p, 2)) for p in model.theta] # round model parameters for printing
    print("Simulated data results:\n" + "-"*4)
    print("Simulated data fitted weights", {feature_names[i]: theta[i] for i in range(len(feature_names))})
    print("R squared simulated data", rsquare(Y, Yhat), "\n")

    # insurance data
    Xmat_train, Y_train, Xmat_test, Y_test, feature_names = load_insurance_data()
    model = LinearRegression()
    model.fit(Xmat_train, Y_train) # only use training data for fitting
    Yhat_test = Xmat_test @ model.theta # evaluate on the test data
    theta = [float(round(p, 2)) for p in model.theta] # round model parameters for printing
    print("Insurance data results:\n" + "-"*4)
    print("Insurance data fitted weights", {feature_names[i]: theta[i] for i in range(len(feature_names))})
    print("R squared insurance data", rsquare(Y_test, Yhat_test))


if __name__ == "__main__":
    main()
