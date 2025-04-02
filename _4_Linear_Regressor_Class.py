import numpy as np
import matplotlib.pyplot as plt


class Linear_Regressor:

    #   Performs Linear Regression using Ordinary Least Square


    def __init__(self):        
        self.Weights = None



    def predict(self, x: np.ndarray):
        #   predicts the dependant variable using the independant variables
        #   this method assumes that coefficients have been found
        #   thus it should only be used after training!

        if self.Weights is None:
            return None

        x = np.append( 1, x )
        Prediction = np.dot( x, self.Weights )

        return Prediction




    def evaluate(self, x: np.ndarray , y: np.ndarray):
        #   Gets a test data set, and evaluates Mean Square Error and Mean Absolute Error 

        if self.Weights is None or x is None or y is None:
            return None

        y_predicted = np.zeros( y.shape )
        for i in range( y.shape[0] ):
            y_predicted[i] = self.predict( x[i] )
        
        Error = y - y_predicted
        

        #   Mean Absolute Error
        MAE = np.abs(Error)
        MAE = np.sum(MAE) / len(y)

        #   Mean Square Error
        MSE = np.square(Error)
        MSE = np.sum(MSE) / len(y)


        return MSE, MAE



    def train(self, x: np.ndarray , y: np.ndarray ):

        #   called to find coefficients based on x and y.
        #   x are the independant variables
        #   y is the dependant viariable
        
        number_of_data = x.shape[0]

        x = np.c_[ np.ones(number_of_data), x ]  #   Adding column of ones at beginning


        Matrix_invertable = True

        #   colculating coefficients
        b_coefficients = np.matmul( x.transpose(), x )  #   x^T * x


        #   checking if matrix is invertible
        b_coefficients_rank = np.linalg.matrix_rank(b_coefficients)

        if ( b_coefficients_rank != min(x.shape) ):
            #   (X^T)*X is singular
            b_coefficients = np.linalg.pinv( b_coefficients )   #   find pseudo inverse instead
            Matrix_invertable = False
        else:
            #   matrix can be inverted
            b_coefficients = np.linalg.inv( b_coefficients )    #   inverting



        b_coefficients = np.matmul( b_coefficients, x.transpose() ) #   * x^T
        b_coefficients = np.matmul( b_coefficients, y ) #   * y

        self.Weights = b_coefficients

        return Matrix_invertable





