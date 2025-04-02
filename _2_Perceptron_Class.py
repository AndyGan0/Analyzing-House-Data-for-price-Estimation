import numpy as np
import matplotlib.pyplot as plt


class Perceptron:




    def __init__(self, learning_rate = 0.001, number_of_epochs = 100):        
        self.Weights = None
        self.bias = 0
        self.learning_rate = learning_rate
        self.number_of_epochs = number_of_epochs

        #   variables used to keep track of the error during training
        self.History_Training_Set_MSE = None
        self.History_Training_Set_MAE = None
        self.History_Test_Set_MSE = None
        self.History_Test_Set_MAE = None


    def _Step_Function( self, Weighted_Sum ):
        if ( Weighted_Sum < 0 ):
            return 0
        else:
            return 1



    def predict(self, x: np.ndarray):
        #   should only be used after training or during training!!!

        if self.Weights is None:
            return None

        Weighted_Sum = np.dot( self.Weights, x ) + self.bias

        Predicted = self._Step_Function(Weighted_Sum) 

        return Predicted



    def evaluate(self, x: np.ndarray , y: np.ndarray):
        #   Gets a test data set, and evaluates Mean Square Error and Mean Absolute Error 

        if self.Weights is None:
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



    def plotTrainingErrorDiagrams(self):

        if ( self.History_Training_Set_MSE is None ):
            return None

        if ( self.History_Test_Set_MSE is None or self.History_Test_Set_MAE is None ):
            #   Only Training data provided
            plt.plot( self.History_Training_Set_MSE , color='red' )
            plt.plot( self.History_Training_Set_MAE , color='C0' )
            plt.title('Training Error')
            plt.ylabel('Error')
            plt.xlabel('Epochs')
            plt.legend( ['Mean Squared Error', 'Mean Absolute Error'], loc='upper right' )
            plt.show()
        else:
            #   Test data is provided       
            plt.plot( self.History_Training_Set_MSE , color='C0' )
            plt.plot( self.History_Test_Set_MSE , color='red' )
            plt.title('Mean Square Error')
            plt.ylabel('Error')
            plt.xlabel('Epochs')
            plt.legend( ['Training Data', 'Test Data'], loc='upper right' )
            
            plt.figure()            
            plt.plot( self.History_Training_Set_MAE , color='C0' )
            plt.plot( self.History_Test_Set_MAE , color='red' )
            plt.title('Mean Absolute Error')
            plt.ylabel('Error')
            plt.xlabel('Epochs')
            plt.legend( ['Training Data', 'Test Data'], loc='upper right' )

            plt.show()



    def train(self, x: np.ndarray , y: np.ndarray,
               Test_data_X: np.ndarray = None, Test_data_Y: np.ndarray = None ):
                
        #   initializing history error arrays for diagrams
        self.History_Training_Set_MSE = np.zeros( (self.number_of_epochs) )  
        self.History_Training_Set_MAE = np.zeros( (self.number_of_epochs) ) 

        if ( Test_data_X is not None and Test_data_Y is not None ):   
            #   Test data is provided
            self.History_Test_Set_MSE = np.zeros( (self.number_of_epochs) )    
            self.History_Test_Set_MAE = np.zeros( (self.number_of_epochs) )
        else:
            self.History_Test_Set_MSE = None
            self.History_Test_Set_MAE = None


        number_of_data, number_of_features = x.shape

        #   Initializing Weights and bias
        self.Weights = np.random.uniform( low=-1, high=1, size=number_of_features)

        #   for each epoch
        for e in range(self.number_of_epochs):

            #   iterate through all data
            for i in range(number_of_data):

                #   predict the value
                Predicted = self.predict(x[i])

                #   calculate the Error
                Error = int(y[i]) - Predicted

                
                #   Update Weights and bias                
                self.bias = self.bias + self.learning_rate * Error
                self.Weights += self.learning_rate * Error * x[i]

            training_set_MSE, training_set_MAE = self.evaluate(x ,y)
            self.History_Training_Set_MSE[e] = training_set_MSE
            self.History_Training_Set_MAE[e] =  training_set_MAE

            if ( Test_data_X is not None and Test_data_Y is not None ):   
                #   Test data is provided

                Test_set_MSE, Test_set_MAE = self.evaluate(Test_data_X , Test_data_Y)
                self.History_Test_Set_MSE[e] = Test_set_MSE
                self.History_Test_Set_MAE[e] =  Test_set_MAE
            


        





                








    
    













