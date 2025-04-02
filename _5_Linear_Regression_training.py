import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from _4_Linear_Regressor_Class import Linear_Regressor

from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold



pd.set_option('display.width', 500)
pd.set_option('display.max_columns', None)



#   importing the data
Data = pd.read_csv('HousingScaled.csv')
Data = Data.sample(frac=1)  #   Shuffling the data to avoid repeating patterns

#   making array with dependant variable
Y_all = Data['median_house_value'].to_numpy()

#   making array with inddependant variables
X_all = Data.drop( "median_house_value" , axis=1).to_numpy().astype('float64')



#   10  Fold Cross Validation
kf = KFold( n_splits= 10 )

Overall_MSE_train = 0
Overall_MAE_train = 0
Overall_MSE_test = 0
Overall_MAE_test = 0

trainings_count = 0

for current_fold, (train_indexes, test_indexes) in enumerate(kf.split(X_all)):

    X_train = X_all[train_indexes]
    Y_train = Y_all[train_indexes]
        
    X_test = X_all[test_indexes]
    Y_test = Y_all[test_indexes]


    model = Linear_Regressor()

    print( "-----------------------------------------------------------------" )
    print( "Current Tested Fold : ", current_fold + 1 )

    model_trained_succesfuly = model.train( X_train, Y_train)

    MSE_train, MAE_train = model.evaluate( X_train, Y_train )
    MSE_test, MAE_test = model.evaluate( X_test, Y_test )


    if ( model_trained_succesfuly ):
        #   model returns true if X^T*X is not singular

        Overall_MSE_train += MSE_train
        Overall_MAE_train += MAE_train
        Overall_MSE_test += MSE_test
        Overall_MAE_test += MAE_test

        trainings_count += 1
    else:               
        print( "\nWARNING!!! TRAINING DATA IS NOT SUITABLE FOR TRAINING\nTHE ERROR WILL NOT BE COUNTED TO THE FINAL AVERAGE ERROR\n" )
    
                
    print( "Mean Square Error (training data) :", MSE_train )
    print( "Mean Absolute Error (training data) :", MAE_train )
    print( "Mean Square Error (test data) :", MSE_test )
    print( "Mean Absolute Error (test data) :", MAE_test, "\n" )
        




if ( trainings_count != 0 ):

    Overall_MSE_train /= trainings_count
    Overall_MAE_train /= trainings_count
    Overall_MSE_test /= trainings_count
    Overall_MAE_test /= trainings_count

    print( "-----------------------------------------------------------------\n" )
    print( "Training experiences that were successful: ", trainings_count, "\n" )
    print( "Overall Score : \n" )
    print( "Mean Square Error (training data) :", Overall_MSE_train, "\n" )
    print( "Mean Absolute Error (training data) :", Overall_MAE_train, "\n" )
    print( "Mean Square Error (test data) :", Overall_MSE_test, "\n" )
    print( "Mean Absolute Error (test data) :", Overall_MAE_test, "\n" )

else:
    print( "-----------------------------------------------------------------\n" )
    print( "Error. Data is not suitable for training" )



