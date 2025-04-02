import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split

from _2_Perceptron_Class import Perceptron



pd.set_option('display.width', 500)
pd.set_option('display.max_columns', None)



#   importing the data
Data = pd.read_csv('HousingScaled.csv')
Data = Data.sample(frac=1)  #   Shuffling the data to avoid repeating patterns

#   finding the median house value
house_value_median = Data["median_house_value"].median()

#   making array with target
#   target = 1 if house value >= median
#            0 if house value < median
Y_all = np.where(Data["median_house_value"] >= house_value_median, 1, 0)

#   converting data to np array
X_all = Data.drop( "median_house_value" , axis=1)
X_all = X_all.to_numpy().astype('float64')



#   10  Fold Cross Validation
kf = KFold( n_splits= 10 )

Overall_MSE_train = 0
Overall_MAE_train = 0
Overall_MSE_test = 0
Overall_MAE_test = 0


for current_fold, (train_indexes, test_indexes) in enumerate(kf.split(X_all)):

    X_train = X_all[train_indexes]
    Y_train = Y_all[train_indexes]
        
    X_test = X_all[test_indexes]
    Y_test = Y_all[test_indexes]

    model = Perceptron( learning_rate= 0.0001 )

    model.train(X_train, Y_train, Test_data_X= X_test, Test_data_Y= Y_test)

    if (current_fold == 0):
        model.plotTrainingErrorDiagrams()

    MSE_train, MAE_train = model.evaluate( X_train, Y_train )
    MSE_test, MAE_test = model.evaluate( X_test, Y_test )

    Overall_MSE_train += MSE_train
    Overall_MAE_train += MAE_train
    Overall_MSE_test += MSE_test
    Overall_MAE_test += MAE_test
    
    print( "-----------------------------------------------------------------" )
    print( "Current Tested Fold : ", current_fold+1 )
    print( "Mean Square Error (training data) :", MSE_train )
    print( "Mean Absolute Error (training data) :", MAE_train )
    print( "Mean Square Error (Test data) :", MSE_test )
    print( "Mean Absolute Error (Test data) :", MAE_test, "\n" )
    


     


Overall_MSE_train /= 10
Overall_MAE_train /= 10
Overall_MSE_test /= 10
Overall_MAE_test /= 10





print( "-----------------------------------------------------------------\n" )
print( "Overall Score : \n" )
print( "Mean Square Error (training data) :", Overall_MSE_train, "\n" )
print( "Mean Absolute Error (training data) :", Overall_MAE_train, "\n" )
print( "Mean Square Error (Test data) :", Overall_MSE_test, "\n" )
print( "Mean Absolute Error (Test data) :", Overall_MAE_test, "\n" )


