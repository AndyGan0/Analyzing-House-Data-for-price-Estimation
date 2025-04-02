import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold

import keras
from keras.optimizers import Adam
from keras.layers import Dense
from keras.layers import BatchNormalization
from keras.layers import LeakyReLU



#
#   Importing the Data
#
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

for current_fold, (train_indexes, test_indexes) in enumerate(kf.split(X_all)):
    
    X_train = X_all[train_indexes]
    Y_train = Y_all[train_indexes]
        
    X_test = X_all[test_indexes]
    Y_test = Y_all[test_indexes]

    #   Initializing the model
    model = keras.Sequential()

    #   Adding layers

    #   Hidden Layers
    model.add( Dense( 100, activation='relu', input_dim=12 ) )
    model.add( BatchNormalization() )
    model.add( Dense( 100, activation='relu') )
    model.add( BatchNormalization() )
    model.add( Dense( 100, activation='relu') )
    model.add( BatchNormalization() )
    model.add( Dense( 50, activation='relu') )


    #   Output Layer
    model.add( Dense( 1, activation='linear') )

    #   Compiling the model
    model.compile(  optimizer=Adam(learning_rate=0.0001) , loss='mse', metrics=['mae'] )

    #   checking model summary
    model.summary()

    history = model.fit(X_train, Y_train, 
                        batch_size=64,
                        epochs=50,
                        verbose=2,
                        shuffle=True,
                        validation_data=(X_test, Y_test)  )


    if (current_fold == 0):
        plt.plot(history.history['loss'], color='C0' )
        plt.plot(history.history['val_loss'], color='red' )
        plt.title('Mean Squared Error')
        plt.ylabel('Error')
        plt.xlabel('Epochs')
        plt.legend( ['Training Data', 'Test Data'], loc='upper right' )


        plt.figure()
        plt.plot(history.history['mae'], color='C0' )
        plt.plot(history.history['val_mae'], color='red' )
        plt.title('Mean Absolute Error')
        plt.ylabel('Error')
        plt.xlabel('Epochs')
        plt.legend( ['Training Data', 'Test Data'], loc='upper right' )

        plt.show()

        
    Overall_MSE_train += history.history['loss'][-1]
    Overall_MAE_train += history.history['mae'][-1]
    Overall_MSE_test += history.history['val_loss'][-1]
    Overall_MAE_test += history.history['val_mae'][-1]

    
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

