"""
Created on Wed Sept 12 10:56:20 2018
@author: Haseeb Khan
@Work: My Workbook
:: Created using TensorFlow and Keras 
:: Created for educational purposes, feel free to use this code with appropriate citations.
:: This code is for demonstration purposes, real accurate stock price prediction is a more complex task, get in touch if you'd like to know.
"""



import os
import numpy as np
import pandas as pd

#Import Plotting Model
import matplotlib.pyplot as plt

#Import Keras Machine Learning and sKlearn scaler + tensorflowbackend
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential, load_model
from keras.layers import LSTM, Dense, Dropout
import tensorflow as tf



# Neural network model Class
class NN_model(Sequential):
    
# Initializes the Neural Network Model to be used for the predictions.
    def __init__(self, * args , ** kwargs):
        super().__init__( * args,** kwargs,)
        
    
    def create_neural_network(self):
        self.model = Sequential
        self.model.add(LSTM(units = 100, return_sequences = True, input_shape=(x_train.shape[1],1)))
        self.model.add(Dropout(0.2))
        self.model.add(LSTM(units = 100, return_sequences = True))
        self.model.add(Dropout(0.2))
        self.model.add(LSTM(units = 100, return_sequences = True))

        self.model.add(Dropout(0.2))
        self.model.add(Dense(units = 100))
        self.model.add(LSTM(units = 100))
        self.model.add(Dense(units =1))

        
    #def custom_compile(self, input_model):
     #   input_model.compile(loss='mean_squared_error', optimizer ='adam')
        


# Post processing class :: Contains multiple delta methods
class Data_Processing():

    def delta_calculator(self, Y_test):

        daily_delta_list = []
        for i in range (Y_test.shape[0] - 1):
            price_current_day = Y_test[i]
            price_current_day = price_current_day[0]

            price_next_day = Y_test[i+1]
            price_next_day = price_next_day[0]

            delta_daily = price_next_day - price_current_day
            delta_daily = round(delta_daily, 7)
            #print(type(delta_daily), delta_daily)
            daily_delta_list.append(delta_daily)

        self.daily_delta_list = np.asarray(daily_delta_list)
        return self.daily_delta_list


    def delta_test_pred_calculator(self, Y_test,Y_pred):

        daily_delta_list = []
        pred_delta_list = []
        for i in range (Y_test.shape[0]-4):
            #Create List of daily delta lists, by subtracting last day price from current day price.
            price_current_day = Y_test[i]
            price_current_day = price_current_day[0]

            price_next_day = Y_test[i+1]
            price_next_day = price_next_day[0]

            delta_daily = price_next_day - price_current_day
            delta_daily = round(delta_daily, 7)
            #print(type(delta_daily), delta_daily)
            daily_delta_list.append(delta_daily)


            #Create List of prediction delta's , by subtracting last day price from current day predicted price.
            price_current_day = Y_test[i]
            price_current_day = price_current_day[0]

            price_next_day = Y_pred[i+1]
            price_next_day = price_next_day[0]

            delta_daily = price_next_day - price_current_day
            delta_daily = round(delta_daily, 7)
            pred_delta_list.append(delta_daily)

        #Convert to np.Array
        self.daily_delta_list = np.asarray(daily_delta_list)
        self.pred_delta_list = np.asarray(pred_delta_list)

        return self.daily_delta_list, self.pred_delta_list


    def correct_sign_percentage_prediction(self, input_list):

        bool_list = []
        for i in (input_list):

                if ((i[0] > 0 and i[1] > 0) or (i[0] < 0
                and i[1] < 0) or (i[0] == 0 and i[1] == 0)):
                    bool_list.append(1)

                else:
                    bool_list.append(0)

        return (bool_list)

    #Function to create dataset
    def create_my_dataset(self, df):
    
        # Declare Empty arrays for zipper
        ind_var_x = []
        dep_var_y = []
    
        # Zipper to zip arrays appropriately.
        length = df.shape[0]
        for i in range(25, length):
            ind_var_x.append(df[i-25:i,0])
            dep_var_y.append(df[i,0])

        ind_var_x = np.array(ind_var_x)
        dep_var_y = np.array(dep_var_y)


        return ind_var_x,dep_var_y

"""IF USING INTERACTIVE PYTHON RUN EACH SECTION IN SEPERATE CELL, 
ELSE SELECTIVELY RUN IN MAIN FUNCTION"""

df = pd.read_csv('AAPL3.csv')
df_Op_Cl_Vol = df[['Open', 'Close', 'Volume']]
df_price_open = df['Open'].values
df_price_open = df_price_open.reshape(-1,1)


#Train test data_split 30% to 70% split, with a 25 part LSTM overlap margin.
dataset_train_input = df_price_open[:int(df_price_open.shape[0]*0.7)]
dataset_test_input = df_price_open[int(df.shape[0]*0.7)-25:]
dataset_train_shaped = np.array(dataset_train_input)
dataset_test_shaped = np.array(dataset_test_input)


#Scaling the training data
new_scalar =  MinMaxScaler(feature_range = (0,1))
dataset_train_shaped = new_scalar.fit_transform(dataset_train_shaped)

#Scaling the test data
new_scalar = MinMaxScaler(feature_range = (0,1))
dataset_test_shaped = new_scalar.fit_transform(dataset_test_shaped)



# Training Set Creation
x_train, y_train = Data_Processing().create_my_dataset(dataset_train_shaped)
x_train.shape

# Test Set Creation
x_test, y_test = Data_Processing().create_my_dataset(dataset_test_shaped)
x_test[0:2]


# Compiling the model and fitting the data to it.
model_instance.compile(loss = 'mean_squared_error', optimizer = 'adam')
model_instance.fit(x_train, y_train, epochs = 30, batch_size = 30)


# Visualizng the predictions
predictions = model.predict(x_test)
predictions = new_scalar.inverse_transform(predictions)


fig, ax = plt.subplots(figsize=(10,5))
plt.plot(df, color='green', label='Stock Price')
lead_markov = 25 #Lead is definied by our markov process and our markov limit.

ax.plot(range(len(y_train)+lead_markov, len(y_train)+lead_markov+len(predictions)),predictions,color='red', label='Predicted')
plt.legend()
print(range(len(y_train)+lead_markov, len(y_train)+lead_markov+len(predictions)))

# Zoomed In Visualization
y_test_scaled = scaler.inverse_transform(y_test.reshape(-1,1))

fig, ax = plt.subplots(figsize=(10,5))
ax.plot(y_test_scaled, color='green', label = 'Actual Stock Price')
plt.plot(predictions, color = 'blue', label='Predicted Stock Price')
plt.legend()

