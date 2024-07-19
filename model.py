import os
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import math
import pandas as pd
import openpyxl
from matplotlib import pyplot as plt

from keras._tf_keras.keras.models import load_model, Sequential
from keras._tf_keras.keras.layers import Dense, Dropout, InputLayer
from keras._tf_keras.keras.callbacks import EarlyStopping, ModelCheckpoint
import keras._tf_keras.keras.optimizers as optimizers
from sklearn.model_selection import train_test_split

def load_or_run_model(scalers:dict, X_scaled:np.ndarray, y_scaled:np.ndarray):

    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_scaled, test_size=0.2, random_state=42)

    file_path = 'mlsOracle/data_and_models/basic_model.keras'

    if not os.path.exists(file_path):
        
        ## Model is basic version without any callibration including no change to hyperparameters ##

        model = Sequential([
            InputLayer(input_shape = (X_scaled.shape[1],)),
            Dense(units=2, activation = 'relu')
        ])  

        opt = optimizers.Adam()
        model.compile(optimizer = opt, loss='mean_squared_error')
        es = EarlyStopping(monitor = 'loss', mode = 'min', patience = 6)
        mcp_save = ModelCheckpoint(file_path, save_best_only=True, monitor='loss', mode='min')
        model.fit(X_train, y_train, epochs=150, batch_size=32, callbacks = [es, mcp_save]) 

        predicted_scores_validate = model.predict(X_test)

        # Rescale back to original range    
        home_predicted_scores = np.round(scalers['HT_SC'].inverse_transform(predicted_scores_validate[:, 0].reshape(-1, 1)))     
        away_predicted_scores = np.round(scalers['AT_SC'].inverse_transform(predicted_scores_validate[:, 1].reshape(-1, 1)))
        y_test_home_pred = np.round(scalers['HT_SC'].inverse_transform(y_test[:, 0].reshape(-1, 1)))
        y_test_away_pred = np.round(scalers['AT_SC'].inverse_transform(y_test[:, 1].reshape(-1, 1)))
        
        ### Evaluate ###
        correct_score_counter = 0
        correct_outcome_counter = 0
        total_fixtures = len(home_predicted_scores)
        
        for index in range(0, len(home_predicted_scores), 1):
            if home_predicted_scores[index] == y_test_home_pred[index] and away_predicted_scores[index] == y_test_away_pred[index]:
                correct_score_counter += 1
                correct_outcome_counter += 1
            elif home_predicted_scores[index] > away_predicted_scores[index] and y_test_home_pred[index] > y_test_away_pred[index]:
                correct_outcome_counter += 1
            elif home_predicted_scores[index] < away_predicted_scores[index] and y_test_home_pred[index] < y_test_away_pred[index]:
                correct_outcome_counter += 1
            elif home_predicted_scores[index] == away_predicted_scores[index] and y_test_home_pred[index] == y_test_away_pred[index]:
                correct_outcome_counter += 1
                
        correct_score_pct = (correct_score_counter / total_fixtures) * 100
        correct_outcome_pct = (correct_outcome_counter / total_fixtures) * 100
        print('Successful score prediction pct = ' + str(round(correct_score_pct, 2)) + ', Successful outcome prediction pct = ' + str(round(correct_outcome_pct, 2)))

        ## Home Scores ##
        home_mse_test = mean_squared_error(y_test_home_pred, home_predicted_scores)
        home_MAE_test = mean_absolute_error(y_test_home_pred, home_predicted_scores)
        home_R2val_test = r2_score(y_test_home_pred, home_predicted_scores)
        ## Away Scores
        away_mse_test = mean_squared_error(y_test_away_pred, away_predicted_scores)
        away_MAE_test = mean_absolute_error(y_test_away_pred, away_predicted_scores)
        away_R2val_test = r2_score(y_test_away_pred, away_predicted_scores)
        
        print('RMSE = ' + str(home_mse_test) + ',' + str(away_mse_test) + ' MAE = ' + str(home_MAE_test) + ',' + str(away_MAE_test) + ' R2 val = ' + str(home_R2val_test) + ',' + str(away_R2val_test))
    else:
        model = load_model(file_path)
    
    return model