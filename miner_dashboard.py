from retrieve_data import get_data, scale_data, prep_pred_input
from model import load_or_run_model

import numpy as np

def activate():

    scrape_more_data = False
    data = get_data(scrape_more_data)

    scalers, X_scaled, y_scaled = scale_data(data)

    model = load_or_run_model(scalers, X_scaled, y_scaled)

    ### This is where the new predictions will get read in ###
    list_of_fixtures = [{'DATE': '2024-06-19', 'HT': 'Charlotte FC', 'AT' : 'Orlando City'}, {'DATE': '2024-06-19', 'HT': 'FC Cincinnati', 'AT' : 'Philadelphia Union'}]
    for fixture in list_of_fixtures:

        pred_input, hist_score = prep_pred_input(fixture['DATE'], fixture['HT'], fixture['AT'], scalers)

        predicted_outcome = model.predict(pred_input)
        
        predicted_outcome[:,0] = np.round(scalers['HT_SC'].inverse_transform(predicted_outcome[:, 0].reshape(-1, 1)).reshape(-1))
        predicted_outcome[:,1] = np.round(scalers['AT_SC'].inverse_transform(predicted_outcome[:, 1].reshape(-1, 1)).reshape(-1))
        
        print(fixture, 'predicted score :', predicted_outcome[0], 'actual score :', hist_score)

activate()
