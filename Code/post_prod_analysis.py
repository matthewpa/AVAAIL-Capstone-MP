import os
import numpy as np
from datetime import date
import pandas as pd
from sklearn.metrics import mean_squared_error
from model import model_train, model_load, model_predict
from cslib import fetch_ts, engineer_features

def main():

    # Load actual data and process it
    prod_data_dir = os.path.join(".", "cs-production")

    loaded_items = model_load(data_dir=prod_data_dir,training=False, version=0.1)

    models = loaded_items[1]
    prod_data_eng = loaded_items[0]

    #for country in prod_data:
    #    prod_data_eng[country] = engineer_features(prod_data[country], training=False)

    #models = model_load()

    ## train the model
    df_results = pd.DataFrame(columns=['country','date','target','predicted'])

    for country in prod_data_eng:
        print("--------------- Country %s ---------------"% country)

        if country not in models:
            print("No model identified for country, skipping")
            continue

        dates = prod_data_eng[country]['dates']
        X =  prod_data_eng[country]['X']
        Y =  prod_data_eng[country]['y']

        if len(dates) == 0:
            print("No dates identfied for predictions, skipping ")
            continue

        pred_y = []
        for i in range(0, len(dates)):
            year = dates[i][0:4]
            month = dates[i][5:7]
            day = dates[i][8:10]
            res = model_predict(country,year,month,day, all_models=models, all_data=prod_data_eng)
            pred_y.append(res['y_pred'][0])

        eval_rmse =  round(np.sqrt(mean_squared_error(Y,pred_y)))

        print("--------------- RMSE %s ---------------"% eval_rmse)

        df = pd.DataFrame(columns=['country','date','target','predicted'])
        df.country = [country] * len(Y)
        df.date = dates
        df.target = Y
        df.predicted = pred_y
        df_results = df_results.append(df)

    print("Writing results to production_analysis.csv")
    df_results.to_csv("production_analysis.csv")

if __name__ == "__main__":

    main()
