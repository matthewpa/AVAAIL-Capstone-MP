#!/usr/bin/env python
"""
module with functions to enable logging
"""

import time,os,re,csv,sys,uuid,joblib
from datetime import date

if not os.path.exists(os.path.join(".","logs")):
    os.mkdir("logs")

def update_train_log(tag, date_range, error_result, params, runtime, model_version, model_version_note, test=False):
    """
    update train log file
    """

    ## name the logfile using something that cycles with date (day, month, year)
    today = date.today()
    if test:
        logfile = os.path.join(".", "logs", "train-test.log")
    else:
        logfile = os.path.join(".", "logs", "train-{}-{}.log".format(today.year, today.month))

    ## write the data to a csv file
    header = ['unique_id','timestamp','tag','date_range', 'error_result','best_params', 'model_version',
              'model_version_note','runtime']
    write_header = False
    if not os.path.exists(logfile):
        write_header = True
    with open(logfile, 'a') as csvfile:
        writer = csv.writer(csvfile, delimiter=',', quotechar="|")
        if write_header:
            writer.writerow(header)

        to_write = map(str, [uuid.uuid4(), time.time(), tag, date_range, error_result, params,
                            model_version, model_version_note, runtime])
        writer.writerow(to_write)

def update_predict_log(country, y_pred, y_proba, target_date, runtime, model_version, test=False):
    """
    update predict log file
    """
    ## name the logfile using something that cycles with date (day, month, year)
    today = date.today()
    if test:
        logfile = os.path.join(".", "logs", "predict-test.log")
    else:
        logfile = os.path.join(".", "logs", "predict-{}-{}.log".format(today.year, today.month))

    ## write the data to a csv file
    header = ['unique_id','timestamp','country','y_pred','y_proba','target_date','model_version','runtime']
    write_header = False
    if not os.path.exists(logfile):
        write_header = True
    with open(logfile,'a') as csvfile:
        writer = csv.writer(csvfile, delimiter=',', quotechar="|")
        if write_header:
            writer.writerow(header)

        to_write = map(str,[uuid.uuid4(), time.time(), country, y_pred, y_proba,target_date,
                            model_version, runtime])
        writer.writerow(to_write)
    return logfile

if __name__ == "__main__":

    """
    basic test procedure for logger.py
    """

    from model import MODEL_VERSION, MODEL_VERSION_NOTE

    ## train logger
    update_train_log(str((100,10)),"{'rmse':0.5}","00:00:01",
                     0.1, "Initiating code", test=True)
    ## predict logger
    update_predict_log("[0]", "[0.6,0.4]","['united_states', 24, 'aavail_basic', 8]",
                       "00:00:01", 0.1, test=True)
