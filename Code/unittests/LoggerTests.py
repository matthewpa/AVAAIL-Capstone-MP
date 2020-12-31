#!/usr/bin/env python
"""
model tests
"""

import os, sys
import csv
import unittest
from ast import literal_eval
import pandas as pd
from datetime import date, timedelta
import datetime
sys.path.insert(1, os.path.join('..', os.getcwd()))

## import model specific functions and variables
from logger import update_train_log, update_predict_log



class LoggerTest(unittest.TestCase):
    """
    test the essential functionality
    """
    TEST_END_DATE = date.today()
    TEST_START_DATE = date.today() - timedelta(days=400)

    TEST_TAG = "test_tag"
    TEST_DATE_RANGE = (TEST_START_DATE, TEST_END_DATE)
    TEST_RUNTIME = "000:00:00"
    TEST_MODEL_V = 0.1
    TEST_MODEL_V_NOTE = "Unit Testing"
    TEST_SHAPE = (100,2)
    TEST_PARAMS = {'label 1': {'precision':0.5, 'recall':1.0, 'f1-score':0.67, 'support':1} }

    TEST_COUNTRY = "United Kingdom"
    TEST_PREDICTION = "C"
    TEST_PROB = 0.1
    TEST_TARGET_DATE = date.today() + timedelta(days=90)


    def test_01_train(self):
        """
        ensure log file is created
        """

        log_file = os.path.join("logs", "train-test.log")
        if os.path.exists(log_file):
            os.remove(log_file)

        ## YOUR CODE HERE
        ## Call the update_train_log() function from logger.py with arbitrary input values and test if the log file
        ## exists in you file system using the assertTrue() base method from unittest.


        update_train_log(self.TEST_TAG, self.TEST_DATE_RANGE, self.TEST_PARAMS, self.TEST_RUNTIME,
                     self.TEST_MODEL_V, self.TEST_MODEL_V_NOTE, test=True)

        self.assertTrue(os.path.exists(log_file))


    def test_02_train(self):
        """
        ensure that content can be retrieved from log file
        """

        log_file = os.path.join("logs", "train-test.log")


        ## update the log
        update_train_log(self.TEST_TAG, self.TEST_DATE_RANGE, self.TEST_PARAMS, self.TEST_RUNTIME,
                     self.TEST_MODEL_V, self.TEST_MODEL_V_NOTE, test=True)

        df = pd.read_csv(log_file, delimiter=',', quotechar="|", index_col=False)
        last_row = df.tail(1)
        self.assertEqual(self.TEST_TAG, last_row['tag'].item())
        self.assertEqual(self.TEST_DATE_RANGE, eval(last_row['date_range'].item()))
        self.assertEqual(self.TEST_PARAMS, [literal_eval(i) for i in last_row['params'].copy()][0])
        self.assertEqual(self.TEST_RUNTIME, df[-1:]['runtime'].item())
        self.assertEqual(self.TEST_MODEL_V, last_row['model_version'].item())
        self.assertEqual(self.TEST_MODEL_V_NOTE, last_row['model_version_note'].item())

    def test_03_predict(self):
        """
        ensure log file is created
        """

        log_file = os.path.join("logs", "predict-test.log")
        if os.path.exists(log_file):
            os.remove(log_file)

        ## YOUR CODE HERE
        ## Call the update_predict_log() function from logger.py with arbitrary input values and test if the log file
        ## exists in you file system using the assertTrue() base method from unittest.
        shape = (100,2)



        update_predict_log(self.TEST_COUNTRY, self.TEST_PREDICTION, self.TEST_PROB, \
                self.TEST_TARGET_DATE, self.TEST_RUNTIME, self.TEST_MODEL_V, test=True)


        self.assertTrue(os.path.exists(log_file))


    def test_04_predict(self):
        """
        ensure that content can be retrieved from log file
        """

        log_file = os.path.join("logs", "predict-test.log")

        ## YOUR CODE HERE
        ## Log arbitrary values calling update_predict_log from logger.py. Then load the data
        ## from this log file and assert that the loaded data is the same as the data you logged.
        update_predict_log(self.TEST_COUNTRY, self.TEST_PREDICTION, self.TEST_PROB, \
                self.TEST_TARGET_DATE, self.TEST_RUNTIME, self.TEST_MODEL_V, test=True)

        df = pd.read_csv(log_file, delimiter=',', quotechar="|", index_col=False)

        last_row = df.tail(1)
        self.assertEqual(self.TEST_COUNTRY, last_row['country'].item())
        self.assertEqual(self.TEST_PREDICTION, last_row['y_pred'].item())
        self.assertEqual(self.TEST_PROB, last_row['y_proba'].item())
        self.assertEqual(self.TEST_TARGET_DATE, date.fromisoformat(last_row['target_date'].item()))
        self.assertEqual(self.TEST_MODEL_V, last_row['model_version'].item())
        self.assertEqual(self.TEST_RUNTIME, last_row['runtime'].item())


### Run the tests
if __name__ == '__main__':
    unittest.main()
