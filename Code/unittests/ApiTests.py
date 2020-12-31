#!/usr/bin/env python
"""
api tests

these tests use the requests package however similar requests can be made with curl

e.g.

data = '{"key":"value"}'
curl -X POST -H "Content-Type: application/json" -d "%s" http://localhost:8080/predict'%(data)
"""

import sys
import os
import unittest
import requests
import re
from ast import literal_eval
import numpy as np
import pandas as pd

port = 5000

try:
    requests.post('http://127.0.0.1:{}/'.format(port))
    server_available = True
except:
    server_available = False

## test class for the main window function
class ApiTest(unittest.TestCase):
    """
    test the essential functionality
    """

    @unittest.skipUnless(server_available, "local server is not running")
    def test_01_train(self):
        """
        test the train functionality
        """

        request_json = {'mode':'test'}
        r = requests.post('http://127.0.0.1:{}/train'.format(port), json=request_json)
        train_complete = re.sub("\W+","",r.text)
        self.assertEqual(train_complete,'true')

    @unittest.skipUnless(server_available, "local server is not running")
    def test_01_01_train_with_data(self):
        """
        test the train functionality
        """
        request_json = {'mode':'test'}
        request_json['tag'] = 'API_TEST'
        columns = ["date","purchases","unique_invoices","unique_streams","total_views","year_month","revenue"]
        data = [
            ["2019-06-26",1358,67,999,6420,"2019-06",4903.17],
            ["2019-06-27",1620,80,944,9435,"2019-06",5499.38],
            ["2019-06-28",1027,70,607,5539,"2019-06",3570.60]
            ]
        df_test = pd.DataFrame(data, columns=columns)
        request_json['data'] = df_test.to_dict()

        r = requests.post('http://127.0.0.1:{}/train'.format(port), json=request_json)
        train_complete = re.sub("\W+","",r.text)
        self.assertEqual(train_complete,'true')


    @unittest.skipUnless(server_available, "local server is not running")
    def test_02_predict_empty(self):
        """
        ensure appropriate failure types
        """

        ## provide no data at all
        r = requests.post('http://127.0.0.1:{}/predict'.format(port))
        self.assertEqual(re.sub('\n|"','',r.text),"[]")

        ## provide improperly formatted data
        r = requests.post('http://127.0.0.1:{}/predict'.format(port), json={"key":"value"})
        self.assertEqual(re.sub('\n|"', '', r.text), "[]")


    @unittest.skipUnless(server_available, "local server is not running")
    def test_03_predict(self):
        """
        test the predict functionality
        """

        query_data =   {
                        'country':"all",
                        'year':2021,
                        'month':2,
                        'day':1
                        }

        query_type = 'dict'
        request_json = {'query':query_data, 'type':query_type}

        r = requests.post('http://127.0.0.1:{}/predict'.format(port), json=request_json)
        response = literal_eval(r.text)

        for p in response['y_pred']:
            self.assertTrue(p in [0.0, 1.0])


    @unittest.skipUnless(server_available,"local server is not running")
    def test_04_logs(self):
        """
        test the log functionality
        """

        file_name = 'train-test.log'
        request_json = {'file':'train-test.log'}
        r = requests.get('http://127.0.0.1:{}/logs/{}'.format(port, file_name))

        with open(file_name, 'wb') as f:
            f.write(r.content)

        self.assertTrue(os.path.exists(file_name))

        if os.path.exists(file_name):
            os.remove(file_name)


    @unittest.skipUnless(server_available, "local server is not running")
    def test_03_predict_with_version(self):
        """
        test the predict functionality
        """

        query_data =   {
                        'country':"all",
                        'year':2021,
                        'month':2,
                        'day':1
                        }


        query_type = 'dict'
        request_json = {'query':query_data, 'type':query_type, 'version':0.1}

        r = requests.post('http://127.0.0.1:{}/predict'.format(port), json=request_json)
        response = literal_eval(r.text)

        for p in response['y_pred']:
            self.assertTrue(p in [0.0, 1.0])

### Run the tests
if __name__ == '__main__':
    unittest.main()
