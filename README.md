# WQU Capstone Project - Group 23

This is an open source library implementing Bayesian updating for intraday volume prediction. This library can be used in replay and real-time by trading algos sensitive to trading volume availability and thus requiring intra-day trading volume forecast.

To train and test our model we used one-minute bars of the following stocks: FB, MSFT, GOOGL, JNJ, V downloaded from [IEX Cloud](https://iexcloud.io/).

## Components

Library consists of the following components:

* **Calibrator** generating config file for the model
* **Model** generating intraday volume predictions based on historical data and intraday volume updates 
* **WebSocket Server** initializing the model instances by loading config generated by configurator, listening for market data updates and notifying clients with the updated predictions.
* **Marketdata publisher** publishing intraday one-minute volume updates. This component is useful in replay mode as it loads data from pickle files and sends updates to the server with predefined interval.  
* **Test Client** listening to volume predictions published by the websocket server and visualising it in a simple html page. Client is using [plotly](https://plotly.com/javascript/) graphical library to visualise volume curve.

In order to test the library, please calibrate the model for required ticker and date, start server, market data publisher and client using commands from **Running** section. 

## Installing

1. Create new virtual environment
    ```
    python3 -m venv vol-prediction
    ```
2. Activate created environment
    ```
    source vol-prediction/bin/activate
    ```
3. Install required libraries
    ```
    pip3 install -r requirements.txt
    ```

## Running 

1. To calibrate the model for the particular date
    ```
    python3 run.py start calibrator --date=2019-12-18 
    ```
2. To start the server 
    ```
    python3 run.py start server 
    ```
3. To start market data publisher in replay mode
    ```
    python3 run.py start publisher --date=2019-12-19 --delay=500
    ```
4. To start client (this will open an html page in browser)
    ```
    python3 run.py start client --ticker=V
    ```

## Message format
Examples of json messages should be supported by market data publisher and client in the real live applications:

1. Market data publisher<br>These one-minute bars should be sent to server once a minute:

    ```json
    { "action" : "bar", "ticker" : "FB", "datetime" : "2019-12-19 09:30:00", "volume" : 51930 }
    ```

2. Client<br>Server sends to client volume curve on connection establishment:

    ```json
    { "type": "curve", "times" : ["09:30", "09:31", "09:32", "09:33"], "values" : [0.51, 0.39, 0.32, 0.28] }
    ```

    Server sends to client updated volume prediction and realized volume at given minute on each market data tick from the publisher:

    ```json
    { "type" : "volume", "time" : "09:31", "volume" : 52637, "prediction" : 2311962 }
    ```

## Authors

* **Andrey Vershinin**
* **Marina Duma**

See also the list of [contributors](https://github.com/manirka/wqu-capstone-vol-prediction/blob/master/CONTRIBUTORS.md) who participated in this project.

## Outstanding Tasks

1. Add an abstract class for model implementations which should allow users to implement and plug in via config file different model types.
2. Support of client subscription to time interval. As the main use-case for our library is trading algo requiring precise forecast of the trading volume within some time interval, such as VWAP, we need to implement support of client subscription to time interval when client receives volume curve normalized to requested time interval.
3. Add an abstract class for market data publisher with just one abstract method which should load data from external source and implemented by the users.
4. Convert current replay client into proper monitoring tool.
