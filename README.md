# WQU Capstone Project - Group 23

This is an open source library which implements Bayesian updating for intraday volume prediction. Library consists of the following components:

* WebSocket Server
* Model
* Calibrator
* Marketdata publisher
* Test Client (html page)


### Installing

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

To start server 
```
python3 run.py start server 
```

To start market data publisher
```
python3 run.py start publisher --date=2019.05.02 --delay=1000
```

To start client (this will open an html page in browser)

```
python3 run.py start client --ticker=GOOGL
```

## Authors

* **Andrey Vershinin**
* **Marina Duma**

See also the list of [contributors](https://github.com/manirka/wqu-capstone-vol-prediction/blob/master/CONTRIBUTORS.md) who participated in this project.

