# WQU Capstone Project

One Paragraph of project description goes here

## Getting Started

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes. See deployment for notes on how to deploy the project on a live system.

### Prerequisites

What things you need to install the software and how to install them

```
Give examples
```

### Installing

1. Create new virtual environment
    ```
    python3 -m venv capstone
    ```
2. Activate created environment
    ```
    source capstone/bin/activate
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


### Break down into end to end tests

Explain what these tests test and why

```
Give an example
```

### And coding style tests

Explain what these tests test and why

```
Give an example
```

## Deployment

Add additional notes about how to deploy this on a live system

## Built With

* [Dropwizard](http://www.dropwizard.io/1.0.2/docs/) - The web framework used
* [Maven](https://maven.apache.org/) - Dependency Management
* [ROME](https://rometools.github.io/rome/) - Used to generate RSS Feeds

## Contributing

Please read [CONTRIBUTING.md](https://gist.github.com/PurpleBooth/b24679402957c63ec426) for details on our code of conduct, and the process for submitting pull requests to us.

## Versioning

We use [SemVer](http://semver.org/) for versioning. For the versions available, see the [tags on this repository](https://github.com/your/project/tags). 

## Authors

* **Andrey Vershinin**
* **Marina Duma**

See also the list of [contributors](https://github.com/manirka/wqu-capstone-vol-prediction/blob/master/CONTRIBUTORS.md) who participated in this project.

## Acknowledgments

* Hat tip to anyone whose code was used
* Inspiration
* etc
