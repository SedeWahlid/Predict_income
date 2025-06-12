# Predict_income

A supervised learning model with an simple html based interface . It predicts the adults income based on the training data set of adult from UCI . 

![Screenshot (1)](https://github.com/user-attachments/assets/7143c330-23ed-420a-b860-9d71fb2ba71d)

## Table of Contents

-   [Overview](#overview)
-   [Features](#features)
-   [How it Works](#how-it-works)
-   [Tech Stack](#tech-stack)
-   [Prerequisites](#prerequisites)
-   [Setup and Installation](#setup-and-installation)
-   [Running the Application](#running-the-application)
-   [Usage Guide](#usage-guide)
-   [License](#license)


## Overview

A fun simple prediction of an adults income based on some features like age , workclass, capital-gain etc. 


## Features

*   **Gradient-Boost model :** uses the Gradient-Boost model/algorithm to make a non-linear approach to predict an outcome.
*   **simple interface :** uses a simple local web interface. It is also possible to use the pred.py file to run the whole thing on the terminal
*   **Pre-trained model :** there is a pre-trained model with an 88%-accuracy , when you use the gradient_model.pkl file .
*   **Open Dataset :** the model is trained with an open dataset from UCI called adult . You can also trained it further if you want just use the train.py file to additionally train the model .

## How it Works

1. **reading and Cleaning data  :** first it reads data via pandas and then it cleans the data by rearranging string elements to numbers since the computer is more familiar with numbers. 
2. **Training the model :** via the train.py file it trains the model with a large training set , and computes a model file called gradient_model.pkl where the model is saved via joblib pipelining. We only need to do this step once since we save the model.
3. **Prediction :** based on the training , it will predict by computing/creating decision trees by correcting the weaknesses/errors of the beforehand trees . So we basically create trees one after another and correct each tree with the errors that the trees before had.

## Tech Stack

*   **Python 3.8+**
*   **Scikit-learn**
*   **Joblib**
*   **Flask**
*   **Pandas**
*   **Numpy**

## Prerequisites

*   Python 3.8 or higher installed.
*   `pip` (Python package installer).

## Setup and Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/SedeWahlid/Predict_income.git
    cd Predict_income
    ```

2.  **Create and activate a virtual environment (recommended):**
    ```bash
    python -m venv venv
    # On Windows
    # venv\Scripts\activate
    # On macOS/Linux
    # source venv/bin/activate
    ```

3.  **Install the required Python packages:**
    ```bash
    pip install joblib numpy scikit-learn flask pandas
    ```

## Running the Application

Once the setup is complete, navigate to your project directory in the terminal and run the app.py application using the following command:

```bash
python app.py
```
or when you want to use it on your terminal:
```bash
python pred.py
```

## Usage Guide of the web application 

1.  **Enter URL in your browser :** Navigate to the application in your web browser and paste the https... address given from your terminal.

2.  **Interface :** you will see a simple interface with input fields that you can experiment with .

## Usage Guide of the terminal application 

1.  **Run the pred.py file :** after running the pred.py , you will notice that you have to enter the features manually via terminal .

2.  **User input restrictions :** you will notice that the terminal application is really poorly , since we don't handle input as "good" as in the web application. Furthermore you will notice that there is no guideline on what you really have to put in as an input which is just poorly made for terminal usage .

## NOTE 

  **Recommendation**

  **Please use the web application to have a clean and understandable trial of the application since the terminal application is just not good. Lastly, I stopped working on it so there will be no updates etc. .*

## License 

MIT License

Copyright (c) 2025 SedeWahlid

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
    
