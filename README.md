# Simple Linear Regression Demo (CRISP-DM)

This interactive web application demonstrates a simple linear regression model built using Python. The application follows the **CRISP-DM (Cross-Industry Standard Process for Data Mining)** methodology to structure the process.

Users can interactively adjust the parameters of synthetically generated data and observe in real-time how the linear regression model adapts and performs.

## Features

-   **Interactive Data Generation**: Adjust the slope, number of data points, and noise level of the dataset.
-   **CRISP-DM Framework**: The application is structured around the six phases of the CRISP-DM model:
    1.  Business Understanding
    2.  Data Understanding
    3.  Data Preparation
    4.  Modeling
    5.  Evaluation
    6.  Deployment
-   **Model Visualization**: See a scatter plot of the generated data and the resulting regression line from the model.
-   **Performance Metrics**: Evaluate the model's performance with metrics like R-squared (R²) and Mean Squared Error (MSE).
-   **Coefficient Comparison**: Compare the model's learned coefficients (slope and intercept) to the original parameters used to generate the data.

## How to Run

### Prerequisites

-   Python 3.7+

### 1. Clone the repository or download the code

### 2. Install the required libraries

Open your terminal and run the following command to install the necessary Python libraries:

```bash
pip install -r requirements.txt
```

### 3. Run the Streamlit application

In your terminal, navigate to the project directory and run:

```bash
streamlit run linear_regression_app.py
```

Your web browser should open with the application running.

## File Descriptions

-   `linear_regression_app.py`: The main Python script containing the Streamlit application code.
-   `requirements.txt`: A list of the Python libraries required to run the application.
-   `README.md`: This file, providing an overview and instructions.
