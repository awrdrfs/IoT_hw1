
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

# --- 1. Business Understanding ---
# The goal is to understand and visualize simple linear regression.
# We want to see how changing the underlying data's characteristics (slope, noise, sample size)affects the model's ability to find the original relationship.

def main():
    st.set_page_config(layout="wide", page_title="CRISP-DM Linear Regression")

    # --- CRISP-DM Phase Display ---
    st.title("Simple Linear Regression Demo following CRISP-DM")
    st.markdown("""
    This interactive application demonstrates the **CRISP-DM (Cross-Industry Standard Process for Data Mining)** methodology
    by building a simple linear regression model. You can adjust the parameters of the synthetic data and see how the model performs.
    """)

    st.sidebar.title("CRISP-DM Steps")
    st.sidebar.info("""
    1.  **Business Understanding**: Define the project objective.
    2.  **Data Understanding**: Examine the data.
    3.  **Data Preparation**: Clean and transform data.
    4.  **Modeling**: Select and apply modeling techniques.
    5.  **Evaluation**: Assess the model's quality.
    6.  **Deployment**: Integrate the model into operations.
    """)

    # --- User Controls in Sidebar ---
    st.sidebar.header("Data Generation Parameters")
    st.sidebar.markdown("Modify the parameters to generate new data.")
    
    # Allow user to modify parameters for y = ax + b
    a_param = st.sidebar.slider("Slope (a)", min_value=-5.0, max_value=5.0, value=2.0, step=0.1)
    b_param = 5.0  # Fixed intercept for simplicity
    n_points = st.sidebar.slider("Number of Data Points", min_value=10, max_value=1000, value=100, step=10)
    noise_level = st.sidebar.slider("Noise Level", min_value=0.0, max_value=20.0, value=5.0, step=0.5)

    # --- 2. Data Understanding ---
    st.header("1. & 2. Business & Data Understanding")
    st.markdown(f"""
    **Objective**: Our goal is to model a linear relationship. We'll generate data based on the formula **y = {a_param:.1f}x + {b_param} + noise**.
    
    **Data Exploration**: Below is a scatter plot of the data we've just generated. We expect to see a general trend, but the 'noise'
makes the relationship imperfect, just like in the real world.
    """)

    # Generate synthetic data
    @st.cache_data
    def generate_data(a, b, n, noise):
        x = np.random.rand(n, 1) * 10  # Generate x values between 0 and 10
        noise_values = np.random.randn(n, 1) * noise
        y = a * x + b + noise_values
        df = pd.DataFrame(data={'x': x.flatten(), 'y': y.flatten()})
        return df

    df = generate_data(a_param, b_param, n_points, noise_level)

    # Plot the generated data
    fig, ax = plt.subplots()
    ax.scatter(df['x'], df['y'], alpha=0.7, label='Generated Data')
    ax.set_title("Generated Data Scatter Plot")
    ax.set_xlabel("X Value")
    ax.set_ylabel("Y Value")
    ax.grid(True)
    ax.legend()
    st.pyplot(fig)
    
    with st.expander("Show Raw Data"):
        st.write(f"Displaying the first 10 of {len(df)} data points:")
        st.dataframe(df.head(10))

    # --- 3. Data Preparation ---
    st.header("3. Data Preparation")
    st.markdown("""
    In this step, we prepare the data for modeling. For this example, our data is already clean and in a good format.
    We just need to separate our features (the independent variable, `X`) from our target (the dependent variable, `y`).
    - **Feature (X)**: The 'x' column.
    - **Target (y)**: The 'y' column.
    """)
    X = df[['x']]
    y = df['y']
    st.code(f"""X = df[['x']]
y = df['y']""", language='python')

    # --- 4. Modeling ---
    st.header("4. Modeling")
    st.markdown("""
    We will use the **Linear Regression** algorithm from the `scikit-learn` library. This model tries to find the best-fitting
    straight line through the data points by minimizing the sum of the squared differences between the actual and predicted `y` values.
    """)
    
    # Create and fit the model
    model = LinearRegression()
    model.fit(X, y)
    
    st.code("from sklearn.linear_model import LinearRegression\n\nmodel = LinearRegression()\nmodel.fit(X, y)", language='python')

    # --- 5. Evaluation ---
    st.header("5. Evaluation")
    st.markdown("""
    Now we evaluate how well our model performed. We will:
    1.  **Visualize the Fit**: Plot the model's regression line over the original data.
    2.  **Check the Coefficients**: Compare the model's learned slope and intercept to the original parameters.
    3.  **Calculate Metrics**: Use R-squared (R²) and Mean Squared Error (MSE) to quantify performance.
    """)

    # Get model predictions
    y_pred = model.predict(X)
    
    # Get model coefficients
    learned_a = model.coef_[0]
    learned_b = model.intercept_

    # Calculate metrics
    mse = mean_squared_error(y, y_pred)
    r2 = r2_score(y, y_pred)

    # Display results in columns
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Model Fit Visualization")
        fig2, ax2 = plt.subplots()
        ax2.scatter(df['x'], df['y'], alpha=0.7, label='Actual Data')
        ax2.plot(df['x'], y_pred, color='red', linewidth=2, label='Regression Line')
        ax2.set_title("Model's Fit vs. Actual Data")
        ax2.set_xlabel("X Value")
        ax2.set_ylabel("Y Value")
        ax2.grid(True)
        ax2.legend()
        st.pyplot(fig2)

    with col2:
        st.subheader("Performance Metrics")
        st.metric(label="Original Slope (a)", value=f"{a_param:.2f}")
        st.metric(label="Learned Slope (a)", value=f"{learned_a:.2f}")
        
        st.metric(label="Original Intercept (b)", value=f"{b_param:.2f}")
        st.metric(label="Learned Intercept (b)", value=f"{learned_b:.2f}")
        
        st.metric(label="R-squared (R²)", value=f"{r2:.3f}", help="Closer to 1 is better. Represents the proportion of variance in y predictable from x.")
        st.metric(label="Mean Squared Error (MSE)", value=f"{mse:.2f}", help="Lower is better. The average squared difference between actual and predicted values.")

    # --- 6. Deployment ---
    st.header("6. Deployment")
    st.success("""
    This Streamlit application itself is the deployment! It provides a user interface for interacting with the trained model.
    
    **How to Run This App:**
    1. Save this code as a Python file (e.g., `linear_regression_app.py`).
    2. Install the required libraries: `pip install streamlit pandas scikit-learn matplotlib`.
    3. Open your terminal and run: `streamlit run linear_regression_app.py`.
    """)

if __name__ == '__main__':
    main()
    print('hello')
