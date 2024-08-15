import streamlit as st
import base64

selected_button = "Introduction"


# Function to convert image to base64
def get_base64_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode()
    
pathname = "https://raw.githubusercontent.com/Pranavsharma13/RoadSense--Advanced-Predictive-Modelling-for-Traffic-Safety/main/RoadSenseAppDeployment/"



side_logo_path = f"{pathname}SideLogo.png"

converted_side_Logo = get_base64_image(side_logo_path)

st.sidebar.image("Logo_Round2222.png")

#logo_base64 = get_base64_image(side_logo_path)
st.markdown(
    f"""
    <style>
    .main-header {{
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 32px;
        font-weight: bold;
        color: #333;
        text-align: center;
        padding: 20px 0;
        background-color: #f7f7f7;
        border-radius: 10px;
    }}
    .main-header img {{
        margin-right: 20px;
        width: 110px; /* Adjust the width as needed */
        height: auto;
    }}
    </style>
    <div class="main-header">
        <img src="data:image/png;base64,{side_logo_path}" alt="logo">
        RoadSense - Advance Predictive Modeling of Traffic Saftey
    </div>
    """,
    unsafe_allow_html=True
)


# Define custom button style
st.sidebar.markdown(
    """
    <style>
    .stButton button {
        font-size: 18px;
        padding: 8px 20px;
        margin: 8px 0;
        width: 100%; /* Ensures all buttons have the same width */
        border-radius: 20px;
        border: none;
        background-color: #283b56; /* A modern green color */
        color: white;
        box-shadow: 0 8px 8px rgba(0, 0, 0, 0.2); /* Soft shadow for depth */
        transition: background-color 0.3s ease, transform 0.3s ease; /* Smooth transitions */
    }
    
    .stButton button:hover {
        background-color: #203046; /* Slightly darker green on hover */
        transform: translateY(-2px); /* Lift effect on hover */
    }
    
    .stButton button:active {
        background-color: #3e8e41; /* Even darker green on click */
        transform: translateY(0); /* Return to original position */
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Create buttons
if st.sidebar.button(" Introduction "):
    selected_button = "Introduction"

if st.sidebar.button(" Exploratory Data Analysis (EDA)"):
    selected_button = "EDA"

if st.sidebar.button("Weather and Time Impact on Accident Severity"):
    selected_button = "Weather and Time Impact on Accident Severity"

if st.sidebar.button("Clustering Accident Patterns by Location and Traffic Conditions"):
    selected_button = "Clustering Accident Patterns by Location and Traffic Conditions"

if st.sidebar.button("Daily Severity Forecasting by Location"):
    selected_button = "Daily Severity Forecasting by Location"

if st.sidebar.button("Location-Based Traffic Pattern Forecasting"):
    selected_button = "Location-Based Traffic Pattern Forecasting"


# Handle each button selection
if selected_button == "Introduction":
    st.header("About Dataset")
    st.write("""
        **Project Description:**
        This project, "RoadSense," is focused on developing advanced predictive models using machine learning techniques to analyze and forecast various aspects of road traffic accidents across the United States. By leveraging this extensive dataset, we aim to uncover insights and patterns that can help improve road safety and reduce accident severity.

        **Data Source:**
        The data used is an open-source dataset from Kaggle for our project analysis. This dataset encompasses accident records across 49 states in the USA, spanning from February 2016 to March 2023. The data were collected via various APIs that stream traffic incident information from multiple sources, including the US and state departments of transportation, law enforcement agencies, traffic cameras, and road sensors. The dataset includes approximately 7.7 million accident records. For more details, you can access the dataset here:

        **Kaggle**: [https://www.kaggle.com/datasets/sobhanmoosavi/us-accidents?resource=download](https://www.kaggle.com/datasets/sobhanmoosavi/us-accidents?resource=download)

        **Citations:**
        - Moosavi, Sobhan, Mohammad Hossein Samavatian, Srinivasan Parthasarathy, and Rajiv Ramnath. “A Countrywide Traffic Accident Dataset.”, 2019.
        - Moosavi, Sobhan, Mohammad Hossein Samavatian, Srinivasan Parthasarathy, Radu Teodorescu, and Rajiv Ramnath. "Accident Risk Prediction based on Heterogeneous Sparse Data: New Dataset and Insights." In proceedings of the 27th ACM SIGSPATIAL International Conference on Advances in Geographic Information Systems, ACM, 2019.
        """)
    
elif selected_button == "EDA":
    st.header("Exploratory Data Analysis (EDA)")                                                                                                                                                          
    tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs(["**Accidents by State, City, Zipcode and Days**", "**Accident Locations and Infrastructure**", "**Correlation Matrix**", "**Distribution of Weather Features**", "**Accident Severity Distribution**", "**Temperature Trends**", "**Visibility Trends**"])

    
    # Create tabs for Weather and Time Impact   
    with tab1:
        st.image(f"{pathname}EDA1.jpeg", caption="By State")
        st.image(f"{pathname}EDA2.jpeg", caption="By City")
        st.image(f"{pathname}EDA3.jpeg", caption="By Zipcode")
        st.image(f"{pathname}EDA10.jpeg", caption="By Days")

    with tab2:
        st.image(f"{pathname}EDA4.jpeg", caption="Accident Locations")
        st.image("Presence_of_Traffic_Signals.png", caption="Presence of Traffic Signals")
        st.image("Traffic_Amenity.png", caption="Presence of Amenities")

    
    with tab3:
        st.image(f"{pathname}EDA5.jpeg", caption="Correlation Matrix")
    
    with tab4:
        st.image(f"{pathname}EDA7.jpeg", caption="Distribution of Weather Features")
        st.image(f"{pathname}EDA8.jpeg", caption="Distribution of Weather Features")
        st.image(f"{pathname}WindSpeedVariation_Monthly.png", caption="WindSpeed Variation Monthly")
        st.image(f"{pathname}WindSpeedVariation_Yearly.png", caption="WindSpeed Variation Yearly")
        st.image(f"{pathname}EDA9.jpeg", caption="Count of Wind Directions")

    with tab5:
        with open("Accident_Severity_pie.html", "r") as f:
            html_content = f.read()
        st.subheader("Interact with the Graph ⏯️ ")
        st.components.v1.html(html_content, height=800)
        

    with tab6:    
        with open("temperature_trends.html", "r") as f:
            html_content = f.read()
        st.subheader("Interact with the Graph ⏯️ ")
        st.components.v1.html(html_content, height=800)

        with open("Avg_Temp_Day_Night.html", "r") as f:
            html_content = f.read()
        st.components.v1.html(html_content, height=800)

    with tab7:
        with open("visibility_trends.html", "r") as f:
            html_content = f.read()
        st.subheader("Interact with the Graph ⏯️ ")
        st.components.v1.html(html_content, height=800)

elif selected_button == "Weather and Time Impact on Accident Severity":
    st.header("Weather and Time Impact on Accident Severity")

    # Create tabs for Weather and Time Impact
    tab1, tab2, tab3 = st.tabs(["**Classification Matrix**", " **📍 Feature Importance (PCA) 📍** ", " **💡 What's Happening Here? 💡** "])


    with tab1:
        st.image(f"{pathname}weather_time_output.png", caption="Classification Matrix")
    with tab2:
        st.image(f"{pathname}weather_time_output2.png", caption="Feature Importance (PCA)")

    with tab3:
        st.write("""
            ### 1. **Best Parameters**
            - **Parameters:** `{'n_estimators': 200, 'min_samples_split': 2, 'min_samples_leaf': 1, 'max_depth': 20}`
            - **n_estimators**: Number of trees in the forest.
            - **min_samples_split**: Minimum number of samples required to split an internal node.
            - **min_samples_leaf**: Minimum number of samples in a leaf node.
            - **max_depth**: Maximum depth of the tree.

            ### 2. **Accuracy**
            - **Accuracy: 88% (0.88)**

            ### 3. **Classification Report**
            - **Precision**: Proportion of true positives relative to total predicted positives.
            - **Recall**: Proportion of true positives relative to actual total positives.
            - **F1-score**: Harmonic mean of precision and recall.

            ### 4. **PCA Components Contribution**
            - **PC1 to PC6**: Principal components ranked by variance they explain.

            ### 5. **Confusion Matrix**
            - **Diagonal Elements**: True positives.
            - **Off-Diagonal Elements**: Misclassifications.

            ### **Overall Performance**
            - **Strengths**: Good performance on Classes 1 and 4.
            - **Weaknesses**: Struggles with Class 2.
            """)

elif selected_button == "Clustering Accident Patterns by Location and Traffic Conditions":
    st.header("Clustering Accident Patterns by Location and Traffic Conditions")

    # Create tabs for Clustering
    tab1, tab2 = st.tabs(["**Output**", "**💡 What's Happening Here? 💡** "])

    with tab1:
        st.image(f"{pathname}clustering_output.png", caption="Clustering Output")

    with tab2:
        st.write("""
            ### **Model Explanation**
            - **Principal Component Analysis (PCA)**: Reduces dimensionality while retaining variance.
            - **K-Means Clustering**: Groups data into clusters based on feature similarity.
            - **DBSCAN Clustering**: Identifies clusters based on density.

            ### **Output Explanation**
            - **PCA Components with Feature Contributions**: Principal components and their feature contributions.
            - **Silhouette Scores**: Measures of cluster separation.

            ### **Sample Data Predictions**
            - Shows how samples are clustered by K-Means and DBSCAN.
            """)

elif selected_button == "Daily Severity Forecasting by Location":
    st.header("Daily Severity Forecasting by Location")

    # Create tabs for Forecasting
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["**Forecast of Average Severity**", "**Trend Over Years**", "**Average Severity by Day of the Week and Yearly**", "**Actual vs. Predicted Accident Severity**", " **💡 What's Happening here? 💡** "])


    with tab1:
        st.image(f"{pathname}severity_forecast.png", caption="Forecast of Average Severity")

    with tab2:
        st.image(f"{pathname}trend_over_years.png", caption="Trend Over Years")

    with tab3:
        st.image(f"{pathname}severity_by_day_of_week_year.png", caption="Average Severity by Day of the Week and Yearly")

    with tab4:
        st.image(f"{pathname}actual_vs_predicted.png", caption="Actual vs. Predicted Accident Severity")
    with tab5:
        st.write("""
        ### **Forecast of Average Severity**
        - **X-Axis (Date)**: Timeline from 2016 to 2024.
        - **Y-Axis (Average Severity)**: Forecasted average severity values.

        ### **Trend Over Years**
        - **X-Axis (Years)**: Years from 2016 to 2024.
        - **Y-Axis (Average Severity)**: Average severity values over years.

        ### **Average Severity by Day of the Week**
        - **X-Axis (Days of the Week)**: From Sunday to Saturday.
        - **Y-Axis (Average Severity)**: Average severity per day.

        ### **Average Severity by Day of the Year**
        - **X-Axis (Day of the Year)**: Each day from January 1 to December 31.
        - **Y-Axis (Average Severity)**: Severity values for each day.

        ### **Actual vs. Predicted Accident Severity**
        - **X-Axis (Date)**: Timeline from 2016 to 2024.
        - **Y-Axis (Average Severity)**: Comparison of actual and predicted severity values.
        """)

elif selected_button == "Location-Based Traffic Pattern Forecasting":
    st.header("Location-Based Traffic Pattern Forecasting")

    # Create tabs for Forecasting
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["**Loss over Epochs**", "**MAE over Epochs**", "**Predictions vs. Actual Values**", "**Residuals Plot**", " **💡 What's Happening Here? 💡** "])


    with tab1:
        st.image(f"{pathname}loss_over_epochs.png", caption="Loss over Epochs")

    with tab2:
        st.image(f"{pathname}mae_over_epochs.png", caption="MAE over Epochs")

    with tab3:
        st.image(f"{pathname}predictions_vs_actual2.png", caption="Predictions vs. Actual Values")

    with tab4:
        st.image(f"{pathname}residuals_plot.png", caption="Residuals Plot")

    with tab5:
        st.write("""
            ### Training Summary
            **Epochs:**
            - The model is trained over 100 epochs, with the output showing the results for the first six epochs.
            - Each epoch includes:
                - **Training Steps:** 2493 steps per epoch.
                - **Loss:** The training loss decreases over the epochs, indicating improved model performance.
                - **Mean Absolute Error (MAE):** A metric that also decreases, showing that the model's predictions are getting closer to the actual values.
                - **Validation Loss and MAE:** These metrics are also reported, providing insight into how well the model generalizes to unseen data.

            **Final Test Metrics:**
            - After training, the model is evaluated on a test set, yielding a test MAE of 0.31.
            - Sample predictions are provided alongside the true values, showing how closely the model's predictions align with the actual values.

            ### Visualizations
            **Loss over Epochs:**
            - A plot showing the training and validation loss across epochs.
            - The training loss should ideally decrease over time, indicating that the model is learning.

            **MAE over Epochs:**
            - A plot displaying the training and validation MAE over epochs.
            - Similar to the loss plot, a decreasing trend in MAE indicates improved model performance.

            **Predictions vs. Actual Values:**
            - A scatter plot comparing predicted values against actual values.
            - This visualization helps assess how well the model's predictions match the true values, with a closer alignment indicating better performance.

            **Residuals Plot:**
            - A plot showing the residuals (the difference between actual and predicted values) against the actual values.
            - This plot helps identify any patterns in the residuals, which can indicate model bias or areas where the model may not be performing well.
            """)

