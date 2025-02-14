# 🚦 RoadSense: Advanced Predictive Modeling for Traffic Safety

Access the RoadSense App here: https://roadsense-trafficsaftey.streamlit.app/

![Screenshot 2025-02-13 at 8 08 19 PM](https://github.com/user-attachments/assets/3793d03b-8597-4ff0-af5b-427ade836b0d)


## 🏆 Project Overview
RoadSense is a machine learning-driven initiative designed to predict and analyze traffic accident severity using the US-Accidents dataset, which contains over **2.25 million records** from 2016 to 2023. This project employs **Random Forest, LSTM, CNNs, Prophet**, and clustering techniques to assess accident patterns and severity based on weather, traffic, and temporal data.

The project utilized **Python, Scikit-learn, TensorFlow, and Tableau** for **data processing, feature engineering, and visualization**. With a focus on actionable insights for traffic safety, the application was deployed using **Streamlit**, providing real-time predictive capabilities. In-depth feature engineering, including **dimensionality reduction** with **PCA **and **handling class imbalance using SMOTE**, resulted in the model achieving **86% accuracy**, with a strong emphasis on interpretability to drive better policy and safety decisions.

The goal is to provide **actionable insights for policymakers, urban planners, and traffic safety authorities** by identifying high-risk factors and improving accident prevention strategies through **real-time predictive analytics**.

## 📌 Table of Contents
1. [Introduction](#introduction)
2. [Problem Statement](#problem-statement)
3. [Project Goals & Objectives](#project-goals--objectives)
4. [Dataset Overview](#dataset-overview)
5. [Data Preprocessing & Feature Engineering](#data-preprocessing--feature-engineering)
6. [Exploratory Data Analysis (EDA)](#exploratory-data-analysis-eda)
7. [Predictive Modeling](#predictive-modeling)
   - [Random Forest Classifier](#random-forest-classifier)
   - [Unsupervised Learning: K-Means & DBSCAN](#unsupervised-learning-k-means--dbscan)
   - [Time Series Forecasting: Prophet](#time-series-forecasting-prophet)
   - [Deep Learning: Long Short-Term Memory (LSTM)](#deep-learning-long-short-term-memory-lstm)
8. [Results & Insights](#results--insights)
9. [Model Interpretability](#model-interpretability)
10. [Conclusion & Future Work](#conclusion--future-work)

---

## 🔹 Introduction
Traffic accidents are a major global concern, causing approximately **1.35 million deaths** annually. Traditional safety measures are often reactive and costly. **RoadSense leverages machine learning** to predict accident severity based on **weather, traffic conditions, time, and geographical attributes**, allowing proactive accident prevention and improved traffic management.

### **Key Research Questions**
- What are the most critical factors influencing **severe** traffic accidents?
- How can **real-time weather and traffic data** enhance predictive accuracy?
- What are the **spatiotemporal patterns** of traffic accidents?

---

## 🔹 Dataset Overview
📂 **Dataset**: US-Accidents (2016 - 2023) ([Kaggle](https://www.kaggle.com/datasets/sobhanmoosavi/us-accidents))  
📊 **Size**: 2.25 million records, 46 features  
🌍 **Geographical Coverage**: Entire contiguous US states  

### **Key Features**
- **Severity**: Categorized into 4 levels (1 = Least Severe, 4 = Most Severe)
- **Weather Conditions**: Rain, Snow, Fog, Temperature, Wind Speed, Visibility
- **Time & Location**: Date, Time of Day, City, State, Road Type
- **Traffic Infrastructure**: Presence of Traffic Signals, Stop Signs, Crosswalks

---

## 🔹 Exploratory Data Analysis (EDA)
📌 **Key Findings**:
- **California (22,702 accidents) has the highest accident frequency**.
- **Miami (2,410 accidents) has the highest accident density in California**.
- **High accident rates during peak traffic hours (7-9 AM, 4-6 PM)**.
- **Weather Factors:**
  - **Low visibility and heavy precipitation significantly increase severity**.
  - **Temperature extremes correlate with high accident counts**.
- **Traffic Signals & Stop Signs:**
  - Locations **without signals have higher accident severity**.
- **Urban vs. Rural:**
  - **Urban areas** show more frequent but lower severity accidents.
  - **Rural highways** experience fewer but **higher severity accidents**.

---

## 🔹 Data Preprocessing & Feature Engineering
-   ✔ **Handling Missing Values**: 
   - Imputation techniques used for missing weather and location data.
   - Precipitation, Wind Speed, and Humidity filled with median values.
- ✔ **Feature Engineering**:
   - Created **‘Accident Hotspot Score’** based on high-frequency accident locations.
   - Derived **‘Time of Day Segmentation’** (Morning, Afternoon, Evening, Night).
   - Added **binary flags** for adverse weather conditions.
- ✔ **Dimensionality Reduction**:
   - **Principal Component Analysis (PCA)** applied for optimization.
- ✔ **Class Imbalance Handling**:
   - **Synthetic Minority Over-sampling Technique (SMOTE)** used to balance severity levels.

---

## 🔹 Predictive Modeling
🧠 **Machine Learning Models Used**:

### **Accident Severity Classification Models**
- **Random Forest** (Best Performing - **86% Accuracy**)
- **Gradient Boosting Machines (GBM)**
- **Deep Neural Networks (DNNs)**
- **Convolutional Neural Networks (CNNs)**
- **Long Short-Term Memory Networks (LSTM)**

### **High-Risk Zone Clustering Models**
- **K-Means**
- **DBSCAN (Density-Based Clustering)**

### **Time Series Forecasting for Accident Trends**
- **Prophet** (Daily severity forecasting by location)

📌 **Performance Highlights**:
- **Random Forest achieved the best classification performance**:
  - **F1-Score: 0.91, ROC-AUC: 0.95**
  - **Most influential factors**: Weather Conditions, Visibility, Time of Day
- **K-Means & DBSCAN identified high-risk zones**:
  - **Clusters of high accident severity in major intersections and highways.**

---

## 🔹 Model Interpretability
📌 **Feature Importance Analysis (Random Forest & XGBoost)**
- **Top Contributing Features to Severity Predictions**:
  - **Weather Conditions** (Rain, Snow, Fog, Low Visibility)
  - **Time of Day** (Rush Hours show more severe accidents)
  - **Road Type** (Highways have the highest severity levels)

📌 **SHAP (SHapley Additive Explanations) Analysis**
- Provides **human-interpretable explanations** of model predictions.
- Highlights how features like **traffic signals & weather influence accident severity**.

📌 **LIME (Local Interpretable Model-Agnostic Explanations)**
- Used to analyze **individual accident case predictions**.

---
## 🔹 Results & Insights
### **Results Overview**
This section summarizes the performance of the various machine learning models developed to predict traffic accident severity, cluster accident patterns, and forecast daily severity based on historical data.

### **Model Performance**
| Model | Type | Metrics | Strengths | Weaknesses |
|--------|---------|-------------------------|--------------------------------------------------|--------------------------------------------------|
| **Random Forest** | Classification | - Accuracy: **86%**  | - Strong performance for **Classes 1 and 4** | - Difficulty distinguishing between **Classes 2 and 3** |
|  |  | - Precision: Highest for **Class 1 and 4 (91%)** | - High recall for **Class 1 (99%)** | - Overlapping feature patterns affecting prediction accuracy |
|  |  | - F1-Score: High for **Class 1 and 4** | - Performs well in predicting **low and high severity accidents** |  |
| **K-Means Clustering** | Clustering | - Silhouette Score: **0.65** | - Well-defined and **separated clusters** | - Sensitive to **initial centroid selection** |
|  |  |  | - Effective in identifying general clusters | - Struggles with **non-globular clusters** |
| **DBSCAN Clustering** | Clustering | - No fixed metric due to algorithm nature | - Identifies clusters with **varying densities** | - Performance depends on **eps and min_samples** |
|  |  |  | - Captures **outliers and core points** effectively |  |
| **Prophet** | Time Series Forecasting | - Accuracy: Strong **alignment of actual vs. predicted** | - Captures **temporal patterns and trends** | - Depends on **data quality and availability** |
|  |  |  | - Provides insights into **seasonal patterns** | - Choice of **regressors impacts accuracy** |
| **LSTM** | Time Series Analysis | - **MAE: 0.31** | - Accurate predictions of **traffic patterns** | - Requires **significant data preprocessing** |
|  |  | - Loss: **Decreased over 100 epochs** | - Effective in **learning complex temporal dependencies** | - Sensitive to **hyperparameters and model architecture** |

📌 **Key Takeaways:**
- **Random Forest performed best for classification**, accurately predicting accident severity with **86% accuracy**.
- **K-Means successfully identified general accident zones**, but struggled with **non-globular clusters**.
- **DBSCAN effectively detected accident hotspots**, particularly in **dense urban areas**.
- **Prophet forecasted seasonal accident patterns**, aiding in long-term **traffic safety planning**.
- **LSTM demonstrated strong predictive capabilities**, particularly in understanding **traffic flow trends** over time.

---

## 🔹 Conclusion & Future Work
🚀 **Next Steps for RoadSense**:
✅ **Integrate real-time traffic & weather data** for dynamic risk analysis.
✅ **Extend the RoadSense App with a web-based interactive dashboard for traffic authorities**.
✅ **Collaborate with municipalities to implement data-driven road safety measures**.

---

## 📌 Get in Touch
📧 pranavsharma1395@gmail.com  
🌐 [LinkedIn](https://www.linkedin.com/in/pranav-harish-sharma/)  
🔗 **Project Repository**: [GitHub Link](https://github.com/user/RoadSense)

💬 *"Errors using inadequate data are much less than those using no data at all." - Charles Babbage"* 🚗💨
