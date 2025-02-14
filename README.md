**RoadSense: Advanced Predictive Modeling for Traffic Safety**  

The RoadSense project leverages machine learning and deep learning techniques to predict the occurrence and severity of traffic accidents using the US-Accidents dataset, which contains over 2.25 million records. The project employs models like Random Forest, LSTM, CNNs, and Prophet to analyze key factors such as weather, traffic conditions, and time of day. The project utilized Python, Scikit-learn, TensorFlow, and Tableau for data processing, feature engineering, and visualization. With a focus on actionable insights for traffic safety, the application was deployed using Streamlit, providing real-time predictive capabilities. In-depth feature engineering, including dimensionality reduction with PCA and handling class imbalance using SMOTE, resulted in the model achieving 86% accuracy, with a strong emphasis on interpretability to drive better policy and safety decisions.


Access the RoadSense App here: https://roadsense-trafficsaftey.streamlit.app/
# 🚦 RoadSense: Advanced Predictive Modeling for Traffic Safety

![Screenshot 2025-02-13 at 8 08 19 PM](https://github.com/user-attachments/assets/3793d03b-8597-4ff0-af5b-427ade836b0d)


## 🏆 Project Overview
RoadSense is a machine learning-driven initiative designed to predict and analyze traffic accident severity using the US-Accidents dataset, which contains over **2.25 million records** from 2016 to 2023. This project employs **Random Forest, LSTM, CNNs, Prophet**, and clustering techniques to assess accident patterns and severity based on weather, traffic, and temporal data.

The project utilized **Python, Scikit-learn, TensorFlow, and Tableau** for **data processing, feature engineering, and visualization**. With a focus on actionable insights for traffic safety, the application was deployed using **Streamlit**, providing real-time predictive capabilities. In-depth feature engineering, including **dimensionality reduction** with **PCA **and **handling class imbalance using SMOTE**, resulted in the model achieving **86% accuracy**, with a strong emphasis on interpretability to drive better policy and safety decisions.

The goal is to provide **actionable insights for policymakers, urban planners, and traffic safety authorities** by identifying high-risk factors and improving accident prevention strategies through **real-time predictive analytics**.

## 📌 Table of Contents
1. [Introduction](#introduction)
2. [Dataset Overview](#dataset-overview)
3. [Data Preprocessing & Feature Engineering](#data-preprocessing--feature-engineering)
4. [Exploratory Data Analysis (EDA)](#exploratory-data-analysis-eda)
5. [Predictive Modeling](#predictive-modeling)
6. [Model Interpretability](#model-interpretability)
7. [Results & Insights](#results--insights)
8. [Conclusion & Future Work](#conclusion--future-work)

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

## 🔹 Data Preprocessing & Feature Engineering
✔ **Handling Missing Values**: 
   - Imputation techniques used for missing weather and location data.
   - Precipitation, Wind Speed, and Humidity filled with median values.
✔ **Feature Engineering**:
   - Created **‘Accident Hotspot Score’** based on high-frequency accident locations.
   - Derived **‘Time of Day Segmentation’** (Morning, Afternoon, Evening, Night).
   - Added **binary flags** for adverse weather conditions.
✔ **Dimensionality Reduction**:
   - **Principal Component Analysis (PCA)** applied for optimization.
✔ **Class Imbalance Handling**:
   - **Synthetic Minority Over-sampling Technique (SMOTE)** used to balance severity levels.

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

## 🔹 Predictive Modeling
🧠 **Machine Learning & Deep Learning Models Used**:

### **Classification Models for Accident Severity**
- **Random Forest** (Best Performing - **92% Accuracy**)
- **XGBoost**
- **Logistic Regression**
- **Support Vector Machines (SVM)**

### **Clustering Models for High-Risk Areas**
- **K-Means**
- **DBSCAN (Density-Based Spatial Clustering)**

### **Time Series Forecasting for Accident Trends**
- **Prophet (Daily Severity Forecasting by Location)**
- **LSTM (Long Short-Term Memory Networks for Traffic Flow Prediction)**

### **Key Model Performance Insights**
📌 **Random Forest outperformed all models** with:
- **F1-Score: 0.91, ROC-AUC: 0.95**
- **Most influential features**: Weather Conditions, Visibility, Time of Day

📌 **Prophet identified seasonal accident trends**:
- **Summer & Winter months show peak accident severity**.
- **Weekdays have higher accident frequencies than weekends**.

📌 **K-Means & DBSCAN uncovered accident-prone zones**:
- **High-risk zones found near major intersections & highways**.
- **DBSCAN effectively detected dense accident clusters**.

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
🏆 **Final Model Performance Summary**
| Model | Accuracy | F1-Score | ROC-AUC |
|--------|---------|---------|---------|
| **Random Forest** | **92%** | **0.91** | **0.95** |
| XGBoost | 88% | 0.87 | 0.91 |
| SVM | 85% | 0.84 | 0.89 |
| Logistic Regression | 81% | 0.79 | 0.84 |

📌 **Key Takeaways**:
- **Accident severity is highly correlated with visibility and weather conditions**.
- **Urban intersections are more prone to frequent but lower severity accidents**.
- **Highway accidents tend to be more severe** due to high-speed collisions.
- **Machine learning models effectively identify high-risk factors and locations**.

---

## 🔹 Conclusion & Future Work
🚀 **Next Steps for RoadSense**:
✅ **Integrate real-time traffic & weather data** for dynamic risk analysis.
✅ **Enhance deep learning models (CNNs, Autoencoders) for improved predictions**.
✅ **Develop a web-based interactive dashboard for traffic authorities**.
✅ **Collaborate with municipalities to implement data-driven road safety measures**.

---

## 📌 Get in Touch
📧 pranavsharma1395@gmail.com  
📞 +1 (778) 598-6373  
🌐 [LinkedIn](https://www.linkedin.com/in/pranav-harish-sharma/)  
🔗 **Project Repository**: [GitHub Link](https://github.com/user/RoadSense)

💬 *"Errors using inadequate data are much less than those using no data at all." - Charles Babbage"* 🚗💨
