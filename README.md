# 🚦 RoadSense: Advanced Predictive Modeling for Traffic Safety

Access the RoadSense App here: https://roadsense-trafficsaftey.streamlit.app/


![RoadSense Github Upload_page-0001](https://github.com/user-attachments/assets/b91c50c8-b764-4b70-a358-a9eee308742e)


## 🏆 Project Overview
RoadSense is a machine learning-driven initiative designed to predict and analyze traffic accident severity using the US-Accidents dataset, which contains over **2.25 million records** from 2016 to 2023. This project employs **Random Forest, LSTM, CNNs, Prophet**, and clustering techniques to assess accident patterns and severity based on weather, traffic, and temporal data.

The project utilized **Python, Scikit-learn, TensorFlow, and Tableau** for **data processing, feature engineering, and visualization**. With a focus on actionable insights for traffic safety, the application was deployed using **Streamlit**, providing real-time predictive capabilities. In-depth feature engineering, including **dimensionality reduction** with **PCA **and **handling class imbalance using SMOTE**, resulted in the model achieving **86% accuracy**, with a strong emphasis on interpretability to drive better policy and safety decisions.

The goal is to provide **actionable insights for policymakers, urban planners, and traffic safety authorities** by identifying high-risk factors and improving accident prevention strategies through **real-time predictive analytics**.
![RoadSense Github Upload_page-0002](https://github.com/user-attachments/assets/d4267467-fb64-4ac1-aeb7-4b76ac3b743b)

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
## 🔹 Problem Statement
Traffic accidents remain a significant cause of fatalities and economic loss across the world. Understanding the key factors contributing to accident severity and occurrence is crucial for effective prevention. Traditional accident analysis relies on historical data and human judgment, which often lacks precision and scalability. 

This project aims to develop **machine learning models** that can:
✔ **Predict accident severity** based on historical accident reports, weather data, and road conditions.  
✔ **Identify key contributing factors** to severe accidents, enabling better decision-making for traffic management authorities.  
✔ **Analyze accident-prone areas** using clustering techniques, allowing policymakers to allocate resources efficiently.  
✔ **Forecast accident severity trends** to help urban planners and law enforcement take proactive measures.  

By leveraging **supervised and unsupervised learning techniques**, the project aims to provide a comprehensive solution for predicting and mitigating traffic accidents.

---

## 🔹 Project Goals & Objectives
The primary objective of **RoadSense** is to build a **scalable and interpretable machine learning pipeline** to enhance road safety through predictive modeling.

### **Key Goals:**
✔ **Develop Accurate Models:** Train and evaluate **Random Forest, K-Means, DBSCAN, Prophet, and LSTM models** to predict accident severity and detect high-risk zones.
✔ **Improve Interpretability:** Use feature importance analysis, SHAP values, and visual tools to ensure model insights can be **understood by policymakers**.
✔ **Address Class Imbalance:** Implement **Synthetic Minority Oversampling Technique (SMOTE)** to balance data and improve model reliability.
✔ **Create an Automated Pipeline:** Design a **fully automated system** that integrates real-time accident reports, weather updates, and road conditions.
✔ **Enable Actionable Insights:** Provide **data-driven recommendations** for government agencies and urban planners to improve road safety.

By achieving these goals, **RoadSense** will contribute to reducing road accidents and improving public safety through AI-powered analysis.

---

## 🔹 Dataset Overview
📂 **Dataset**: US-Accidents (2016 - 2023) ([Kaggle](https://www.kaggle.com/datasets/sobhanmoosavi/us-accidents))  
📊 **Size**: 2.25 million records, 46 features  
🌍 **Geographical Coverage**: Entire contiguous US states  
![RoadSense Github Upload_page-0004](https://github.com/user-attachments/assets/09b5beda-a5d5-4345-a08c-8ba878b7ca27)


### **Key Features**
- **Severity**: Categorized into 4 levels (1 = Least Severe, 4 = Most Severe)
- **Weather Conditions**: Rain, Snow, Fog, Temperature, Wind Speed, Visibility
- **Time & Location**: Date, Time of Day, City, State, Road Type
- **Traffic Infrastructure**: Presence of Traffic Signals, Stop Signs, Crosswalks

---

## 🔹 Exploratory Data Analysis (EDA)
![RoadSense Github Upload_page-0005](https://github.com/user-attachments/assets/430ed7c7-e0fc-478c-8cf4-5382b0530d9b)

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
![RoadSense Github Upload_page-0006](https://github.com/user-attachments/assets/c7ab5ffb-36b2-43f2-ba9d-f904b3664670)
![RoadSense Github Upload_page-0007](https://github.com/user-attachments/assets/9d754980-53ca-4e77-bf11-e271620bdebb)
![RoadSense Github Upload_page-0008](https://github.com/user-attachments/assets/ee118d85-2535-4932-9de3-5601fbbe8c61)
![RoadSense Github Upload_page-0009](https://github.com/user-attachments/assets/651e007a-2644-4857-bd15-4477e21bb15e)


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

The predictive modeling phase of **RoadSense** focuses on classifying accident severity, clustering high-risk zones, and forecasting future accident severity trends. This section outlines the methodologies used and their respective performances.

### **1️⃣ Random Forest Classifier (Accident Severity Prediction)**

📌 **Objective:** Predict accident severity based on weather, traffic, and location features.

#### **Data Preprocessing & Feature Engineering**
✔ **Dataset:** Cleaned and sampled US accident data with key features:
   - **Start_Time, Temperature(F), Humidity(%), Pressure(in), Visibility(mi), Wind_Speed(mph)**
✔ **Feature Engineering:**
   - Extracted time-based features: **Start_Hour, Start_Minute, Start_Second**
   - Encoded categorical variables (e.g., **Weather_Condition**) using **Label Encoding**
✔ **Dimensionality Reduction:** Applied **Principal Component Analysis (PCA)** to retain **six principal components**
✔ **Class Imbalance Handling:** Implemented **Synthetic Minority Oversampling Technique (SMOTE)** to balance severity classes
✔ **Dataset Splitting:** 80-20 split into **training and testing sets**

#### **Model Training & Hyperparameter Tuning**
✔ **Base Model:** Implemented **Random Forest Classifier** for severity prediction
✔ **Hyperparameter Optimization:** Conducted **Randomized Search Cross-Validation**
   - **n_estimators:** 200
   - **max_depth:** 20
   - **min_samples_split:** 2
   - **min_samples_leaf:** 1

#### **Model Performance & Evaluation**
✔ **Accuracy:** **86%** indicating strong predictive power
✔ **Classification Metrics:**
   - **Precision:** 91% for Class 1 & Class 4
   - **Recall:** **99% for Class 1** (high sensitivity)
   - **F1-Score:** Strong for Class 1 and Class 4, balancing precision and recall
✔ **Confusion Matrix:**
   - **Performs well for low and high-severity classes** (Class 1 and 4)
   - **Struggles with overlapping feature patterns in Classes 2 and 3**

![RoadSense Github Upload_page-0010](https://github.com/user-attachments/assets/d414744a-8a6d-4d89-ac42-5d2d96e98d4c)

---

### **2️⃣ Unsupervised Learning (K-Means & DBSCAN for High-Risk Zones)**

📌 **Objective:** Identify accident-prone regions based on spatial and temporal patterns.

#### **Clustering Results**
✔ **K-Means Performance:**
   - **Silhouette Score:** 0.65 (indicating well-separated clusters)
   - **Effective at identifying general accident-prone zones**
✔ **DBSCAN Performance:**
   - **No fixed metric** due to density-based clustering approach
   - **Effectively identified accident hotspots and outliers**
✔ **Strengths:**
   - **K-Means:** Best for finding **general risk zones**
   - **DBSCAN:** Best for detecting **high-density accident areas and outliers**
✔ **Weaknesses:**
   - **K-Means:** Sensitive to **initial centroid selection**
   - **DBSCAN:** Requires **fine-tuning of eps and min_samples**
![RoadSense Github Upload_page-0011](https://github.com/user-attachments/assets/10f60619-41df-452a-b67a-86fa9d5f7cbf)

---

### **3️⃣ Time Series Forecasting (Prophet for Trend Analysis)**

📌 **Objective:** Forecast accident severity trends over time.

✔ **Performance:** Strong **alignment between actual and predicted values**
✔ **Key Insights:**
   - Captured **seasonal accident patterns** and peak time intervals
   - Effective in predicting **high-risk periods** for accidents
✔ **Limitations:**
   - **Dependent on data quality and availability**
   - **Requires careful selection of regressors for improved accuracy**
![RoadSense Github Upload_page-0012](https://github.com/user-attachments/assets/8655d77a-77a4-4148-a938-ed40ec664229)

---

### **4️⃣ Deep Learning (LSTM for Traffic Pattern Prediction)**

📌 **Objective:** Model complex temporal dependencies for long-term accident prediction.

✔ **Training Process:**
   - Trained over **100 epochs** with batch size **32**
   - **Gradual decrease in loss function**, demonstrating learning progression
✔ **Final Metrics:**
   - **Mean Absolute Error (MAE):** 0.31 (indicating high prediction accuracy)
✔ **Strengths:**
   - **Excels in capturing intricate temporal relationships**
   - **Highly effective for traffic forecasting and severity prediction**
✔ **Weaknesses:**
   - **Requires extensive preprocessing and hyperparameter tuning**
   - **Computationally expensive compared to traditional models**
![RoadSense Github Upload_page-0013](https://github.com/user-attachments/assets/d3edf901-95dd-49fc-845a-2110b5f82bf3)

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
![ThanksNote_page-0001](https://github.com/user-attachments/assets/0b732265-16b7-4201-979c-c4a2fe6f2574)

