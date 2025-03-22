# Retail Customer Churn Prediction

## 📌 Project Overview
Customer churn is a critical issue in the retail industry. Predicting churn helps retail businesses retain customers by identifying at-risk individuals early and offering targeted interventions. This project aims to develop a machine learning model to predict customer churn in a retail setting, leveraging retail analytics to inform data-driven decisions.

The goal of this project is to predict whether a customer will return to the business, enabling the company to take targeted actions to increase customer loyalty and minimize churn. By exploring various machine learning models and analytical techniques, I aim to create a reliable predictive model that aids in customer retention strategies.

## 📊 Dataset
- **Source:** Open-source retail dataset
- **Purpose:** To analyze customer behavior and predict churn

## 🔍 Exploratory Data Analysis (EDA)
I have extensively analyzed the dataset to understand:
- **Univariate Analysis:** Examining individual variables using histograms, box plots, and summary statistics.
- **Bivariate Analysis:** Understanding relationships between two variables using scatter plots, line graphs, and correlation heatmaps.
- **Multivariate Analysis:** Exploring complex interactions between multiple features.

**Techniques Used:**
- Box plots, scatter plots, line and bar graphs, correlation plots
- Box-Cox transformation for handling skewed variables
- Chi-square test for categorical variables
- Multicollinearity test to identify correlated features

## 🤖 Machine Learning Models Used
To compare the performance of different models, I have trained multiple classification algorithms, including:
- **Logistic Regression**
- **Random Forest**
- **K-Nearest Neighbors (KNN)**
- **XGBoost**
- **Naïve Bayes**

Additionally, I experimented with:
- **SMOTEEN**: Used for handling class imbalances, although the dataset is balanced
- **Principal Component Analysis (PCA)**: Applied to observe changes in accuracy

## 🚀 Model Evaluation
Each model's performance is evaluated based on:
- Accuracy
- Precision, Recall, and F1-score
- AUC-ROC Curve
- Feature importance analysis

## 🛠 Technologies Used
- **Programming Language:** Python
- **Libraries:** Pandas, NumPy, Scikit-learn, Matplotlib, Seaborn, XGBoost, Imbalanced-learn
- **Notebook Environment:** Jupyter Notebook

## 📂 Project Structure
```
📦 Retail_Customer_Churn
├── 📄 notebook_final.ipynb   # Jupyter Notebook with full analysis and model building
├── 📄 Dataset_for_Capstone.csv  # Dataset used for training models
├── 📄 README.md  # Project documentation
```

## 🎯 How to Run the Project
1. Clone the repository:
   ```sh
   git clone git@github.com:your-username/Capstone.git
   ```
2. Navigate to the project folder:
   ```sh
   cd Capstone
   ```
3. Open `notebook_final.ipynb` in Jupyter Notebook and run all cells to see the analysis and model results.

## 🚀 Future Enhancements
- Feature engineering to improve model performance
- Hyperparameter tuning for optimal model selection
- Deployment of the model using Flask or FastAPI

## 🤝 Contributions
This project is solely developed by **Pritam Sarkar**. Feel free to fork the repository and contribute to further enhancements!

## 📧 Contact
For any queries, reach out via [LinkedIn](https://www.linkedin.com/in/your-profile) or email: your-email@example.com.

---

✅ **Now your GitHub repository will look professional with this README!** 🎉 Let me know if you'd like any modifications. 🚀

