# Build-own-Simple-Linear-Regression-Model

## Overview

This project demonstrates the implementation of a **Simple Linear Regression Model** to predict sales based on advertising budgets. Using the Advertising dataset, we explore the relationship between TV advertising spend and sales, train a regression model, and evaluate its performance. The project is designed to highlight both conceptual and practical aspects of regression analysis.

---

## Author

- **Name**: Himel Sarder  
- **Email**: [info.himelcse@gmail.com](mailto:info.himelcse@gmail.com)  
- **GitHub**: [Himel-Sarder](https://github.com/Himel-Sarder)

---

## Dataset Overview

The dataset contains information on advertising budgets across three media types and their corresponding sales:

- **TV**: Budget allocated to TV advertisements (in thousands of dollars).  
- **Radio**: Budget allocated to Radio advertisements (in thousands of dollars).  
- **Newspaper**: Budget allocated to Newspaper advertisements (in thousands of dollars).  
- **Sales**: Generated sales (in thousands of units).

### Sample Data
| TV     | Radio  | Newspaper | Sales |
|--------|--------|-----------|-------|
| 230.1  | 37.8   | 69.2      | 22.1  |
| 44.5   | 39.3   | 45.1      | 10.4  |
| 17.2   | 45.9   | 69.3      | 12.0  |
| 151.5  | 41.3   | 58.5      | 16.5  |
| 180.8  | 10.8   | 58.4      | 17.9  |

---

## Objectives

1. **Data Analysis**: Explore the relationship between advertising spends and sales.  
2. **Model Building**: Develop a simple linear regression model to predict sales based on TV advertising spend.  
3. **Evaluation**: Measure model performance using statistical metrics like RMSE and R-squared.  
4. **Visualization**: Plot the best-fit line and residuals to validate assumptions.  

---

## Methodology

### 1. **Exploratory Data Analysis (EDA)**  
Scatter plots and bar plots were used to visualize relationships between advertising budgets and sales. Key observations include:  
- TV advertising has the strongest correlation with sales.  
- Outliers were minimal, and the data was clean (no missing values).  

### 2. **Custom Linear Regression Implementation**  
A custom regression model was implemented from scratch using Python.  

#### Code Snippet:  
```python
class MyLinearRegression:
    def __init__(self):
        self.m = None
        self.c = None
        
    def fit(self, X_train, y_train):
        num = sum((X_train - X_train.mean()) * (y_train - y_train.mean()))
        den = sum((X_train - X_train.mean()) ** 2)
        self.m = num / den
        self.c = y_train.mean() - self.m * X_train.mean()
        
    def predict(self, X_test):
        return self.m * X_test + self.c
```

### 3. **Model Training and Evaluation**  
- **Training**: The dataset was split into 70% training and 30% testing subsets using `train_test_split`.  
- **Evaluation Metrics**:  
  - **Root Mean Squared Error (RMSE)**: Measures prediction accuracy.  
  - **R-squared**: Indicates the proportion of variance explained by the model.  

#### Key Results:  
- **RMSE**: 2.019  
- **R-squared**: 0.816  

### 4. **Statistical Analysis with Statsmodels**  
Using `statsmodels`, we validated the model and obtained the following summary:  

| Metric           | Value  |  
|-------------------|--------|  
| Coefficient (TV) | 0.0555 |  
| Intercept        | 6.9748 |  
| P-value          | 0.000  |  
| R-squared        | 0.812  |  

---

## Conclusion

- TV advertising has a significant positive impact on sales, explaining over 81% of the variance.  
- The custom regression model performs well and aligns with results from standard libraries like `statsmodels`.  
- Future work could involve extending the analysis to include Radio and Newspaper budgets.

---

## References

- [Kaggle Advertising Dataset](https://www.kaggle.com/)  
- Python libraries: `pandas`, `numpy`, `matplotlib`, `seaborn`, `sklearn`, `statsmodels`  

For detailed analysis and code, visit [GitHub](https://github.com/Himel-Sarder).
😊
