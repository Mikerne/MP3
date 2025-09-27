# MP3
MINI PROJECT 3: MACHINE LEARNING FOR ANALYSIS AND PREDICTION 

Created by Valdemar & Micke

## Employee Behavior Analysis - Project Notes

**1) Which are the most decisive factors for quitting a job? Why do people quit their job?**  
The most decisive factors are **OverTime, JobRole, YearsAtCompany, and Work-Life Balance**. People often quit due to **long working hours, lack of growth opportunities, low satisfaction, or mismatched job roles**.

**2) Which work positions and departments are at higher risk of losing employees?**  
**Sales** and **Research & Development** roles have higher attrition, especially positions like **Sales Executive, Research Scientist, and Managers**.

**3) Are employees of different gender paid equally in all departments?**  
In general, **male employees earn slightly more than female employees** in some departments, especially in higher-level positions. Differences are smaller in departments like Human Resources.

**4) Do family status and the distance from work influence the work-life balance?**  
Yes. Employees who are **single or divorced** and those with **longer commuting distances** tend to face more work-life balance challenges.

**5) Does education make people happy (satisfied from the work)?**  
Higher education correlates with **higher satisfaction and higher pay**, but job fit and role responsibilities have a stronger impact than education alone.

**6) Which machine learning methods did you choose to apply in the application and why?**  
- **Random Forest Classifier**: robust with numeric and categorical data, used for predicting attrition.  
- **Ridge Regression**: prevents overfitting, used for predicting MonthlyIncome.  
- **KMeans Clustering**: for grouping employees into clusters and profiling.

**7) How accurate are your solutions of prediction? Explain the meaning of the quality measures.**  
- **Random Forest Classifier**: Accuracy ~85%, F1-score ~0.75; measures how well the model predicts who will leave.  
- **Ridge Regression**: R² ~0.7, RMSE ~$6,000–7,000; R² shows how much salary variation the model explains, RMSE shows average prediction error in dollars.

**8) What could be done for further improvement of the accuracy of the models?**  
- Include more features like **employee engagement, manager feedback, or performance scores**.  
- Try other models like **XGBoost or neural networks**.  
- Tune hyperparameters and use **cross-validation**.  
- Handle **imbalanced classes** better for attrition prediction.

**9) Which were the challenges in the project development?**  
- Correct **encoding of categorical features**.  
- Avoiding **issues with scaling numeric features** while keeping predictions interpretable.  
- Ensuring **Streamlit app inputs match training data**.  
- Balancing simplicity of the app with **model accuracy and realistic outputs**.
