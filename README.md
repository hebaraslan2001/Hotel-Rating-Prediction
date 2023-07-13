# Hotel-Rating-Prediction
Can you make your trip cozier by using data science? Can you predict what score a reviewer will give a hotel using features about the hotel in combination with the reviewer history and each review’s language? this question is more important than ever to the industry. Using machine learning algorithms: Linear Regression and Polynomial Regression, we can predict which hotels will be highly rated by using Reviewer_Score column.
# Preprocessing Techniques:
  • Outliers Removal
  • handle_duplicated
  • Filling the missing values
  • Encoding
  • Feature scaling
   Feature selection
# Regression Techniques:
  • With Linear Regression:
    Mean Square Error (test data) is: 1.82, 
    Mean Absolute Error using the test data is: 1.03,
    R2_score: 0.31
  • With Polynomial Regression:
    Mean Square Error using the test data is: 1.70, 
    Mean Absolute Error using the test data is: 0.97
    R2_score: 0.36
# Conclusion:
  •	We predict the best Hotel Rating based on  a number of best features
  •	In 2 models we selected the Most 1 Features with high score by using SelectKBest(f_regression, k=7) .
  •	The best model get the highest Accuracy is polynomial Regression 
  Mean Square Error=1.70

# Classification Techniques & hyper parameter tuning:
  • With logistic regression:
      We find that the accuracy = 0.57
  • With Decision Tree:
      We find that the accuracy (max_depth=5) = 0.70
      We find that the accuracy (max_depth=50) = 0.70
      We find that the accuracy (max_depth=100) = 0.70
  • With Random forest:
      We find that the accuracy (max_depth=10) = 0.72
      We find that the accuracy (max_depth=50) = 0.71
      We find that the accuracy (max_depth=90) = 0.71
