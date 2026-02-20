Brief Overview:
- This project explores how data preprocessing and model optimization impact performance, with four experimental setups:
- Not Optimized: Baseline models trained on raw data without any hyperparameter tuning.
- Not Optimized but Scaled: Baseline models trained on scaled features to observe the effect of normalization.
- Optimized with Outliers: Models tuned with hyperparameter optimization including all data, keeping outliers.
- Optimized without Outliers: Models tuned after removing outliers, showing the impact of noise reduction on performance.
- Only the “Optimized with Outliers” results would discussed to keep the analysis brief, since the final model was selected from this experiment.

Method:
- Studied correlation between inputs and output to validate feature relevance
- Observed how scaling, outlier removal, and tuning affect each model differently
- Models used were Logistic Regression (LR), Random Forest Classifier (RF), and Support Vector Classification (SVC)

Results from Optimized with Outliers Present:
- LR: Train 80% of data score: 0.80028 || Test 20% of data score: 0.792135
- LR Recall: 0.79710
- LR Precision: 0.70513
- LR Accuracy: 0.792135

- SVC: Train 80% of data score: 0.83405 || Test 20% of data score: 0.81461
- SVC Recall: 0.68116
- SVC Precision: 0.81034
- SVC Accuracy: 0.81461

- RF: Train 80% of data score: 0.92686 || Test 20% of data score: 0.81461
- RF Recall: 0.78261
- RF Precision: 0.75000
- RF Accuracy: 0.81461

Model Selection: 
- The Support Vector Classification model was selected as the final model. Logistic Regression had a higher recall, meaning it caught more true positives, but SVC achieved better overall performance with higher accuracy and precision, making it more balanced and reliable. Random Forest showed signs of overfitting, reducing its generalization to unseen data.

Discussion Question
- For this Titanic dataset, the main competition is between Logistic Regression and SVC. Considering that Logistic Regression has higher recall but SVC shows better overall accuracy and precision, which model would you have chosen and why?
