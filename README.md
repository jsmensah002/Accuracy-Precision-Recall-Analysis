Brief Overview:
- This project explores how data preprocessing and model optimization impact performance, with four experimental setups:
- Not Optimized: Baseline models trained on raw data without any hyperparameter tuning.
- Not Optimized but Scaled: Baseline models trained on scaled features to observe the effect of normalization.
- Optimized with Outliers: Models tuned with hyperparameter optimization including all data, keeping outliers.
- Optimized without Outliers: Models tuned after removing outliers, showing the impact of noise reduction on performance.
- Only the “Optimized with Outliers” results would be discussed to keep the analysis brief, since the final model was selected from this experiment.

Method:
- Studied correlation between inputs and output to validate feature relevance
- Observed how scaling, outlier removal, and tuning affect each model differently
- Models used were Logistic Regression (LR), Random Forest Classifier (RF), and Support Vector Classification (SVC)

Results from Optimized with Outliers Present:
- LR: Train 80% of data score: 0.80028 || Test 20% of data score: 0.792135
- LR Recall: 0.79710
- LR Precision: 0.70513
- LR Accuracy: 0.79213

- SVC: Train 80% of data score: 0.83405 || Test 20% of data score: 0.81461
- SVC Recall: 0.68116
- SVC Precision: 0.81034
- SVC Accuracy: 0.81461

- RF: Train 80% of data score: 0.92686 || Test 20% of data score: 0.81461
- RF Recall: 0.78261
- RF Precision: 0.75000
- RF Accuracy: 0.81461

Model Selection: 
- Comparing Logistic Regression and SVC, Logistic Regression achieved higher recall at 0.79710, meaning it caught more true positives, but its precision was lower at 0.70513 and overall accuracy was 0.79213. SVC had slightly lower recall at 0.6812 but higher precision at 0.81034 and better accuracy at 0.81461, making it more balanced in performance. Both models had small train-test gaps, indicating stable generalization.
- Random Forest, on the other hand, overfitted the data with a large train-test gap: 0.92686 on training data versus 0.81461 on test data, showing a weak generalization.
- Based on these results, SVC was chosen as the final model for its overall balance between precision, recall, and accuracy.

Discussion Question
- For this Titanic dataset, the main competition is between Logistic Regression and SVC. Considering that Logistic Regression has higher recall but SVC shows better overall accuracy and precision, which model would you have chosen and why?
