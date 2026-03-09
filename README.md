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

Logistic Regression Results:
- LR: Train 80% of data score: 0.800 || Test 20% of data score: 0.792
- LR Recall: 0.793
- LR Precision: 0.783
- LR Accuracy: 0.790

Support Vector Classification Results:
- SVC: Train 80% of data score: 0.834 || Test 20% of data score: 0.815
- SVC Recall: 0.790
- SVC Precision: 0.813
- SVC Accuracy: 0.810

Random Forest Classifier Results:
- RF: Train 80% of data score: 0.927 || Test 20% of data score: 0.815
- RF Recall: 0.809
- RF Precision: 0.804
- RF Accuracy: 0.810

Model Selection: 
- Comparing Logistic Regression and SVC, Logistic Regression achieved a slightly higher recall at 0.793 but had a lower precision and accuracy. Both models had small train-test gaps, indicating stable generalization.
- Random Forest, on the other hand, overfitted the data with a large train-test gap: 0.927 on training data versus 0.815 on test data, showing a weak generalization.
- Based on these results, SVC was chosen as the final model for its overall balance between precision, recall, accuracy, and train-test score gap.
