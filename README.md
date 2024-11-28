# Microsoft: Classifying Cybersecurity Incidents with Machine Learning

## Project Description
This project aims to enhance the efficiency of Security Operation Centers (SOCs) by developing a machine learning model capable of accurately predicting the triage grade of cybersecurity incidents. Using the comprehensive GUIDE dataset, the model categorizes incidents as true positive (TP), benign positive (BP), or false positive (FP) based on historical evidence and customer responses. The ultimate goal is to support guided response systems in providing SOC analysts with precise, context-rich recommendations, thereby improving the overall security posture of enterprise environments.

## Skills Learned
- Data Preprocessing and Feature Engineering
- Machine Learning Classification Techniques
- Model Evaluation Metrics (Macro-F1 Score, Precision, Recall)
- Handling Imbalanced Datasets
- Model Benchmarking and Optimization

## Approach
1. **Data Exploration and Understanding**
   - Initial Inspection: Load the `train.csv` dataset and inspect its structure.
   - Exploratory Data Analysis (EDA): Identify patterns, correlations, and potential anomalies.

2. **Data Preprocessing**
   - Handling Missing Data: Identify and handle missing values.
   - Feature Engineering: Create new features or modify existing ones to improve performance.
   - Encoding Categorical Variables: Convert categorical features into numerical representations.

3. **Data Splitting**
   - Train-Validation Split: Split the `train.csv` data into training and validation sets.
   - Stratification: Use stratified sampling for imbalanced target variables.

4. **Model Selection and Training**
   - Baseline Model: Start with a simple model to establish a performance benchmark.
   - Advanced Models: Experiment with more sophisticated models like Random Forests and Gradient Boosting Machines.
   - Cross-Validation: Implement cross-validation to ensure consistent model performance.

5. **Model Evaluation and Tuning**
   - Performance Metrics: Evaluate the model using macro-F1 score, precision, and recall.
   - Hyperparameter Tuning: Fine-tune hyperparameters to optimize performance.
   - Handling Class Imbalance: Use techniques like SMOTE and adjusting class weights.

6. **Model Interpretation**
   - Feature Importance: Analyze which features contribute most to predictions.
   - Error Analysis: Identify common misclassifications for potential improvements.

7. **Final Evaluation on Test Set**
   - Testing: Evaluate the model on the `test.csv` dataset and report final metrics.
   - Comparison to Baseline: Compare performance on the test set to the baseline model.

8. **Documentation and Reporting**
   - Model Documentation: Document the entire process and key findings.
   - Recommendations: Provide recommendations for integration into SOC workflows.
  
## Project Evaluation metrics:
The success and effectiveness of the project will be evaluated based on the following
metrics:
- **Macro-F1 Score**: A balanced metric that accounts for the performance across
 all classes (TP, BP, FP), ensuring that each class is treated equally.
- **Precision**: Measures the accuracy of the positive predictions made by the
 model, which is crucial for minimizing false positives.
- **Recall**: Measures the model's ability to correctly identify all relevant instances
 (true positives), which is important for ensuring that real threats are not missed.

## Business Use Cases
The solution developed can be implemented in various business scenarios, including:
- **Security Operation Centers (SOCs)**: Automating the triage process.
- **Incident Response Automation**: Suggesting appropriate actions for incidents.
- **Threat Intelligence**: Enhancing threat detection capabilities.
- **Enterprise Security Management**: Improving the overall security posture.

## Results

By the end of the project, the machine learning model achieved the following performance metrics:

### Performance Metrics for New Train Sample Dataset (RandomForestClassifier Model)

Accuracy : 0.98
Precision : 0.98
Recall : 0.98
F1_Score : 0.98

Classification Report:

              precision    recall  f1-score   support

           0       0.97      0.99      0.98    415850
           1       0.98      0.96      0.97    203214
           2       0.99      0.98      0.98    332620

    accuracy                           0.98    951684
   macro avg       0.98      0.97      0.98    951684
weighted avg       0.98      0.98      0.98    951684



### Performance Metrics for GUIDE Test Dataset (RandomForestClassifier Model)
-----------------------------------------------------------------------------

Accuracy : 0.99
Precision : 0.99
Recall : 0.99
F1_Score : 0.99

Classification Report:

              precision    recall  f1-score   support

           0       0.98      0.99      0.99    350403
           1       0.98      0.98      0.98    180282
           2       0.99      0.99      0.99    298914

    accuracy                           0.99    829599
   macro avg       0.99      0.98      0.98    829599
weighted avg       0.99      0.99      0.99    829599



### Key Findings
- The model demonstrates a high level of accuracy, precision, recall, and macro F1 score, indicating its effectiveness in classifying the triage grades of cybersecurity incidents (TP, BP, FP).
- The XGBoost model performed particularly well on the GUIDE test dataset, achieving an accuracy of 93.32%, which is an improvement over the performance on the new train sample.

### Comparison to Baseline
- The results from the training and testing datasets indicate that the implemented model significantly outperforms baseline models, showcasing the effectiveness of the selected machine learning techniques and preprocessing strategies.

### Limitations
- One of the challenges faced during the project was related to local GPU issues while running the model, which affected the training time and efficiency.
- This issue can be solved by using colab notebook for executing the models.
 
### Conclusion
The project successfully developed a machine learning model capable of accurately predicting the triage grade of cybersecurity incidents, with comprehensive analysis and documentation provided. Future improvements could focus on addressing the limitations encountered during model training and exploring additional features for further enhancing model performance.

