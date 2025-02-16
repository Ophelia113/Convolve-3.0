# Convolve-3.0: A Pan IIT AI/ML Hackathon
Convolve 3.0 is a prestigious AI/ML hackathon organized by nine premier IITs—Bombay, Delhi, Madras, Kanpur, Kharagpur, Guwahati, Roorkee, Varanasi (BHU), and Hyderabad. Open to participants worldwide, this hackathon presents an opportunity to tackle real-world problems using advanced Machine Learning and Data Analytics techniques.

# Participation in Convolve 3.0
My team participated in Convolve 3.0 and advanced through multiple rounds, working on challenging AI/ML problem statements. Below is a summary of my experience in both rounds.

# Round 1: Credit Card Behaviour Score
Problem Statement:
Bank A wanted to develop a robust risk management framework by creating a "Behaviour Score" model. This model aimed to predict the probability of a credit card customer defaulting in the future.

**Approach:**

Exploratory Data Analysis (EDA):
-Loaded and inspected a dataset of 96,806 rows and 1,216 columns.
-Addressed missing values by removing columns with excessive NaNs and imputing missing values where necessary.
-Conducted feature selection using variance thresholding and correlation analysis.
-Reduced features from 1,216 to 189, ensuring a balance between performance and dimensionality.

Model Development & Training:
-Implemented a Sequential Neural Network as a baseline model.
-Designed a ResNet-inspired Neural Network with residual connections for improved gradient flow.
-Fine-tuned hyperparameters such as dropout rates, learning rates, and number of neurons.
-Evaluation Metrics: Precision, Recall, F1 Score, and Mean Absolute Error (MAE) were prioritized due to class imbalance.
-Final Results: The ResNet-inspired model improved recall and precision over the baseline, making it more effective in identifying high-risk customers.

Results of the Baseline Neural Network: 
● Accuracy: 76.60% 
● Precision: 4.44% 
● Recall: 69.32% 
● F1 Score: 8.35% 
● Confusion Matrix: 
[ [ 17285  5246 ] 
[ 108   244 ] ] 
While the accuracy was relatively high, the low precision and F1 score highlighted the 
model's difficulty in identifying the minority class. 


Results of the ResNet-Inspired Neural Network: 
● Accuracy: 77.94% 
● Precision: 4.72% 
● Recall: 69.60% 
● F1 Score: 8.85% 
● Confusion Matrix: 
[ [ 17589  4942 ] 
[ 107   245 ] ] 
Compared to the baseline, the ResNet-inspired model improved precision, recall, and F1 
score, particularly in identifying the minority class (positive samples). 


**Observations from MAE:**

For the sequential neural network, the calculated MAE was **0.2339**, indicating an average 
error rate of approximately **23.39%** per sample. After implementing the ResNet-inspired 
neural network, the MAE further decreased to **0.2206**, signifying an improvement in the 
model's overall prediction accuracy. 
This reduction in MAE aligned with the observed enhancements in classification metrics, 
such as precision, recall, and F1 score, confirming that the ResNet-inspired model was not 
only better at classification but also at minimizing absolute prediction errors. 


**Conclusion:** Given its ability to effectively handle class imbalance and leverage residual connections for 
capturing complex patterns, we are selecting the ResNet-inspired model for final validation 
and submission.


# Round 2: Predicting Mail Delivery Time Slots


Problem Statement:
Given various input features related to orders, customers, and logistics, predict the most probable time slot for mail delivery. The goal is to rank the all the 28 slots in decreasing order of likelihood of a customer opening the mail.

Data Loading and Initial Inspection:

This project was initially provided with 2 datasets for training data, i.e. 
train_action_history.csv & train_cdna_data.csv.

1. Preprocessing Report for train_action_history.csv 
The project commenced with loading the train_action_history dataset into a Pandas 
DataFrame for initial exploration and analysis.The data had  8797911  rows and 8 columns.

2. Preprocessing Report for train_cdna.csv 
Loading the train_cdna_data dataset into a Pandas DataFrame for initial exploration and 
analysis. The dataset consisted of  12,85,402 rows  and 303 columns  . 
This dataset had columns with numerical data, categorical data and also few columns with 
mixed data types.

The first step involved addressing two primary concerns: 
- Handling Missing Values  : Several columns contained missing (  NaN  ) values, which 
required appropriate cleaning strategies before training the model. 
- Handling Categorical Data:  Columns with categorical or  mixed data types  had to 
be handled separately using one hot encoding.

**MODEL DEVELOPMENT AND TRAINING:**

Data Preparation: We split the dataset into training and testing sets, allocating 80% of the data for training and 
20% for testing.
-Since the XGBoost  num_classes  parameter, when set to 28, expects class labels 
ranging from  0 to 27  , the target column, which initially  contained values from  1 to 28  , 
was transformed using  label encoding 
-Features in the dataset, stored in  X  , were normalized using the  StandardScaler  to 
ensure that all variables contributed equally to the model training. This transformation 
standardized the features to have a mean of 0 and a standard deviation of 1. 



**Model Selection:**

Throughout our analysis, we explored a range of models to identify the best approach for our 
dataset. We experimented with popular boosting algorithms like  XGBoost  , recommendation 
systems such as  LightFM  and  Neural Collaborative Filtering (NCF)  , and more complex 
deep learning models like  Neural Networks  . 
However, working with such a large dataset and within a limited timeframe posed significant 
challenges. While we gave LightFM and NCF a genuine try, their training processes couldn’t 
be completed because of the heavy computational power they demanded. 
In the end, we focused on comparing  XGBoost  and  Neural Networks  , both of which struck 
a balance between computational feasibility and performance. 


**XGBoost**

Class Weight Calculation: 
To address class imbalance, we calculated weights for each class using the 
compute_class_weight  function. These weights were  applied during training to ensure 
fair learning across all classes. But this was  not used  finally since it  did not improve model 
metrics. 
The XGBoost model was configured with the following parameters: 
●  Objective:  multi:softmax  to handle multi-class classification. 
●  Number of Classes:  Set to 28, as per the problem's  requirements. 
●  Evaluation Metric:  Multi-class log-loss (  mlogloss  ) for performance tracking. 
●  Max Depth:  10, to allow the model to capture complex patterns. 
●  Number of Estimators:  10, to balance training time and performance. 


**Neural Network Architecture**

●  Model Definition: 
A sequential neural network was defined with the following layers: 
○  Input Layer: Accepts input with dimensions equal to the number of features in 
X2_train. 
○  Hidden Layers: Two dense layers with 128 and 64 neurons, respectively, 
each using ReLU activation and followed by a 30% dropout to prevent 
overfitting. 
○  Output Layer: A softmax layer with 28 neurons, one for each class.

We fine-tuned the Sequential model by adjusting: 
●  Number of neurons in input layer: 128. 
●  Dropout rates: 0.3 for the input layer and hidden layers. 


**Use of Mean Absolute Error (MAE) in Model Evaluation:**
For the  XGBoost, the calculated MAE was  **6.31**.  After implementing the Sequential neural network, the MAE further increased to  **7.85**.
Since we did not have ground truth rankings we could not use Mean Average Precision as an evaluation metric. 

**TESTING DATA ANALYSIS AND PREPARATION** 

Testing Data Preparation:

The test data was cleaned using the same steps as the training data, including handling 
missing values, encoding categorical variables, and aligning features. Columns were 
adjusted to ensure compatibility with the trained model. 

Model Prediction on Test Data:

The trained XGBoost model was applied to the test data (  X_test  )  to predict the probabilities 
for each of the 28 email slots using the  predict_proba()  function. These probabilities 
were decoded using a label encoder and sorted in descending order to rank the slots for 
each customer.


**LIMITATIONS AND CHALLENGES FACED**


We initially considered using LightFM and Neural Collaborative Filtering (NCF) for this task, 
as these models take into account  customer behaviors across all slots  rather than relying 
on a single target variable.
However, due to  computational limitations  , we were unable to train these models on such 
a large dataset efficiently  . Additionally, a key challenge with LightFM was that it generates 
personalized rankings based on interactions  within the same dataset  . In our case, the train 
and test datasets had  completely different customer codes  , meaning the model trained 
on the train dataset could not be used directly for generating personalized rankings on the 
test data.


**COMPARISON & CONCLUSION**


Both  XGBoost  and  Neural Networks  struggled to deliver strong performance in terms of 
accuracy; however, the  Mean Absolute Error (MAE)  showed a slight advantage for 
XGBoost over Neural Networks. Given the constraints that prevented us from utilizing 
advanced models like LightFM and NCF, we opted for XGBoost as the final model to predict 
on the test data.
