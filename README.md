# CSE151A
## Overview
This is a project for CSE151A at UCSD. (The link to our <a href="https://github.com/xgui17/CSE151A/blob/main/Group%20Project%20Milestone%20%233.ipynb">jupyter notebook</a>)

This repository contains the analysis of crop yield data for India. The datasets used in the analysis include information on various crops grown in different states and districts of India and the corresponding yields of these crops in different years.

The dataset is obtained from <a href=https://www.kaggle.com/datasets/zsinghrahulk/india-crop-yield/data>kaggle</a>.

### Dataset Information
The dataset consists of the following columns:

  1. `State`: Indicates the State of India in which the crop yield is measured.
  2. `District`: Indicate the district of the State in which the crop is grown.
  3. `Crop`: Indicates the specific crop grown in the field. 
  4. `Crop_Year`: The year the crop was harvested or produced. 
  5. `Season`: Divides the cropping period of the crop into different seasons. 
  6. `Area`: Measurement of the area of a crop field. 
  7. `Production`: Indicates the total production of a crop field in a given season. 
  8. `Yield`: Calculates the yield per unit area for that particular observation.

## Insights

The dataset comprises 345,336 observations, covering 37 unique states, 707 districts, 55 different types of crops, 24 years, and 6 agricultural seasons.

The top states in the proportion of observations include Uttar Pradesh, Madhya Pradesh, Karnataka, Bihar, and Rajasthan.

Rice, maize, moong (green gram), urad, and sesamum are among the most cultivated crops.
The dataset exhibits variations in crop yield across different states, districts, crops, and seasons.

## Data Preprocessing
Preprocessing is a crucial step in preparing data for machine learning models. In our dataset, we applied several preprocessing techniques to enhance the quality and interpretability of the data.

### Handling of missing values: 

We dropped rows with missing values, considering they represent only 1.4% of the total data.

### Feature Transformation/Regularization:

Outliers in the `Area` and `Production` columns were retained, as they are considered representative of real-world agricultural practices.

We excluded the `Yield` column from our dataset since it directly corresponds to the ratio of `Area` to `Production`, which is already present. This eliminates redundancy and potential multicollinearity issues.

The `Production` and `Area` column underwent a log-transformation. Log-transformation helps make skewed data more symmetric and manageable for the model. It reduces the impact of outliers and ensures a more normal distribution, thereby improving the model's stability and performance.

### Handling Categorical Features:

Categorical features such as `Crop`, `Season`, `State` and `District` need to be converted into numerical values to be interpretable by machine learning models. We employed one-hot encoding to achieve this conversion. Notably, we undertook an aggregation of smaller `District` entities to reduce training and predictive burden. Districts representing less than 10% of the state's total were grouped into a collective category named "(state name)_other." This aggregation approach reduced the number of unique `District` categories from 707 to 80.

After one-hot encoding, the `Crop` feature expanded into 56 features, the `State` feature expanded into 37 features, the `Season` feature expanded into 6 features, and the `District` feature expanded into 80 features. This transformation ensures that categorical data are appropriately represented for analysis and modeling purposes.

By applying these preprocessing techniques, we have made the dataset more suitable for machine learning algorithms, enabling them to learn patterns and make accurate predictions or classifications effectively.

## Model 1: Polynomial Regression

### Introduction
This section presents an analytical approach to predicting crop productions using polynomial regression. Polynomial regression allows us to capture the nonlinear relationship between the independent and target variables, providing a more flexible modeling approach than simple linear regression.

### Polynomial Features
To capture the nonlinear relationships in the dataset, we introduce polynomial features. These features consist of a polynomial combination of the original features. We can better approximate the underlying data distribution by including a certain degree of polynomial features. 

### Model Training and Evaluation
We trained the polynomial regression model using the preprocessed dataset, including the polynomial features. The model was evaluated using Mean Squared Error (MSE) on both the training and test sets to assess its performance and generalization capability.

### Results
The performance metrics of the model are as follows:
- Mean squared error testing (MSE): about 0.3872
- Mean Squared Error Training (MSE): about 0.3852

These results show that the model performs consistently on the training and test sets, indicating it is not overfitting. More details can be seen in the <a href="https://github.com/xgui17/CSE151A/blob/main/Group%20Project%20Milestone%20%233.ipynb">jupyter notebook</a>.

<p align="center">
  <img src="assets/model1_actual_vs_pred.png" width="50%" height="50%">
</p>

<p align="center" style="font-size:85%">
  Figure 1: Actual vs. Predicted Production. It shows the alignment of our model's predictions with the actual `Production` values.
</p>

<p align="center">
  <img src="assets/model1_fitting_graph.png" width="50%" height="50%">
</p>

<p align="center" style="font-size:85%">
  Figure 2: the fitting graph. Our model with 2 interaction terms has improved performance compared to models with fewer interaction terms.
</p>


### Conclusion
The polynomial regression model is the first step in our analysis, allowing us to capture nonlinear relationships and gain insight into the factors that affect crop production. We will further refine and improve the model, such as feature selection, regularization, and exploring other regression models to improve prediction accuracy and deepen our understanding of crop production dynamics.

## Future models

In addition to the polynomial regression model discussed earlier, we are considering exploring two other models: dense layer neural networks and random forests.

### Neural networks with dense layers
Neural networks are powerful models that capture the complexity of nonlinear relationships in data. By using dense layers, neural networks can automatically learn the interactions between features, thus reducing the need for manual feature engineering. This makes them ideal for datasets like ours, where traditional linear models or simple polynomials may not adequately capture underlying patterns. Neural network modeling is flexible and has the potential to improve prediction accuracy.

### Random Forests
Random Forest is another model we plan to explore further. It is known for its robustness to overfitting and its ability to handle complex interactions between features. Random forests are particularly suitable for datasets with mixed feature types, as they can efficiently handle numeric and categorical variables. In addition, Random Forest can handle high dimensionality due to the single coding of categorical variables, which makes it an ideal candidate model for our dataset.

### Evaluation of Data, Labels, and Loss Function

The similarity in performance metrics suggests that the data and labels were sufficiently represented by the training and test datasets. The consistent performance also implies that the data was properly cleaned and preprocessed, including the effective use of one-hot encoding for categorical variables, ensuring that no significant bias or noise was introduced. The choice of Mean Squared Error (MSE) as the loss function appears to have been appropriate for this regression task, as evidenced by the model's ability to generalize well from the training data to unseen data.

## Model 2: Random Forest Regression

### Introduction
In this section, we explore the application of Random Forest Regression as a modeling technique for crop production forecasting. Random forest regression is a powerful ensemble learning method capable of capturing complex relationships in data by combining multiple decision trees.

### Model Training and Evaluation
We trained the random forest regression model using the first model's processed data and features. The dataset was split into training and test sets to facilitate model evaluation. The model is then fitted to the training data and used to make predictions on the test set.

### Results
The performance metrics of the model are as follows:
- Mean squared error Testing (MSE): about 0.7894
- Mean Squared Error Training (MSE): about 0.7853

This indicates that the test set has a slightly higher error than the training set, suggesting slight overfitting.

### Hyperparameter Tuning
To optimize the performance of the model, we tuned the hyperparameters with 5-fold cross-validation using GridSearchCV. This involved exploring different combinations of hyperparameters, such as the number of estimators and the maximum depth of the tree, to determine the optimal configuration.

### Results
- Best Parameters: 'max_depth': 15, 'n_estimators': 200
- Best Model Mean Squared Error: 0.5588088610648668

<p align="center">
  <img src="assets/model2_fittingG.png" width="50%" height="50%">
</p>

<p align="center" style="font-size:85%">
  Figure 3: the fitting graph of Model 2. Our model after Hyperparameter Tuning has improved performance.
</p>

### Conclusion.
The random forest regression model initially exhibited a degree of overfitting, which was reflected in the higher MSE on the test set than on the training set. However, by tuning the hyperparameters, we significantly improved the performance of the model, thereby reducing the MSE on both the training and test sets.

The random forest regression model showed better performance after hyperparameter tuning than the polynomial regression model. Despite encountering runtime issues due to the model's computational complexity, we optimized the model's performance through effective hyperparameter tuning techniques.

To further enhance the random forest regression model, we can explore the dimensionality reduction technique and try to use the ensemble approach to improve the prediction accuracy while solving the runtime limitation problem. In conclusion, the random forest regression model shows better prediction of crop yields. 

### Next Model Plan
We plan to use Neural Networks as our next model. Neural networks are good at modeling complex, non-linear relationships between features, which might not be effectively captured by polynomial regression or even random forests, as used in our first two models. Production can be influenced by intricate interactions between various factors, such as area, season, and so on. A neural network's ability to capture these non-linearities could lead to better predictive performance. Additionally, after one-hot encoding of categorical variables, our datasets become high-dimensional. Neural networks are adept at handling high-dimensional data and automatically learning feature representations that are most relevant for prediction, potentially leading to more accurate predictions.

## Model 3: Neural Net

### Introduction
In this section, we employ a neural network with dense layers to predict crop production. By analyzing patterns in crop data, we aim to develop a model that can forecast production with high accuracy. Initial models were simplistic, setting the stage for more advanced neural networks and hyperparameter tuning to refine predictions.

### Model Training and Evaluation
The model training involved preparing the dataset, defining the architecture with dense layers, and selecting activation functions and optimizers. The use of Keras Tuner facilitated the fine-tuning of hyperparameters. We employed techniques like k-fold cross-validation to ensure that the model's performance was robust across different subsets of data.

### Results
The trained neural network, optimized through random search and early stopping, demonstrated a high level of accuracy. The hyperparameter tuning concluded with an optimal set of parameters, including:
- Learning rate: 0.001
- Number of nodes per layer: 32
- Activation Function: ReLU activation function
- Optimizer: Adam optimizer

And we got the result:
- Mean squared error Testing (MSE): about 0.202
- Mean Squared Error Training (MSE): about 0.193

<p align="center">
  <img src="assets/model3_fitting_graph.png" width="50%" height="50%">
</p>

<p align="center" style="font-size:85%">
  Figure 4: the fitting graph of Model 3. The performance of our model has improved as number of epochs increases.
</p>

### Conclusion
The model significantly improved the predictive accuracy for crop production compared to initial versions. The success of the model was further confirmed through k-fold cross-validation, ensuring consistent performance across various data segments and bolstering confidence in the model's generalizability. This suggests not only a high level of accuracy but also an ability to adapt to new, unseen data, thereby avoiding common issues such as overfitting and underfitting. 

Future improvements for enhancing model performance include data augmentation, advanced feature engineering, and exploring ensemble methods or more sophisticated neural network architectures. Additionally, integrating external datasets could further improve predictions, making the model an even more powerful tool for agricultural planning and forecasting. These steps will aim to solidify the model's applicability in real-world scenarios, ensuring it can serve as a reliable decision-support tool in agricultural production systems.
