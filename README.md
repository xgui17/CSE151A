# CSE151A
This is a project for CSE151A at UCSD.

The dataset is obtained from kaggle (https://www.kaggle.com/datasets/zsinghrahulk/india-crop-yield/data).

## Data Preprocessing
Preprocessing is a crucial step in preparing data for machine learning models. In our dataset, we applied several preprocessing techniques to enhance the quality and interpretability of the data:

### Feature Transformation/Regularization:

We excluded the 'yield' column from our dataset since it directly corresponds to the ratio of 'Area' to 'Production', which are already present. This eliminates redundancy and potential multicollinearity issues.
Normalization was applied to the 'Area' column. Normalization scales the values of the 'Area' column to a standard range, typically between 0 and 1, to prevent outliers from disproportionately affecting the model's performance.
The 'Product' column underwent a log-transformation. Log-transformation helps in making skewed data more symmetric and manageable for the model. It reduces the impact of outliers and ensures a more normal distribution, thereby improving the model's stability and performance.
### Handling Categorical Features:

Categorical features such as 'Crop', 'Season', and 'State' need to be converted into numerical values to be interpretable by machine learning models. We employed one-hot encoding to achieve this conversion.
One-hot encoding generates binary columns for each category within a categorical feature. For instance, if there were 'n' unique categories in a feature, it creates 'n' binary columns, each indicating the presence or absence of that category for a particular data point.
After one-hot encoding, the 'Crop' feature expanded into 56 features, the 'State' feature expanded into 37 features, and the 'Season' feature expanded into 6 features. This transformation ensures that categorical data are appropriately represented for analysis and modeling purposes.
By applying these preprocessing techniques, we have made the dataset more suitable for machine learning algorithms, enabling them to effectively learn patterns and make accurate predictions or classifications.
