# Regression Techniques on 'California Housing Prices' Dataset

This repository contains code for applying regression techniques on the 'California Housing Prices dataset after performing EDA and data preprocessing. The goal is to build regression models to predict house prices based on the given features.

## Dataset

The data pertains to the houses found in a given California district and some summary stats about them based on the 1990 census data.
The "California Housing Prices" dataset contains housing data for various districts in California. It has the following features:

- longitude: A measure of how far west a house is; a higher value is farther west
- latitude: A measure of how far north a house is; a higher value is farther north
- housingMedianAge: Median age of a house within a block; a lower number is a newer building
- totalRooms: Total number of rooms within a block
- totalBedrooms: Total number of bedrooms within a block
- population: Total number of people residing within a block
- households: Total number of households, a group of people residing within a home unit, for a block
- medianIncome: Median income for households within a block of houses (measured in tens of thousands of US Dollars)
- medianHouseValue: Median house value for households within a block (measured in US Dollars)
- oceanProximity: Location of the house w.r.t ocean/sea

## Workflow

1. **Data Preprocessing and EDA**: 

The dataset is loaded and being analyzed. There are 20640 data with 10 features. All except the 'ocean_proximity' feature have numerical data. I used one hot encoding, a technique to represent categorical variables as numerical values in a  machine learning model. I have observed that there are 207 null values in 'total_bedrooms' feature. I handled the missing values using median.

In 'total_bedrooms' feature, I have observed that the Mean is greater than the Median(50th percentile). It means that it is right skewed. This explains that the majority of the data points are concentrated on the left side of the distribution. Hence, I chose median.

There are also some outliers on the dataset. Outliers may represent valid and meaningful data points in some cases. If the outliers are genuine and not the result of data entry errors or measurement issues, those can be kept and included in the analysis. As I am not sure about those, particularly about data generation process, I kept them as they are.

I have also done feature scaling on the data. Feature scaling helps to normalize the range of the features and make them comparable, ensuring that different features do not dominate the learning process based solely on their magnitudes. Here I used Robust Scalar as it uses the median and the interquartile range (IQR) to scale the features, making it more robust and resistant to the influence of outliers compared to StandardScaler.

2. **Splitting the dataset**:

The dataset is split in a 80-20 ratio for training and testing, and then the training data is further divided into a training set and a validation set in a 80-20 ratio. After fitting the training dataset, I used validation dataset for initial testing and experimenting parameter tuning. I used the test set for final prediction result.

3. **Regression Models**: 

I have used the following algorithms. I have provided brief background details of each technique I used :

* **Linear Regression**:
Linear regression is a simple and widely used regression algorithm that models the relationship between the dependent variable and one or more independent variables by fitting a linear equation to the data. It works well when the relationship between the features and the target variable is linear. Linear regression can be prone to overfitting if there are too many features or multicollinearity among the features.

* **Ridge Regression**:
Ridge regression is a linear regression technique that incorporates L2 regularization to address multicollinearity issues and prevent overfitting. It adds a penalty term to the linear regression cost function that includes the squared values of the coefficients. The regularization term helps to shrink the coefficients, effectively reducing the impact of less important features on the model. Ridge regression is suitable when there are highly correlated features in the dataset, and it provides more stable and interpretable results compared to standard linear regression.

* **Support Vector Regression (SVR)**:
SVR is a regression algorithm based on Support Vector Machines (SVM) that aims to find the best hyperplane that maximizes the margin while allowing for some tolerance for errors (epsilon-insensitive loss). SVR is particularly effective when dealing with non-linear relationships between features and the target variable. It transforms the input data into a higher-dimensional space using a kernel function and finds the optimal hyperplane in that space. SVR can handle both linear and non-linear regression tasks and is robust to outliers.

* **K-Nearest Neighbors (KNN) Regression**:
KNN regression is a non-parametric algorithm that predicts the target value by averaging the values of its k nearest neighbors in the feature space. KNN is simple and easy to understand, making no assumptions about the underlying data distribution. However, it can be sensitive to the choice of k, and the prediction time can be computationally expensive for large datasets. KNN regression performs well when the data has local patterns, and there is no clear linear relationship between features and the target variable.

* **Decision Tree Regression**:
Decision tree regression is a non-linear algorithm that models the target variable by recursively splitting the feature space into subsets based on the most informative features. Each split creates a node in the tree, and the predicted value for a new sample is the average of the target values in the leaf node that the sample falls into. Decision tree regression is interpretable and can handle both numerical and categorical features. However, it tends to overfit the training data when the tree depth is not controlled.

* **AdaBoost Regression**:
AdaBoost stands for Adaptive Boosting, and AdaBoost regression is an ensemble method that combines multiple weak learners (typically decision trees) to create a strong predictor. The weak learners are trained sequentially, and each subsequent model focuses on correcting the errors made by the previous ones. AdaBoost assigns weights to each sample, emphasizing misclassified samples during each training iteration. It is effective in reducing bias and variance and performs well when the weak learners complement each other.

* **XGBoost Regression**:
XGBoost (Extreme Gradient Boosting) is an advanced ensemble method based on gradient boosting. It uses gradient-based optimization techniques and regularization to achieve high accuracy and computational efficiency. XGBoost builds a sequence of decision trees and combines their predictions to make the final prediction. It incorporates various hyperparameter tuning options, making it highly customizable and powerful. XGBoost is widely used in competitions and real-world applications due to its performance and versatility.

* **Multi-Layer Perceptron (MLP) Regression and Sequential Model**:
Multi-Layer Perceptron (MLP) Regression: MLP is a type of artificial neural network that consists of multiple layers of interconnected nodes (neurons). The input is passed through the hidden layers, and each neuron applies an activation function to its input. MLP regression can learn complex non-linear relationships between features and the target variable. It uses backpropagation and optimization algorithms to adjust the weights during training to minimize the error between actual and predicted values. MLP regression is flexible and capable of handling large-scale data and complex tasks, but it requires careful tuning and may suffer from overfitting without appropriate regularization. It is type of sequential model. In Keras, a Sequential model is a linear stack of layers, where you can simply add one layer at a time. A Sequential model is a suitable choice for feedforward neural networks, where the data flows sequentially through the layers from the input to the output. Each layer in the model has weights that are learned during training. On the other hand, a MLP is a type of feedforward neural network, which consists of multiple layers of interconnected nodes (neurons). An MLP can be represented as a sequence of layers, where each layer connects to the next one in a linear manner. The input to each neuron in a layer is the output from the previous layer. Since an MLP can be represented as a linear stack of layers, it naturally fits the concept of a Sequential model in Keras. In Keras, One can use the Sequential class to build an MLP by adding layers to it one after the other. This allows to define the architecture of the network in a sequential manner, which is simple and intuitive.

4. **Evaluation Metrics**:

For evaluation, I have used RMSE (Root Mean Squared Error) and R-squared (Coefficient of Determination). 

RMSE measures the average magnitude of the error (residuals) made by the model in predicting the target values. It represents the square root of the average of squared differences between the predicted values and the actual target values. Smaller RMSE values indicate that the model's predictions are closer to the actual values, and hence, a smaller RMSE is better.

R-squared represents the proportion of the variance in the target variable that is explained by the model. Larger R-squared values suggest that the model is better at explaining the variation in the target variable and hence, a larger R-squared is better.

5. **Result**

| Algorithm         | RMSE           | R-squared      |
|-------------------|----------------|----------------|
| XGBoost           | 46217.383609   | 0.836994       |
| KNN Regression    | 59133.839278   | 0.733151       |
| Decision Tree     | 60362.551481   | 0.721947       |
| Sequential Model  | 63995.289662   | 0.687472       |
| Linear Regression | 70020.860473   | 0.625848       |
| Ridge Regression  | 70021.120930   | 0.625845       |
| AdaBoost          | 74513.736715   | 0.576293       |
| SVM Regression    | 79829.201680   | 0.513686       |

For RMSE, a smaller value is better, while for R-squared, a larger value is better. So, here, XGBoost performed better.


## Conclusion

In conclusion, XGBoost demonstrated the best performance in predicting California housing prices, providing the lowest RMSE and the highest R-squared value. The results highlight the effectiveness of XGBoost as a powerful regression algorithm for this particular dataset. Further optimization and fine-tuning of hyperparameters may improve the performance of other models. Generally, the best technique to use for a specific regression problem will depend on the following factors:

- The complexity of the relationship between the features and the target variable
- The amount of data available
- The desired accuracy of the predictions

Sometimes in industry, combination of algorithms are used for better performance.



