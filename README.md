Developed a model to predict monthly precipitation in Mumbai (2000-2020) based on meteorological data, including Specific Humidity, Relative Humidity, Temperature, and Precipitation.

Utilized a Recurrent Neural Network (RNN) for its ability to capture temporal dependencies in sequential time-series data, unlike Artificial Neural Networks (ANNs) which are less suited for such tasks.

Code Overview:
-Loaded and preprocessed the data, handling missing values and normalizing features.

-Split the data into training and testing sets, creating DataLoader objects for efficient batch processing.

-Built and trained an RNN model with the Adam optimizer and Mean Squared Error loss function.

-Evaluated the model using MSE, MAE, and R², achieving strong predictive accuracy.

Model Performance:
MSE: 0.0098
MAE: 0.0670
R²: 0.8587
