# Analyzing-House-Data-for-price-Estimation
This was a project created during my university's course "PATTERN RECOGNITION". <br>
Language: Python<br>
<br>

In this project, house data from the state of california were analyzed.<br><br>

File "_1_Preprocess_&_Opticalization.py" performs pre-processing in the data. <br>
Arithmetic Data has been scaled and categorical data has been turned into vector representation using One Hot Vector encoding. <br>
Missing values were also replaced with the median values.<br>
Processed data is saved in file "HousingScaled.csv" <br>
The opticalization of the data also happens in this file. Data visualizations are produced that show if there is correlation between different variables. Some Graphs have more than 2 variables.<br><br>

File "_2_Perceptron_Class.py" contains a custom implementation of Perceptron. <br>
This is used in the File "_3_Perceptron_training.py" to perform linear Classification between the houses that are above the median price and under the median price.<br><br>

File "_4_Linear_Regressor_Class.py" contains a custom implementation of Ordinary Least Squares Algorithm. <br>
This is used in the File "_5_Linear_Regression_training.py" to perform linear regression and predict the price of a house using the other variables.<br><br>

File "_6_MLP_Regression.py" tries to perform non-linear regression to predict house prices.<br>
Unlike file "_5_Linear_Regression_training.py", this file uses multi-layer neural network to achieve this. <br>
Keras module has been used.
