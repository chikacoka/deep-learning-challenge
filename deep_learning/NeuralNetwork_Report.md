# The Neural Network Model Report

The aim of this project is to develop a model that screens applicants for funding based on the best chances of success in their chosen ventures. Applying knowledge in machine learning and neural networks concepts, a binary classifier was created using selected features from a CSV containing historical data of over 34,000 organizations that had received funding from a non-profit organization, Alphabet Soup, in past years. Below are the column headers that capture metadata about each organization:

•	EIN and NAME—Identification columns
•	APPLICATION_TYPE—Alphabet Soup application type
•	AFFILIATION—Affiliated sector of industry
•	CLASSIFICATION—Government organization classification
•	USE_CASE—Use case for funding
•	ORGANIZATION—Organization type
•	STATUS—Active status
•	INCOME_AMT—Income classification
•	SPECIAL_CONSIDERATIONS—Special considerations for application
•	ASK_AMT—Funding amount requested
•	IS_SUCCESSFUL—Was the money used effectively

### Developing the neural network model involved the following four critical steps:

#### Step 1: Preprocess the data using Pandas and Scikit-learn’s StandardScaler() function: 
The provided CSV was read into Pandas DataFrame, non-beneficial ID columns were dropped (e.g., the NAME and EID columns), and categorical data were converted to numeric using “pd.get_dummies.”
Split the preprocessed data into features array x, and a target array, y, using “sklearn.” This process is followed by using the array and the “train_test_split” function to split the data into training and testing sub-datasets. Within the context of this model, the target array is the “IS_SUCCESSFUL”column representing the final outcome of the campaign venture, i.e., 1 representing “ funds were used effectively” and 0 representing “failure in using the funds effectively”. The features variables are the other nine remaining columns: APPLICATION_TYPE, AFFILIATION, CLASSIFICATION, USE_CASE, ORGANIZATION, STATUS, INCOME_AMT, SPECIAL_CONSIDERATIONS, ASK_AMT. 
The training and testing features datasets are scaled by creating a ‘StandardScaler’ instance, and then fitted to the training data using the ‘transform’ function.

#### Step 2: Compiling, Training, and Evaluating the Model:
The next step is the design of a neural network using TensorFlow, and creating a binary classification that can predict whether an organization sourcing funds from Alphabet Soup will be successful based on the features in the dataset. The number of input features variable was set to the number of input features in the training data ‘X-train ‘. Two hidden layers 1 and 2 were chosen, set at 80, and 30 number of nodes respectively. This was followed by compiling, training, and evaluating the binary classification model to calculate the model’s loss and accuracy. 

#### Step 3: The final step involved attempts to optimize the model to achieve a target predictive accuracy using any or all of the following methods of optimization such as;

1.	Adding one more layer to the existing two layers in the model (AlphabethSoupCity_Optimization.ipynb)
2.	Adding regularization techniques such as L2 regularization to the model to prevent overfitting (AlphabethSoupCity_Optimization2.ipynb)
3.	Calculating the importance of each feature and ranking them in descending order, and then incorporating only the first 20 most important features into the model(AlphabethSoupCity_Optimization3.ipynb).
4.	Combining all three optimization steps listed above, 1-3, but using the 25 most important features instead, and still keeping the existing number of neurons (AlphabethSoupCity_Optimization4.ipynb).

### Summary
Overall, the best optimization improvement on the model so far resulted when the importance of each feature was evaluated and ranked in descending order, and then incorporating only the first 20 most important features into the model(AlphabethSoupCity_Optimization3.ipynb). This resulted to an accuracy of 72.65% (with loss of 55.28%) in contrast with the initial result in AlphabethSoupCity.ipynb which had an accuracy of 72.52% (with loss of 55.65%). 

The second best improvement occurred when the L2 regularization technique was added to the model to prevent overfitting (AlphabethSoupCity_Optimization2.ipynb), thus increasing accuracy slightly to 72.61%, up from the initial result of 72.52%. 

### Recommendations

•	More experimentation could be done using different values for hyperparameters such as the number of nodes in each layer, the batch size, and the number of epochs
•	Trying different activation functions could potentially improve the performance of a model, as each function could have an advantage over others depending on specific prevailing factors
•	A different type of neural model architectural could be tried to ascertain if it performs better than the existing model
However, optimization should be carried out in a systematic way by making one change at a time and evaluating the impact on the model's performance. Incorporating the above techniques could result an enhanced model with better results
