# Spark--Peugeot-Sports-investment-in-racing-cars
Peugeot Sports' investment in racing cars analysis by using databricks
The report is to model and analyze the influencing factors of Peugeot Sports' investment in racing cars. The first is to predict which racers will complete the race through machine learning. Secondly, through deep learning Predict how many pitstops a driver will need per race.
Through data processing and building machine learning model, the K-fold Logistic Regression model and Random Forest Model were built respectively, and it was found that the first model performed better, with an AUC of 0.67. 60 of the 834 drivers were predicted to complete the game. 
After establishing the Single-label Multiclass Classification deep learning model, and comparing the performance of the 3 layer and 5 layer models, it is found that the 3 layer model performs better, has a smaller loss and a better fitting accuracy curve.
The machine learning model has a low AUC, and the performance of the model can be improved by adding new variables in the future. 
The deep learning model shows some overfitting, which can be optimized by Regularization or Dropout later.

