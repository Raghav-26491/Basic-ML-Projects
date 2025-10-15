# Iris-Classification-System

This repository contains Python code for an Iris Classification System. The system performs classification tasks using the Iris dataset. The code involves data analysis, visualization, model training, and a Flask app for predictions.

Files:

•	Main.py

•	This file contains the main code for data analysis, visualization, model training, and saving the trained model using various machine learning algorithms.

•	deploy.py

•	This file contains the Flask app for deployment, where the saved model is loaded to predict the Iris species based on user inputs through a web interface.

•	index.html

•	This HTML file provides a simple form for user input. It contains input fields for Sepal Length, Sepal Width, Petal Length, and Petal Width. 
  Upon form submission, the predicted class will be displayed.
  
•	saved_model.sav

•	This file contains the trained model serialized using the pickle module. It is loaded by the Flask app for making predictions.

Instructions:

•	Running the Code: Ensure you have Python installed along with the required libraries (such as pandas, sklearn, Flask, seaborn, etc.).

•	Data: The code assumes the presence of the Iris dataset in a file named "Iris.csv." Adjust the file name or path if needed.

•	Main.py: Run this file to execute the data analysis, model training, and save the trained model as "saved_model.sav".

•	deploy.py: Run this file to deploy the trained model as a Flask web app. Access the predictions via the web interface by opening the index.html file in a browser.

Usage:

•	Main.py: Execute this file to perform data analysis, visualize the dataset, train machine learning models, and save the trained model.

•	deploy.py: Run this file to deploy the trained model as a Flask web app. Use the provided web interface (index.html) to enter input data and get predictions.


Requirements:

Ensure you have the necessary Python libraries installed. You can install them using pip install -r requirements.txt.
