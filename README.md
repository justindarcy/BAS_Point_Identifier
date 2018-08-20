# BAS_Point_Identifier

This script uses Tkinter to create a GUI that gives the user two options. 

# Creating the model with machine learning
The first option is build a machine learning model using textblob and a Naive Bayes Classifier which is all done through file selection and GUI prompts. Once the model is created a pickled model file will automatically be saved in the python file source folder. The format for training the machine learning model is to have the target name as the first column and the existing name in the second column.

# Utilizing the model for text classification
The second option is to utilize an already created and pickled model to classify text input from a excel file.
The intended use for this program is to identify and rename building automation system points names from the programmatic point name to a commonly used abbreviation.  The input data for renaming should be formatted as headers (ie: top row) of an excel document. The abbreviation will be added to the second row of the document upon export.  
