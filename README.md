# Artifical-Intelligence-
# HATE SPEECH DETECTION

# GROUP MEMBERS:
AROOBA MINHAS (21K- 3094)
BUSHRA KHAN (21K-3081)

# INTRODUCTION TO THE PROJECT:
This project implements a machine learning model for hate speech detection using
Scikit-learn, Pandas, NumPy, and NLTK libraries in Python. The code reads a dataset
containing tweets and their corresponding labels (Hate Speech, Offensive Language, or
No Hate and Offensive) from a CSV file, preprocesses the text by cleaning and
stemming the tweets, and converts the text into a numerical representation using
CountVectorizer. The data is then split into training and testing sets and fed into a
Decision Tree Classifier model, which is trained on the training set and evaluated on the
testing set.
The project also includes a Streamlit app that allows users to enter a tweet and receive
a prediction from the model on whether it is classified as Hate Speech, Normal
Language, or No Hate. This app demonstrates how the machine learning model can be
deployed in a user-friendly interface for practical use.

# METHODOLOGY:
The methodology of the code can be summarized as follows:
1. Import the necessary libraries for data processing and modeling, including
pandas, numpy, Scikit-learn, and NLTK.
2. Load the Twitter dataset into a pandas DataFrame and add a column to
represent the labels of the tweets.
3. Define a function to clean the text data of each tweet, including removing URLs,
punctuation, digits, and stop words, and performing stemming.
4. Apply the text cleaning function to the tweets in the DataFrame
5. Convert the cleaned tweets into a bag-of-words representation using
CountVectorizer.
6. Split the dataset into training and testing sets.
7. Train a DecisionTreeClassifier model on the training set.
8. Evaluate the accuracy of the model on the testing set.
9. Define a function to allow users to input a tweet and get a prediction of whether
the tweet contains hate speech or offensive language.
10. Implement the function in a Streamlit app to provide an interactive interface for
users to test the model.

# TOOLS, TECHNOLOGIES AND PROGRAMMING CONCEPTS
USED:
IDE: pycharm
LIBRARIES:
⦁ pandas
⦁ numpy
⦁ sklearn
⦁ nltk
⦁ streamlit

# CONCEPTS:
⦁ decision tree classifier
⦁ natural language processing

# HOW TO RUN THE PROGRAM:
As we are using the streamlit library in Python here so you cannot run this application
the same way you run your other Python programs. You need to write the command
mentioned below in your command prompt or terminal:
streamlit run filename.py

Once the above command is executed, it will open a link on your default web browser
which will show an end-to-end application where you have to write some text and it will
detect if the text contains hate speech or not, as shown in the results below.

# RESULTS:
The output for the )project is:
![p1](https://github.com/Arooba-Minhas/Artifical-Intelligence-/assets/123541169/6c37a05a-e451-475f-bb40-d9c354686949)
![p2](https://github.com/Arooba-Minhas/Artifical-Intelligence-/assets/123541169/0ca8093c-a613-48c3-903e-168e8a655a2f)
![p3](https://github.com/Arooba-Minhas/Artifical-Intelligence-/assets/123541169/3cdaa4b5-a621-482a-b444-47599b7dd6d6)

# SUMMARY:
The project demonstrates an effective approach to detecting hate speech and offensive
language in social media data, and the implementation in a Streamlit app provides a
user-friendly interface for testing the model.
