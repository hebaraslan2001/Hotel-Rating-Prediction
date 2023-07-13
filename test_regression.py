import string
import matplotlib.pyplot as plt
import pandas as pd
import time
from sklearn import metrics
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from nltk import PorterStemmer, word_tokenize
from nltk.corpus import stopwords
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.linear_model import LinearRegression, Ridge, Lasso, LogisticRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn import linear_model
import random
from sklearn.preprocessing import StandardScaler, LabelEncoder, PolynomialFeatures, OneHotEncoder, MultiLabelBinarizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import _pickle as pickle
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
# C:/Users/ALSHARKAOY/PycharmProjects/projectml2/dtree_model
# lin_model = pickle.load(open('C:/Users/ALSHARKAOY/PycharmProjects/projectml2/linear_model', 'rb'))
# poly_model = pickle.load(open('C:/Users/ALSHARKAOY/PycharmProjects/projectml2/poly_model', 'rb'))

cols = ('Reviewer_Nationality', 'Negative_Review', "Hotel_Name", 'Positive_Review', 'Hotel_Address')
def Feature_Encoder(X, cols):
    for c in cols:
        lbl = pickle.load(open('C:/Users/ALSHARKAOY/PycharmProjects/projectml2/Feature_Encoder2' + c, 'rb'))
        X[c] = lbl.transform(list(X[c].values))
    return X



def handle(data):

    # handle day and days word
    data['Review_Date'] = pd.to_datetime(data['Review_Date'])
    data['days_since_review'] = data['days_since_review'].astype(str)
    data['days_since_review'] = data['days_since_review'].str.replace(' day', '')
    data['days_since_review'] = data['days_since_review'].str.replace('s', '')
    data['days_since_review'] = data['days_since_review'].astype('int64')

    # handle space
    data['Reviewer_Nationality'] = data['Reviewer_Nationality'].astype(str)
    data['Reviewer_Nationality'] = data['Reviewer_Nationality'].str.replace(' ', '')


    data['Review_Date'] = pd.to_datetime(data['Review_Date'])
    data['Review_Date'] = (data['Review_Date']).dt.year
    data['Review_Date'] = data['Review_Date'].astype('int64')

    data['Leisure'] = data['Tags'].map(lambda x: 1 if ' Leisure trip ' in x else 0)
    data['Business'] = data['Tags'].map(lambda x: 2 if ' Business trip ' in x else 0)
    data['Trip_type'] = data['Leisure'] + data['Business']
    data['Trip_type'] = data[data['Trip_type'] == 0]['Trip_type'].map(lambda x: 1 if random.random() > 0.2 else 2)
    data['Trip_type'] = data['Trip_type'].fillna(0)
    data['Trip_type'] = data['Trip_type'] + data['Business'] + data['Leisure']
    data.drop(['Leisure', 'Business'], axis=1, inplace=True)

    data['lat'] = data['lat'].fillna(data['lat'].mean())
    data['lng'] = data['lng'].fillna(data['lng'].mean())

    # Couple or Solo or Group or Family_with_older children or Family with younger Children
    data['Solo'] = data['Tags'].map(lambda x: 1 if ' Solo traveler ' in x else 0)
    data['Couple'] = data['Tags'].map(lambda x: 2 if ' Couple ' in x else 0)
    data['Group'] = data['Tags'].map(lambda x: 3 if ' Group ' in x else 0)
    data['Family_with_young_children'] = data['Tags'].map(lambda x: 4 if ' Family with young children ' in x else 0)
    data['Family_with_older_children'] = data['Tags'].map(lambda x: 5 if ' Family with older children ' in x else 0)
    data['guests'] = data['Solo'] + data['Couple'] + data['Group'] + data['Family_with_young_children'] + data[
        'Family_with_older_children']

    data.drop(['Solo', 'Couple', 'Family_with_young_children', 'Group', 'Family_with_older_children'], axis=1,
              inplace=True)
    data.drop('Tags', axis=1, inplace=True)

    data['Review_Total_Positive_Word_Counts'] = data['Review_Total_Positive_Word_Counts'].astype(int)
    data['Review_Total_Negative_Word_Counts'] = data['Review_Total_Negative_Word_Counts'].astype(int)

    return data



def preprocess_and_select_features( df_test):

    num_features = ['Average_Score', 'Review_Total_Negative_Word_Counts',
                    'Review_Total_Positive_Word_Counts', 'Total_Number_of_Reviews',
                    'Total_Number_of_Reviews_Reviewer_Has_Given']

    scaler = pickle.load(open('C:/Users/ALSHARKAOY/PycharmProjects/projectml2/select_feaatuers1ml1', 'rb'))
    df_test[num_features] = scaler.transform(df_test[num_features])

    selector = pickle.load(open('C:/Users/ALSHARKAOY/PycharmProjects/projectml2/select_feaatuers2ml1', 'rb'))
    X_test = selector.transform(df_test)

    # Return the preprocessed and feature-selected data
    return  X_test


def testFile(path):

    test_data = pd.read_csv(path)
    print(1)

    print(test_data.head())
    #test_data.fillna(test_data.mean(), inplace=True) #filling nan values with mean
    # # Rate encoding
    # lE = LabelEncoder()
    # lE.fit(test_data['Rate'])
    # test_data['Rate'] = lE.transform(test_data['Rate'])

    X_testt = test_data.drop("Reviewer_Score",axis=1)
    Y_testt = test_data['Reviewer_Score'].values
    print(2)

    cols = ('Reviewer_Nationality', 'Negative_Review', "Hotel_Name", 'Positive_Review', 'Hotel_Address')

    def Feature_Encoder2(X_testt, cols):
        for c in cols:
            lbl = LabelEncoder()
            lbl.fit(list(X_testt[c].values))
            X_testt[c] = lbl.transform(list(X_testt[c].values))

        return X_testt

    X_testt = Feature_Encoder2(X_testt, cols)


    print(3)

    X_testt = handle(X_testt)  # preprocessing
    # X_testt[X_testt < 0] = 0
    X_testt = preprocess_and_select_features(X_testt)  # feature selection
    print (4)

    print(X_testt)



    # LR = linear_model.Ridge()
    # LR.fit(X_testt,Y_testt)
    # y_pred = LR.predict(X_testt)
    # mse = mean_squared_error(Y_testt, y_pred)
    # r2 = r2_score(Y_testt, y_pred)
    # print(f"MSE: {mse:.2f}")
    # print(f"R2 Score: {r2 * 100:.2f}%")
    #
    # print(10)
    # # Polynomial Regression Model
    # poly_reg = PolynomialFeatures(degree=5, include_bias=False)
    # X_poly_test = poly_reg.fit_transform(X_testt)
    # poly_reg_model = LinearRegression()
    # poly_reg_model.fit(X_poly_test, Y_testt)
    # y_pred1 = poly_reg_model.predict(X_poly_test)
    # accuracy = r2_score(Y_testt, y_pred1)
    # print("Polynomial Regression Performance:")
    # print("Accuracy: {:.2f}%".format(accuracy * 100))
    # poly_reg_mse = mean_squared_error(Y_testt, y_pred1)
    # print(f"Polynomial Regression MSE: {poly_reg_mse}")

    X_train=pickle.load(open('x_train', 'rb'))
    y_train = pickle.load(open('y_train', 'rb'))

    LR = linear_model.Ridge()
    LR.fit(X_testt, Y_testt)
    y_pred = LR.predict(X_testt)
    mse = mean_squared_error(Y_testt, y_pred)
    r2 = r2_score(Y_testt, y_pred)
    print(f"MSE: {mse:.2f}")
    print(f"R2 Score: {r2 * 100:.2f}%")

    print(10)
    # Polynomial Regression Model
    poly_reg = PolynomialFeatures(degree=2, include_bias=False)
    X_poly_train = poly_reg.fit_transform(X_train)
    X_poly_test = poly_reg.transform(X_testt)
    poly_reg_model = LinearRegression()
    poly_reg_model.fit(X_poly_train, y_train)
    y_pred1 = poly_reg_model.predict(X_poly_test)
    accuracy = r2_score(Y_testt, y_pred1)
    print("Polynomial Regression Performance:")
    print("Accuracy: {:.2f}%".format(accuracy * 100))
    poly_reg_mse = mean_squared_error(Y_testt, y_pred1)
    print(f"Polynomial Regression MSE: {poly_reg_mse}")

    print('Using The Test File:')







testFile('C:/Users/ALSHARKAOY/Desktop/hotel-tas-test-regression.csv')