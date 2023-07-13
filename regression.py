import string
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import metrics
import _pickle as pickle
from sklearn.impute import SimpleImputer
from nltk import PorterStemmer, word_tokenize
from nltk.corpus import stopwords
from scipy.stats._mstats_basic import winsorize
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import mean_squared_error, r2_score
from sklearn import linear_model
import random
from sklearn.preprocessing import StandardScaler, LabelEncoder, PolynomialFeatures, OneHotEncoder, MultiLabelBinarizer
from textblob import TextBlob
from xgboost import XGBRegressor
#read data

df=pd.read_csv(r"C:\Users\ALSHARKAOY\Downloads\hotel-regression-dataset.csv")
num_cols = ['Additional_Number_of_Scoring', 'Average_Score', 'Total_Number_of_Reviews_Reviewer_Has_Given',
            'Total_Number_of_Reviews', 'lat', 'lng']

# removing outliers with Interquartile Range
for i in num_cols:
    Q1 = df[i].quantile(0.25)
    Q3 = df[i].quantile(0.75)
    IQR = Q3 - Q1
    up_limit = Q3 + 1.5 * IQR
    low_limit = Q1 - 1.5 * IQR
    df.loc[df[i] > up_limit, i] = up_limit
    df.loc[df[i] < low_limit, i] = low_limit


cols = ('Reviewer_Nationality', 'Negative_Review', "Hotel_Name", 'Positive_Review', 'Hotel_Address')
def Feature_Encoder(X, cols):
    for c in cols:
        lbl = LabelEncoder()
        lbl.fit(list(X[c].values))
        X[c] = lbl.transform(list(X[c].values))
        filename = 'Feature_Encoder2'+c
        pickle.dump(lbl, open(filename, 'wb'))
    return X
df = Feature_Encoder(df, cols)


# Split data into features and target
X = df.drop(columns=['Reviewer_Score'])
y = df['Reviewer_Score']
df_train, df_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


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

    cols = ('Reviewer_Nationality', 'Negative_Review', "Hotel_Name", 'Positive_Review', 'Hotel_Address')
    data = Feature_Encoder(data, cols)

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

    data['lat'] = data['lat'].fillna(df['lat'].mean())
    data['lng'] = data['lng'].fillna(df['lng'].mean())

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


###########################################################################################
df_train = handle((df_train))
df_test = handle(df_test)

def preprocess_and_select_features(df_train, df_test, y_train):
    scaler = StandardScaler()
    num_features = ['Average_Score', 'Review_Total_Negative_Word_Counts',
                    'Review_Total_Positive_Word_Counts', 'Total_Number_of_Reviews',
                    'Total_Number_of_Reviews_Reviewer_Has_Given']

    df_train[num_features] = scaler.fit_transform(df_train[num_features])
    filename = 'select_feaatuers1ml1'
    pickle.dump(scaler, open(filename, 'wb'))

    df_test[num_features] = scaler.transform(df_test[num_features])

    selector = SelectKBest(k=7)
    X_train = selector.fit_transform(df_train, y_train)
    print (X_train)
    filename = 'select_feaatuers2ml1'
    pickle.dump(selector, open(filename, 'wb'))
    X_test = selector.transform(df_test)

    # Return the preprocessed and feature-selected data
    return selector,X_train, X_test



class linearRegression:
    def __init__(self, X_train, y_train, X_test, y_test):
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test


    def save_model(self):
        linear_regression = linear_model.LinearRegression()
        linear_regression.fit(self.X_train, self.y_train)
        filename = 'linear_model'
        pickle.dump(linear_regression, open(filename, 'wb'))


class polynomial:

    def __init__(self, X_train, y_train, X_test, y_test):
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test


    def save_model(self):
        poly_features = PolynomialFeatures(degree=2)
        X_train_poly = poly_features.fit_transform(self.X_train)
        X_test_poly = poly_features.transform(self.X_test)
        poly_model = LinearRegression()
        poly_model.fit(X_train_poly, self.y_train)

        filename = 'poly_model'
        pickle.dump(poly_model, open(filename, 'wb'))
        return X_test_poly



selector,X_train,X_test=preprocess_and_select_features(df_train, df_test, y_train)

# Simple linear Regression
LR = linearRegression(X_train, y_train, X_test, y_test)
LR.save_model()
lin_model = pickle.load(open('linear_model', 'rb'))
LR_Predicted = lin_model.predict(X_test)



print(f"The Mean Squared Error for Linear Regression (Test): {metrics.mean_squared_error(y_test, LR_Predicted):.2f}")
print(f"The Mean Absolute Error for Linear Regression (Test) : {metrics.mean_absolute_error(y_test, LR_Predicted):.2f}")
print(f"The R2_Score for Linear Regression (Test) : {metrics.r2_score(y_test, LR_Predicted):.2f}")
print()

plt.scatter(y_test, LR_Predicted)
plt.plot(y_test, y_test, color='red')
plt.xlabel('y-test data')
plt.ylabel('predicted data(linear)')
plt.show()


#Polynomial Regression
poly = polynomial(X_train, y_train, X_test, y_test)
X_test_poly = poly.save_model()

poly_model = pickle.load(open('poly_model', 'rb'))
poly_prediction = poly_model.predict(X_test_poly)


print(np.shape(X_test_poly))
print(f"The Mean Squared Error for Polynomial Regression (Test) : {metrics.mean_squared_error(y_test, poly_prediction):.2f}")
print(f"The Mean Absolute Error for Polynomial Regression (Test) : {metrics.mean_absolute_error(y_test, poly_prediction):.2f}")
print(f"The R2_Score for Polynomial Regression (Test) : {metrics.r2_score(y_test, poly_prediction):.2f}")

plt.scatter(y_test, poly_prediction)
plt.plot(y_test, y_test, color='red')
plt.xlabel('y-test data')
plt.ylabel('predicted data (polynomial)')
plt.show()


# def select_features( df_test):
#
#     num_features = ['Average_Score', 'Review_Total_Negative_Word_Counts',
#                     'Review_Total_Positive_Word_Counts', 'Total_Number_of_Reviews',
#                     'Total_Number_of_Reviews_Reviewer_Has_Given']
#
#     scaler = pickle.load(open('C:/Users/ALSHARKAOY/PycharmProjects/projectml2/select_feaatuers1ml1', 'rb'))
#     df_test[num_features] = scaler.transform(df_test[num_features])
#
#     selector = pickle.load(open('C:/Users/ALSHARKAOY/PycharmProjects/projectml2/select_feaatuers2ml1', 'rb'))
#     X_test = selector.transform(df_test)
#
#     # Return the preprocessed and feature-selected data
#     return  X_test





# LR = linear_model.Ridge()
# LR.fit(X_train, y_train)
# y_pred = LR.predict(X_test)
# mse = mean_squared_error(y_test, y_pred)
# r2 = r2_score(y_test, y_pred)
# print(f"MSE: {mse:.2f}")
# print(f"R2 Score: {r2 * 100:.2f}%")
#
#
# # Polynomial Regression Model
# poly_reg = PolynomialFeatures(degree=3, include_bias=False)
# X_poly_train = poly_reg.fit_transform(X_train)
# X_poly_test = poly_reg.transform(X_test)
# poly_reg_model = LinearRegression()
# poly_reg_model.fit(X_poly_train, y_train)
# y_pred1 = poly_reg_model.predict(X_poly_test)
# accuracy = r2_score(y_test, y_pred1)
# print("Polynomial Regression Performance:")
# print("Accuracy: {:.2f}%".format(accuracy * 100))
# poly_reg_mse = mean_squared_error(y_test, y_pred1)
# print(f"Polynomial Regression MSE: {poly_reg_mse}")
filename = 'x_train'
pickle.dump(X_train, open(filename, 'wb'))
filename = 'y_train'
pickle.dump(y_train, open(filename, 'wb'))