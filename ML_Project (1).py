import numpy as np
import pandas as pd
import re
# import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn import linear_model
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import accuracy_score, mean_squared_error
from sklearn import metrics
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif
from joblib import Parallel, delayed
import joblib
import random

df=pd.read_csv(r"C:\Users\ALSHARKAOY\Downloads\hotel-regression-dataset.csv")
df, df_test = train_test_split(df, test_size=0.2, shuffle=True, random_state=44)
y_test = df_test['Reviewer_Score']

# get the first five rows
df.head()
# check for NA values in dataset
print(df.isnull().sum())

# checking the % of NAs columnwise
df.isnull().sum() * 100 / df.shape[0]  #################################

df['lat'] = df['lat'].fillna(df['lat'].mean())
df['lng'] = df['lng'].fillna(df['lng'].mean())
df['Review_Date'] = pd.to_datetime(df['Review_Date'])

df.info()

df['days_since_review'] = df['days_since_review'].astype(str)
df['days_since_review'] = df['days_since_review'].str.replace(' day', '')
df['days_since_review'] = df['days_since_review'].str.replace('s', '')
df['days_since_review'] = df['days_since_review'].astype('int64')

# I will split the Date and pick the year
import datetime as dt

df['Review_Date'] = pd.to_datetime(df['Review_Date'])
df['Review_Date'] = (df['Review_Date']).dt.year
df['Review_Date'] = df['Review_Date'].astype('int64')

# Plotting the Average scores of the hotels
df_sd = df[['Hotel_Name', 'Average_Score']].drop_duplicates()  # Dropping any duplicates
plt.figure(figsize=(14, 4))
# sns.countplot(x = 'Average_Score',data = df_sd,color = 'green')
# From the graph below, we can notice that most hotels were given scores ranging from 8.1 to 8.9
df.Average_Score.describe()
# There are 34 unique average scores
# Minimum Average score is 5.2
# Maximum Average score is 9.8
# 25% of the hotels have an Average_score of 8.1 - 5.2
# 50% of the hotels have an Average_score of 8.4 - 8.2
# 75% of the hotels have an Average_score of 8.8 - 8.5

df[df.Average_Score >= 8.8][['Hotel_Name'
    , 'Average_Score'
    , 'Total_Number_of_Reviews']].drop_duplicates().sort_values(by='Total_Number_of_Reviews', ascending=False)[:10]
# We now attempt to find the 10 most popular hotels based on 'Total number of reviews, Average score greater than 8.8, and the Hotel names'
listt=[]
for l in df ["Tags"]:
    if len (re.findall("Stayed\s+(\d+)\s+nights",l)) != 0 :
        listt.append(int(re.findall("Stayed\s+(\d+)\s+nights",l)[0]))
    else :
        listt.append(1)

df['stat_nights']=listt
print(df["stat_nights"].dtype)
df['Leisure'] = df['Tags'].map(lambda x: 1 if ' Leisure trip ' in x else 0)
df['Business'] = df['Tags'].map(lambda x: 2 if ' Business trip ' in x else 0)
df['Trip_type'] = df['Leisure'] + df['Business']

df['Trip_type'] = df[df['Trip_type'] == 0]['Trip_type'].map(lambda x: 1 if random.random() > 0.2 else 2)
df['Trip_type'] = df['Trip_type'].fillna(0)
df['Trip_type'] = df['Trip_type'] + df['Business'] + df['Leisure']
df.drop(['Leisure', 'Business'], axis=1, inplace=True)

df['Trip_type'].value_counts()

# Couple or Solo or Group or Family_with_older children or Family with younger Children
df['Solo'] = df['Tags'].map(lambda x: 1 if ' Solo traveler ' in x else 0)
df['Couple'] = df['Tags'].map(lambda x: 2 if ' Couple ' in x else 0)
df['Group'] = df['Tags'].map(lambda x: 3 if ' Group ' in x else 0)
df['Family_with_young_children'] = df['Tags'].map(lambda x: 4 if ' Family with young children ' in x else 0)
df['Family_with_older_children'] = df['Tags'].map(lambda x: 5 if ' Family with older children ' in x else 0)
df['guests'] = df['Solo'] + df['Couple'] + df['Group'] + df['Family_with_young_children'] + df[
    'Family_with_older_children']
df.drop(['Solo', 'Couple', 'Family_with_young_children', 'Group', 'Family_with_older_children'], axis=1, inplace=True)

df.guests.value_counts()

df.head()

# Replacing "United Kingdom with "UK"
df.Hotel_Address = df.Hotel_Address.str.replace("United Kingdom", "UK")
# Now I will split the address and pick the last word in the address to identify the country
df["countries"] = df.Hotel_Address.apply(lambda x: x.split(' ')[-1])
print(df.countries.unique())

sc = StandardScaler()

cools = df.drop('Reviewer_Score', axis=1).select_dtypes(include=['int', 'float']).columns
df[cools] = sc.fit_transform(df[cools])
# joblib.dump(sc, 'sc.pkl')

# plt.boxplot(df)

df.drop('Tags', axis=1, inplace=True)
for column in df.select_dtypes(include=['int', 'float']).columns:
    plt.figure()
    df.boxplot([column])


def drop_outliers(df, field_name):
    distance = 1.5 * (np.percentile(df[field_name], 95) - np.percentile(df[field_name], 5))
    df.drop(df[df[field_name] > distance + np.percentile(df[field_name], 75)].index, inplace=True)
    df.drop(df[df[field_name] < np.percentile(df[field_name], 25) - distance].index, inplace=True)
    return df


for column in df.select_dtypes(include=['int', 'float']).columns:
    df = drop_outliers(df, column)

Encoding_Columns = df.select_dtypes(exclude=['int', 'float']).columns
Encoding_Columns
df['Hotel_Address'] = df['Hotel_Address'].str.replace(' ', '')

from sklearn.preprocessing import OrdinalEncoder

enc = OrdinalEncoder()


class MultiColumnLabelEncoder:
    def __init__(self, columns=None):
        self.columns = columns  # array of column names to encode

    def fit(self, X, y=None):
        return self  # not relevant here

    def transform(self, X):
        '''
        Transforms columns of X specified in self.columns using
        LabelEncoder(). If no columns specified, transforms all
        columns in X.
        '''
        output = X.copy()
        if self.columns is not None:
            for col in self.columns:
                output[col] = LabelEncoder().fit_transform(output[col])
        else:
            for colname, col in output.iteritems():
                output[colname] = LabelEncoder().fit_transform(col)
        return output

    def fit_transform(self, X, y=None):
        return self.fit(X, y).transform(X)


mu = MultiColumnLabelEncoder(columns=Encoding_Columns)
df = mu.fit_transform(df)
X_train = df.iloc[:, :-1]
Y_train = df.iloc[:, -1]
# Set up the matplot figure

# Draw the heatmap using seaborn

# #Feature Selection
# #Get the correlation between the features
# corr = df.corr()
# #Top 50% Correlation training features with the Value
# top_feature = corr.index[abs(corr['Reviewer_Score'])>0.02]
# #Correlation plot
# top_corr = df[top_feature].corr()
# sns.heatmap(top_corr, annot=True)
# top_feature = top_feature.delete(-1)
# corr_df = corr_df[top_feature]

from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_regression

# corr = df.corr()
# top_feature = corr.index[abs(corr['Reviewer_Score']) > 0.05]
# plt.subplots(figsize=(10, 6))
# top_corr = df[top_feature].corr()
# sns.heatmap(top_corr, annot=True)
# plt.show()
sbest = SelectKBest(f_regression, k=16)
X_new = sbest.fit_transform(X_train, Y_train)
import joblib

joblib.dump(sbest, 'sbest.pkl')
joblib.dump(mu, 'mu.pkl')
joblib.dump(sc, 'sc.pkl')


def data_pre(df, sc, sbest, mu):



    df['lat'] = df['lat'].fillna(df['lat'].mean())  # wep scaping
    df['lng'] = df['lng'].fillna(df['lng'].mean())

    df['Review_Date'] = pd.to_datetime(df['Review_Date'])
    df['Review_Date'] = (df['Review_Date']).dt.year
    df['Review_Date'] = df['Review_Date'].astype('int64')
    df['days_since_review'] = df['days_since_review'].astype(str)
    df['days_since_review'] = df['days_since_review'].str.replace(' day', '')
    df['days_since_review'] = df['days_since_review'].str.replace('s', '')
    df['days_since_review'] = df['days_since_review'].astype('int64')
    df['Review_Date'] = df['Review_Date'].astype('int64')
    listt = []
    for l in df["Tags"]:
        if len(re.findall("Stayed\s+(\d+)\s+nights", l)) != 0:
            listt.append(int(re.findall("Stayed\s+(\d+)\s+nights", l)[0]))
        else:
            listt.append(1)

    df['stat_nights'] = listt
    print(df["stat_nights"].dtype)
    df['Leisure'] = df['Tags'].map(lambda x: 1 if ' Leisure trip ' in x else 0)
    df['Business'] = df['Tags'].map(lambda x: 2 if ' Business trip ' in x else 0)
    df['Trip_type'] = df['Leisure'] + df['Business']

    import random

    df['Trip_type'] = df[df['Trip_type'] == 0]['Trip_type'].map(lambda x: 1 if random.random() > 0.2 else 2)
    df['Trip_type'] = df['Trip_type'].fillna(0)
    df['Trip_type'] = df['Trip_type'] + df['Business'] + df['Leisure']
    df.drop(['Leisure', 'Business'], axis=1, inplace=True)

    df['Trip_type'].value_counts()

    # Couple or Solo or Group or Family_with_older children or Family with younger Children
    df['Solo'] = df['Tags'].map(lambda x: 1 if ' Solo traveler ' in x else 0)
    df['Couple'] = df['Tags'].map(lambda x: 2 if ' Couple ' in x else 0)
    df['Group'] = df['Tags'].map(lambda x: 3 if ' Group ' in x else 0)
    df['Family_with_young_children'] = df['Tags'].map(lambda x: 4 if ' Family with young children ' in x else 0)
    df['Family_with_older_children'] = df['Tags'].map(lambda x: 5 if ' Family with older children ' in x else 0)
    df['guests'] = df['Solo'] + df['Couple'] + df['Group'] + df['Family_with_young_children'] + df[
        'Family_with_older_children']
    df.drop(['Solo', 'Couple', 'Family_with_young_children', 'Group', 'Family_with_older_children', 'Tags'], axis=1,
            inplace=True)

    df.guests.value_counts()

    df.head()

    # Replacing "United Kingdom with "UK"
    df.Hotel_Address = df.Hotel_Address.str.replace("United Kingdom", "UK")
    # Now I will split the address and pick the last word in the address to identify the country
    df["countries"] = df.Hotel_Address.apply(lambda x: x.split(' ')[-1])
    print(df.countries.unique())

    # # Plotting with matplotlib
    # plt.figure(figsize=(12, 3))
    # plt.title('Hotel distribution in European countries')
    # df.countries.value_counts().plot.barh(color='green')
    Encoding_Columns = df.select_dtypes(exclude=['int', 'float']).columns

    # scaling

    cools = df.drop('Reviewer_Score', axis=1).select_dtypes(include=['int', 'float']).columns
    # sc = joblib.load('SC.pkl')
    df[cools] = sc.transform(df[cools])
    df = df.replace(r"\s*\.*", "", regex=True)

    Encoding_Columns = df.select_dtypes(exclude=['int', 'float']).columns
    df = mu.transform(df)

    if "Reviewer_Score" in df.columns:
        X = df.iloc[:, :-1]
        Y = df.iloc[:, -1]
    else:
        X = df

    X_new = sbest.transform(X)
    return X_new


sbest = joblib.load('sbest.pkl')
mu = joblib.load('mu.pkl')
sc = joblib.load('sc.pkl')
df_test_new = data_pre(df_test, sc, sbest, mu)

# In[817]:


x_test, X_val, y_new_tes, y_val = train_test_split(df_test_new, y_test, test_size=0.4, random_state=42)

# In[818]:


df_test_new

# Simple linear Regression
LR = linear_model.Ridge()
LR.fit(X_new, Y_train)

prediction_V = LR.predict(X_val)
prediction = LR.predict(x_test)
prediction_x = LR.predict(X_new)
print( f"The Mean Squared Error for Linear Regression (Validation) : {metrics.mean_squared_error(y_val, prediction_V):.2f}")
print( f"The Mean Squared Error for Linear Regression (train) : {metrics.mean_squared_error(Y_train, prediction_x):.2f}")
print(f"The Mean Squared Error for Linear Regression (Test): {metrics.mean_squared_error(y_new_tes, prediction):.2f}")
print(f"The Mean Absolute Error for Linear Regression (Test) : {metrics.mean_absolute_error(y_new_tes, prediction):.2f}")
print(f"The R2_Score for Linear Regression (Test) : {LR.score(X_new, Y_train):.2f}")
print(f"The R2_Score for Linear Regression (Test) : {LR.score(X_val, y_val):.2f}")
plt.scatter(y_new_tes, prediction)
plt.plot(y_new_tes, y_new_tes, color='red')
plt.xlabel('y-test data')
plt.ylabel('predicted data(linear)')
plt.show()

# Polynomial Regression
poly_features = PolynomialFeatures(degree=3)
X_train_poly = poly_features.fit_transform(X_new)
X_test_poly = poly_features.transform(x_test)
X_val_poly = poly_features.transform(X_val)
poly_model = LinearRegression()
poly_model.fit(X_train_poly, Y_train)
poly_prediction = poly_model.predict(X_test_poly)
poly_prediction_V = poly_model.predict(X_val_poly)
prediction_x = poly_model.predict(X_train_poly)

print(
    f"The Mean Squared Error for Polynomial Regression (Validation) : {metrics.mean_squared_error(y_val, poly_prediction_V):.2f}")
print(
    f"The Mean Squared Error for Linear Regression (train) : {metrics.mean_squared_error(Y_train, prediction_x):.2f}")
print(
    f"The Mean Squared Error for Polynomial Regression (Test) : {metrics.mean_squared_error(y_new_tes, poly_prediction):.2f}")
print(
    f"The Mean Absolute Error for Polynomial Regression (Test) : {metrics.mean_absolute_error(y_new_tes, poly_prediction):.2f}")
print(f"The R2_Score for Polynomial Regression (Test) : {metrics.r2_score(y_new_tes, poly_prediction):.2f}")

plt.scatter(y_new_tes, poly_prediction)
plt.plot(y_new_tes, y_new_tes, color='red')
plt.xlabel('y-test data')
plt.ylabel('predicted data (polynomial)')
plt.show()