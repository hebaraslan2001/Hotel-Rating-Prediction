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

# read data
df=pd.read_csv(r"C:\Users\ALSHARKAOY\Desktop\hotel-tas-test-classification.csv");
num_cols = ['Additional_Number_of_Scoring', 'Average_Score', 'Total_Number_of_Reviews_Reviewer_Has_Given',
            'Total_Number_of_Reviews', 'lat', 'lng']
df.head()
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
        filename = 'Feature_Encoder'+c
        pickle.dump(lbl, open(filename, 'wb'))
    return X
df = Feature_Encoder(df, cols)



dictionary = dict({'Low_Reviewer_Score': 0, 'Intermediate_Reviewer_Score': 1, 'High_Reviewer_Score': 2})
df['Reviewer_Score'] = df['Reviewer_Score'].map(dictionary).astype("int64")
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
#handle null
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
    filename = 'select_feaatuers1'
    pickle.dump(scaler, open(filename, 'wb'))

    df_test[num_features] = scaler.transform(df_test[num_features])

    selector = SelectKBest(f_regression, k=7)
    X_train = selector.fit_transform(df_train, y_train)
    filename = 'select_feaatuers2'
    pickle.dump(selector, open(filename, 'wb'))
    X_test = selector.transform(df_test)

    # Return the preprocessed and feature-selected data
    return selector,X_train, X_test


training_time = []
testing_time = []


class Logistic:
    def __init__(self, X_train, y_train, X_test, y_test):
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test

    def save_model(self):
        logistic_regression = LogisticRegression(random_state=0, max_iter=20)
        start_time_train = time.time()
        logistic_regression.fit(self.X_train, self.y_train)
        end_time_train = time.time()
        training_times = end_time_train - start_time_train
        training_time.append(training_times)
        filename = 'logistic_model'
        pickle.dump(logistic_regression, open(filename, 'wb'))


class DecisionTree:
    def __init__(self, X_train, y_train, X_test, y_test):
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test

    def save_model(self):
        Decision_tree = DecisionTreeClassifier(max_depth=5)
        start_time_train = time.time()
        Decision_tree.fit(self.X_train, self.y_train)
        end_time_train = time.time()
        training_times = end_time_train - start_time_train
        training_time.append(training_times)
        filename = 'dtree_model'
        pickle.dump(Decision_tree, open(filename, 'wb'))

class DecisionTree2:
    def __init__(self, X_train, y_train, X_test, y_test):
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test

    def save_model(self):
        Decision_tree = DecisionTreeClassifier(max_depth=50)
        start_time_train = time.time()
        Decision_tree.fit(self.X_train, self.y_train)
        end_time_train = time.time()
        training_times = end_time_train - start_time_train
        training_time.append(training_times)
        filename = 'dtree_model2'
        pickle.dump(Decision_tree, open(filename, 'wb'))

class DecisionTree3:
    def __init__(self, X_train, y_train, X_test, y_test):
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test

    def save_model(self):
        Decision_tree = DecisionTreeClassifier(max_depth=100)
        start_time_train = time.time()
        Decision_tree.fit(self.X_train, self.y_train)
        end_time_train = time.time()
        training_times = end_time_train - start_time_train
        training_time.append(training_times)
        filename = 'dtree_model3'
        pickle.dump(Decision_tree, open(filename, 'wb'))






class Random_forest:
    def __init__(self, X_train, y_train, X_test, y_test):
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test

    def save_model(self):
        Random_forest = RandomForestClassifier(max_depth=10, random_state=0)
        start_time_train = time.time()
        Random_forest.fit(self.X_train, self.y_train)
        end_time_train = time.time()
        training_times = end_time_train - start_time_train
        training_time.append(training_times)
        filename = 'random_model'
        pickle.dump(Random_forest, open(filename, 'wb'))

class Random_forest2:
    def __init__(self, X_train, y_train, X_test, y_test):
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test

    def save_model(self):
        Random_forest = RandomForestClassifier(max_depth=50, random_state=0)
        start_time_train = time.time()
        Random_forest.fit(self.X_train, self.y_train)
        end_time_train = time.time()
        training_times = end_time_train - start_time_train
        training_time.append(training_times)
        filename = 'random_model2'
        pickle.dump(Random_forest, open(filename, 'wb'))




class Random_forest3:
    def __init__(self, X_train, y_train, X_test, y_test):
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test

    def save_model(self):
        Random_forest = RandomForestClassifier(max_depth=90, random_state=0)
        start_time_train = time.time()
        Random_forest.fit(self.X_train, self.y_train)
        end_time_train = time.time()
        training_times = end_time_train - start_time_train
        training_time.append(training_times)
        filename = 'random_model3'
        pickle.dump(Random_forest, open(filename, 'wb'))


# class KNN:
#     def __init__(self, X_train, y_train, X_test, y_test):
#         self.X_train = X_train
#         self.y_train = y_train
#         self.X_test = X_test
#         self.y_test = y_test
#
#     def save_model(self):
#         knn = KNeighborsClassifier(n_neighbors=50)
#         start_time_train = time.time()
#         knn.fit(self.X_train, self.y_train)
#         end_time_train = time.time()
#         training_times = end_time_train - start_time_train
#         training_time.append(training_times)
#         filename = 'knn_model'
#         pickle.dump(KNN, open(filename, 'wb'))


selector,X_train,X_test=preprocess_and_select_features(df_train, df_test, y_train)


# Logistic Regression

LR = Logistic(X_train, y_train, X_test, y_test)
LR.save_model()
logistic_model = pickle.load(open('logistic_model', 'rb'))
start_time_test = time.time()
LR_predicted = logistic_model.predict(X_test)
end_time_test = time.time()
testing_times = end_time_test - start_time_test
testing_time.append(testing_times)
print(f"The Accuracy for Logistic Regression: {metrics.accuracy_score(y_test, LR_predicted):.2f}")



# Decision Tree
DT = DecisionTree(X_train, y_train, X_test, y_test)
DT.save_model()
dt_model = pickle.load(open('dtree_model', 'rb'))
start_time_test = time.time()
dt_predicted = dt_model.predict(X_test)
end_time_test = time.time()
testing_times = end_time_test - start_time_test
testing_time.append(testing_times)
print(f"The Accuracy for Decision tree Regression: {metrics.accuracy_score(y_test, dt_predicted):.2f}")

# Decision Tree
DT2 = DecisionTree2(X_train, y_train, X_test, y_test)
DT2.save_model()
dt_model2 = pickle.load(open('dtree_model2', 'rb'))
start_time_test = time.time()
dt_predicted2 = dt_model.predict(X_test)
end_time_test = time.time()
testing_times = end_time_test - start_time_test
testing_time.append(testing_times)
print(f"The Accuracy for Decision tree Regression: {metrics.accuracy_score(y_test, dt_predicted2):.2f}")



# Decision Tree
DT3 = DecisionTree3(X_train, y_train, X_test, y_test)
DT3.save_model()
dt_model3 = pickle.load(open('dtree_model3', 'rb'))
start_time_test = time.time()
dt_predicted3 = dt_model.predict(X_test)
end_time_test = time.time()
testing_times = end_time_test - start_time_test
testing_time.append(testing_times)
print(f"The Accuracy for Decision tree Regression: {metrics.accuracy_score(y_test, dt_predicted3):.2f}")





# Random Forest
RF = Random_forest(X_train, y_train, X_test, y_test)
RF.save_model()
RF_model = pickle.load(open('random_model', 'rb'))
start_time_test = time.time()
RF_predicted = RF_model.predict(X_test)
end_time_test = time.time()
testing_times = end_time_test - start_time_test
testing_time.append(testing_times)
print(f"The Accuracy for Random Forest: {metrics.accuracy_score(y_test, RF_predicted):.2f}")


# Random Forest2
RF2 = Random_forest2(X_train, y_train, X_test, y_test)
RF2.save_model()
RF_model2 = pickle.load(open('random_model2', 'rb'))
start_time_test = time.time()
RF_predicted2 = RF_model2.predict(X_test)
end_time_test = time.time()
testing_times = end_time_test - start_time_test
testing_time.append(testing_times)
print(f"The Accuracy for Random Forest2: {metrics.accuracy_score(y_test, RF_predicted2):.2f}")



# Random Forest3
RF3 = Random_forest3(X_train, y_train, X_test, y_test)
RF3.save_model()
RF_model3 = pickle.load(open('random_model3', 'rb'))
start_time_test = time.time()
RF_predicted3 = RF_model3.predict(X_test)
end_time_test = time.time()
testing_times = end_time_test - start_time_test
testing_time.append(testing_times)
print(f"The Accuracy for Random Forest3: {metrics.accuracy_score(y_test, RF_predicted3):.2f}")

# #KNN
# # create a KNN object with k = 5
# knn = KNeighborsClassifier(n_neighbors=50)
# # fit the model to your training data
# knn.fit(X_train, y_train)
# # save the model using pickle
# pickle.dump(knn, open('knn_model.pkl', 'wb'))
# # load the model using pickle
# knn_model = pickle.load(open('knn_model.pkl', 'rb'))
# # make predictions on your test data
#
# # calculate the accuracy of the model
#
# start_time_test = time.time()
# knn_predicted = knn_model.predict(X_test)
# end_time_test = time.time()
# testing_times = end_time_test - start_time_test
# testing_time.append(testing_times)
# accuracy = metrics.accuracy_score(y_test, knn_predicted)
# print(f"The Accuracy for KNN1: {accuracy:.2f}")

models = ['LR', 'dt','dt2','dt3', 'RF','RF2','RF3']
plt.bar(models, training_time, color='lightblue', label='Training Time')
plt.xlabel('Models')
plt.ylabel('Time (seconds)')
plt.title('Training Times for Each Model')
plt.legend()
plt.show()

models_test = ['LR', 'dt','dt2','dt3', 'RF','RF2','RF3']
plt.bar(models_test, testing_time, color='limegreen', label='Testing Time')
plt.xlabel('Models')
plt.ylabel('Time (seconds)')
plt.title(' Testing Times for Each Model')
plt.legend()
plt.show()

accuracy_scores = [metrics.accuracy_score(y_test, LR_predicted),
                   metrics.accuracy_score(y_test, dt_predicted),
                   metrics.accuracy_score(y_test, dt_predicted2),
                   metrics.accuracy_score(y_test, dt_predicted3),
                   metrics.accuracy_score(y_test, RF_predicted),
                   metrics.accuracy_score(y_test, RF_predicted2),
                   metrics.accuracy_score(y_test, RF_predicted3)]

models_acc = ['LR', 'dt','dt2','dt3', 'RF','RF2','RF3']
plt.bar(models_acc, accuracy_scores, color='pink')
plt.xlabel('Models')
plt.ylabel('Accuracy')
plt.title('Classification Accuracy for Each Model')

# Display the accuracy values on top of each bar
for i, v in enumerate(accuracy_scores):
    plt.text(i, v, str(round(v, 2)), ha='center', va='bottom')
plt.show()
print(training_time)
print(testing_time)





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
# poly_reg = PolynomialFeatures(degree=2, include_bias=False)
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
















