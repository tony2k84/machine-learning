#python3 -m pip install --upgrade pip
#python3 -m pip install matplotlib seaborn pandas numpy sklearn tensorflow

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from sklearn.metrics import classification_report,confusion_matrix
from tensorflow.keras.callbacks import EarlyStopping


df  = pd.read_csv('./BankChurners.csv')
gender_map = {'M':0, 'F': 1}
attrition_map = {'Existing Customer':0, 'Attrited Customer': 1}
education_map = {'High School':0,'College':1,'Graduate':2,'Post-Graduate':3,'Doctorate':4, 'Unknown':5, 'Uneducated':6}
marital_map = {'Married': 2, 'Single': 1, 'Unknown': 0, 'Divorced':3}
income_map = {
    '$60K - $80K': 3, 
    'Less than $40K': 1, 
    '$80K - $120K': 4, 
    '$40K - $60K': 2,
    '$120K +': 5,
    'Unknown': 0}
card_map = {'Blue': 0, 'Gold': 2, 'Silver': 1, 'Platinum': 2}

# # check missing data
# print(df.isnull().sum())

# Normalization/Transformation
df = df.drop('Naive_Bayes_Classifier_Attrition_Flag_Card_Category_Contacts_Count_12_mon_Dependent_count_Education_Level_Months_Inactive_12_mon_1',axis=1)
df = df.drop('Naive_Bayes_Classifier_Attrition_Flag_Card_Category_Contacts_Count_12_mon_Dependent_count_Education_Level_Months_Inactive_12_mon_2',axis=1)

df['gender'] = df['Gender'].map(gender_map)
df['attrition_flag'] = df['Attrition_Flag'].map(attrition_map)
df['education_level'] = df['Education_Level'].map(education_map)
df['marital_status'] = df['Marital_Status'].map(marital_map)
df['income_category'] = df['Income_Category'].map(income_map)
df['card_category'] = df['Card_Category'].map(card_map)

df = df.drop('Gender', axis=1)
df = df.drop('Attrition_Flag', axis=1)
df = df.drop('Education_Level', axis=1)
df = df.drop('Marital_Status', axis=1)
df = df.drop('Income_Category', axis=1)
df = df.drop('Card_Category', axis=1)


print(df.dtypes)
print("normalization: done")

# # Statistical analysis
# print(df.describe().transpose())
# print(df['attrition_flag'].unique())
# sns.countplot(df['attrition_flag'])
# plt.show()

# #Correlation
# print(df.corr()['attrition_flag'].sort_values()) # Total_Trans_Ct
# sns.scatterplot(x='attrition_flag',y='Total_Trans_Ct', data=df)
# plt.show()

# values
X = df.drop('attrition_flag', axis=1).values
y = df['attrition_flag'].values
print("values: done")

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=101)
print(X_train.shape) # 20
print(X_test.shape) # 20
print("splitting: done")

# Scaling
scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
print("scaling: done")

# Model
model = Sequential()

# NN
model.add(Dense(20, activation='relu'))
model.add(Dropout(0.3))

model.add(Dense(10, activation='relu'))
model.add(Dropout(0.25))

#outout
model.add(Dense(1, activation='sigmoid')) #as binary classification

print("model instantiation: done")

model.compile(
    optimizer='adam',
    loss='binary_crossentropy'
)

print("model compilation: done")

early_stop = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=25)
model.fit(
    x=X_train, 
    y=y_train,
    validation_data=(X_test, y_test),
    batch_size=128,
    epochs=600,
    callbacks=[early_stop]
)
print("model fitting: done")

# # View Losses
# losses = pd.DataFrame(model.history.history)
# losses.plot()
# plt.show()

# Prediction & Accuracy
predictions = model.predict_classes(X_test)
print(classification_report(y_test, predictions))
print(confusion_matrix(y_test, predictions))