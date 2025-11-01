from sklearn.tree import DecisionTreeClassifier 
from sklearn.preprocessing import LabelEncoder 
import pandas as pd
#read data from .csv file
data=pd.read_csv('/home/ebrahemhany/Desktop/Data Warahaouse/Salaries.csv')
# drop target colmun from traning data 
inputs=data.drop('salary',axis='columns')
#tareget data
taregt=data['salary']
#trnsform all string inputs to integer inputs using encoded
inputs['rank_d']=LabelEncoder().fit_transform(inputs['rank'])
inputs['discipline_d']=LabelEncoder().fit_transform(inputs['discipline'])
inputs['gender_d']=LabelEncoder().fit_transform(inputs['gender'])

#drop string values from target data 
clear_input=inputs.drop(['rank','discipline','gender'],axis='columns')
#tran model using DecisionTree
model=DecisionTreeClassifier().fit(clear_input.values,taregt.values)

print(model.predict([[2,54,65,43,43]]))

