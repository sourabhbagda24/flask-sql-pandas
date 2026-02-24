import pymysql
import sqlalchemy
from sqlalchemy import create_engine
import numpy as np
import pandas as pd


engine = create_engine("mysql+pymysql://root:newpassword@localhost:3306/social")
query="select * from social_network"
df = pd.read_sql(query,engine)
print(df)


df.dropna()

from sklearn.preprocessing import LabelEncoder
lb=LabelEncoder()
df["Gender"]=lb.fit_transform(df["Gender"])
x= df.drop(columns=["Purchased"])
y=df["Purchased"]

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,random_state=42,test_size=0.2)

from sklearn. linear_model import LogisticRegression

lr = LogisticRegression()

lr.fit(x_train,y_train)
import joblib
model = joblib.dump(lr,'lr_model.pkl')
