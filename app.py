import streamlit as st
import pandas as pd
from PIL import Image
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# Load the dataset
df = pd.read_csv(r'C:\Users\hp\Desktop\My Python\diabetes.csv')

# App title
st.title('Diabetes Checkup')

# Display data information
st.subheader('Training Data')
st.write(df.describe())

# Visualize the dataset
st.subheader('Visualization')
st.bar_chart(df)

# Split data into features (X) and target (Y)
x = df.drop(['Outcome'], axis=1)
y = df.iloc[:, -1]

# Train/test split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)

# Function to collect user input via sidebar
def user_report():
    pregnancies = st.sidebar.slider('Pregnancies', 0, 17, 3)
    glucose = st.sidebar.slider('Glucose', 0, 200, 120)
    bp = st.sidebar.slider('Blood Pressure', 0, 122, 70)
    skinthickness = st.sidebar.slider('Skin Thickness', 0, 100, 20)
    insulin = st.sidebar.slider('Insulin', 0, 846, 79)
    bmi = st.sidebar.slider('BMI', 0, 67, 20)
    dpf = st.sidebar.slider('Diabetes Pedigree Function', 0.0, 2.4, 0.47)
    age = st.sidebar.slider('Age', 21, 88, 33)

    user_report_data = {
        'Pregnancies': pregnancies,  
        'Glucose': glucose,          
        'BloodPressure': bp,         
        'SkinThickness': skinthickness,        
        'Insulin': insulin,          
        'BMI': bmi,                  
        'DiabetesPedigreeFunction': dpf,  
        'Age': age                   
    }
    report_data = pd.DataFrame(user_report_data, index=[0])
    return report_data

# Get user input data
user_data = user_report()

# Build and train the model
rf = RandomForestClassifier()
rf.fit(x_train, y_train)

# Show model accuracy
st.subheader('Accuracy: ')
st.write(f'{accuracy_score(y_test, rf.predict(x_test)) * 100:.2f}%')

# Predict user outcome
user_result = rf.predict(user_data)
st.subheader('Your Report: ')

if user_result[0] == 0:
    st.write('You are healthy')
else:
    st.write('You are not healthy')
