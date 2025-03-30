import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import pickle
import random

# Title of the Streamlit app
st.title('Glass Classification Data Analysis & Prediction')

# Load the dataset
dataset = pd.read_csv("C:\\Users\\KUBER\\OneDrive\\Documents\\Team Losser\\ITR\\glass.csv")

# Display the dataset
with st.expander('Show Dataset', expanded=False):
    st.write(dataset)

# Show dataset description
with st.expander('Show Dataset Description', expanded=False):
    st.write(dataset.describe())   

# Elements for analysis
elements = ['RI', 'Na', 'Mg', 'Al', 'Si', 'K', 'Ca', 'Ba', 'Fe']

# Plot type selection 
plot_type = st.selectbox('Select plot type:', ['Select Any One Plot','Histogram', 'Bar Plot', 'Box Plot', 'Scatter Plot', 'Pie Chart', 'Line Plot'])

# Conditionally render element selection
if plot_type in ['Histogram', 'Bar Plot','Pie Chart']:
    selected_element = st.selectbox('Select element:', elements)

# Generate plots based on user selection
if plot_type == 'Histogram':
    st.header(f'Histogram of {selected_element}')
    plt.figure(figsize=(10, 6))
    plt.hist(dataset[selected_element], bins=50)
    plt.title(f'Histogram of {selected_element}')
    plt.xlabel(selected_element)
    plt.ylabel('Frequency')
    st.pyplot(plt)

elif plot_type == 'Bar Plot':
    st.header(f'Bar Plot of {selected_element}')
    value_counts = dataset[selected_element].value_counts().head(10)
    plt.figure(figsize=(10, 6))
    plt.bar(value_counts.index.astype(str), value_counts)
    plt.xlabel(selected_element)
    plt.ylabel('Counts')
    plt.title(f'Bar Plot of {selected_element}')
    plt.xticks(rotation=45)
    st.pyplot(plt)

elif plot_type == 'Box Plot':
    st.header('Box Plot of All Elements')
    plt.figure(figsize=(10, 6))
    plt.boxplot([dataset[element] for element in elements], tick_labels=elements)
    plt.title('Box Plot of All Elements')
    plt.xlabel('Elements')
    plt.grid(True)
    st.pyplot(plt)

elif plot_type == 'Scatter Plot':
    st.header('Scatter Plot of All Elements')
    plt.figure(figsize=(20, 12))
    for element in elements:
        plt.scatter(dataset.index, dataset[element], label=element)
    plt.title('Scatter Plot of Elements')
    plt.xlabel('Index')
    plt.ylabel('Element Values')
    plt.legend()
    plt.grid(True)
    st.pyplot(plt)

elif plot_type == 'Pie Chart':
    st.header(f'Pie Chart of {selected_element}')
    value_counts = dataset[selected_element].value_counts().head(10)
    plt.figure(figsize=(10, 6))
    plt.pie(value_counts, labels=value_counts.index.astype(str), autopct='%1.1f%%', startangle=140)
    plt.title(f'Pie Chart of {selected_element}')
    st.pyplot(plt)

elif plot_type == 'Line Plot':
    st.header('Line Plot of All Elements')
    plt.figure(figsize=(20, 12))
    for element in elements:
        plt.plot(dataset.index, dataset[element], label=element)
    plt.title('Line Plot of Elements')
    plt.xlabel('Index')
    plt.ylabel('Element Values')
    plt.legend()
    plt.grid(True)
    st.pyplot(plt)
else:
    st.header('Not Selected')
    st.write('Please select a plot type from the dropdown menu')

from sklearn.metrics import accuracy_score          
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

clf = RandomForestClassifier(n_estimators=100, random_state=42)

x = dataset.drop(columns = ['Type'])
y = dataset['Type']

xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size = 0.2, random_state = 1)

clf.fit(xtrain,ytrain)

ypred = clf.predict(xtest)

accuracy = accuracy_score(ytest, ypred)
print(f'Accuracy: {accuracy:.2f}')

# Prediction Section
st.header('Predict Glass Type')

# Function to predict glass type
def predict_glass_type(model, new_data):

    new_df = pd.DataFrame(new_data)
    
    predicted_labels = model.predict(new_df)
    
    glass_names = {
        1: "building_windows_float_processed",
        2: "building_windows_non_float_processed",
        3: "vehicle_windows_float_processed",
        4: "vehicle_windows_non_float_processed",
        5: "containers",
        6: "tableware",
        7: "headlamps"
    }
    
    predicted_label = predicted_labels[0]
    predicted_name = glass_names.get(predicted_label, "Unknown class")
    
    return predicted_label, predicted_name

# # Load the pre-trained model
# with open('clf.pkl', 'rb') as model_file:
#     clf = pickle.load(model_file)

# Method selection for prediction
method = st.selectbox('Select Prediction Method:', ['Static Data', 'User Input', 'Random Data'])

if method == 'Static Data':
    # Static Data
    st.subheader('Static Data Prediction')
    static_data = {
        'RI': [1.52],
        'Na': [13.64],
        'Mg': [4.49],
        'Al': [1.10],
        'Si': [71.78],
        'K':  [0.06],
        'Ca': [8.75],
        'Ba': [0.0],
        'Fe': [0.0]
    }

    if st.button('Predict Static Data'):
        predicted_label, predicted_name = predict_glass_type(clf, static_data)
        st.write(f'Predicted Glass Type (Label): {predicted_label}')
        st.write(f'Predicted Glass Type (Name): {predicted_name}')

elif method == 'User Input':
    # User Input
    st.subheader('User Input Prediction')
    user_data = {}
    for element in elements:
        user_data[element] = [st.number_input(f'Enter value for {element}', value=0.0)]

    if st.button('Predict User Input'):
        predicted_label, predicted_name = predict_glass_type(clf, user_data)
        st.write(f'Predicted Glass Type (Label): {predicted_label}')
        st.write(f'Predicted Glass Type (Name): {predicted_name}')

elif method == 'Random Data':
    # Random Data
    st.subheader('Random Data Prediction')
    
    random_data = {
        'RI': [random.choice(dataset['RI'])],
        'Na': [random.choice(dataset['Na'])],
        'Mg': [random.choice(dataset['Mg'])],
        'Al': [random.choice(dataset['Al'])],
        'Si': [random.choice(dataset['Si'])],
        'K':  [random.choice(dataset['K'])],
        'Ca': [random.choice(dataset['Ca'])],
        'Ba': [random.choice(dataset['Ba'])],
        'Fe': [random.choice(dataset['Fe'])]
    }
    
    if st.button('Predict Random Data'):
        predicted_label, predicted_name = predict_glass_type(clf, random_data)
        st.write('Random Sample Data:')
        st.dataframe(pd.DataFrame([random_data]))
        st.write(f'Predicted Glass Type (Label): {predicted_label}')
        st.write(f'Predicted Glass Type (Name): {predicted_name}')
