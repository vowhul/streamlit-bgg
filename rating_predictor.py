import streamlit as st
import pandas as pd
import pickle
from sklearn import datasets
from sklearn.ensemble import RandomForestRegressor

# Load the bgg dataset
url = "new_bgg_dataset.csv"
bgg_dataset=pd.read_csv(url)
X = bgg_dataset.drop(columns='Rating Average')
y = bgg_dataset['Rating Average']

# Train the RandomForestRegressor
rf = RandomForestRegressor(
    n_estimators= 120, 
    min_samples_split= 3, 
    min_samples_leaf= 3,  
    max_depth= 20, 
    criterion= 'squared_error', 
    bootstrap= True
)
rf.fit(X, y)

# Save the model
pickle.dump(rf, open('model_rf.pkl', 'wb'))

# Load the model
load_rf = pickle.load(open('model_rf.pkl', 'rb'))

# Streamlit app
st.write("""
# Board Game Prediction App
This app predicts the rating of a Board Game
""")

st.sidebar.header('User Input Parameters')

def user_input_features():
    year_published = st.sidebar.slider('Year Published', 0, 2022, 2000)
    users_rated = st.sidebar.slider('Users Rated', 30, 155312, 0),
    bgg_rank = st.sidebar.slider('BGG Rank', 1, 20344, 1),
    complexity_average = st.sidebar.slider('Complexity Average', 0.0, 5.0, 2.5)
    owned_users = st.sidebar.slider('Owned Users', 0, 155312, 0)
  
    data = {'Year Published': year_published,
            'Users Rated': users_rated,
            'BGG Rank': bgg_rank,
            'Complexity Average': complexity_average,
            'Owned Users': owned_users
           }
    features = pd.DataFrame(data, index=[0])
    return features

df = user_input_features()

st.subheader('User Input parameters')
st.write(df)

# Apply model to make predictions
prediction = load_rf.predict(df)

st.subheader('Prediction')
st.write(prediction)