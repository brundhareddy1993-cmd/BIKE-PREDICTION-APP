import pickle
import pandas as pd
import streamlit as st
import numpy as np
import joblib
import os
# 1. Load the saved model

 
#os.system('pip install scikit-learn==1.6.1')


@st.cache_resource
def load_model():
    return joblib.load("gradient_boost_model.joblib")


model = load_model()

st.title('Bike Rental / Prediction App')
st.write('Select the conditions below to generate a prediction.')

# --- 2. USER INPUTS ---

# Numerical / Continuous Inputs (Adjust or add any remaining features from X_train)
col1, col2, col3 = st.columns(3)
with col1:
    hr = int(st.number_input('Hour of Day (0-23)', min_value=0, max_value=23, value=12))
    temp = st.number_input('Temperature', value=20.0)
with col2:
    humidity = st.number_input('Humidity', value=50.0)
    windspeed = st.number_input('Windspeed', value=10.0)
with col3:
    year = st.selectbox('Year', options=[2011, 2012])  # Assuming model trained on these years
    workingday = st.selectbox('Working Day', options=['No', 'Yes'])  # Assuming model trained on these categories
    

# Categorical Inputs
season = st.selectbox('Season', options=['Springer', 'Summer', 'Winter'])  # (Add 'Fall' if your model has it)
month = st.selectbox('Month', options=list(range(1, 13)))  # 1 to 12
mnth_num = float(month)  # Store the numeric month for cyclical encoding
holiday = st.selectbox('Holiday', options=['No', 'Yes'])
weekday = st.selectbox(
    'Weekday',
    options=["Sunday", "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday"],
    
)
weather = st.selectbox('Weather Condition', options=['Clear / Partly Cloudy', 'Mist', 'Light Snow', 'Heavy Rain'])
print(type(weekday))
print("The selected weekday is ")
print(weekday)

# Feature column names (ensure these match the model's expected features)
feature_names = [
    'hr_cos','hr_sin' 'weekday', 'atemp', 'hum', 'windspeed', 'season_springer',
    'season_summer', 'season_winter', 'yr_2012', 
    'mnth_cos','mnth_sin',
    'mnth_8', 'mnth_9', 'holiday_Yes', 'workingday_Working Day',
    'weathersit_Heavy Rain', 'weathersit_Light Snow', 'weathersit_Mist'

]


# --- 4. BUILD DATAFRAME & PREDICT ---
if st.button('Predict'):
    # Combine all features into a single dictionary


    # --- 3. ONE-HOT ENCODING LOGIC (Setting chosen to 1, others to 0) ---

    #year encoding:
    year_2012 = True if year == 2012 else False
    workingday_Working_Day= True if workingday == 'Yes' else False

    # Season encoding
    season_springer = True if season == 'Springer' else False
    season_summer = True if season == 'Summer' else False
    season_winter = True if season == 'Winter' else False

    hr_sin = np.sin(2 * np.pi * hr / 24.0)
    hr_cos = np.cos(2 * np.pi * hr / 24.0)


    # Month encoding (Creates month_1 through month_12, setting only chosen month to 1)
    # Convert 'mnth' to numeric (since it is stored as text/object in your dataset)

    # 2. Create cyclical sine and cosine features for month (1 to 12)
    mnth_sin = np.sin(2 * np.pi * mnth_num / 12.0)
    mnth_cos = np.cos(2 * np.pi * mnth_num / 12.0)

    def get_time_of_day(hour):
        if 5 <= hour < 10:
            return 'Morning_Rush'
        elif 10 <= hour < 16:
            return 'Midday'
        elif 16 <= hour < 20:
            return 'Evening_Rush'
        else:
            return 'Night'
    time_of_day = get_time_of_day(hr)



    # Holiday encoding (Yes = 1, No = 0)
    holiday_Yes = True if holiday == 'Yes' else False

    # Weather encoding
    weather_mist = True if weather == 'Mist' else False
    weather_light_snow = True if weather == 'Light Snow' else False
    weather_heavy_rain = True if weather == 'Heavy Rain' else False
    # Time of the day encoding

    time_of_day_Midday = True if time_of_day == 'Midday' else False
    time_of_day_Morning_Rush = True if time_of_day == 'Morning_Rush' else False
    time_of_day_Night = True if time_of_day == 'Night' else False

    is_weekend = 1 if weekday in ["Sunday", "Saturday"] else 0

    input_dict = {
       
        'atemp': int(temp),
        'hum': humidity,
        'windspeed': windspeed,
        'holiday_Yes': holiday_Yes,
        'yr_2012': year_2012,
        'season_springer': season_springer,
        'season_summer': season_summer,
        'season_winter': season_winter,
        'weathersit_Mist': weather_mist,
        'weathersit_Light Snow': weather_light_snow,
        'weathersit_Heavy Rain': weather_heavy_rain,
        'workingday_Working Day': workingday_Working_Day,
        'time_of_day_Midday': time_of_day_Midday,
        'time_of_day_Morning_Rush': time_of_day_Morning_Rush,
        'time_of_day_Night': time_of_day_Night, 
        'hr_sin': hr_sin,
        'hr_cos': hr_cos,
        'mnth_sin': mnth_sin,
        'mnth_cos': mnth_cos,   
        'is_weekend': is_weekend,
    }

    print(input_dict)
    input_df = pd.DataFrame([input_dict])
    print("The input dataframe is ")
    print(input_df)

    # IMPORTANT: Automatically align column names & exact order with what the model expects!
    if hasattr(model, 'feature_names_in_'):
        # Fills any missing columns with 0 and orders them identically to X_train
        input_df = input_df.reindex(columns=model.feature_names_in_, fill_value=0)

    print("The input dataframe after reindexing is ")
    print(input_df)
    # Make prediction
    prediction = model.predict(input_df)
    st.success(f'Predicted Output: **{prediction[0]:.2f}**')
