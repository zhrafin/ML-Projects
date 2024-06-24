import streamlit as st
import pickle
import pandas as pd
import zipfile

zip_file_path = 'pipe.zip'
pickle_file_name = 'pipe.pkl'

with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
    with zip_ref.open(pickle_file_name) as pickle_file:
        model = pickle.load(pickle_file)

teams = ['Australia', 'India', 'Bangladesh', 'New Zealand', 'South Africa',
         'England', 'West Indies', 'Afghanistan', 'Pakistan', 'Sri Lanka']

cities = ['Colombo', 'Mirpur', 'Johannesburg', 'Dubai', 'Auckland', 'Cape Town', 'London', 'Pallekele',
          'Barbados', 'Sydney', 'Melbourne', 'Durban', 'St Lucia', 'Wellington', 'Lauderhill', 'Hamilton',
          'Centurion', 'Manchester', 'Abu Dhabi', 'Mumbai', 'Nottingham', 'Southampton', 'Mount Maunganui',
          'Chittagong', 'Kolkata', 'Lahore', 'Delhi', 'Nagpur', 'Chandigarh', 'Adelaide', 'Bangalore',
          'St Kitts', 'Cardiff', 'Christchurch', 'Trinidad']

st.title('Cricket Score Predictor')

col1, col2 = st.columns(2)

with col1:
    batting_team = st.selectbox('Select batting team', sorted(teams))
with col2:
    bowling_team = st.selectbox('Select bowling team', sorted(teams))

city = st.selectbox('Select city', sorted(cities))

col3, col4, col5 = st.columns(3)

with col3:
    current_score = st.number_input('Current Score', min_value=0, step=1)
with col4:
    overs_input = st.text_input('Overs done', '1.0')  # Default to 1 over
with col5:
    wickets = st.number_input('Wickets out', min_value=0, max_value=10, step=1)

# Ensure runs scored in last 5 overs is not higher than current score
last_five_max = min(current_score,
                    999999)  # set a very high number, Streamlit doesn't support direct max_value for number_input
last_five = st.number_input('Runs scored in last 5 overs', min_value=0, max_value=last_five_max, step=1)


# Function to convert overs input to float format (e.g., 1.5 to 1.5)
def parse_overs(overs_input):
    try:
        overs, balls = overs_input.split('.')
    except:
        overs, balls = int(overs_input), 0
    overs, balls = int(overs), int(balls)
    while balls >= 6:
        balls -= 6
        overs += 1
    overs = float(overs) + float(balls) / 10.0
    return overs


# Function to calculate additional features and predict the score
def predict_score(batting_team, bowling_team, city, current_score, overs_input, wickets, last_five):
    # Convert overs input to float format
    overs = parse_overs(overs_input)

    # Calculate additional features
    wickets_left = 10 - wickets
    balls_left = (20 - overs) * 6  # 20 overs per match
    crr = current_score / overs if overs > 0 else 0.0  # Calculate current run rate

    if wickets_left == 0 or balls_left == 0:
        return current_score

    # Create a DataFrame with all required columns
    data = pd.DataFrame({
        'batting_team': [batting_team],
        'bowling_team': [bowling_team],
        'city': [city],
        'current_score': [current_score],
        'overs': [overs],
        'wickets': [wickets],
        'last_five': [last_five],
        'wickets_left': [wickets_left],
        'balls_left': [balls_left],
        'crr': [crr]
    })

    # Use the loaded model to predict the score
    predicted_score = model.predict(data)

    return abs(int(predicted_score[0]))


# Call the prediction function when 'Predict' button is clicked
if st.button('Predict'):
    predicted_score = predict_score(batting_team, bowling_team, city, current_score, overs_input, wickets, last_five)
    st.write(f"{batting_team}'s predicted final score: {abs(int(predicted_score))}")
