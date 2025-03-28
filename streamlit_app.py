import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from scipy.stats import ttest_ind

# Load the data
df = pd.read_csv("https://raw.githubusercontent.com/ZHM89/MidProject_ML/main/nba_salary_stats.csv")

# Encode categorical variables
le_position = LabelEncoder()
le_shoots = LabelEncoder()
df['position'] = le_position.fit_transform(df['position'])
df['shoots'] = le_shoots.fit_transform(df['shoots'])

# Select features and target
features = ['season_start', 'career_AST', 'career_FG%', 'career_FG3%', 'career_FT%',
            'career_G', 'career_PER', 'career_PTS', 'career_WS', 'height', 'weight',
            'draft_age', 'position', 'shoots']
target = 'salary'

X = df[features]
y = df[target]

# Train model
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Streamlit UI
st.title("NBA Salary Predictor 💲")
st.write("Adjust the inputs to predict the player's salary.")

with st.expander("Adjust Features"):
    season_start = st.slider("Season Start Year", int(df['season_start'].min()), int(df['season_start'].max()), 2020)
    career_AST = st.slider("Career Assists", float(df['career_AST'].min()), float(df['career_AST'].max()), 2.5)
    career_FG = st.slider("Career Field Goal %", float(df['career_FG%'].min()), float(df['career_FG%'].max()), 45.0)
    career_FG3 = st.slider("Career 3PT FG %", float(df['career_FG3%'].min()), float(df['career_FG3%'].max()), 35.0)
    career_FT = st.slider("Career Free Throw %", float(df['career_FT%'].min()), float(df['career_FT%'].max()), 75.0)
    career_G = st.slider("Career Games Played", int(df['career_G'].min()), int(df['career_G'].max()), 500)
    career_PER = st.slider("Career Player Efficiency Rating", float(df['career_PER'].min()), float(df['career_PER'].max()), 15.0)
    career_PTS = st.slider("Career Points Per Game", float(df['career_PTS'].min()), float(df['career_PTS'].max()), 10.0)
    career_WS = st.slider("Career Win Shares", float(df['career_WS'].min()), float(df['career_WS'].max()), 20.0)
    height = st.slider("Height (cm)", float(df['height'].min()), float(df['height'].max()), 200.0)
    weight = st.slider("Weight (kg)", float(df['weight'].min()), float(df['weight'].max()), 100.0)
    draft_age = st.slider("Draft Age", float(df['draft_age'].min()), float(df['draft_age'].max()), 19.0)
    position = st.selectbox("Position", le_position.classes_)
    shoots = st.selectbox("Shooting Hand", le_shoots.classes_)

# Convert categorical inputs
position_encoded = le_position.transform([position])[0]
shoots_encoded = le_shoots.transform([shoots])[0]

# Prepare input data
input_data = np.array([[season_start, career_AST, career_FG, career_FG3, career_FT,
                         career_G, career_PER, career_PTS, career_WS, height, weight,
                         draft_age, position_encoded, shoots_encoded]])

# Predict salary
predicted_salary = model.predict(input_data)[0]

st.subheader("Predicted Salary:")
st.write(f"${predicted_salary:,.2f}")

# Hypothesis Testing
st.subheader("Hypothesis Testing: Does Higher PER Mean Higher Salary?")
median_per = df['career_PER'].median()
low_per_salaries = df[df['career_PER'] < median_per]['salary']
high_per_salaries = df[df['career_PER'] >= median_per]['salary']

t_stat, p_value = ttest_ind(high_per_salaries, low_per_salaries, equal_var=False)

st.write(f"T-Statistic: {t_stat:.4f}")
st.write(f"P-Value: {p_value:.4f}")

if p_value < 0.05:
    st.write("Conclusion: Reject the null hypothesis. Players with higher PER tend to have significantly higher salaries.")
else:
    st.write("Conclusion: Fail to reject the null hypothesis. No significant difference in salaries between high and low PER players.")