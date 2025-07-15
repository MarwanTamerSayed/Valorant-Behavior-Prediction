import streamlit as st
import pandas as pd
import numpy as np
import joblib

model = joblib.load('Final_model2.pkl')
power = joblib.load('Transformer2.pkl')
robust = joblib.load('RobustScaler2.pkl')
feature_columns = ['rating', 'damage_round', 'headshot_percent', 'aces', 'clutches',
       'flawless', 'first_bloods', 'win_percent', 'kd_ratio', 'kills_round',
       'most_kills', 'score_round', 'agent_1_Breach', 'agent_1_Brimstone',
       'agent_1_Chamber', 'agent_1_Cypher', 'agent_1_Fade', 'agent_1_Jett',
       'agent_1_KAY/O', 'agent_1_Killjoy', 'agent_1_Neon', 'agent_1_Omen',
       'agent_1_Phoenix', 'agent_1_Raze', 'agent_1_Reyna', 'agent_1_Sage',
       'agent_1_Skye', 'agent_1_Sova', 'agent_1_Viper', 'agent_1_Yoru',
       ]


rating_map = {
    'Unrated': 0, 'Bronze 2': 1, 'Bronze 3': 2, 'Silver 1': 3, 'Silver 2': 4,
    'Silver 3': 5, 'Gold 1': 6, 'Gold 2': 7, 'Gold 3': 8, 'Platinum 1': 9,
    'Platinum 2': 10, 'Platinum 3': 11, 'Diamond 1': 12, 'Diamond 2': 13,
    'Diamond 3': 14, 'Immortal 1': 15, 'Immortal 2': 16, 'Immortal 3': 17, 'Radiant': 18
}


agent_options = ['Breach', 'Brimstone', 'Chamber', 'Cypher', 'Fade', 'Jett', 'KAY/O', 'Killjoy', 'Neon', 'Omen',
                 'Phoenix', 'Raze', 'Reyna', 'Sage', 'Skye', 'Sova', 'Viper', 'Yoru']


cluster_map = {1:'Aggressive Carry Player',2:'Balanced Team-Oriented Player',0:'Support Role Player'}

st.title("🎮 Valorant Player Behavior Predictor")
st.markdown("Enter your stats and select your agent and rating to predict your playstyle.")


rating_input = st.selectbox("Rating", list(rating_map.keys()))
damage_round = st.number_input("Damage per Round", min_value=0.0)
headshot_percent = st.number_input("Headshot %", min_value=0.0, max_value=100.0)
aces = st.number_input("Aces", min_value=0)
clutches = st.number_input("Clutches", min_value=0)
flawless = st.number_input("Flawless Rounds", min_value=0)
first_bloods = st.number_input("First Bloods", min_value=0)
win_percent = st.number_input("Win %", min_value=0.0, max_value=100.0)
kd_ratio = st.number_input("K/D Ratio", min_value=0.0)
kills_round = st.number_input("Kills per Round", min_value=0.0)
most_kills = st.number_input("Most Kills in a Map",min_value=0)
score_round = st.number_input("AVG Score",min_value=0)
agent = st.selectbox("Agent Played", agent_options)


if st.button("Predict Cluster"):
    
    numeric_input = pd.DataFrame([[
    rating_map[rating_input], damage_round, headshot_percent, aces,
    clutches, flawless, first_bloods, win_percent, kd_ratio,
    kills_round, most_kills, score_round
    ]], columns=[
    'rating', 'damage_round', 'headshot_percent', 'aces',
    'clutches', 'flawless', 'first_bloods', 'win_percent',
    'kd_ratio', 'kills_round', 'most_kills', 'score_round'
    ])

    
    expected_agents = ['Breach', 'Brimstone', 'Chamber', 'Cypher', 'Fade', 'Jett', 'KAY/O',
                'Killjoy', 'Neon', 'Omen', 'Phoenix', 'Raze', 'Reyna', 'Sage',
                'Skye', 'Sova', 'Viper', 'Yoru']
    agent_onehot = {f'agent_1_{a}': int(agent == a) for a in expected_agents}
    agent_df = pd.DataFrame([agent_onehot])

   
    full_input = pd.concat([numeric_input, agent_df], axis=1)

    
    for col in feature_columns:
        if col not in full_input.columns:
            full_input[col] = 0

    
    full_input = full_input[feature_columns]
    columns_to_transform = ['damage_round','headshot_percent','clutches',
       'flawless','first_bloods', 'win_percent', 'kd_ratio','most_kills', 'score_round']
   
    full_input[columns_to_transform] = power.transform(full_input[columns_to_transform])
    scaled = robust.transform(full_input)
    final_input = pd.DataFrame(scaled, columns=feature_columns)

   
    prediction = model.predict(final_input)[0]
    st.success(f"🔮 Predicted Playstyle Cluster: **{cluster_map[prediction]}**")
        