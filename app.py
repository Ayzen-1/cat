import streamlit as st
import pandas as pd
import numpy as np
from catboost import CatBoostRegressor

# Загрузка модели
@st.cache_resource()
def load_model():
    return CatBoostRegressor().load_model("catboost_model.cbm")

model = load_model()

# Устанавливаем стиль кофейни с фоном и логотипом
st.markdown(
    """
    <style>
        body {
            background-image: url('https://source.unsplash.com/1600x900/?coffee-shop');
            background-size: cover;
            color: #6f4e37;
            font-family: 'Arial', sans-serif;
        }
        .stButton>button {
            background-color: #d2b48c;
            color: white;
            border-radius: 8px;
            padding: 10px;
        }
        .stButton>button:hover {
            background-color: #a67b5b;
        }
        .logo {
            position: absolute;
            top: 10px;
            left: 10px;
            width: 80px;
        }
    </style>
    <img src="https://upload.wikimedia.org/wikipedia/en/thumb/3/35/Starbucks_Corporation_Logo_2011.svg/1200px-Starbucks_Corporation_Logo_2011.svg.png" class="logo">
    """,
    unsafe_allow_html=True
)

st.title("☕ Coffee Shop Revenue Predictor ☕")
st.write("Введите данные для предсказания и узнайте, сколько прибыли принесёт ваш кофейный бизнес!")

# Загрузка данных
@st.cache_data()
def load_data():
    df = pd.read_csv('coffee_shop_revenue.csv')
    df = df[df['Daily_Revenue'] >= 0]
    df['Revenue_per_Employee'] = np.where(df['Number_of_Employees'] != 0,
                                          df['Daily_Revenue'] / df['Number_of_Employees'],
                                          np.nan)
    df['Marketing_Efficiency'] = np.where(df['Marketing_Spend_Per_Day'] != 0,
                                          df['Daily_Revenue'] / df['Marketing_Spend_Per_Day'],
                                          np.nan)
    df['Foot_Traffic_Conversion'] = np.where(df['Location_Foot_Traffic'] != 0,
                                             df['Number_of_Customers_Per_Day'] / df['Location_Foot_Traffic'],
                                             np.nan)
    df['Revenue_per_Operating_Hour'] = np.where(df['Operating_Hours_Per_Day'] != 0,
                                                df['Daily_Revenue'] / df['Operating_Hours_Per_Day'],
                                                np.nan)
    return df

df = load_data()

# Генерация ползунков на основе данных
st.sidebar.header("Настройки кофейни")
feature_sliders = {}
features = df.drop(columns=['Daily_Revenue', 'Number_of_Customers_Per_Day', 'Operating_Hours_Per_Day']).columns
for feature in features:
    min_val = float(df[feature].min())
    max_val = float(df[feature].max())
    mean_val = float(df[feature].mean())
    feature_sliders[feature] = st.sidebar.slider(feature, min_val, max_val, mean_val)

# Создаем DataFrame с введенными значениями
input_df = pd.DataFrame([feature_sliders])

if st.button("☕ Предсказать выручку ☕"):
    prediction = model.predict(input_df)[0]
    st.success(f"### 💰 Предсказанное значение: {prediction:.2f} $ в день")