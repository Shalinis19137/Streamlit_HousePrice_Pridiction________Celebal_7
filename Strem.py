import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# ------------------ App Setup ------------------
st.set_page_config(page_title="House Price App", layout="wide")
st.sidebar.title("🔀 Navigation")
page = st.sidebar.radio("Go to", ["🏡 Welcome", "🏠 House Price Predictor", "📊 Data Visualization", "ℹ️ About"])

# ------------------ Load Dataset ------------------
@st.cache_data
def load_data():
    df = pd.read_csv(r"C:\Users\lorem\Downloads\data.csv")
    df = df.dropna()
    return df

data = load_data()

# ------------------ Train Model ------------------
@st.cache_resource
def train_model(df):
    df = df.copy()
    X = df.drop(columns=["price"])
    y = df["price"]
    
    label_encoders = {}
    for col in X.select_dtypes(include=["object"]).columns:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col])
        label_encoders[col] = le
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    return model, label_encoders, X.columns

model, label_encoders, feature_columns = train_model(data)

# ------------------ Prediction Logic ------------------
def predict(model, label_encoders, columns, user_input):
    df_input = pd.DataFrame([user_input])
    for col in label_encoders:
        le = label_encoders[col]
        df_input[col] = le.transform(df_input[col])
    df_input = df_input[columns]
    prediction = model.predict(df_input)[0]
    return prediction

# ------------------ Page: Welcome ------------------
if page == "🏡 Welcome":
    st.markdown("""
        <h1 style='text-align: center; color: #1f77b4;'>🏡 Smart House Price Prediction App</h1>
        <p style='text-align: center; font-size: 18px;'>A powerful  platform for predicting housing prices and exploring real estate trends </p>
        """, unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        st.image("1751455096841-d8e77222-aaae-4396-a493-2073f287e6d7_1.jpg", use_column_width=True)
    with col2:
        st.markdown("""
        ###  What You Can Do:
        - Predict house prices based on features like:
            - 🛏️ Bedrooms, 🛁 Bathrooms, 📏 Area
        - Explore detailed visual insights:
            - 🔥 Price Trends
            - 📊 Correlation Heatmaps
            - 📉 Distribution Charts
        - All in one sleek and powerful app

        ###  Start Now
        Use the left sidebar to begin.
        """)

    st.markdown("---")

# ------------------ Page: House Price Predictor ------------------
elif page == "🏠 House Price Predictor":
    st.title("🏠 House Price Prediction App")
    st.write("Fill in the details below to estimate the house price.")

    user_input = {}
    for col in feature_columns:
        if data[col].dtype == "object":
            user_input[col] = st.selectbox(f"{col}", data[col].unique())
        else:
            min_val = float(data[col].min())
            max_val = float(data[col].max())
            avg_val = float(data[col].mean())
            user_input[col] = st.number_input(f"{col}", min_value=min_val, max_value=max_val, value=avg_val)

    if st.button("Predict Price"):
        price = predict(model, label_encoders, feature_columns, user_input)
        st.success(f"🏷️ Estimated House Price: ₹ {price:,.2f}")

# ------------------ Page: Data Visualization ------------------
elif page == "📊 Data Visualization":
    st.title("📊 Data Visualization")

    st.subheader("Dataset Preview")
    st.dataframe(data.head())
          
    # ------- Correlation Heatmap -------
    st.subheader(" Correlation Heatmap (Numerical Columns Only)")
    numeric_df = data.select_dtypes(include=[np.number])
    if numeric_df.shape[1] >= 2:
        fig1, ax1 = plt.subplots(figsize=(10, 6))
        corr_matrix = numeric_df.corr()
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', ax=ax1)
        st.pyplot(fig1)
    else:
        st.warning("Not enough numeric columns to generate a correlation heatmap.")

        
 # ------- Price Distribution -------
    st.subheader(" Price Distribution")
    fig2, ax2 = plt.subplots()
    sns.histplot(data['price'], kde=True, ax=ax2, color='green')
    st.pyplot(fig2)

    Chart_data = pd.DataFrame(
        np.random.randn(100, 2),
        columns=['x', 'y']
    )

    st.line_chart(Chart_data)
    st.bar_chart(Chart_data)
    

# ------------------ Page: About ------------------
elif page == "ℹ️ About":
    st.title("ℹ️ About This App")

    st.markdown("""
This application is designed to help users **predict house prices** based on relevant property features using a **machine learning model**.

###  Dataset Description:
The dataset used in this app typically contains the following:
- `area` – Built-up area of the house
- `bedrooms`, `bathrooms` – Number of rooms
- `location` – Categorical feature representing neighborhood
- `price` – Target variable representing house cost (₹)

> *Note: The app will automatically detect all features and handle them accordingly, so your custom dataset will still work.*

###  How It Works:
1. When the app loads, it automatically:
   - Loads and cleans the data
   - Encodes categorical variables
   - Trains a Linear Regression model
2. The **prediction page** lets you enter feature values and see a live estimate.
3. The **visualization page** helps you understand the data distribution, correlation, and trends.

###  Built With:
- [Streamlit](https://streamlit.io/)
- [scikit-learn](https://scikit-learn.org/)
- [pandas](https://pandas.pydata.org/)
- [matplotlib](https://matplotlib.org/)
- [seaborn](https://seaborn.pydata.org/)

---
""")
