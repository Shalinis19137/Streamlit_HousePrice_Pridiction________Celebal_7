🏠House Price Prediction App using Streamlit


This project is a Machine Learning-based Web App built using Streamlit that predicts house prices based on user-input features such as number of bedrooms, bathrooms, square footage, location, and more. It allows users to interactively input house-related parameters and receive an instant price prediction with visualizations and model insights.






📌 Project Objective
The goal of this project is to:


Predict the price of a house using regression models.

Build an interactive and user-friendly web application using Streamlit.

Demonstrate the end-to-end data science pipeline, from data preprocessing to model deployment.







🧠 Tech Stack

🐍 Python

📊 Pandas, NumPy, Matplotlib, Seaborn

🤖 Scikit-learn (Machine Learning models)

🌐 Streamlit (for web app UI)

📁 Jupyter Notebook / VS Code

✅ Git & GitHub for version control
















📈 Machine Learning Pipeline

1. Data Preprocessing
Missing Values Handling: Filled or removed missing entries.

Feature Engineering: Created new useful features or transformed existing ones.

Encoding Categorical Variables: Used Label Encoding or One-Hot Encoding.

Feature Scaling: Normalized or standardized features for better model performance.

2. Model Selection
Trained and tested multiple regression models:

Linear Regression

Decision Tree Regressor

Random Forest Regressor

Gradient Boosting Regressor

Used evaluation metrics:

MAE (Mean Absolute Error)

MSE (Mean Squared Error)

RMSE (Root Mean Squared Error)

R² Score (Coefficient of Determination)

3. Hyperparameter Tuning
Used GridSearchCV or RandomizedSearchCV to find the optimal parameters.









📊 App Features and Parameters (Explained)
The Streamlit app lets users input the following parameters:

Parameter	Description
Square Footage	Total built-up area of the house in square feet. A strong predictor of price.
Number of Bedrooms	Total bedrooms in the house. More rooms usually mean higher prices.
Number of Bathrooms	Total bathrooms, an important feature for price and comfort level.
Location	City or area name. One of the most influential price factors.
BHK	Number of Bedrooms, Hall, Kitchen – used as a housing type classification.
Balcony	Number of balconies in the property.
Parking	Availability of car parking space(s).
Furnishing Status	Whether the house is Furnished / Semi-Furnished / Unfurnished.
Availability	Whether the house is ready to move or under construction.

The model uses these parameters to predict the price using the trained regression model.









🎯 How to Run Locally
Clone the repository:

bash
Copy
Edit
git clone https://github.com/Shalinis19137/Streamlit_HousePrice_Prediction________Celebal_7.git
cd Streamlit_HousePrice_Prediction________Celebal_7
Install the dependencies:

bash
Copy
Edit
pip install -r requirements.txt
Run the Streamlit app:

bash
Copy
Edit
streamlit run app.py









📌 Output




A live prediction of the house price based on real-time user input.

Data visualization (such as heatmaps, distribution plots) to help users understand features and correlations.

Clean and responsive UI using Streamlit.













📦 Future Improvements
Integrate more location-specific data (like crime rate, school ratings).

Add user authentication for saving predictions.

Connect to a real-estate database for real-time listings.

Deploy the app on the cloud (e.g., Heroku, AWS, or Streamlit Cloud).









🧑‍💻 Made with ❤️ by Shalini
This project was developed during my internship at Celebal Technologies as part of my major machine learning project. I faced many errors, iterations, and learnings , but finally achieved 65% model success and gained a deep understanding of ML workflow and error handling
