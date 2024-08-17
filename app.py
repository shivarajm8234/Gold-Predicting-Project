from flask import Flask, request, render_template
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

app = Flask(__name__)

# Load and prepare the dataset
file_path = 'C:\\Users\\srina\\OneDrive\\Desktop\\vscode\\Gold Prediction\\Daily_Gold_Price_on_World.csv'
df = pd.read_excel(file_path)

# Convert 'Date' to datetime and set as index
df['Date'] = pd.to_datetime(df['Date'])
df.set_index('Date', inplace=True)

# Handle missing values by interpolation
df['Gold Price'] = df['Gold Price'].replace(0, np.nan)
df['Gold Price'].interpolate(method='time', inplace=True)

# Log transformation to stabilize variance
df['Log_Gold_Price'] = np.log(df['Gold Price'])
df['Log_Gold_Price'] = df['Log_Gold_Price'].interpolate(method='time')
df = df[np.isfinite(df['Log_Gold_Price'])]

# Calculate Moving Averages
df['MA50'] = df['Gold Price'].rolling(window=50).mean()
df['MA200'] = df['Gold Price'].rolling(window=200).mean()

# Calculate RSI
def calculate_rsi(data, window):
    delta = data.diff(1)
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(window=window).mean()
    avg_loss = loss.rolling(window=window).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

df['RSI'] = calculate_rsi(df['Gold Price'], window=14)

# Calculate MACD
df['EMA12'] = df['Gold Price'].ewm(span=12, adjust=False).mean()
df['EMA26'] = df['Gold Price'].ewm(span=26, adjust=False).mean()
df['MACD'] = df['EMA12'] - df['EMA26']
df['Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()

# Drop rows with NaN values resulting from feature calculations
df.dropna(inplace=True)

# Feature engineering
df['Day'] = df.index.day
df['Month'] = df.index.month
df['Year'] = df.index.year

# Define features and target
X = df[['Day', 'Month', 'Year', 'MA50', 'MA200', 'RSI', 'MACD', 'Signal']]
y = df['Gold Price']

# Handle missing values in features
X.fillna(X.mean(), inplace=True)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train the model with a more robust model (Random Forest)
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluate the model
y_pred = model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
print(f"Mean Absolute Error: {mae}")

def predict_gold_price(day, month, year):
    future_df = pd.DataFrame({'Day': [day], 'Month': [month], 'Year': [year]})
    future_df['MA50'] = df['MA50'].iloc[-1]
    future_df['MA200'] = df['MA200'].iloc[-1]
    future_df['RSI'] = df['RSI'].iloc[-1]
    future_df['MACD'] = df['MACD'].iloc[-1]
    future_df['Signal'] = df['Signal'].iloc[-1]
    future_df.fillna(future_df.mean(), inplace=True)
    predicted_price = model.predict(future_df)
    return predicted_price[0]

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    day = int(request.form['day'])
    month = int(request.form['month'])
    year = int(request.form['year'])
    predicted_price = predict_gold_price(day, month, year)
    return render_template('result.html', day=day, month=month, year=year, price=predicted_price) 

if __name__ == '__main__':
    app.run(debug=True)
