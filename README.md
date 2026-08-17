# 🤖 AI Powered Google Search Trend Prediction And Analysis Dashboard

## 📌 Project Overview

This project is an **interactive data analytics dashboard** built using **Streamlit** that analyzes and visualizes Google search trends for multiple keywords.
The dashboard allows users to explore trends, compare keyword popularity, analyze correlations, and predict future search interest using a **Deep Learning LSTM model**.

The goal of this project is to demonstrate **time series analysis, data visualization, and machine learning** techniques using real-world search trend data.

---

# 🚀 Features

### 📊 Keyword Trend Comparison

Compare multiple keywords and visualize their popularity over time using interactive line charts.

### 📌 Key Metrics

Displays important information including:

* Total dataset records
* Number of keywords available
* Latest date in dataset

### 🔥 Correlation Heatmap

Shows correlation between different keywords to identify relationships and search behavior patterns.

### 🤖 AI Prediction (LSTM Model)

Uses a **Long Short-Term Memory (LSTM)** deep learning model to predict the next search interest value for the selected keyword.

### 🏆 Top Search Interest Days

Displays the top 10 days with the highest search interest.

### ⬇ Dataset Download

Allows users to download the processed dataset directly from the dashboard.

---

# 🛠 Technologies Used

* **Python**
* **Streamlit** – Web dashboard framework
* **Pandas** – Data processing
* **Plotly** – Interactive data visualization
* **Seaborn & Matplotlib** – Heatmap visualization
* **TensorFlow / Keras** – LSTM Deep Learning model
* **Scikit-learn** – Data normalization

---

# 📂 Project Structure

```
Google_Search_Trend_Project
│
├── app.py
├── multiTimeline.csv
├── requirements.txt
└── README.md
```

---

# ⚙ Installation

### 1️⃣ Clone the repository

```bash
git clone https://github.com/yourusername/google-trend-dashboard.git
cd google-trend-dashboard
```

### 2️⃣ Install required libraries

```bash
pip install -r requirements.txt
```

Or install manually:

```bash
pip install streamlit pandas plotly tensorflow scikit-learn seaborn matplotlib
```

---

# ▶ Running the Application

Run the Streamlit dashboard:

```bash
streamlit run app.py
```

Then open your browser and go to:

```
http://localhost:8501
```

---

# 📊 Dataset

The dataset used in this project is exported from **Google Trends** and contains search interest values for multiple keywords over time.

Example keywords analyzed:

* Artificial Intelligence
* Machine Learning
* Data Science
* Python
* Deep Learning
* Big Data
* Neural Networks
* NLP

---

# 📈 Example Dashboard Insights

Using this dashboard, users can:

* Identify trending technologies
* Compare popularity between different topics
* Discover relationships between keywords
* Predict future search interest

---

# 🎯 Use Cases

* Market trend analysis
* Keyword popularity research
* Technology trend forecasting
* Data science learning project

---

# 🔮 Future Improvements

* Add **Prophet time-series forecasting**
* Real-time Google Trends API integration
* Advanced trend prediction models
* Interactive filtering and keyword search
* Deployment on **Streamlit Cloud**

---

# 👨‍💻 Author

**Karan Dodia**
Data Science & Data Analytics

---

# ⭐ If you like this project

Consider giving the repository a star ⭐

