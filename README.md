# 📊 Unemployment Analysis with Python – India

![Python](https://img.shields.io/badge/Python-3.11-blue?logo=python&logoColor=white) 
![Dash](https://img.shields.io/badge/Dash-2.10-orange?logo=plotly&logoColor=white) 
![Plotly](https://img.shields.io/badge/Plotly-5.20-red?logo=plotly&logoColor=white) 
![License](https://img.shields.io/badge/License-MIT-green)

---

## 👩‍💻 Author / Contact

| Name          | Email                        | LinkedIn |
|---------------|------------------------------|----------|
| Khadija Rao   | [raoumar0058@gmail.com](mailto:raoumar0058@gmail.com) | [Rao Umar](https://www.linkedin.com/in/rao-umar-904807355) |

---

## 📝 Project Overview

This project provides a **comprehensive analysis of unemployment trends in India** using Python, Plotly, and Dash.  

**Key highlights:**

- 🔝 **State-wise unemployment analysis** – Top & Bottom 10 states  
- 📈 **Yearly trends** – Average unemployment rates  
- 🌡️ **State-Year Heatmap** – Compare states across years interactively  
- 📉 **5-year forecast** – Predict future unemployment rates using Linear Regression  
- 🔽 **Interactive dashboard** – Users can select a state to explore its yearly trend  

This dashboard demonstrates **data cleaning, visualization, forecasting, and interactive deployment** in Python.

---
          ┌──────────────────────────┐
          │  Unemployment in India   │
          │       CSV/XLS File       │
          └────────────┬────────────┘
                       │
                       ▼
          ┌──────────────────────────┐
          │  Unemployment Analysis   │
          │   with Python Script     │
          │ (Data Cleaning, Visuals,│
          │ Forecasting, Dashboard) │
          └────────────┬────────────┘
                       │
      ┌────────────────┼─────────────────┐
      ▼                ▼                 ▼
┌───────────────┐ ┌───────────────┐ ┌───────────────┐
│ Cleaned CSV   │ │ Forecast CSV  │ │ Dash Dashboard│
│(Unemployment_ │ │(Unemployment_ │ │(Interactive, │
│Cleaned.csv)   │ │Forecast.csv)  │ │State Selection│
└───────────────┘ └───────────────┘ └───────────────┘

---

## 🛠️ Tech Stack / Libraries

| Category                  | Libraries |
|----------------------------|-----------|
| **Data Handling**          | pandas, numpy |
| **Visualization**          | matplotlib, seaborn, plotly |
| **Dashboard**              | dash |
| **Modeling / Forecasting** | scikit-learn (Linear Regression) |
| **Excel File Handling**    | openpyxl, xlrd |

---

## 📂 Dataset

- **File Name:** `Unemployment in India.csv.xls`  
- **Required Columns:**  
  - `Region` → renamed as `State`  
  - `Date` → converted to `Year`  
  - `Estimated Unemployment Rate (%)` → renamed as `Unemployment Rate`  
- **Placement:** Dataset should be located in `/Users/mac/Downloads/` or update `file_path` in the script.

---

## ⚡ Installation & Setup

1. **Clone the repository:**

```bash
git clone <your-repo-link>
cd <repo-folder>
````

2. **Install required packages:**

```bash
pip install pandas numpy matplotlib seaborn plotly dash scikit-learn openpyxl xlrd
```

3. **Run the dashboard:**

```bash
python "Unemployment Analysis Dashboard.py"
```

4. **Access the dashboard:**
   Open your browser at: [http://127.0.0.1:8050/](http://127.0.0.1:8050/)

---

## 📊 Features & Visuals

| Feature                     | Description                                                      |
| --------------------------- | ---------------------------------------------------------------- |
| 🔝 **Top & Bottom States**  | Color-coded bar charts for highest and lowest unemployment rates |
| 📈 **Yearly Trend**         | Line chart showing year-wise average unemployment                |
| 🌡️ **State-Year Heatmap**  | Compare unemployment rates for all states across years           |
| 📉 **Forecast**             | 5-year predicted unemployment trends (Linear Regression)         |
| 🔽 **Interactive Dropdown** | Select any state to view its yearly trend interactively          |

---

## 💾 Output

* **Interactive Dashboard:** Fully functional with hover-over details.
* **Cleaned Dataset:** Saved as `Unemployment_Cleaned.csv` in `/Users/mac/Downloads/`.

---

## 🔧 Usage Notes

* Ensure dataset is in the correct location or update `file_path` in the script.
* Dashboard is interactive; hover over charts for detailed insights.
* Forecast is **trend visualization only**, not official prediction.

---

## 📸 Screenshots / Examples

*(Optional: Add screenshots of the dashboard showing bar charts, heatmaps, forecasts, and dropdown interactions.)*

---

## 🏷️ License

**MIT License** – Free to use, modify, and distribute.

---

## ✅ Summary

This **Unemployment Analysis Dashboard** showcases:

* Data cleaning and handling
* Interactive and static visualizations
* Forecasting future trends
* User-friendly **Dash dashboard**

It provides insights into unemployment trends in India and demonstrates practical **Python, data visualization, and dashboard development skills**.
