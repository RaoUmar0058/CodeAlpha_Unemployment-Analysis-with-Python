# ============================
#  Unemployment Analysis & Dashboard 
# ============================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from sklearn.linear_model import LinearRegression
from dash import Dash, dcc, html
import os

# ---------------------------
# 1. File Path Setup
# ---------------------------
file_path = "/Users/mac/Downloads/Unemployment in India.csv.xls"
if not os.path.exists(file_path):
    raise FileNotFoundError(f"❌ File not found: {file_path}")

# ---------------------------
# 2. Load Dataset (Tab-Separated)
# ---------------------------
if file_path.endswith('.csv') or '.csv.' in file_path:
    data = pd.read_csv(file_path, sep='\t')
    print("✅ Loaded as CSV with tab separator")
elif file_path.endswith('.xls') or file_path.endswith('.xlsx'):
    engine = 'xlrd' if file_path.endswith('.xls') else 'openpyxl'
    data = pd.read_excel(file_path, engine=engine)
    print(f"✅ Loaded as Excel using engine '{engine}'")

# Clean column names
data.columns = data.columns.str.strip()
print("✅ Columns after cleaning:", data.columns.tolist())

# ---------------------------
# 3. Rename Columns for Simplicity
# ---------------------------
data.rename(columns={
    'Region': 'State',
    'Date': 'Date',
    'Estimated Unemployment Rate (%)': 'Unemployment Rate'
}, inplace=True)

# Convert Date to Year
data['Year'] = pd.to_datetime(data['Date'], errors='coerce').dt.year

# ---------------------------
# 4. Handle Missing Values
# ---------------------------
num_cols = data.select_dtypes(include=['int64','float64']).columns
data[num_cols] = data[num_cols].fillna(data[num_cols].median())

cat_cols = data.select_dtypes(include=['object']).columns
for col in cat_cols:
    data[col] = data[col].fillna(data[col].mode()[0])

print("✅ Missing values handled")

# ---------------------------
# 5. Basic Plots
# ---------------------------
if 'State' in data.columns and 'Unemployment Rate' in data.columns:
    plt.figure(figsize=(12,6))
    sns.barplot(x='State', y='Unemployment Rate', data=data)
    plt.xticks(rotation=90)
    plt.title("Unemployment Rate by State")
    plt.tight_layout()
    plt.show()

# Yearly Trend
if 'Year' in data.columns and 'Unemployment Rate' in data.columns:
    yearly_avg = data.groupby('Year')['Unemployment Rate'].mean().reset_index()
    plt.figure(figsize=(10,5))
    sns.lineplot(x='Year', y='Unemployment Rate', data=yearly_avg, marker='o')
    plt.title("Year-wise Average Unemployment Rate")
    plt.grid(True)
    plt.show()

# ---------------------------
# 6. Top & Bottom States - Plotly
# ---------------------------
state_avg = data.groupby('State')['Unemployment Rate'].mean().reset_index()
top_states = state_avg.nlargest(10, 'Unemployment Rate')
bottom_states = state_avg.nsmallest(10, 'Unemployment Rate')

bar_top = px.bar(top_states, x='State', y='Unemployment Rate',
                 color='Unemployment Rate', color_continuous_scale='Reds',
                 title="Top 10 States with Highest Unemployment")
bar_bottom = px.bar(bottom_states, x='State', y='Unemployment Rate',
                    color='Unemployment Rate', color_continuous_scale='Greens',
                    title="Bottom 10 States with Lowest Unemployment")

# ---------------------------
# 7. State-Year Heatmap
# ---------------------------
pivot_table = data.pivot_table(index='State', columns='Year', values='Unemployment Rate', aggfunc='mean')
heatmap_state_year = px.imshow(pivot_table,
                               labels=dict(x="Year", y="State", color="Unemployment Rate"),
                               x=pivot_table.columns, y=pivot_table.index,
                               color_continuous_scale='YlOrRd', text_auto=True)
heatmap_state_year.update_xaxes(tickangle=-45)
heatmap_state_year.update_layout(title="State-wise Yearly Unemployment Heatmap")

# ---------------------------
# 8. Forecast Next 5 Years
# ---------------------------
model = LinearRegression()
X = yearly_avg['Year'].values.reshape(-1,1)
y = yearly_avg['Unemployment Rate'].values
model.fit(X, y)

future_years = np.arange(yearly_avg['Year'].max()+1, yearly_avg['Year'].max()+6).reshape(-1,1)
predictions = model.predict(future_years)
forecast_df = pd.DataFrame({'Year': future_years.flatten(), 'Predicted Unemployment Rate': predictions})

forecast_plot = px.line(yearly_avg, x='Year', y='Unemployment Rate', markers=True, title="Unemployment Forecast")
forecast_plot.add_scatter(x=forecast_df['Year'], y=forecast_df['Predicted Unemployment Rate'],
                          mode='lines+markers', name='Predicted', line=dict(dash='dash', color='red'))

# ---------------------------
# 9. Dash App
# ---------------------------
app = Dash(__name__)
app.title = "Unemployment Analysis Dashboard"

state_options = [{'label': s, 'value': s} for s in sorted(data['State'].unique())]

app.layout = html.Div([
    html.H1("📊 Unemployment Analysis Dashboard", style={'textAlign':'center'}),

    html.H2("Top 10 States with Highest Unemployment"),
    dcc.Graph(figure=bar_top),

    html.H2("Bottom 10 States with Lowest Unemployment"),
    dcc.Graph(figure=bar_bottom),

    html.H2("State-wise Yearly Heatmap"),
    dcc.Graph(figure=heatmap_state_year),

    html.H2("Forecast for Next 5 Years"),
    dcc.Graph(figure=forecast_plot),

    html.H2("Select State to View Yearly Trend"),
    dcc.Dropdown(id='state-dropdown', options=state_options, value=state_options[0]['value']),
    dcc.Graph(id='state-line')
])

# Callback for state-specific trend
from dash.dependencies import Input, Output

@app.callback(
    Output('state-line', 'figure'),
    Input('state-dropdown', 'value')
)
def update_state_line(selected_state):
    df_state = data[data['State']==selected_state].groupby('Year')['Unemployment Rate'].mean().reset_index()
    fig = px.line(df_state, x='Year', y='Unemployment Rate', markers=True,
                  title=f"Year-wise Unemployment Rate: {selected_state}")
    return fig

# ---------------------------
# 10. Run App
# ---------------------------
if __name__ == '__main__':
    app.run(debug=True)

# ---------------------------
# 11. Save Cleaned Dataset
# ---------------------------
data.to_csv("/Users/mac/Downloads/Unemployment_Cleaned.csv", index=False)
print("✅ Cleaned dataset saved as 'Unemployment_Cleaned.csv'")# ============================
# Final Unemployment Analysis & Dashboard - Smooth Version
# ============================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from sklearn.linear_model import LinearRegression
from dash import Dash, dcc, html
import os

# ---------------------------
# 1. File Path Setup
# ---------------------------
file_path = "/Users/mac/Downloads/Unemployment in India.csv.xls"
if not os.path.exists(file_path):
    raise FileNotFoundError(f"❌ File not found: {file_path}")

# ---------------------------
# 2. Load Dataset (Tab-Separated)
# ---------------------------
if file_path.endswith('.csv') or '.csv.' in file_path:
    data = pd.read_csv(file_path, sep='\t')
    print("✅ Loaded as CSV with tab separator")
elif file_path.endswith('.xls') or file_path.endswith('.xlsx'):
    engine = 'xlrd' if file_path.endswith('.xls') else 'openpyxl'
    data = pd.read_excel(file_path, engine=engine)
    print(f"✅ Loaded as Excel using engine '{engine}'")

# Clean column names
data.columns = data.columns.str.strip()
print("✅ Columns after cleaning:", data.columns.tolist())

# ---------------------------
# 3. Rename Columns for Simplicity
# ---------------------------
data.rename(columns={
    'Region': 'State',
    'Date': 'Date',
    'Estimated Unemployment Rate (%)': 'Unemployment Rate'
}, inplace=True)

# Convert Date to Year
data['Year'] = pd.to_datetime(data['Date'], errors='coerce').dt.year

# ---------------------------
# 4. Handle Missing Values
# ---------------------------
num_cols = data.select_dtypes(include=['int64','float64']).columns
data[num_cols] = data[num_cols].fillna(data[num_cols].median())

cat_cols = data.select_dtypes(include=['object']).columns
for col in cat_cols:
    data[col] = data[col].fillna(data[col].mode()[0])

print("✅ Missing values handled")

# ---------------------------
# 5. Basic Plots
# ---------------------------
if 'State' in data.columns and 'Unemployment Rate' in data.columns:
    plt.figure(figsize=(12,6))
    sns.barplot(x='State', y='Unemployment Rate', data=data)
    plt.xticks(rotation=90)
    plt.title("Unemployment Rate by State")
    plt.tight_layout()
    plt.show()

# Yearly Trend
if 'Year' in data.columns and 'Unemployment Rate' in data.columns:
    yearly_avg = data.groupby('Year')['Unemployment Rate'].mean().reset_index()
    plt.figure(figsize=(10,5))
    sns.lineplot(x='Year', y='Unemployment Rate', data=yearly_avg, marker='o')
    plt.title("Year-wise Average Unemployment Rate")
    plt.grid(True)
    plt.show()

# ---------------------------
# 6. Top & Bottom States - Plotly
# ---------------------------
state_avg = data.groupby('State')['Unemployment Rate'].mean().reset_index()
top_states = state_avg.nlargest(10, 'Unemployment Rate')
bottom_states = state_avg.nsmallest(10, 'Unemployment Rate')

bar_top = px.bar(top_states, x='State', y='Unemployment Rate',
                 color='Unemployment Rate', color_continuous_scale='Reds',
                 title="Top 10 States with Highest Unemployment")
bar_bottom = px.bar(bottom_states, x='State', y='Unemployment Rate',
                    color='Unemployment Rate', color_continuous_scale='Greens',
                    title="Bottom 10 States with Lowest Unemployment")

# ---------------------------
# 7. State-Year Heatmap
# ---------------------------
pivot_table = data.pivot_table(index='State', columns='Year', values='Unemployment Rate', aggfunc='mean')
heatmap_state_year = px.imshow(pivot_table,
                               labels=dict(x="Year", y="State", color="Unemployment Rate"),
                               x=pivot_table.columns, y=pivot_table.index,
                               color_continuous_scale='YlOrRd', text_auto=True)
heatmap_state_year.update_xaxes(tickangle=-45)
heatmap_state_year.update_layout(title="State-wise Yearly Unemployment Heatmap")

# ---------------------------
# 8. Forecast Next 5 Years
# ---------------------------
model = LinearRegression()
X = yearly_avg['Year'].values.reshape(-1,1)
y = yearly_avg['Unemployment Rate'].values
model.fit(X, y)

future_years = np.arange(yearly_avg['Year'].max()+1, yearly_avg['Year'].max()+6).reshape(-1,1)
predictions = model.predict(future_years)
forecast_df = pd.DataFrame({'Year': future_years.flatten(), 'Predicted Unemployment Rate': predictions})

forecast_plot = px.line(yearly_avg, x='Year', y='Unemployment Rate', markers=True, title="Unemployment Forecast")
forecast_plot.add_scatter(x=forecast_df['Year'], y=forecast_df['Predicted Unemployment Rate'],
                          mode='lines+markers', name='Predicted', line=dict(dash='dash', color='red'))

# ---------------------------
# 9. Dash App
# ---------------------------
app = Dash(__name__)
app.title = "Unemployment Analysis Dashboard"

state_options = [{'label': s, 'value': s} for s in sorted(data['State'].unique())]

app.layout = html.Div([
    html.H1("📊 Unemployment Analysis Dashboard", style={'textAlign':'center'}),

    html.H2("Top 10 States with Highest Unemployment"),
    dcc.Graph(figure=bar_top),

    html.H2("Bottom 10 States with Lowest Unemployment"),
    dcc.Graph(figure=bar_bottom),

    html.H2("State-wise Yearly Heatmap"),
    dcc.Graph(figure=heatmap_state_year),

    html.H2("Forecast for Next 5 Years"),
    dcc.Graph(figure=forecast_plot),

    html.H2("Select State to View Yearly Trend"),
    dcc.Dropdown(id='state-dropdown', options=state_options, value=state_options[0]['value']),
    dcc.Graph(id='state-line')
])

# Callback for state-specific trend
from dash.dependencies import Input, Output

@app.callback(
    Output('state-line', 'figure'),
    Input('state-dropdown', 'value')
)
def update_state_line(selected_state):
    df_state = data[data['State']==selected_state].groupby('Year')['Unemployment Rate'].mean().reset_index()
    fig = px.line(df_state, x='Year', y='Unemployment Rate', markers=True,
                  title=f"Year-wise Unemployment Rate: {selected_state}")
    return fig

# ---------------------------
# 10. Run App
# ---------------------------
if __name__ == '__main__':
    app.run(debug=True)

# ---------------------------
# 11. Save Cleaned Dataset
# ---------------------------
data.to_csv("/Users/mac/Downloads/Unemployment_Cleaned.csv", index=False)
print("✅ Cleaned dataset saved as 'Unemployment_Cleaned.csv'")