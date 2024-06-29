import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
import seaborn as sns
import matplotlib.pyplot as plt
import base64  # Added import for base64 encoding
from io import BytesIO
from wordcloud import WordCloud
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

# Load the dataset
df = pd.read_csv('main5k.csv')

# Strip any leading/trailing spaces in column names
df.columns = df.columns.str.strip()

# Check if required columns exist
required_columns = ['Region', 'Month', 'Train Name', 'Food Item', 'Total Sale']
for col in required_columns:
    if col not in df.columns:
        st.error(f"- Column '{col}' not found in the dataset.")
        st.stop()

# Map numeric months to month names
month_mapping = {
    1: 'January', 2: 'February', 3: 'March', 4: 'April', 
    5: 'May', 6: 'June', 7: 'July', 8: 'August', 
    9: 'September', 10: 'October', 11: 'November', 12: 'December'
}
df['Month'] = df['Month'].map(month_mapping)

# Encode categorical variables
df['Region'] = df['Region'].astype('category')
df['Train Name'] = df['Train Name'].astype('category')

# Streamlit App
st.title('Product Sales Analysis and Prediction')

st.sidebar.header('User Input Features')
selected_regions = st.sidebar.multiselect('Regions', df['Region'].unique())
selected_months = st.sidebar.multiselect('Months', df['Month'].unique())
selected_trains = st.sidebar.multiselect('Trains', df['Train Name'].unique())
selected_food_items = st.sidebar.multiselect('Food Items', df['Food Item'].unique())

# Filter the dataset based on user inputs
filtered_df = df[df['Region'].isin(selected_regions) & 
                 df['Month'].isin(selected_months) & 
                 df['Train Name'].isin(selected_trains)]

if filtered_df.empty:
    st.write('No data available for the selected inputs.')
else:
    # Display overall statistics
    st.subheader('Overall Statistics for Selected Criteria')
    total_sales = filtered_df['Total Sale'].sum()
    average_sales = filtered_df['Total Sale'].mean()
    max_sales = filtered_df['Total Sale'].max()
    min_sales = filtered_df['Total Sale'].min()

    st.write(f'- Total Sales: {total_sales}')
    st.write(f'- Average Sales: {average_sales:.2f}')
    st.write(f'- Maximum Sales: {max_sales}')
    st.write(f'- Minimum Sales: {min_sales}')

    # Filter data for selected food items if any
    if selected_food_items:
        filtered_df = filtered_df[filtered_df['Food Item'].isin(selected_food_items)]

    # Find top 5 products for the filtered dataset
    top_5_products = filtered_df.groupby('Food Item')['Total Sale'].sum().nlargest(5)
    top_5_product_names = top_5_products.index.tolist()

    # Display results
    st.write('Top Products for the Selected Inputs:')
    for product in top_5_product_names:
        st.write(f'- {product}: {top_5_products[product]}')

    # Bar chart for top 5 products
    st.subheader('Top 5 Products by Total Sales')
    bar_chart = alt.Chart(top_5_products.reset_index()).mark_bar().encode(
        x=alt.X('Food Item:N', sort='-y'),
        y='Total Sale:Q',
        color='Food Item:N',
        tooltip=['Food Item:N', 'Total Sale:Q']
    ).properties(
        width=600,
        height=400
    ).interactive()

    st.altair_chart(bar_chart, use_container_width=True)

    # Time series chart for sales over months
    st.subheader('Sales Over the Months')
    time_series_data = df[df['Food Item'].isin(top_5_product_names) & 
                          df['Region'].isin(selected_regions) & 
                          df['Train Name'].isin(selected_trains)]

    time_series_chart = alt.Chart(time_series_data).mark_line().encode(
        x=alt.X('Month:O', axis=alt.Axis(title='Month')),
        y=alt.Y('Total Sale:Q', axis=alt.Axis(title='Total Sale')),
        color='Food Item:N',
        tooltip=['Month:O', 'Total Sale:Q', 'Food Item:N']
    ).properties(
        width=600,
        height=400
    ).interactive()

    st.altair_chart(time_series_chart, use_container_width=True)

    # Pie chart for distribution of sales by product
    st.subheader('Sales Distribution by Product')
    pie_chart = alt.Chart(top_5_products.reset_index()).mark_arc().encode(
        theta=alt.Theta('Total Sale:Q', stack=True),
        color=alt.Color('Food Item:N', legend=alt.Legend(title="Food Item")),
        tooltip=['Food Item:N', 'Total Sale:Q']
    ).properties(
        width=600,
        height=400
    ).interactive()

    st.altair_chart(pie_chart, use_container_width=True)

    # Comparison chart for different regions
    st.subheader('Comparison of Sales across Different Regions')
    region_comparison_data = df[df['Food Item'].isin(top_5_product_names) &
                                df['Month'].isin(selected_months) &
                                df['Train Name'].isin(selected_trains)]

    region_comparison_chart = alt.Chart(region_comparison_data).mark_bar().encode(
        x=alt.X('Region:N', sort='-y'),
        y='Total Sale:Q',
        color='Region:N',
        tooltip=['Region:N', 'Total Sale:Q', 'Food Item:N']
    ).properties(
        width=600,
        height=400
    ).interactive()

    st.altair_chart(region_comparison_chart, use_container_width=True)

    # Download filtered data
    st.subheader('Download Filtered Data')
    csv = filtered_df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()  # strings <-> bytes conversions
    href = f'<a href="data:file/csv;base64,{b64}" download="filtered_data.csv">Download CSV File</a>'
    st.markdown(href, unsafe_allow_html=True)

    # Predictive Model (Simple Linear Regression as an example)
    st.subheader('Predictive Model')
    model_type = st.selectbox('Select Model Type', ['Linear Regression', 'Decision Tree', 'Random Forest'])

    # Prepare data for model
    model_data = df[df['Food Item'].isin(top_5_product_names)]
    X = pd.get_dummies(model_data[['Region', 'Month', 'Train Name', 'Food Item']], drop_first=True)
    y = model_data['Total Sale']

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train and predict based on selected model
    if model_type == 'Linear Regression':
        model = LinearRegression()
    elif model_type == 'Decision Tree':
        from sklearn.tree import DecisionTreeRegressor
        model = DecisionTreeRegressor(random_state=42)
    else:
        from sklearn.ensemble import RandomForestRegressor
        model = RandomForestRegressor(random_state=42)

    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    # Display predictions
    st.write('Predictions for the next period:')
    for i, prediction in enumerate(predictions[:5]):
        st.write(f'- {i+1}st prediction: {prediction}')

    # Correlation Heatmap
    st.subheader('Correlation Heatmap')
    numeric_cols = filtered_df.select_dtypes(include=[np.number]).columns
    corr = filtered_df[numeric_cols].corr()
    fig, ax = plt.subplots()
    sns.heatmap(corr, annot=True, ax=ax, cmap='coolwarm', center=0)
    st.pyplot(fig)

    # Sales Trends Comparison for selected food items across all regions
    st.subheader('Sales Trends for Selected Food Items across All Regions')
    sales_trends_data = df[df['Food Item'].isin(selected_food_items)]

    sales_trends_chart = alt.Chart(sales_trends_data).mark_line().encode(
        x=alt.X('Month:O', axis=alt.Axis(title='Month')),
        y=alt.Y('Total Sale:Q', axis=alt.Axis(title='Total Sale')),
        color='Region:N',
        tooltip=['Month:O', 'Total Sale:Q', 'Food Item:N', 'Region:N']
    ).properties(
        width=600,
        height=400
    ).interactive()

    st.altair_chart(sales_trends_chart, use_container_width=True)

    # Histogram for total sales distribution
    st.subheader('Total Sales Distribution')
    hist_chart = alt.Chart(filtered_df).mark_bar().encode(
        x=alt.X('Total Sale:Q', bin=True),
        y='count()',
        tooltip=['Total Sale:Q', 'count()']
    ).properties(
        width=600,
        height=400
    ).interactive()

    st.altair_chart(hist_chart, use_container_width=True)

    # Scatter plot for total sales vs. food items
    st.subheader('Total Sales vs. Food Items')
    scatter_chart = alt.Chart(filtered_df).mark_circle(size=60).encode(
        x='Food Item:N',
        y='Total Sale:Q',
        color='Region:N',
        tooltip=['Food Item:N', 'Total Sale:Q', 'Region:N']
    ).properties(
        width=600,
        height=400
    ).interactive()

    st.altair_chart(scatter_chart, use_container_width=True)

    # Word Cloud for Food Items
    st.subheader('Word Cloud for Food Items')
    wordcloud_data = ' '.join(filtered_df['Food Item'])
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(wordcloud_data)
    fig, ax = plt.subplots()
    ax.imshow(wordcloud, interpolation='bilinear')
    ax.axis('off')
    st.pyplot(fig)
