import streamlit as st
import pandas as pd

excel_file_path = 'train.xlsx'
csv_file_path = 'indian_food.csv'

excel_data = pd.read_excel(excel_file_path, sheet_name='train')
csv_data = pd.read_csv(csv_file_path)

def get_food_details(food_name):
    food_details = csv_data[csv_data['name'].str.lower() == food_name.lower()]
    return food_details

st.set_page_config(layout="wide")
st.title('Train Food Information')

st.sidebar.header('Filter Options')

train_names = ['--Select--'] + list(excel_data['Train Name/Number'].unique())
selected_train = st.sidebar.selectbox('Select a Train', train_names)

if selected_train != '--Select--':
    train_details = excel_data[excel_data['Train Name/Number'] == selected_train]
    
    if not train_details.empty:
        st.sidebar.header('Train Details')
        st.sidebar.markdown('**Route:**')
        st.sidebar.markdown(train_details.iloc[0]['Route'])
        
        st.sidebar.markdown('**Dining Car Availability:**')
        st.sidebar.markdown(train_details.iloc[0]['Dining Car Availability (Yes/No)'])
        
        st.sidebar.markdown('**Concession Stand Availability:**')
        st.sidebar.markdown(train_details.iloc[0]['Concession Stand Availability (Yes/No)'])
        
    st.header('Food Item Details')
    for meal in ['Breakfast', 'Lunch', 'Dinner', 'Snacks']:
        st.subheader(f'{meal}')
        food_items = train_details.iloc[0][meal].split(', ')
        
        for food in food_items:
            food_details = get_food_details(food)
            if not food_details.empty:
                with st.expander(f"{food}"):
                    food_table = pd.DataFrame({
                        'Category': ['Ingredients', 'Diet', 'Preparation Time', 'Cooking Time', 'Flavor Profile', 'Course', 'State', 'Region'],
                        'Details': [
                            food_details.iloc[0]['ingredients'],
                            food_details.iloc[0]['diet'],
                            f"{food_details.iloc[0]['prep_time']} minutes",
                            f"{food_details.iloc[0]['cook_time']} minutes",
                            food_details.iloc[0]['flavor_profile'],
                            food_details.iloc[0]['course'],
                            food_details.iloc[0]['state'],
                            food_details.iloc[0]['region']
                        ]
                    })
                    st.table(food_table)
            else:
                st.markdown(f'**{food}**: No detailed information available.')

