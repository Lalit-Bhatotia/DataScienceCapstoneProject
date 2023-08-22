import pickle
with open('best_model.pkl', 'wb') as f:
    pickle.dump(data_to_save, f)

# Load the pipeline
with open('best_model.pkl', 'rb') as file:
    pipeline = pickle.load(file)

def filter_cars_by_company(selected_company, car_details):
    sorted_df = df.sort_values(['company', 'name'], ascending=True)
    filtered_cars = sorted_df[sorted_df['company'] == selected_company]['name'].unique()
    return filtered_cars

# Create the web app
def main():
    # Set the title and description
    st.title('Car Price Prediction')
    st.write('Enter the details of the car to predict its price.')
    # Add a CSS class to the Streamlit app's root element
    st.markdown(
        """
       <style>
       .stApp {
          background-color: lightblue;
      }
      </style>
      """,
     unsafe_allow_html=True
  )

    # Get user inputs
    company = st.selectbox('Company Name', sorted(df['company'].unique()))
    name = st.selectbox('Car Name',filter_cars_by_company(company,car_details))
    year = st.number_input('Year', min_value=1900, max_value=2023, step=1)
    km_driven = st.number_input('Kilometers Driven', step=1000)
    fuel = st.selectbox('Fuel Type', df['fuel'].unique())
    seller_type = st.selectbox('Seller Type', df['seller_type'].unique())
    transmission = st.selectbox('Transmission', df['transmission'].unique())
    owner = st.selectbox('Owner', df['owner'].unique())

    data = pd.DataFrame({'name': [name],
                         'year': [year],
                         'km_driven': [km_driven],
                         'fuel': [fuel],
                         'seller_type': [seller_type],
                         'transmission': [transmission],
                         'owner': [owner]})

    if st.button('Predict Price'):
        # Extract the transformer and regressor from the pipeline
        transformer, regressor = pipeline

        # Perform one-hot encoding on categorical columns
        data_encoded = transformer.transform(data).toarray()

         # Make predictions
        predictions = regressor.predict(data_encoded)
        success_message = f'Price of the car is {predictions[0]} INR'

        # Add color to the success message
        colored_message =  f'<span style="color: blue; font-size: 24px;font-weight: bold;">{success_message}</span>'
        st.markdown(colored_message, unsafe_allow_html=True)

# Run the web app
if __name__ == '__main__':
    main()
