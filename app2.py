import streamlit as st
import pandas as pd
import joblib
import ollama
from PIL import Image

# 1. Page Config & Title
st.set_page_config(page_title="Fuel Economy ML App", layout="wide")
st.title("🚗 Automotive Efficiency & AI Dashboard")

# 2. Load Assets (Cached so they only load once)
@st.cache_data
def load_data():
    return pd.read_csv('cleaned_vehicles.csv')

@st.cache_resource
def load_models():
    mpg_model = joblib.load('rf_mpg_model.joblib')
    co2_model = joblib.load('rf_co2_model.joblib')
    return mpg_model, co2_model

data = load_data()
rf_mpg_model, rf_co2_model = load_models()

# 3. Sidebar: Vehicle Selection UI
st.sidebar.header("Select a Vehicle")

# Cascading Dropdowns
selected_year = st.sidebar.selectbox("Year", sorted(data['year'].unique(), reverse=True))
year_filtered = data[data['year'] == selected_year]

selected_make = st.sidebar.selectbox("Make", sorted(year_filtered['make'].unique()))
make_filtered = year_filtered[year_filtered['make'] == selected_make]

selected_model = st.sidebar.selectbox("Model", sorted(make_filtered['model'].unique()))
final_filtered = make_filtered[make_filtered['model'] == selected_model]

# Get the specific data for the chosen car (taking the first row if multiple trims exist)
selected_car_data = final_filtered.iloc[0]

# 4. Main Dashboard: Display CV and Predictions
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("YOLOv8 Object Detection")
    try:
        # Load the annotated image you saved in Phase 4
        img = Image.open('annotated_car_image.jpg')
        st.image(img, caption="COCO Class 2 Confirmed", use_column_width=True)
    except FileNotFoundError:
        st.warning("Please ensure 'annotated_car_image.jpg' is in the same folder.")

with col2:
    st.subheader("Model Predictions")
    
    # Reconstruct the feature row for prediction
    X_input = pd.DataFrame({
        'year': [selected_car_data['year']],
        'cylinders': [selected_car_data['cylinders']],
        'displ': [selected_car_data['displ']],
        'is_electrified': [selected_car_data['is_electrified']],
        'gear_count': [selected_car_data['gear_count']],
        'trany_type_Manual': [1 if selected_car_data['trany_type'] == 'Manual' else 0],
        'trany_type_Other/Unknown': [1 if selected_car_data['trany_type'] == 'Other/Unknown' else 0],
        'drive_grouped_AWD_4WD': [1 if selected_car_data['drive_grouped'] == 'AWD_4WD' else 0],
        'drive_grouped_Unknown': [1 if selected_car_data['drive_grouped'] == 'Unknown' else 0]
    })
    
    # Make the predictions
    predicted_mpg = rf_mpg_model.predict(X_input)[0]
    actual_mpg = selected_car_data['comb08']
    
    predicted_co2 = rf_co2_model.predict(X_input)[0]
    actual_co2 = selected_car_data['co2TailpipeGpm']
    
    # Display side-by-side metrics for MPG
    st.markdown("**Combined Fuel Economy (MPG)**")
    metric_col1, metric_col2 = st.columns(2)
    metric_col1.metric("Predicted MPG", f"{predicted_mpg:.1f}")
    metric_col2.metric("Measured MPG", f"{actual_mpg:.1f}", delta=f"{actual_mpg - predicted_mpg:.1f} error")
    
    st.divider()
    
    # Display side-by-side metrics for CO2
    st.markdown("**Tailpipe CO2 Emissions (g/mi)**")
    metric_col3, metric_col4 = st.columns(2)
    metric_col3.metric("Predicted CO2", f"{predicted_co2:.1f}")
    metric_col4.metric("Measured CO2", f"{actual_co2:.1f}", delta=f"{actual_co2 - predicted_co2:.1f} error", delta_color="inverse")
    # Note: delta_color="inverse" makes a positive error (higher CO2) show as red instead of green!

# 5. Local RAG with Ollama
st.divider()
st.subheader("Ask the Local AI about this Vehicle")

user_question = st.text_input("Ask a question about this car's efficiency:")

if st.button("Ask Ollama"):
    if user_question:
        with st.spinner("Thinking..."):
            # Construct the context payload from our dataset
            context = f"""
            Vehicle: {selected_car_data['year']} {selected_car_data['make']} {selected_car_data['model']}
            Engine: {selected_car_data['cylinders']} cylinders, {selected_car_data['displ']}L displacement
            Electrified: {'Yes' if selected_car_data['is_electrified'] == 1 else 'No'}
            Combined MPG: {selected_car_data['comb08']}
            Annual Fuel Cost: ${selected_car_data['fuelCost08']}
            Tailpipe CO2 Emissions: {selected_car_data['co2TailpipeGpm']} g/mi
            """
            
            # Formulate the prompt for the LLM
            prompt = f"""
            You are an automotive data expert. Use ONLY the following vehicle context to answer the user's question. 
            Do not invent outside information.
            
            Context:
            {context}
            
            Question: {user_question}
            """
            
            # Call the local Ollama model (Make sure to change the model name if you use qwen3:8b)
            response = ollama.chat(model='llama3', messages=[
                {'role': 'user', 'content': prompt}
            ])
            
            st.success(response['message']['content'])
            
            # Display retrieved context per rubric requirements
            with st.expander("Show Retrieved Context Used by LLM"):
                st.code(context)