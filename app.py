#!/usr/bin/env python
# coding: utf-8

# In[52]:


import torch
import gradio as gr
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
import pickle
import torch
import gradio as gr
import pandas as pd
import numpy as np

import torch
import gradio as gr
import pandas as pd
import pickle
import numpy as np


# ✅ Load dataset to extract feature names
car_data = pd.read_csv("car_price_dataset.csv")  # Ensure dataset is accessible
car_data["Brand_model"] = car_data["Brand"] + "_" + car_data["Model"]
car_data = car_data.drop(columns=["Brand", "Model"])

# ✅ Load preprocessing objects
with open("fuel_encoder.pkl", "rb") as f:
    ordinal_fuel = pickle.load(f)

with open("trasmission_encoder.pkl", "rb") as f:
    ordinal_transmission = pickle.load(f)

with open("brand_encoder.pkl", "rb") as f:
    brand_encoder = pickle.load(f)

with open("Price_scaler.pkl", "rb") as f:
    Price_scaler = pickle.load(f)

with open("Owner_Count_scaler.pkl", "rb") as f:
    Owner_count_scaler = pickle.load(f)

with open("Mileage_scaler.pkl", "rb") as f:
    Mileage_scaler = pickle.load(f)

# ✅ Identify categorical & numerical columns
categorical_features = ["Fuel_Type", "Transmission", "Brand_model"]
numerical_features = ["Year", "Engine_Size", "Mileage", "Doors", "Owner_Count"]
all_features = categorical_features + numerical_features

# ✅ Define device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ✅ Load the trained model
model = torch.load("car_price_model.pth", map_location=device)
model.to(device)
model.eval()

# ✅ Extract categorical options dynamically from dataset
categorical_options = {
    "Fuel_Type": ["Electric", "Hybrid", "Diesel", "Petrol"],
    "Transmission": ["Automatic", "Semi-Automatic", "Manual"],
    "Brand_model": list(car_data["Brand_model"].unique())  # Get unique (Brand, Model) names
}

# ✅ Initialize session state to store inputs
session_state = {
    "step": 0,
    "inputs": {feature: None for feature in all_features}
}

# ✅ Function to apply preprocessing before prediction
def preprocess_input(inputs):
    """Apply the same preprocessing transformations as in training."""
    # Encode categorical variables
    fuel_transformed = ordinal_fuel.transform([[inputs["Fuel_Type"]]])[0][0]
    transmission_transformed = ordinal_transmission.transform([[inputs["Transmission"]]])[0][0]
    brand_transformed = brand_encoder.transform([inputs["Brand_model"]])[0]  # Encode brand

    # Transform numerical features
    mileage_transformed = np.log(inputs["Mileage"])  # Apply log transformation
    year_transformed = car_data["Year"].max() - inputs["Year"]  # Convert Year to Age

    # Scale numerical values using the correct scalers
    mileage_scaled = Mileage_scaler.transform([[mileage_transformed]])[0][0]
    owner_count_scaled = Owner_count_scaler.transform([[inputs["Owner_Count"]]])[0][0]

    # Include Engine_Size and Doors
    engine_size = inputs["Engine_Size"]
    doors = inputs["Doors"]

    # Combine processed features
    processed_input = [
        fuel_transformed, transmission_transformed, brand_transformed,
        year_transformed, engine_size, mileage_scaled, doors, owner_count_scaled
    ]

    return torch.tensor([processed_input], dtype=torch.float32).to(device)

# ✅ Function to make predictions when all inputs are provided
def predict_price():
    if None in session_state["inputs"].values():
        return "Waiting for all inputs..."

    # Preprocess input
    processed_input = preprocess_input(session_state["inputs"])

    # Make a prediction
    with torch.no_grad():
        prediction_scaled = model(processed_input).cpu().numpy()[0][0]  # Extract scalar value

    # Inverse scale price
    prediction = Price_scaler.inverse_transform([[prediction_scaled]])[0][0]

    return f"Estimated Car Price: ${prediction:,.2f}"

# ✅ Function to handle navigation between features
def update_interface(value=None):
    """Store user input, move to next feature, and update UI."""
    if value is not None:  # Only store value if provided
        current_feature = all_features[session_state["step"]]
        session_state["inputs"][current_feature] = value
        
        # Move to next step if value was provided
        if session_state["step"] < len(all_features) - 1:
            session_state["step"] += 1

    # Get current feature type
    current_feature = all_features[session_state["step"]]
    is_categorical = current_feature in categorical_features

    # Update visibility and values
    dropdown_update = gr.update(
        label=f"Select {current_feature}",
        choices=categorical_options[current_feature] if is_categorical else [],
        visible=is_categorical
    )
    
    number_update = gr.update(
        label=f"Enter {current_feature}",
        visible=not is_categorical
    )

    # Show back button if not on first step
    back_button_update = gr.update(visible=session_state["step"] > 0)

    prediction = predict_price()

    return [dropdown_update, number_update, prediction, back_button_update]

# ✅ Function to handle next button for numeric inputs
def handle_next_button(number_value):
    """Process numeric input when Next button is clicked."""
    current_feature = all_features[session_state["step"]]
    is_categorical = current_feature in categorical_features
    
    # Only process if we're on a numeric input
    if not is_categorical and number_value is not None:
        return update_interface(number_value)
    
    # Otherwise return the current state
    return [
        gr.update(),  # dropdown
        gr.update(),  # number input
        predict_price(),  # prediction
        gr.update(visible=session_state["step"] > 0)  # back button
    ]

# ✅ Function to go back
def go_back():
    """Move back to the previous feature."""
    if session_state["step"] > 0:
        session_state["step"] -= 1
        # Clear the input for the current step
        current_feature = all_features[session_state["step"]]
        session_state["inputs"][current_feature] = None
    
    return update_interface()

# ✅ Define Gradio Interface (Clean & Minimal UI)
with gr.Blocks() as iface:
    gr.Markdown("# 🚗 Car Price Estimator")

    with gr.Row():  # Input and Output in separate columns
        with gr.Column():
            gr.Markdown("### Enter Vehicle Details")

            # ✅ Create both a dropdown and a number input
            dropdown_input = gr.Dropdown(label="Select Feature", choices=[], interactive=True)
            number_input = gr.Number(label="Enter Feature", interactive=True)

            with gr.Row():  # Small, non-intrusive buttons
                back_button = gr.Button("⬅️ Back", size="sm", visible=False)
                next_button = gr.Button("➡️ Next", size="sm")

        with gr.Column():
            gr.Markdown("### Estimated Price")
            output_text = gr.Textbox(value="Waiting for all inputs...", interactive=False)

    # ✅ Button Actions
    next_button.click(
        handle_next_button,
        inputs=[number_input],
        outputs=[dropdown_input, number_input, output_text, back_button]
    )

    dropdown_input.change(
        update_interface,
        inputs=[dropdown_input],
        outputs=[dropdown_input, number_input, output_text, back_button]
    )

    number_input.submit(
        update_interface,
        inputs=[number_input],
        outputs=[dropdown_input, number_input, output_text, back_button]
    )

    back_button.click(
        go_back,
        outputs=[dropdown_input, number_input, output_text, back_button]
    )

    # Initialize the interface
    iface.load(
        update_interface,
        outputs=[dropdown_input, number_input, output_text, back_button]
    )

# ✅ Launch the Gradio app
iface.launch()


# In[13]:


import torch

# Check what is inside the file
model_data = torch.load("car_price_model2.pth", map_location="cpu")

print(type(model_data))  # Expected output should be <class 'torch.nn.Module'> if it's the full model

