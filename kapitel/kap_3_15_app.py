import streamlit as st
import pandas as pd
import joblib

df = pd.read_csv("data/pruned_car_price_dataset.csv")

column_names = df.columns.values

column_names = column_names[1:]

brands = []
models = []
fuel_types = []
transmissions = []
others = []

ai_model = joblib.load("models/kap_3_15_model.joblib")

for name in column_names:
    if name[:5] == "Brand":
        brands.append(name)
    elif name[:5] == "Model":
        models.append(name)
    elif name[:4] == "Fuel":
        fuel_types.append(name)
    elif name[:12] == "Transmission":
        transmissions.append(name)
    else:
        others.append(name)

st.title("Bilvärderare")

def remove_label(lst):
    temp_lst = []
    for name in lst:
        temp = name.split("_")
        temp_lst.append(temp[-1])
    return temp_lst

brands = remove_label(brands)
models = remove_label(models)
fuel_types = remove_label(fuel_types)
transmissions = remove_label(transmissions)

input_brand = st.selectbox("Märke", brands)
input_model = st.selectbox("Modell", models)
input_transmission = st.selectbox("Växellåda", transmissions)
input_fuel_type = st.selectbox("Drivmedel", fuel_types)

input_year = st.number_input(
    label="Årsmodell",
    value=0,
    min_value=0,
    format="%0f")
input_mileage = st.number_input(
    label="Miltal",
    min_value=0,
    value=0,
    format="%0f")

input_dict = {}

for name in range(1,len(column_names)):
    input_dict.update({column_names[name]:""})

input_dict.update({"Year":input_year})
input_dict.update({"Mileage":input_mileage})

selectbox_inputs = [input_brand, input_model, input_fuel_type, input_transmission]

for name in input_dict:
    suffix = name.split("_")[-1]
    if suffix not in others:
        input_dict[name] = 1 if suffix in selectbox_inputs else 0

final_df = pd.DataFrame([input_dict])
final_df = final_df[ai_model.feature_names_in_]

if st.button("Värdera", type="primary"):
    prediction = ai_model.predict(final_df)
    prediction = round(float(prediction[0]), 2)
    st.write(prediction, "kr")