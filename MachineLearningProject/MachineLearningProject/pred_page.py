import streamlit as st
import pickle
import numpy as np

def load_model():
    with open("saved_steps.pkl", "rb") as file:
        data = pickle.load(file)
    return data

data = load_model()

model_loaded = data["model"]
le_gender = data["le_gender"]
le_ct = data["le_ct"]
le_tot = data["le_tot"]
le_class = data["le_class"]
scaler_age = data["scaler_age"]
scaler_fd = data["scaler_fd"]
scaler_dd = data["scaler_dd"]

def show_predict_page():
    st.title("Airline Customer Satisfaction Prediction Using Logistic Regression")
    st.image('plane.jpg', caption='Airline', use_column_width=True)
    st.write("Customer satisfaction is the heartbeat of an airline company. It's not just about happy passengers; it's about building trust, loyalty, and positive word-of-mouth. Satisfied customers are more likely to fly again and recommend the airline to others. By focusing on customer satisfaction, airlines can create better travel experiences, strengthen their brand, and stay ahead in a competitive market.")
    st.write("This app utilize logistic regression to predict customer satisfaction based on various factors. This predictive model helps airlines understand what drives satisfaction, allowing them to make data-driven decisions to enhance the overall passenger experience.")
    st.write("""### Enter the following customer data to predict their satisfaction""")

    Genders = {
        "Male",
        "Female"
    }

    Gender = st.selectbox("**Gender**", Genders)

    CustomerTypes = {
        "Loyal Customer",
        "disloyal Customer"
    }

    Customer = st.selectbox("**Customer Type**", CustomerTypes)

    Age = st.number_input('**Age**', min_value=7, max_value=85)

    TypeOfTravels = {
        "Business travel",
        "Personal Travel"
    }

    TypeOfTravel = st.selectbox("**Type of Travel**", TypeOfTravels)

    Classes = {
        "Business",
        "Eco",
        "Eco Plus"
    }

    Class = st.selectbox("**Class**", Classes)

    FlightDistance = st.number_input('**Flight Distance**', min_value=50, max_value=6951)

    SeatComfort = st.slider("**Seat Comfort**", 0, 5, 0, step=1)
    
    DepartureArrivaltimeconvenient = st.slider("**Departure/Arrival time convenient**", 0, 5, 0, step=1)
    
    Foodanddrink = st.slider("**Food and Drink**", 0, 5, 0, step=1)

    GateLocation = st.slider("**Gate Location**", 0, 5, 0, step=1)

    InflightWifiService = st.slider("**Inflight Wi-Fi Service**", 0, 5, 0, step=1)

    InflightEntertainment = st.slider("**Inflight Entertainment**", 0, 5, 0, step=1)

    OnlineSupport = st.slider("**Online Support**", 0, 5, 0, step=1)

    EaseOfOnlineBooking = st.slider("**Ease of Online Booking**", 0, 5, 0, step=1)

    OnBoardService = st.slider("**On-board Service**", 0, 5, 0, step=1)

    LegRoomService = st.slider("**Leg Room Service**", 0, 5, 0, step=1)

    BaggageHandling = st.slider("**Baggage Handling**", 0, 5, 0, step=1)

    CheckinService = st.slider("**Check-in Service**", 0, 5, 0, step=1)

    Cleanliness = st.slider("**Cleanliness**", 0, 5, 0, step=1)

    OnlineBoarding = st.slider("**Online Boarding**", 0, 5, 0, step=1)

    DepartureDelayInMinutes = st.number_input('**Departure Delay in Minutes**', min_value=0, max_value=1584)

    
    st.markdown("&nbsp;")
    ok = st.button("**Let's predict!**")
    if ok:
        X = np.array([[Gender, Customer, Age, TypeOfTravel, Class, FlightDistance, SeatComfort, DepartureArrivaltimeconvenient, Foodanddrink, GateLocation, InflightWifiService, InflightEntertainment, OnlineSupport, EaseOfOnlineBooking, OnBoardService, LegRoomService, BaggageHandling, CheckinService, Cleanliness, OnlineBoarding, DepartureDelayInMinutes]])
        X[:, 0] = le_gender.transform(X[:, 0])
        X[:, 1] = le_ct.transform(X[:, 1])
        X[:, 3] = le_tot.transform(X[:, 3])
        X[:, 4] = le_class.transform(X[:, 4])
        X[:, 2] = scaler_age.fit_transform(X[:, [2]].astype(float)).flatten()
        X[:, 5] = scaler_fd.fit_transform(X[:, [5]].astype(float)).flatten()
        X[:, 20] = scaler_dd.fit_transform(X[:, [20]].astype(float)).flatten()
        X = X.astype(float)

        predict = model_loaded.predict(X)

        prediction = "Prediction: Satisfied" if predict == 'satisfied' else "Prediction: Dissatisfied"
        if predict == 'satisfied':
            st.subheader(f"{prediction}")
            st.write("Keep up the good work!")
        else:
            st.subheader(f"{prediction}")
            st.write("Looks like the customer dissatisfied! Let's evaluate and keep improving!")
        

        