import numpy as np
import pickle
import streamlit as st
#from sklearn.preprocessing import StandardScaler
from streamlit_option_menu import option_menu

#st.title('Ship-Fuel-Consumption')

#grid_trained_model = pickle.load(open("knnpickle_file", "rb"))
grid_knn_model = pickle.load(open("grid_knn_model.sav", "rb"))
grid_RF_model = pickle.load(open("knnpickle_file", "rb"))
#grid_DT_model  = pickle.load(open("grid_dt_model.sav", "rb"))
grid_SVR_model  = pickle.load(open("grid_svr_model.sav", "rb"))


def create_month_array(selected_month):
    months = ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December']
    month_index = months.index(selected_month or 'January')
    month_array = np.zeros(12)
    month_array[month_index] = 1
    return month_array

# Creating a function for prediction
def consumption_LSHFO_prediction(input_data, month, model):
    import pandas as pd
    import numpy as np
    month_encoder = create_month_array(month)
    input_data = np.asarray(input_data, dtype=float)
    #st.success(month_encoder)
    #st.success(input_data)
    #st.write(input_data)
    
    X = np.concatenate((month_encoder, input_data))

    speed_values = np.arange(0, 40)
    num_repeats = len(speed_values)
    
    result = np.concatenate((np.tile(X, (num_repeats, 1)), np.expand_dims(speed_values, axis=1)), axis=1)
    
    if(model == "All"):
        prediction_knn = grid_knn_model.predict(result)
        prediction_RF = grid_RF_model.predict(result)
        prediction_SVR = grid_SVR_model.predict(result)
        prediction = [prediction_knn, prediction_RF, prediction_SVR]
        speed = result[:, -1]
        df = pd.DataFrame({'Speed': speed,
                           'KNN': prediction_knn,
                           'RF': prediction_RF,
                           'SVR': prediction_SVR,})

    else :
        if(model == "KNN"):
            prediction = grid_knn_model.predict(result)
        elif(model == "RF"):
            prediction = grid_RF_model.predict(result)
        elif(model == "SVR"):
            prediction = grid_SVR_model.predict(result)  
        else :
            prediction = "No model selected"
        speed = result[:, -1]
        df = pd.DataFrame({'Speed': speed,'Consumption': prediction})

    # print(prediction)
    
    #first_column_second = prediction
    #new_array = np.column_stack((last_column_first, first_column_second))
    
    return df


import pandas as pd
import numpy as np 


def predict(Consumption_LSHFO, Month, Model):

    df = consumption_LSHFO_prediction(Consumption_LSHFO, Month, Model)
    
    #st.write(df)
    #df = df.rename(columns={'x':'index'}).set_index('index')
    # x="col1", y=["col2", "col3"],
    if(Model=="All"):
        st.line_chart(df, x="Speed", y=["KNN", "RF", "SVR"] ,  color=["#FFFF00", "#FF0000", "#FF00FF"])
    else:
        st.line_chart(df, x="Speed", y="Consumption",  color="#FFFF00")
    st.write(df)


def main():
    # Giving Title
    #st.title("Ship-Fuel-Consumption")
    cols=st.columns(3)
    # Getting input data from the user
    # Wind Power,Amount of Cargo,Auxiliary Engine Power,Main Engine Power,LOA,Breath,Draught,Light Weight,Month,Latitude,Longitude,Speed,Consumption_LSHFO
    with cols[0]:
        WindPower = st.sidebar.text_input("Wind Power", value="2")
        AmountOfCargo = st.sidebar.text_input("Amount of Cargo", value="50999.821")
        AuxiliaryEnginePower = st.sidebar.text_input("Auxiliary Engine Power", value="770")
        MainEnginePower = st.sidebar.text_input("Main Engine Power", value="9480")
    
    with cols[1]:
        LOA = st.sidebar.text_input("LOA", value="190")
        Breath = st.sidebar.text_input("Breath", value="32.26")
        Draught = st.sidebar.text_input("Draught", value="12.54")
        LightWeight = st.sidebar.text_input("Light Weight", value="11149.4")
    
    with cols[2]:
        #Month = st.text_input("Month")
        Month = st.sidebar.selectbox(label="Month", options=('January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December'))
        Latitude = st.sidebar.text_input("Latitude", value="26.995")
        Longitude = st.sidebar.text_input("Longitude", value="56.366")
        #Speed = st.sidebar.text_input("Speed")
        Model = st.sidebar.selectbox(label="Model", options=('KNN', 'RF', 'SVR', 'All'))

    
    # Code for prediction
    Consumption_LSHFO = [
    WindPower, AmountOfCargo, AuxiliaryEnginePower, MainEnginePower,
    LOA, Breath, Draught, LightWeight,
        Latitude, Longitude
    ]

    # Creating button for Prediction
    if st.sidebar.button("Consumption_LSHFO") :
        predict(Consumption_LSHFO, Month, Model)
        
if __name__ == "__main__":
    main()