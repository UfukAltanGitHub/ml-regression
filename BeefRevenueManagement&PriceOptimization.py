import streamlit as st
import numpy as np
import pandas as pd
import joblib
import time


randomForest = joblib.load('BeefRandomForest.pkl')
scaler = joblib.load('BeefRandomForestScaler.pkl')


def append_data(variables):
        
    variable_list = []
    
    for variable in variables:
        for data_point in variable:
            variable_list.append(data_point)
            
    return variable_list      


def preprocess_entry(data_entry):

    data = []

    #Quarter
    
    if data_entry[1] == 'Q1':
        q = [0,0,0]
        
    elif data_entry[1] == 'Q2':
        q = [1,0,0]
    
    elif data_entry[1] == 'Q3':
        q = [0,1,0]
    
    elif data_entry[1] == 'Q4':
        q = [0,0,1]
    
    #Brand
    
    if data_entry[2] == 'A':
        b = [0,0,0]
    
    elif data_entry[2] == 'B':
        b = [1,0,0]
    
    elif data_entry[2] == 'C':
        b = [0,1,0]
    
    elif data_entry[2] == 'D':
        b = [0,0,1]
            
    variables = [q, b]
    data = append_data(variables)
    data = np.array(data)
    data = np.insert(data, 0, [data_entry[0]])    
    
    return data 

def create_prediction(data,scaler= scaler, model= randomForest):
    
    data = np.reshape(data, (1,7))
    data = scaler.transform(data)   
    
    prediction = model.predict(data)[0]

    return round(prediction)


def optimize_values(data_entry, rangeA = (230, 300), rangeB = (180, 260), rangeC = (130, 210), rangeD = (130, 160)):
    
    quantityA, quantityB, quantityC, quantityD = [], [], [], []
    revenueA, revenueB, revenueC, revenueD = [], [], [], []
    
    #Determine size

    sizeA, sizeB, sizeC, sizeD = rangeA[1]-rangeA[0], rangeB[1]-rangeB[0], rangeC[1]-rangeC[0], rangeD[1]-rangeD[0]
    sizes = [sizeA, sizeB, sizeC, sizeD]
    sizes = np.array(sizes)
    
    #Generate Values

    priceA = np.random.choice(range(rangeA[0], rangeA[1]+1), sizes.min(), replace=False)
    priceA.sort()

    
    priceB = np.random.choice(range(rangeB[0], rangeB[1]+1), sizes.min(), replace=False)
    priceB.sort()

    
    priceC = np.random.choice(range(rangeC[0], rangeC[1]+1), sizes.min(), replace=False)
    priceC.sort()

    priceD = np.random.choice(range(rangeD[0], rangeD[1]+1), sizes.min(), replace=False)
    priceD.sort()


    for pA,pB,pC,pD in zip(priceA,priceB,priceC,priceD):

        #Calculate A
        dataA = [pA, data_entry, 'A']
        quanA = create_prediction(preprocess_entry(dataA))
        revA = quanA*pA

        quantityA.append(quanA)
        revenueA.append(revA)

        #Calculate B    
        dataB = [pB, data_entry, 'B']
        quanB = create_prediction(preprocess_entry(dataB))
        revB = quanB*pB

        quantityB.append(quanB)
        revenueB.append(revB)

        #Calculate C
        dataC = [pC, data_entry, 'C']
        quanC = create_prediction(preprocess_entry(dataC))
        revC = quanC*pC

        quantityC.append(quanC)
        revenueC.append(revC)

        #Calculate D
        dataD = [pD, data_entry, 'D']
        quanD = create_prediction(preprocess_entry(dataD))
        revD = quanD*pD

        quantityD.append(quanD)       
        revenueD.append(revD)



    table = pd.DataFrame({'PriceA': priceA, 'QuantityA': quantityA, 'RevenueA': revenueA,
                          'PriceB': priceB, 'QuantityB': quantityB, 'RevenueB': revenueB,
                          'PriceC': priceC, 'QuantityC': quantityC, 'RevenueC': revenueC,
                          'PriceD': priceD, 'QuantityD': quantityD, 'RevenueD': revenueD,})                    

    valuesA = table[table['RevenueA'] == table['RevenueA'].max()][['PriceA', 'QuantityA', 'RevenueA']]
    valuesB = table[table['RevenueB'] == table['RevenueB'].max()][['PriceB', 'QuantityB', 'RevenueB']]
    valuesC = table[table['RevenueC'] == table['RevenueC'].max()][['PriceC', 'QuantityC', 'RevenueC']]
    valuesD = table[table['RevenueD'] == table['RevenueD'].max()][['PriceD', 'QuantityD', 'RevenueD']]    
    
    return valuesA, valuesB, valuesC, valuesD, table




quarter = st.selectbox('Which quarter do you intend to research',["Q1","Q2", "Q3", "Q4"])

A_low = st.text_input(label="Lowest Price for Brand A", value = '230')
A_high = st.text_input(label="Highest Price for Brand A", value = '300')

B_low = st.text_input(label="Lowest Price for Brand B", value = '180')
B_high = st.text_input(label="Highest Price for Brand B", value = '260')

C_low = st.text_input(label="Lowest Price for Brand C", value = '160')
C_high = st.text_input(label="Highest Price for Brand C", value = '210')

D_low = st.text_input(label="Lowest Price for Brand D", value = '130')
D_high = st.text_input(label="Highest Price for Brand D", value = '170')


a, b, c, d, table = optimize_values(quarter, rangeA = (int(A_low), int(A_high)), rangeB = (int(B_low), int(B_high)),
 								   rangeC = (int(C_low), int(C_high)), rangeD = (int(D_low), int(D_high)))


total_revenue = a['RevenueA'].values[0] + b['RevenueB'].values[0] + c['RevenueC'].values[0] + d['RevenueD'].values[0]

if st.button('Optimize Values'):
	with st.spinner("Calculating Results..."):
		time.sleep(1)
	
	st.header("Total Estimated Revenue is: ${}".format(int(total_revenue)))

	if not st.checkbox("Show details"):

		st.dataframe(a)
		st.dataframe(b)
		st.dataframe(c)
		st.dataframe(d)