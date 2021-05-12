import joblib
import streamlit as st
import pandas as pd
import numpy as np

st.title('App Empréstimo POLI USP')
st.text(" ")
st.text(" ")

from PIL import Image
img = Image.open('Emprestimo.jpg')
st.image(img)

# Variáveis
st.sidebar.header('Preencha o Requerimento')
credit = st.sidebar.selectbox('Medelo Preditivo', list(['Sem Hitórico de Crédito', 'Com Hitórico de Crédito']))
Gender = st.sidebar.selectbox('Gênero', list(['Masculino', 'Feminino']))
Married = st.sidebar.selectbox('Casado', list(['Não', 'Sim']))
Dependents = st.sidebar.selectbox("Nº Dependentes", list([0, 1, 2, '3+']))
Education = st.sidebar.selectbox('Escolaridade', list(['Ensino Médio', 'Superior']))
Self_Employed = st.sidebar.selectbox('Autônomo', list(['Não', 'Sim']))
ApplicantIncome = st.sidebar.number_input("Sua Renda", 0)
CoapplicantIncome = st.sidebar.number_input("Renda Do Fiador", 0)
LoanAmount = st.sidebar.number_input("Montante do Empréstimo em Milhares", 0)
Loan_Amount_Term = st.sidebar.slider("Prazo do Empréstimo (Meses)", 1, 360, 1)
Credit_History = st.sidebar.selectbox('Histórico de Crédito', list(['Não', 'Sim']))
Property_Area = st.sidebar.selectbox('Localização da Propriedade', list(['Urbano', 'Semi Urbano', 'Rural']))
btn_predict = st.sidebar.button("REALIZAR PREDIÇÃO")

if btn_predict:
    #DataFrame
    head = [{'Gender': Gender, 'Married': Married, 'Dependents': Dependents, 'Education': Education,
             'Self_Employed': Self_Employed, 'ApplicantIncome': ApplicantIncome,'CoapplicantIncome': CoapplicantIncome,
         'LoanAmount': LoanAmount,'Loan_Amount_Term': Loan_Amount_Term, 'Credit_History': Credit_History,
         'Property_Area': Property_Area}]

    df = pd.DataFrame(head)

    x = df['ApplicantIncome']
    y = df['LoanAmount']

    if x[0] == 0 | y[0] == 0:
        st.header("Sua Renda e Montande de Empréstimo Precisam sem Diferentes de 0!")

    else:
        df['Gender'] = df['Gender'].map({'Masculino': 1, 'Feminino': 0})
        df['Married'] = df['Married'].map({'Não': 1, 'Sim': 0})
        df['Dependents'] = df['Dependents'].map({0:0, 1:1, 2:2, '3+': 3})
        df['Education'] = df['Education'].map({'Ensino Médio': 1, 'Superior': 0})
        df['Self_Employed'] = df['Self_Employed'].map({'Não': 1, 'Sim': 0})
        df['Credit_History'] = df['Credit_History'].map({'Sim': 1, 'Não': 0})
        df['Property_Area'] = df['Property_Area'].map({'Semi Urbano': 1, 'Rural': 0, 'Urbano':2})

        # Features
        df['LoanAmount_log'] = np.log(df['LoanAmount'])
        df['TotalIncome'] = df['ApplicantIncome'] + df['CoapplicantIncome']
        df['TotalIncome_log'] = np.log(df['TotalIncome'])
        df['Conditions_to_pay'] = df['LoanAmount'] / df['TotalIncome']
        df["EMI"] = df["LoanAmount"] / df["Loan_Amount_Term"]
        df["Balance_Income"] = df["TotalIncome"] - df["EMI"]*1000

        if credit == 'Sem Hitórico de Crédito':
            # Carregando Modelo e Prevendo
            joblib_file = "predict_model.pkl"
            pred = joblib.load(joblib_file)

            predictor_var = ['Gender', 'Married', 'Dependents', 'Education',
                             'Self_Employed', 'Loan_Amount_Term',
                             'Property_Area', 'LoanAmount_log','TotalIncome_log',
                            'EMI', 'Balance_Income', 'Conditions_to_pay']
        else:
            joblib_file = "predict_model2.pkl"
            pred = joblib.load(joblib_file)

            predictor_var = ['Gender', 'Married', 'Dependents', 'Education',
                             'Self_Employed', 'Loan_Amount_Term', 'Credit_History',
                             'Property_Area', 'LoanAmount_log', 'TotalIncome_log',
                             'EMI', 'Balance_Income', 'Conditions_to_pay']

        df = df[predictor_var]
        y_predict = pred.predict(df)

        if y_predict == 1:
            st.header('Seu Empréstimo Foi Aprovado!')
            img2 = Image.open('Aprovado.jpg')
            st.image(img2)
        else:
            st.header('Sorry! Seu Empréstimo Não Foi Aprovado!')
            img3 = Image.open('not-approved.jpg')
            st.image(img3)

