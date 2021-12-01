import streamlit as st
import pandas as pd
import numpy as np
import csv
from network import Network

global dataframe
dataframe = None


st.write("""# My Neural network""")
st.caption("An application of Multilayer Perceptron and backpropagation")

@st.cache
def run_mlp(data_train, data_test, n_hiddens = 1, learning=0.01, epochs=200, error=0.1, out_mode='Linear'):
    mlp = Network(epoch=epochs, error_rate=error, n_hiddens=n_hiddens, output_mode=out_mode, learning_rate=learning)
    mlp.train(data_train)
    errors = mlp.listErrors
    accuracy, confusion_mat = mlp.test(data_test)
    return confusion_mat, accuracy, errors

## Setup GUI

# Número de neuronios
linha1 = st.columns((1,1,1))
with linha1[0]:
    st.text_input("Camada de Entrada:")
with linha1[1]:
    st.text_input("Camada de Saída:")
with linha1[2]:
    hidden_layer = st.number_input("Camada de Oculta:", min_value=0, step=1)

linha3 = st.columns((1))

# Valor de aprendizado
with linha3[0]:
    learning_rate = st.slider(label="Taxa de Aprendizado", format="%.4f", min_value=0.0000, max_value=1.0000, step=0.0010)
# Valor do erro
error_rate = st.number_input("Valor do erro:", format="%.4f", min_value=0.0010, step=0.0010)


linha3 = st.columns((1,1))
with linha3[0]:
    # Número de épocas
    epochs_num = st.number_input('Número de épocas:', min_value=100, step=100)
with linha3[1]:
    # Função de transferência
    transfer_function = st.radio("Função de transferência",('Linear', 'Logistica', 'Hiperbólica'))

st.text("                                                              ")
st.text("                                                              ")
st.text("                                                              ")

linha2 = st.columns((1,1))
with linha2[0]:
    train_file = st.file_uploader("Arquivo de Treinamento", 'csv', help='Faça upload de um arquivo csv para treinar a rede neural')
with linha2[1]:
    test_file = st.file_uploader("Arquivo de Teste", 'csv', help='Faça upload de um arquivo csv para testar a rede neural')

show_file = st.empty()
if train_file is None and test_file is None:
    show_file.info("")
elif not(train_file is None) and not(test_file is None):
    ##to know type of file
    #type = from_buffer(train_file.getvalue())
    show_file.info(train_file.name)
    show_file.info(test_file.name)
    dataframe_train = pd.read_csv(train_file)
    dataframe_test = pd.read_csv(test_file)
    input_size = len(dataframe_train.columns) - 1
    tupla_ones = tuple(np.ones(input_size))
                       
    checkbox_row = st.columns(tupla_ones)
    
    columns_train = dataframe_train.columns[:input_size]
    columns_test = dataframe_test.columns[:input_size]
    checkbox = []
    i = 0
    for col in columns_train:
        with checkbox_row[i]:
            i = i + 1
            checkbox.append({"checked": st.checkbox(col, value=1, key=col, help="Selecione ou não esse atributo para usar na rede"), "attribute": col})
       
    
    for check in checkbox:
        if not(check["checked"]):
            dataframe_train = dataframe_train.drop(columns=[check['attribute']])
            dataframe_test = dataframe_test.drop(columns=[check['attribute']])
    # columns length is equal for the datasets validation
    if len(dataframe_train.columns) == len(dataframe_test.columns):
        with st.expander('Dados de Treino:'):
            st.dataframe(dataframe_train)
        with st.expander('Dados de Teste:'):
            st.dataframe(dataframe_test)
    else:
        st.text("Entrada Inválida!")
    
    if st.button('Treinar rede'):
        confusin_matrix, accuracy, errors = run_mlp(data_test=dataframe_test,
                                                    data_train=dataframe_train,
                                                    n_hiddens=hidden_layer,
                                                    learning=learning_rate,
                                                    epochs=epochs_num,
                                                    error=error_rate,
                                                    out_mode=transfer_function)
        st.line_chart(data=errors)
        st.subheader("Matriz de Confusão")
        st.dataframe(data=confusin_matrix)
        st.text('Acurácia: '+ str(accuracy))
    