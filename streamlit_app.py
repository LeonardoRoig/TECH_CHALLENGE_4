import streamlit as st
import joblib
import pandas as pd

st.title('Previsão de Obesidade')

# Load the trained model
model = joblib.load('obesity_model.joblib')

st.header('Insira seus dados para a previsão')

# Mapeamento de nomes de colunas para perguntas mais amigáveis e tipos de widget
perguntas_amigaveis_widgets = {
    'vegetais': {"pergunta": "Com que frequência você come vegetais? (1 a 3): ", "tipo": "number_input", "min_value": 1, "max_value": 3, "step": 1},
    'ref_principais': {"pergunta": "Quantas refeições principais você faz por dia? (1 a 4): ", "tipo": "number_input", "min_value": 1, "max_value": 4, "step": 1},
    'agua': {"pergunta": "Quantos litros de água você bebe por dia? (1 a 3): ", "tipo": "number_input", "min_value": 1, "max_value": 3, "step": 1},
    'atv_fisica': {"pergunta": "Com que frequência você pratica atividade física? (0 a 3): ", "tipo": "number_input", "min_value": 0, "max_value": 3, "step": 1},
    'atv_eletronica': {"pergunta": "Com que frequência você usa dispositivos eletrônicos para lazer? (0 a 2): ", "tipo": "number_input", "min_value": 0, "max_value": 2, "step": 1},
    'idade': {"pergunta": "Qual a sua idade? (inteiro): ", "tipo": "number_input", "min_value": 0, "step": 1},
    'peso': {"pergunta": "Qual o seu peso em kg? (inteiro): ", "tipo": "number_input", "min_value": 0, "step": 1},
    'altura': {"pergunta": "Qual a sua altura em metros? (ex: 1.75): ", "tipo": "number_input", "min_value": 0.0, "format": "%.2f"},
    'historico': {"pergunta": "Você tem histórico familiar de obesidade? ", "tipo": "radio", "opcoes": {0: 'Não', 1: 'Sim'}},
    'al_calorico': {"pergunta": "Você consome frequentemente alimentos calóricos? ", "tipo": "radio", "opcoes": {0: 'Não', 1: 'Sim'}},
    'ctrl_caloria': {"pergunta": "Você monitora a ingestão de calorias? ", "tipo": "radio", "opcoes": {0: 'Não', 1: 'Sim'}},
    'entre_ref': {"pergunta": "Você come entre as refeições principais? ", "tipo": "radio", "opcoes": {0: 'Não', 1: 'Sim'}},
    'fumante': {"pergunta": "Você é fumante? ", "tipo": "radio", "opcoes": {0: 'Não', 1: 'Sim'}},
    'alcool': {"pergunta": "Você consome álcool? ", "tipo": "radio", "opcoes": {0: 'Não', 1: 'Sim'}},
    'transporte': {"pergunta": "Seu meio de transporte principal envolve caminhada ou bicicleta? ", "tipo": "radio", "opcoes": {0: 'Sim', 1: 'Não'}},
    'feminino': {"pergunta": "Seu gênero é feminino? ", "tipo": "radio", "opcoes": {0: 'Não', 1: 'Sim'}},
    'masculino': {"pergunta": "Seu gênero é masculino? ", "tipo": "radio", "opcoes": {0: 'Não', 1: 'Sim'}}
}

dados_entrada = {}

# Use X_train from the notebook to get the correct column order
# In a real app, you'd save and load the column list or infer it from the model
# For this context, assuming X_train is available from the notebook state
# If X_train is not available, you would need to load the column names from a saved file or infer from the model if possible.
# Let's assume X_train.columns is accessible for this step.
# In a standalone app, you would load the column names, e.g., from a file saved alongside the model.
# For this exercise, we'll simulate accessing the column names.
try:
    # This assumes X_train is in the notebook's global scope
    colunas_features = X_train.columns.tolist()
except NameError:
    # Fallback: If X_train is not available, use a predefined list or load from a file
    # In a real app, you MUST have a reliable way to get the feature names and order
    # This is a placeholder for demonstration within the notebook's context
    colunas_features = list(perguntas_amigaveis_widgets.keys()) # Fallback to dictionary keys order

for coluna in colunas_features:
    if coluna in perguntas_amigaveis_widgets:
        widget_info = perguntas_amigaveis_widgets[coluna]
        pergunta = widget_info["pergunta"]
        tipo_widget = widget_info["tipo"]

        if tipo_widget == "number_input":
            min_value = widget_info.get("min_value")
            max_value = widget_info.get("max_value")
            step = widget_info.get("step")
            format_str = widget_info.get("format")
            dados_entrada[coluna] = st.number_input(pergunta, min_value=min_value, max_value=max_value, step=step, format=format_str, key=coluna)
        elif tipo_widget == "radio":
            opcoes = list(widget_info["opcoes"].keys()) # Use keys (0, 1) as internal values
            opcoes_labels = list(widget_info["opcoes"].values()) # Use values ('Não', 'Sim') as labels
            # Streamlit radio returns the selected label, so we need to map it back to the key (0 or 1)
            selected_label = st.radio(pergunta, opcoes_labels, key=coluna)
            # Find the key corresponding to the selected label
            dados_entrada[coluna] = opcoes[opcoes_labels.index(selected_label)]
    else:
        # Handle columns not in the friendly questions map, if any
        st.warning(f"Widget não definido para a coluna: {coluna}")
        # Add a generic text input as a fallback, though ideally all columns should be mapped
        dados_entrada[coluna] = st.text_input(f"Insira o valor para '{coluna}': ", key=coluna)

# Format the input data into a DataFrame
# Ensure the order of columns matches the training data
novo_dado_df = pd.DataFrame([dados_entrada])
novo_dado_df = novo_dado_df[colunas_features]

# Add a button to trigger the prediction
if st.button('Prever Obesidade'):
    # Make the prediction
    previsao = model.predict(novo_dado_df)
    previsao_proba = model.predict_proba(novo_dado_df)[:, 1] # Get probability of the positive class (1)

    # Display the prediction result
    st.subheader('Resultado da Previsão:')
    if previsao[0] == 1:
        st.write(f"A previsão é: **Obeso**")
    else:
        st.write(f"A previsão é: **Não Obeso**")

    st.write(f"Probabilidade de ser Obeso: **{previsao_proba[0]:.2f}**")
