import streamlit as st
import pandas as pd
import joblib
import os

st.title('Previsão de Obesidade')

# Verifica se o modelo está disponível
model_path = 'obesity_model.joblib'
if not os.path.isfile(model_path):
    st.error("Modelo 'obesity_model.joblib' não encontrado. Verifique se ele está no diretório correto.")
    st.stop()

# Carrega o modelo
model = joblib.load(model_path)

st.header('Insira seus dados para a previsão')

perguntas_amigaveis_widgets = {
    'vegetais': {"pergunta": "Com que frequência você come vegetais? (1 a 3): ", "tipo": "number_input", "min_value": 1, "max_value": 3, "step": 1},
    'ref_principais': {"pergunta": "Quantas refeições principais você faz por dia? (1 a 4): ", "tipo": "number_input", "min_value": 1, "max_value": 4, "step": 1},
    'agua': {"pergunta": "Quantos litros de água você bebe por dia? (1 a 3): ", "tipo": "number_input", "min_value": 1, "max_value": 3, "step": 1},
    'atv_fisica': {"pergunta": "Com que frequência você pratica atividade física? (0 a 3): ", "tipo": "number_input", "min_value": 0, "max_value": 3, "step": 1},
    'atv_eletronica': {"pergunta": "Com que frequência você usa dispositivos eletrônicos para lazer? (0 a 2): ", "tipo": "number_input", "min_value": 0, "max_value": 2, "step": 1},
    'idade': {"pergunta": "Qual a sua idade? (inteiro): ", "tipo": "number_input", "min_value": 0, "step": 1},
    'peso': {"pergunta": "Qual o seu peso em kg? (inteiro): ", "tipo": "number_input", "min_value": 0, "step": 1},
    'altura': {"pergunta": "Qual a sua altura em metros? (ex: 1.75): ", "tipo": "number_input", "min_value": 0.0, "step": 0.01},
    'historico': {"pergunta": "Você tem histórico familiar de obesidade? ", "tipo": "radio", "opcoes": {0: 'Não', 1: 'Sim'}},
    'al_calorico': {"pergunta": "Você consome frequentemente alimentos calóricos? ", "tipo": "radio", "opcoes": {0: 'Não', 1: 'Sim'}},
    'ctrl_caloria': {"pergunta": "Você monitora a ingestão de calorias? ", "tipo": "radio", "opcoes": {0: 'Não', 1: 'Sim'}},
    'entre_ref': {"pergunta": "Você come entre as refeições principais? ", "tipo": "radio", "opcoes": {0: 'Não', 1: 'Sim'}},
    'fumante': {"pergunta": "Você é fumante? ", "tipo": "radio", "opcoes": {0: 'Não', 1: 'Sim'}},
    'alcool': {"pergunta": "Você consome álcool? ", "tipo": "radio", "opcoes": {0: 'Não', 1: 'Sim'}},
    'transporte': {"pergunta": "Seu meio de transporte principal envolve caminhada ou bicicleta? ", "tipo": "radio", "opcoes": {1: 'Sim', 0: 'Não'}},
    'feminino': {"pergunta": "Seu gênero é feminino? ", "tipo": "radio", "opcoes": {1: 'Sim', 0: 'Não'}},
    'masculino': {"pergunta": "Seu gênero é masculino? ", "tipo": "radio", "opcoes": {1: 'Sim', 0: 'Não'}}
}

dados_entrada = {}

# Obtém a lista de colunas esperadas pelo modelo
try:
    colunas_features = model.feature_names_in_.tolist()
except AttributeError:
    colunas_features = list(perguntas_amigaveis_widgets.keys())  # fallback

for coluna in colunas_features:
    if coluna in perguntas_amigaveis_widgets:
        widget_info = perguntas_amigaveis_widgets[coluna]
        pergunta = widget_info["pergunta"]
        tipo_widget = widget_info["tipo"]

        if tipo_widget == "number_input":
            step = widget_info.get("step", 1)
            min_value = widget_info.get("min_value")
            max_value = widget_info.get("max_value")
            dados_entrada[coluna] = st.number_input(pergunta, min_value=min_value, max_value=max_value, step=step, key=coluna)
        elif tipo_widget == "radio":
            opcoes_dict = widget_info["opcoes"]
            opcoes_labels = list(opcoes_dict.values())
            selected_label = st.radio(pergunta, opcoes_labels, key=coluna)
            dados_entrada[coluna] = [k for k, v in opcoes_dict.items() if v == selected_label][0]
    else:
        st.warning(f"Widget não definido para a coluna: {coluna}")
        dados_entrada[coluna] = st.text_input(f"Insira o valor para '{coluna}': ", key=coluna)

# Cria o DataFrame com os dados de entrada
novo_dado_df = pd.DataFrame([dados_entrada])
novo_dado_df = novo_dado_df[colunas_features]  # Garante ordem correta

if st.button('Prever Obesidade'):
    try:
        previsao = model.predict(novo_dado_df)
        previsao_proba = model.predict_proba(novo_dado_df)[:, 1]
        st.subheader('Resultado da Previsão:')
        if previsao[0] == 1:
            st.success("A previsão é: **Obeso**")
        else:
            st.success("A previsão é: **Não Obeso**")
        st.info(f"Probabilidade de ser Obeso: **{previsao_proba[0]:.2f}**")
    except Exception as e:
        st.error(f"Erro ao realizar a previsão: {e}")