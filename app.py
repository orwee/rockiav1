import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from langchain_openai import ChatOpenAI
from langchain_experimental.agents import create_pandas_dataframe_agent
from langchain.agents.agent_types import AgentType

# Configuración de la página
st.set_page_config(
    page_title="Asistente de Portafolio Cripto",
    page_icon="🤖",
    layout="wide"
)

# Inicializar estados de sesión
if 'messages' not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Hola 👋 Soy tu asistente de análisis de portafolio cripto. ¿En qué puedo ayudarte hoy?"}
    ]

if 'openai_api_key' not in st.session_state:
    st.session_state.openai_api_key = None

if 'show_visualization' not in st.session_state:
    st.session_state.show_visualization = {
        'show': False,
        'type': None,      # puede ser 'dashboard', 'bar', 'pie'
        'group_by': None,  # wallet, chain, etc.
        'data': None       # para almacenar DataFrame para tablas
    }

# Cargar datos del portafolio
@st.cache_data
def load_portfolio_data():
    portfolio_data = [
        {
            "id": 1,
            "wallet": "Wallet #1",
            "chain": "base",
            "protocol": "Uniswap V3",
            "token": "ODOS",
            "usd": 21.91
        },
        {
            "id": 2,
            "wallet": "Wallet #1",
            "chain": "mantle",
            "protocol": "Pendle V2",
            "token": "ETH/cmETH",
            "usd": 554.81
        },
        {
            "id": 3,
            "wallet": "Wallet #2",
            "chain": "base",
            "protocol": "aave",
            "token": "USDT",
            "usd": 2191
        },
        {
            "id": 4,
            "wallet": "Wallet #3",
            "chain": "solana",
            "protocol": "meteora",
            "token": "JLP/SOL",
            "usd": 551
        },
    ]
    return pd.DataFrame(portfolio_data)

# Cargar datos
df = load_portfolio_data()

# Clasificar tokens
def classify_token(token):
    stablecoins = ['USDT', 'USDC', 'DAI', 'BUSD']
    bluechips = ['ETH', 'BTC', 'SOL']

    for stable in stablecoins:
        if stable in token:
            return 'Stablecoin'

    for blue in bluechips:
        if blue in token:
            return 'Bluechip'

    return 'Altcoin'

df['category'] = df['token'].apply(classify_token)

# Configurar el agente de LangChain
@st.cache_resource
def setup_agent(_df):
    try:
        api_key = st.secrets["openai"]["api_key"]
    except:
        api_key = st.session_state.get('openai_api_key', None)

    if not api_key:
        return None

    try:
        llm = ChatOpenAI(temperature=0, model="gpt-4o-mini", api_key=api_key)
        agent = create_pandas_dataframe_agent(
            llm,
            _df,
            verbose=True,
            agent_type=AgentType.OPENAI_FUNCTIONS,
            handle_parsing_errors=True,
            allow_dangerous_code=True
        )
        return agent
    except Exception as e:
        st.error(f"Error al configurar el agente: {e}")
        return None

# Crear agente
agent = setup_agent(df)

# Título principal
st.title("💬 Asistente de Portafolio Cripto")

# Sidebar para configuración
with st.sidebar:
    st.header("⚙️ Configuración")

    # API key
    api_key_input = st.text_input(
        "OpenAI API Key",
        type="password",
        placeholder="sk-..."
    )

    if api_key_input:
        st.session_state.openai_api_key = api_key_input
        agent = setup_agent(df)

    st.subheader("Consultas Rápidas")

    # Botones de acciones rápidas
    if st.button("📊 Distribución por Wallet"):
        st.session_state.messages.append({"role": "user", "content": "Muestra la distribución por wallet"})
        st.session_state.show_visualization = {
            'show': True,
            'type': 'specific',
            'group_by': 'wallet'
        }

    if st.button("🔗 Análisis por Blockchain"):
        st.session_state.messages.append({"role": "user", "content": "Visualiza mi exposición por blockchain"})
        st.session_state.show_visualization = {
            'show': True,
            'type': 'specific',
            'group_by': 'chain'
        }

    if st.button("💰 Categorías de Token"):
        st.session_state.messages.append({"role": "user", "content": "Distribución por categorías de token"})
        st.session_state.show_visualization = {
            'show': True,
            'type': 'specific',
            'group_by': 'category'
        }

    if st.button("🔄 Dashboard Completo"):
        st.session_state.messages.append({"role": "user", "content": "Muestra un dashboard completo"})
        st.session_state.show_visualization = {
            'show': True,
            'type': 'dashboard'
        }

    if st.button("💸 Valor Total"):
        st.session_state.messages.append({"role": "user", "content": "¿Cuál es el valor total de mi portafolio?"})
        # No mostrar visualización para esta consulta

    st.markdown("---")
    st.caption("Este asistente analiza tu portafolio de criptomonedas y genera visualizaciones.")

# Mostrar chat
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

# Área de visualización (si está activada)
if st.session_state.show_visualization['show']:
    viz_type = st.session_state.show_visualization['type']
    group_by = st.session_state.show_visualization['group_by']

    viz_container = st.container()

    with viz_container:
        if viz_type == 'specific' and group_by:
            st.subheader(f"Visualización por {group_by.capitalize()}")

            # Agregar datos
            grouped_data = df.groupby(group_by)['usd'].sum()
            total = grouped_data.sum()

            # Gráficos
            col1, col2 = st.columns(2)

            with col1:
                fig, ax = plt.subplots(figsize=(8, 5))
                grouped_data.plot(kind='bar', ax=ax)
                ax.set_title(f"USD por {group_by.capitalize()}")
                ax.set_xlabel(group_by)
                ax.set_ylabel("USD")
                st.pyplot(fig)

            with col2:
                fig, ax = plt.subplots(figsize=(8, 5))
                grouped_data.plot(kind='pie', autopct='%1.1f%%', ax=ax)
                ax.set_title(f"Distribución por {group_by.capitalize()}")
                ax.axis('equal')
                st.pyplot(fig)

            # Tabla de datos
            data_df = pd.DataFrame({
                group_by.capitalize(): grouped_data.index,
                "USD": grouped_data.values.round(2),
                "Porcentaje (%)": [(v/total*100).round(2) for v in grouped_data.values]
            })
            st.dataframe(data_df, hide_index=True)

        elif viz_type == 'dashboard':
            st.subheader("Dashboard del Portafolio")

            # Primera fila: métricas generales
            total_value = df['usd'].sum()
            avg_value = df['usd'].mean()
            unique_chains = df['chain'].nunique()

            col1, col2, col3 = st.columns(3)
            col1.metric("Valor Total", f"${total_value:.2f}")
            col2.metric("Promedio por Inversión", f"${avg_value:.2f}")
            col3.metric("Blockchains", f"{unique_chains}")

            # Segunda fila: visualizaciones principales
            col1, col2 = st.columns(2)

            with col1:
                # Gráfico de Wallet
                fig, ax = plt.subplots(figsize=(8, 5))
                wallet_data = df.groupby('wallet')['usd'].sum()
                wallet_data.plot(kind='pie', autopct='%1.1f%%', ax=ax)
                ax.set_title("Distribución por Wallet")
                ax.axis('equal')
                st.pyplot(fig)

            with col2:
                # Gráfico de Chain
                fig, ax = plt.subplots(figsize=(8, 5))
                chain_data = df.groupby('chain')['usd'].sum().sort_values(ascending=False)
                chain_data.plot(kind='bar', ax=ax)
                ax.set_title("USD por Blockchain")
                ax.set_xlabel("Chain")
                ax.set_ylabel("USD")
                st.pyplot(fig)

            # Tercera fila
            col1, col2 = st.columns(2)

            with col1:
                # Gráfico de Categoría
                fig, ax = plt.subplots(figsize=(8, 5))
                cat_data = df.groupby('category')['usd'].sum()
                cat_data.plot(kind='pie', autopct='%1.1f%%', ax=ax)
                ax.set_title("Distribución por Categoría")
                ax.axis('equal')
                st.pyplot(fig)

            with col2:
                # Gráfico de Protocolo
                fig, ax = plt.subplots(figsize=(8, 5))
                protocol_data = df.groupby('protocol')['usd'].sum().sort_values(ascending=False)
                protocol_data.plot(kind='bar', ax=ax)
                ax.set_title("USD por Protocolo")
                ax.set_xlabel("Protocolo")
                ax.set_ylabel("USD")
                st.pyplot(fig)

# Entrada de usuario
prompt = st.chat_input("Escribe tu consulta...")

if prompt:
    # Añadir mensaje del usuario
    st.session_state.messages.append({"role": "user", "content": prompt})

    # Analizar la consulta para determinar si es de visualización
    query_lower = prompt.lower()
    viz_terms = ["gráfico", "grafico", "visualiza", "visualizar", "mostrar", "ver", "distribución", "distribucion", "dashboard"]
    is_viz_query = any(term in query_lower for term in viz_terms)

    # Variables de agrupación
    group_vars = {
        "wallet": "wallet", "billetera": "wallet",
        "blockchain": "chain", "chain": "chain", "cadena": "chain",
        "categoria": "category", "categoría": "category", "tipo de token": "category",
        "protocolo": "protocol", "protocol": "protocol",
        "token": "token"
    }

    group_by = None
    for term, var in group_vars.items():
        if term in query_lower:
            group_by = var
            break

    # Decidir si mostrar visualización
    if is_viz_query:
        if group_by:
            st.session_state.show_visualization = {
                'show': True,
                'type': 'specific',
                'group_by': group_by
            }
            asst_response = f"Aquí tienes la visualización de la distribución por {group_by}:"
        else:
            st.session_state.show_visualization = {
                'show': True,
                'type': 'dashboard'
            }
            asst_response = "Aquí tienes un dashboard con diferentes visualizaciones de tu portafolio:"
    else:
        # No mostrar visualización, usar el agente para responder
        st.session_state.show_visualization['show'] = False

        if agent:
            try:
                asst_response = agent.run(prompt)
            except Exception as e:
                asst_response = f"Error al procesar tu consulta: {str(e)}"
        else:
            asst_response = "No puedo responder sin una API key válida. Por favor, configura la API key en la barra lateral."

    # Añadir respuesta del asistente
    st.session_state.messages.append({"role": "assistant", "content": asst_response})

    # Recargar para mostrar la respuesta completa
    st.rerun()
