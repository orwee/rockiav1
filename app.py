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

# Estilos CSS para el chat
st.markdown("""
<style>
.chat-message {
    padding: 1.5rem; border-radius: 0.5rem; margin-bottom: 1rem; display: flex;
    flex-direction: row; align-items: center; gap: 0.75rem;
}
.chat-message.user {
    background-color: #2b313e;
}
.chat-message.assistant {
    background-color: #475063;
}
.chat-message .avatar {
    width: 2.5rem; height: 2.5rem; border-radius: 0.5rem; display: flex;
    align-items: center; justify-content: center; font-size: 1.5rem;
}
.chat-message .avatar.user {
    background-color: #19C37D;
}
.chat-message .avatar.assistant {
    background-color: #9013FE;
}
.chat-message .message {
    flex-grow: 1; padding-left: 0.5rem;
}
</style>
""", unsafe_allow_html=True)

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

# Inicializar estados de sesión
if 'messages' not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Hola 👋 Soy tu asistente de análisis de portafolio cripto. ¿En qué puedo ayudarte hoy?"}
    ]

if 'processing' not in st.session_state:
    st.session_state.processing = False

if 'openai_api_key' not in st.session_state:
    st.session_state.openai_api_key = None

# Cargar datos
df = load_portfolio_data()
df['category'] = df['token'].apply(classify_token)

# Función para mostrar mensajes de chat
def display_chat_message(role, content):
    if role == "user":
        st.markdown(f"""
        <div class="chat-message user">
            <div class="avatar user">👤</div>
            <div class="message">{content}</div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class="chat-message assistant">
            <div class="avatar assistant">🤖</div>
            <div class="message">{content}</div>
        </div>
        """, unsafe_allow_html=True)

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

# Función para visualización
def plot_portfolio(group_by=None, measure='usd', agg_func='sum', chart_type='bar',
                 sort_values=True, ascending=False, title=None, figsize=(10, 5)):
    fig, ax = plt.subplots(figsize=figsize)

    if group_by is None:
        data = df
        x = df.columns[0]
        y = measure
    else:
        if agg_func == 'sum':
            data = df.groupby(group_by)[measure].sum()
        elif agg_func == 'mean':
            data = df.groupby(group_by)[measure].mean()
        elif agg_func == 'count':
            data = df.groupby(group_by)[measure].count()
        else:
            data = getattr(df.groupby(group_by)[measure], agg_func)()

        if sort_values:
            data = data.sort_values(ascending=ascending)

    if title is None:
        agg_name = {'sum': 'Suma', 'mean': 'Promedio', 'count': 'Conteo'}.get(agg_func, agg_func)
        measure_name = measure.upper()
        group_name = group_by.capitalize() if group_by else "Sin agrupar"
        title = f"{agg_name} de {measure_name} por {group_name}"

    if chart_type == 'pie':
        data.plot(kind='pie', autopct='%1.1f%%', title=title, ax=ax)
        ax.axis('equal')
    elif chart_type == 'bar':
        if group_by is None:
            sns.barplot(x=x, y=y, data=data, ax=ax)
            ax.set_title(title)
        else:
            data.plot(kind='bar', title=title, ax=ax)
            ax.set_xlabel(group_by)
            ax.set_ylabel(measure)
    else:
        data.plot(kind=chart_type, title=title, ax=ax)

    ax.tick_params(axis='x', rotation=45)
    plt.tight_layout()

    return fig, data

# Función para dashboard
def plot_portfolio_dashboard():
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()

    # Gráfico 1: Pie chart de wallets
    data = df.groupby('wallet')['usd'].sum()
    data.plot(kind='pie', autopct='%1.1f%%', title="Distribución por Wallet", ax=axes[0])
    axes[0].axis('equal')

    # Gráfico 2: Bar chart de blockchains
    data = df.groupby('chain')['usd'].sum().sort_values(ascending=False)
    data.plot(kind='bar', title="USD por Blockchain", ax=axes[1])
    axes[1].set_xlabel('chain')
    axes[1].set_ylabel('usd')

    # Gráfico 3: Pie chart de categorías
    data = df.groupby('category')['usd'].sum()
    data.plot(kind='pie', autopct='%1.1f%%', title="Distribución por Categoría", ax=axes[2])
    axes[2].axis('equal')

    # Gráfico 4: Bar chart de protocolos
    data = df.groupby('protocol')['usd'].sum().sort_values(ascending=False)
    data.plot(kind='bar', title="USD por Protocolo", ax=axes[3])
    axes[3].set_xlabel('protocol')
    axes[3].set_ylabel('usd')

    for ax in axes:
        ax.tick_params(axis='x', rotation=45)

    plt.tight_layout()
    return fig

# Función para procesar la consulta del usuario
def process_query(query, agent):
    query_lower = query.lower()

    # Términos para detectar consultas de visualización
    viz_terms = ["gráfico", "grafico", "visualiza", "visualizar", "mostrar",
                "ver", "distribución", "distribucion", "dashboard"]

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

    # Si es consulta de visualización
    if is_viz_query:
        response_text = ""
        viz_container = st.container()

        with viz_container:
            if group_by:
                # Si se especifica una variable, mostrar gráficos específicos
                col1, col2 = st.columns(2)

                with col1:
                    st.subheader(f"Distribución por {group_by.capitalize()}")
                    fig1, data = plot_portfolio(group_by=group_by, chart_type="bar")
                    st.pyplot(fig1)

                with col2:
                    st.subheader(f"Proporción por {group_by.capitalize()}")
                    fig2, _ = plot_portfolio(group_by=group_by, chart_type="pie")
                    st.pyplot(fig2)

                # Datos en tabla
                if isinstance(data, pd.Series):
                    total = data.sum()
                    data_df = pd.DataFrame({
                        group_by.capitalize(): data.index,
                        "USD": data.values.round(2),
                        "Porcentaje (%)": [(v/total*100).round(2) for v in data.values]
                    })
                    st.dataframe(data_df, use_container_width=True, hide_index=True)

                response_text = f"Aquí tienes las visualizaciones de la distribución por {group_by}."
            else:
                # Si no se especifica, mostrar dashboard general
                st.subheader("Dashboard General del Portafolio")
                fig = plot_portfolio_dashboard()
                st.pyplot(fig)

                # Total del portafolio
                total_value = df['usd'].sum()
                st.metric("Valor Total del Portafolio", f"${total_value:.2f}")

                response_text = "Aquí tienes un dashboard general de tu portafolio."

        return response_text
    else:
        # Para consultas no visuales, usar el agente
        try:
            response = agent.run(query)
            return response
        except Exception as e:
            return f"Error al procesar tu consulta: {str(e)}"

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

    st.subheader("Consultas Rápidas")

    # Botones de acciones rápidas
    if st.button("📊 Distribución por Wallet"):
        st.session_state.messages.append({"role": "user", "content": "Muestra la distribución por wallet"})
        st.session_state.processing = True

    if st.button("🔗 Análisis por Blockchain"):
        st.session_state.messages.append({"role": "user", "content": "Visualiza mi exposición por blockchain"})
        st.session_state.processing = True

    if st.button("💰 Categorías de Token"):
        st.session_state.messages.append({"role": "user", "content": "Distribución por categorías de token"})
        st.session_state.processing = True

    if st.button("💸 Valor Total"):
        st.session_state.messages.append({"role": "user", "content": "¿Cuál es el valor total de mi portafolio?"})
        st.session_state.processing = True

    st.markdown("---")
    st.caption("Este asistente analiza tu portafolio de criptomonedas y genera visualizaciones automáticamente.")

# Mostrar historial de mensajes
chat_container = st.container()
with chat_container:
    for message in st.session_state.messages:
        display_chat_message(message["role"], message["content"])

# Configurar el agente
agent = setup_agent(df)

# Area para entrada de texto
col1, col2 = st.columns([6, 1])
with col1:
    user_input = st.text_input(
        "Escribe tu mensaje:",
        placeholder="Ej: Muestra la distribución por wallet o ¿Cuánto tengo invertido en stablecoins?",
        key="user_input",
        label_visibility="collapsed"
    )
with col2:
    send_button = st.button("Enviar")

# Manejar entrada del usuario
if send_button and user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})
    st.session_state.processing = True
    st.experimental_rerun()

# Procesar la respuesta si está pendiente
if st.session_state.processing:
    if not agent and st.session_state.openai_api_key:
        agent = setup_agent(df)

    if agent:
        # Procesar la última consulta
        last_user_message = next((m["content"] for m in reversed(st.session_state.messages)
                               if m["role"] == "user"), None)

        if last_user_message:
            with st.spinner("Procesando..."):
                response = process_query(last_user_message, agent)
                st.session_state.messages.append({"role": "assistant", "content": response})
    else:
        st.session_state.messages.append({
            "role": "assistant",
            "content": "Por favor, configura una API key válida para usar el asistente."
        })

    # Marcar como procesado para evitar bucles
    st.session_state.processing = False
    st.experimental_rerun()
