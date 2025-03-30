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

# Estilos CSS personalizados para el chat
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

# Clasificar tokens en categorías
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

# Función para visualización de portafolio
def plot_portfolio(
    group_by=None,         # Variable para agrupar (wallet, chain, category, etc.)
    measure='usd',         # Variable a medir (generalmente usd)
    agg_func='sum',        # Función de agregación (sum, mean, count, etc.)
    chart_type='bar',      # Tipo de gráfico (bar, pie, line, etc.)
    sort_values=True,      # Ordenar valores
    ascending=False,       # Orden ascendente o descendente
    title=None,            # Título personalizado
    figsize=(10, 5),       # Tamaño del gráfico
    **kwargs               # Parámetros adicionales para personalización
):
    """Función genérica para visualizar datos del portafolio"""
    fig, ax = plt.subplots(figsize=figsize)

    # Si no se especifica grupo, mostrar datos sin agrupar
    if group_by is None:
        data = df
        x = kwargs.get('x', df.columns[0])
        y = measure
    else:
        # Agrupar y agregar datos
        if agg_func == 'sum':
            data = df.groupby(group_by)[measure].sum()
        elif agg_func == 'mean':
            data = df.groupby(group_by)[measure].mean()
        elif agg_func == 'count':
            data = df.groupby(group_by)[measure].count()
        else:
            data = getattr(df.groupby(group_by)[measure], agg_func)()

        # Ordenar valores si se solicita
        if sort_values:
            data = data.sort_values(ascending=ascending)

    # Configurar título automático si no se proporciona
    if title is None:
        agg_name = {'sum': 'Suma', 'mean': 'Promedio', 'count': 'Conteo'}.get(agg_func, agg_func)
        measure_name = measure.upper()
        group_name = group_by.capitalize() if group_by else "Sin agrupar"
        title = f"{agg_name} de {measure_name} por {group_name}"

    # Generar el gráfico según el tipo solicitado
    if chart_type == 'pie':
        data.plot(kind='pie', autopct='%1.1f%%', title=title, ax=ax, **kwargs)
        ax.axis('equal')
    elif chart_type == 'bar':
        if group_by is None:
            sns.barplot(x=x, y=y, data=data, ax=ax, **kwargs)
            ax.set_title(title)
        else:
            data.plot(kind='bar', title=title, ax=ax, **kwargs)
            ax.set_xlabel(group_by)
            ax.set_ylabel(measure)
    elif chart_type == 'line':
        data.plot(kind='line', title=title, ax=ax, **kwargs)
        ax.set_xlabel(group_by)
        ax.set_ylabel(measure)
    else:
        # Cualquier otro tipo soportado por pandas
        data.plot(kind=chart_type, title=title, ax=ax, **kwargs)

    ax.tick_params(axis='x', rotation=kwargs.get('rotation', 45))
    plt.tight_layout()

    return fig, data

# Función para dashboard con múltiples gráficos
def plot_portfolio_dashboard(plots_config, figsize=(12, 8), grid=(2, 2)):
    """Genera un dashboard con múltiples visualizaciones"""
    fig, axes = plt.subplots(grid[0], grid[1], figsize=figsize)
    axes = axes.flatten() if hasattr(axes, 'flatten') else [axes]

    for i, config in enumerate(plots_config):
        position = config.pop("position", i+1) - 1
        if position < len(axes):
            plt.sca(axes[position])

            # Extraer parámetros
            group_by = config.pop("group_by", None)
            measure = config.pop("measure", "usd")
            agg_func = config.pop("agg_func", "sum")
            chart_type = config.pop("chart_type", "bar")
            title = config.pop("title", None)

            # Generar gráfico en el subplot actual
            if group_by is None:
                data = df
                x = config.get('x', df.columns[0])
                y = measure
            else:
                # Agrupar y agregar datos
                if agg_func == 'sum':
                    data = df.groupby(group_by)[measure].sum()
                elif agg_func == 'mean':
                    data = df.groupby(group_by)[measure].mean()
                elif agg_func == 'count':
                    data = df.groupby(group_by)[measure].count()
                else:
                    data = getattr(df.groupby(group_by)[measure], agg_func)()

                # Ordenar valores
                if config.get('sort_values', True):
                    data = data.sort_values(ascending=config.get('ascending', False))

            # Configurar título
            if title is None:
                agg_name = {'sum': 'Suma', 'mean': 'Promedio', 'count': 'Conteo'}.get(agg_func, agg_func)
                measure_name = measure.upper()
                group_name = group_by.capitalize() if group_by else "Sin agrupar"
                title = f"{agg_name} de {measure_name} por {group_name}"

            # Generar gráfico
            if chart_type == 'pie':
                data.plot(kind='pie', autopct='%1.1f%%', title=title, ax=axes[position], **config)
                axes[position].axis('equal')
            elif chart_type == 'bar':
                if group_by is None:
                    sns.barplot(x=x, y=y, data=data, ax=axes[position], **config)
                    axes[position].set_title(title)
                else:
                    data.plot(kind='bar', title=title, ax=axes[position], **config)
                    axes[position].set_xlabel(group_by)
                    axes[position].set_ylabel(measure)
            else:
                data.plot(kind=chart_type, title=title, ax=axes[position], **config)

            axes[position].tick_params(axis='x', rotation=config.get('rotation', 45))

    plt.tight_layout()
    return fig

# Configurar el agente de LangChain
@st.cache_resource
def setup_agent(_df):
    # Obtener API key
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

# Función para mostrar mensajes de chat
def display_chat_message(role, content, avatar=None):
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

# Analizar la consulta y determinar qué visualizaciones mostrar
def analyze_query(query, agent):
    query_lower = query.lower()

    # Verificar si la consulta pide una visualización
    visualization_terms = [
        "gráfico", "grafico", "visualiza", "visualizar", "mostrar",
        "ver", "distribución", "distribucion", "dashboard"
    ]

    is_visualization_query = any(term in query_lower for term in visualization_terms)

    # Verificar si menciona alguna variable para agrupar
    group_vars = {
        "wallet": "wallet",
        "blockchain": "chain",
        "chain": "chain",
        "cadena": "chain",
        "categoria": "category",
        "categoría": "category",
        "tipo de token": "category",
        "protocolo": "protocol",
        "protocol": "protocol",
        "token": "token"
    }

    group_by = None
    for term, var in group_vars.items():
        if term in query_lower:
            group_by = var
            break

    # Verificar tipos de agregación mencionados
    agg_funcs = {
        "promedio": "mean",
        "media": "mean",
        "mean": "mean",
        "avg": "mean",
        "total": "sum",
        "suma": "sum",
        "sum": "sum",
        "contar": "count",
        "count": "count",
        "cantidad": "count"
    }

    agg_func = "sum"  # Predeterminado
    for term, func in agg_funcs.items():
        if term in query_lower:
            agg_func = func
            break

    # Determinar respuesta
    if is_visualization_query:
        return {
            "type": "visualization",
            "group_by": group_by,
            "agg_func": agg_func
        }
    else:
        # Usar el agente para responder consultas generales
        try:
            response = agent.run(query)
            return {
                "type": "text",
                "content": response
            }
        except Exception as e:
            return {
                "type": "error",
                "content": f"Error al procesar la consulta: {str(e)}"
            }

# Generar una visualización basada en el análisis de la consulta
def generate_visualization(analysis):
    group_by = analysis.get("group_by")
    agg_func = analysis.get("agg_func", "sum")

    if group_by:
        # Si se identifica una variable de agrupación, mostrar múltiples visualizaciones
        col1, col2 = st.columns(2)

        with col1:
            st.subheader(f"Gráfico de Barras - {group_by.capitalize()}")
            fig1, data = plot_portfolio(
                group_by=group_by,
                chart_type="bar",
                agg_func=agg_func,
                title=f"{agg_func.capitalize()} de USD por {group_by}"
            )
            st.pyplot(fig1)

        with col2:
            st.subheader(f"Gráfico Circular - {group_by.capitalize()}")
            fig2, _ = plot_portfolio(
                group_by=group_by,
                chart_type="pie",
                agg_func=agg_func,
                title=f"Distribución por {group_by}"
            )
            st.pyplot(fig2)

        # Mostrar datos en formato de tabla
        if isinstance(data, pd.Series):
            total = data.sum()
            data_df = pd.DataFrame({
                group_by.capitalize(): data.index,
                "USD": data.values.round(2),
                "Porcentaje": [(v/total*100).round(2) for v in data.values]
            })
            st.dataframe(data_df, use_container_width=True, hide_index=True)

        return f"He generado visualizaciones para {group_by} usando {agg_func} como método de agregación."
    else:
        # Si no se específica agrupación, mostrar dashboard completo
        st.subheader("Dashboard del Portafolio")
        plots_config = [
            {"group_by": "wallet", "chart_type": "pie", "position": 1, "title": "Distribución por Wallet"},
            {"group_by": "chain", "chart_type": "bar", "position": 2, "title": "USD por Blockchain"},
            {"group_by": "category", "chart_type": "pie", "position": 3, "title": "Distribución por Categoría"},
            {"group_by": "protocol", "chart_type": "bar", "position": 4, "title": "USD por Protocolo"}
        ]
        fig = plot_portfolio_dashboard(plots_config)
        st.pyplot(fig)

        return "He generado un dashboard completo con las principales visualizaciones de tu portafolio."

# Inicializar el estado de la sesión
if 'messages' not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Hola 👋 Soy tu asistente de análisis de portafolio cripto. ¿En qué puedo ayudarte hoy?"}
    ]

if 'openai_api_key' not in st.session_state:
    st.session_state.openai_api_key = None

# Cargar los datos
df = load_portfolio_data()
df['category'] = df['token'].apply(classify_token)

# Título principal
st.title("💬 Asistente de Portafolio Cripto")

# Sidebar para la configuración
with st.sidebar:
    st.header("⚙️ Configuración")

    # API key
    api_key_input = st.text_input(
        "OpenAI API Key",
        type="password",
        placeholder="sk-...",
        help="Introduce tu API key de OpenAI para activar el asistente"
    )

    if api_key_input:
        st.session_state.openai_api_key = api_key_input

    # Botones de acción rápida
    st.subheader("Acciones Rápidas")
    quick_actions = st.container()

    with quick_actions:
        if st.button("📊 Distribución por Wallet"):
            new_msg = "Muestra la distribución por wallet"
            st.session_state.messages.append({"role": "user", "content": new_msg})

        if st.button("🔗 Análisis por Blockchain"):
            new_msg = "Visualiza mi exposición por blockchain"
            st.session_state.messages.append({"role": "user", "content": new_msg})

        if st.button("💰 Categorías de Token"):
            new_msg = "Distribución por categorías de token"
            st.session_state.messages.append({"role": "user", "content": new_msg})

        if st.button("💸 Valor Total"):
            new_msg = "¿Cuál es el valor total de mi portafolio?"
            st.session_state.messages.append({"role": "user", "content": new_msg})

    st.markdown("---")
    st.caption("Este asistente puede responder consultas sobre tu portafolio y generar visualizaciones automáticamente.")

# Configurar el agente
agent = setup_agent(df)

# Área principal: Conversación
chat_container = st.container()
input_container = st.container()

# Mostrar todo el historial de mensajes
with chat_container:
    for message in st.session_state.messages:
        display_chat_message(message["role"], message["content"])

# Área de entrada de usuario
with input_container:
    if not agent and not st.session_state.openai_api_key:
        st.warning("Por favor, introduce tu API key de OpenAI en la barra lateral para activar el asistente.")
    else:
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

        # Procesar la entrada del usuario
        if user_input and (send_button or st.session_state.get('auto_send', True)):
            # Agregar mensaje del usuario al historial
            st.session_state.messages.append({"role": "user", "content": user_input})

            # Refrescar para mostrar el mensaje del usuario
            st.rerun()

# Procesar la respuesta del asistente (se ejecuta después de rerun)
if st.session_state.messages and st.session_state.messages[-1]["role"] == "user":
    with st.spinner("Procesando..."):
        user_query = st.session_state.messages[-1]["content"]

        # Si no hay agente configurado, intenta configurarlo de nuevo
        if not agent and st.session_state.openai_api_key:
            agent = setup_agent(df)

        if agent:
            # Analizar la consulta
            analysis = analyze_query(user_query, agent)

            # Generar respuesta según el tipo de consulta
            if analysis["type"] == "visualization":
                asst_response = generate_visualization(analysis)
            elif analysis["type"] == "text":
                asst_response = analysis["content"]
            else:  # Error
                asst_response = analysis["content"]
        else:
            asst_response = "Lo siento, no puedo procesar tu consulta sin una API key válida. Por favor, configura la API key en la barra lateral."

        # Agregar respuesta del asistente al historial
        st.session_state.messages.append({"role": "assistant", "content": asst_response})

        # Refrescar para mostrar la respuesta
        st.rerun()
