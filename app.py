import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import random
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
        'type': None,      # puede ser 'dashboard', 'specific', 'positions'
        'group_by': None   # wallet, chain, etc. (None para dashboard)
    }

# Respuestas conversacionales para cada tipo de consulta rápida
def get_conversational_response(query_type):
    responses = {
        'wallet': [
            "Aquí tienes la distribución de tus fondos por wallet. Se observa una interesante concentración en algunos wallets:",
            "Analizando tus wallets... Esto es interesante. Te muestro cómo están distribuidos tus fondos entre diferentes wallets:",
            "He revisado tus datos y aquí te presento la distribución por wallet. Hay patrones claros de concentración."
        ],
        'chain': [
            "He analizado tu exposición a diferentes blockchains. Aquí tienes el detalle de cómo están distribuidos tus fondos:",
            "Análisis de diversificación blockchain: Estos datos muestran en qué cadenas tienes invertido actualmente y cómo se distribuye el valor:",
            "Aquí está el análisis por blockchain. Es interesante ver la distribución entre diferentes ecosistemas:"
        ],
        'category': [
            "He categorizado tus tokens y aquí tienes la distribución. Esto muestra tu balance entre stablecoins, bluechips y altcoins:",
            "Veamos la distribución por categoría de tokens... Esto es interesante. La proporción entre activos de diferente naturaleza es notable:",
            "Aquí tienes el análisis por categoría. La distribución refleja ciertos patrones de inversión:"
        ],
        'dashboard': [
            "Aquí tienes un dashboard con las principales métricas y visualizaciones de tu portafolio:",
            "Un panorama general siempre es útil. He generado este dashboard con diferentes perspectivas de tu portafolio para visualizar las distribuciones:",
            "Presentando un resumen completo de tu portafolio con diferentes visualizaciones para entender mejor la posición actual:"
        ],
        'total': [
            "He calculado el valor total de tu portafolio. Actualmente tienes invertido:",
            "Según mis cálculos, el valor total de tu portafolio en este momento es:",
            "Revisando tus posiciones, el valor total de tu portafolio es:"
        ],
        'positions': [
            "Aquí tienes el detalle de todas tus posiciones. Puedes filtrar por cualquier criterio y por rango de valor. Los porcentajes se calculan sobre la selección actual:",
            "He preparado una tabla interactiva con todas tus posiciones. Usa los filtros para encontrar exactamente lo que buscas y el rango de valores que te interesa:",
            "Estas son todas tus posiciones actuales. Los filtros te permiten analizar segmentos específicos de tu portafolio. La columna de porcentaje muestra la proporción dentro de tu selección:"
        ]
    }

    return random.choice(responses.get(query_type, ["Aquí tienes lo que me pediste:"]))

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
        {
            "id": 5,
            "wallet": "Wallet #1",
            "chain": "ethereum",
            "protocol": "Aave V3",
            "token": "ETH",
            "usd": 3.50
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
        response = get_conversational_response('wallet')
        st.session_state.messages.append({"role": "user", "content": "Muestra la distribución por wallet"})
        st.session_state.messages.append({"role": "assistant", "content": response})
        st.session_state.show_visualization = {
            'show': True,
            'type': 'specific',
            'group_by': 'wallet'
        }

    if st.button("🔗 Análisis por Blockchain"):
        response = get_conversational_response('chain')
        st.session_state.messages.append({"role": "user", "content": "Visualiza mi exposición por blockchain"})
        st.session_state.messages.append({"role": "assistant", "content": response})
        st.session_state.show_visualization = {
            'show': True,
            'type': 'specific',
            'group_by': 'chain'
        }

    if st.button("💰 Categorías de Token"):
        response = get_conversational_response('category')
        st.session_state.messages.append({"role": "user", "content": "Distribución por categorías de token"})
        st.session_state.messages.append({"role": "assistant", "content": response})
        st.session_state.show_visualization = {
            'show': True,
            'type': 'specific',
            'group_by': 'category'
        }

    if st.button("🔄 Dashboard Completo"):
        response = get_conversational_response('dashboard')
        st.session_state.messages.append({"role": "user", "content": "Muestra un dashboard completo"})
        st.session_state.messages.append({"role": "assistant", "content": response})
        st.session_state.show_visualization = {
            'show': True,
            'type': 'dashboard',
            'group_by': None
        }

    if st.button("📋 Mostrar Posiciones"):
        response = get_conversational_response('positions')
        st.session_state.messages.append({"role": "user", "content": "Muestra todas mis posiciones"})
        st.session_state.messages.append({"role": "assistant", "content": response})
        st.session_state.show_visualization = {
            'show': True,
            'type': 'positions',
            'group_by': None
        }

    if st.button("💸 Valor Total"):
        response = get_conversational_response('total')
        total_value = df['usd'].sum()
        st.session_state.messages.append({"role": "user", "content": "¿Cuál es el valor total de mi portafolio?"})
        st.session_state.messages.append({"role": "assistant", "content": f"{response} ${total_value:.2f} USD"})
        st.session_state.show_visualization = {
            'show': False,
            'type': None,
            'group_by': None
        }

    st.markdown("---")
    st.caption("Este asistente analiza tu portafolio de criptomonedas y genera visualizaciones.")

# Mostrar chat
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

# Área de visualización (si está activada)
if st.session_state.show_visualization['show']:
    viz_type = st.session_state.show_visualization['type']
    group_by = st.session_state.show_visualization.get('group_by', None)

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

            # Añadir resumen descriptivo
            st.subheader("Resumen del Análisis")

            # Preparar información para el resumen
            top_item = grouped_data.idxmax()
            top_value = grouped_data.max()
            top_percent = (top_value/total*100).round(2)

            # Calcular índice de concentración (Herfindahl-Hirschman simplificado)
            hhi = ((grouped_data / total) ** 2).sum() * 100

            # Texto con formato
            st.markdown(f"""
            ### Análisis de distribución por {group_by}

            - **Valor total:** ${total:.2f} USD
            - **Número de {group_by}s:** {len(grouped_data)}
            - **Mayor concentración:** {top_item} con ${top_value:.2f} ({top_percent}% del total)
            - **Valor promedio por {group_by}:** ${(total/len(grouped_data)).round(2)} USD
            - **Índice de concentración:** {hhi:.1f}/100 (valores más altos indican mayor concentración)
            - **Distribución porcentual:** {', '.join([f"**{idx}:** {(val/total*100).round(1)}%" for idx, val in grouped_data.items()])}
            """)

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

            # Añadir resumen descriptivo para el dashboard
            st.subheader("Resumen del Portafolio")

            # Calcular datos para el resumen
            top_wallet = wallet_data.idxmax()
            top_wallet_value = wallet_data.max()
            top_wallet_percent = (top_wallet_value/total_value*100).round(2)

            top_chain = chain_data.idxmax()
            top_chain_value = chain_data.max()
            top_chain_percent = (top_chain_value/total_value*100).round(2)

            top_category = cat_data.idxmax()
            top_category_value = cat_data.max()
            top_category_percent = (top_category_value/total_value*100).round(2)

            # Calcular índices de concentración
            wallet_hhi = ((wallet_data / total_value) ** 2).sum() * 100
            chain_hhi = ((chain_data / total_value) ** 2).sum() * 100
            category_hhi = ((cat_data / total_value) ** 2).sum() * 100

            # Calcular métricas de diversificación
            coef_var = (df['usd'].std() / df['usd'].mean() * 100)  # Coeficiente de variación
            positions_per_chain = round(len(df) / unique_chains, 1)  # Usar función round en lugar del método

            # Crear resumen descriptivo
            st.markdown(f"""
            ### Estadísticas del Portafolio

            El portafolio tiene un valor total de **${total_value:.2f}** distribuido en **{len(df)}** posiciones a través de **{unique_chains}** blockchains diferentes.

            #### Distribución Principal:
            - **Wallet**: Mayor concentración en **{top_wallet}** con **${top_wallet_value:.2f}** (**{top_wallet_percent}%** del total)
            - **Blockchain**: Predominio de **{top_chain}** con **${top_chain_value:.2f}** (**{top_chain_percent}%** del total)
            - **Categoría**: Mayor presencia de **{top_category}** con **${top_category_value:.2f}** (**{top_category_percent}%**)

            #### Métricas de Diversificación:
            - **Índice de concentración por wallet**: **{wallet_hhi:.1f}**/100
            - **Índice de concentración por blockchain**: **{chain_hhi:.1f}**/100
            - **Índice de concentración por categoría**: **{category_hhi:.1f}**/100
            - **Coeficiente de variación**: **{coef_var}%** (dispersión de valores)
            - **Posiciones por blockchain**: **{positions_per_chain}** (promedio)

            #### Distribución por Blockchain:
            {', '.join([f"**{chain}**: **{(value/total_value*100).round(1)}%**" for chain, value in chain_data.items()])}
            """)

        elif viz_type == 'positions':
            st.subheader("📋 Todas las Posiciones")

            # Enriquecer el DataFrame con datos para mostrar
            df_display = df.copy()

            # Crear filtros
            st.write("#### Filtros")
            col1, col2, col3, col4 = st.columns(4)

            with col1:
                # Filtro de Wallet
                wallet_options = ['Todos'] + sorted(df_display['wallet'].unique().tolist())
                wallet_filter = st.selectbox('Wallet', wallet_options)

            with col2:
                # Filtro de Blockchain
                chain_options = ['Todos'] + sorted(df_display['chain'].unique().tolist())
                chain_filter = st.selectbox('Blockchain', chain_options)

            with col3:
                # Filtro de Categoría
                category_options = ['Todos'] + sorted(df_display['category'].unique().tolist())
                category_filter = st.selectbox('Categoría', category_options)

            with col4:
                # Filtro de Protocolo
                protocol_options = ['Todos'] + sorted(df_display['protocol'].unique().tolist())
                protocol_filter = st.selectbox('Protocolo', protocol_options)

            # Aplicar filtros
            if wallet_filter != 'Todos':
                df_display = df_display[df_display['wallet'] == wallet_filter]

            if chain_filter != 'Todos':
                df_display = df_display[df_display['chain'] == chain_filter]

            if category_filter != 'Todos':
                df_display = df_display[df_display['category'] == category_filter]

            if protocol_filter != 'Todos':
                df_display = df_display[df_display['protocol'] == protocol_filter]

            # Rango para filtrar por valor en USD (mínimo y máximo)
            min_usd = float(df['usd'].min())
            max_usd = float(df['usd'].max())

            usd_range = st.slider(
                "Rango de Valor (USD)",
                min_value=min_usd,
                max_value=max_usd,
                value=(max(min_usd, 5.0), max_usd),  # Valor predeterminado: mínimo $5
                step=1.0
            )

            # Aplicar filtro de rango USD
            df_display = df_display[(df_display['usd'] >= usd_range[0]) & (df_display['usd'] <= usd_range[1])]

            # Mostrar número de resultados
            st.write(f"Mostrando {len(df_display)} de {len(df)} posiciones")

            # Calcular los porcentajes DESPUÉS de todos los filtros, basados en la tabla filtrada
            if not df_display.empty:
                filtered_total = df_display['usd'].sum()
                df_display['% del Total'] = (df_display['usd'] / filtered_total * 100).round(2)
            else:
                df_display['% del Total'] = 0  # Manejo de caso vacío

            # Reorganizar columnas para mejor visualización
            df_display = df_display[['wallet', 'chain', 'protocol', 'token', 'category', 'usd', '% del Total']]

            # Renombrar columnas para mejor presentación
            df_display.columns = ['Wallet', 'Blockchain', 'Protocolo', 'Token', 'Categoría', 'USD', '% de Selección']

            # Tabla interactiva con filtrado y ordenación
            st.dataframe(
                df_display,
                column_config={
                    "USD": st.column_config.NumberColumn(
                        format="$%.2f",
                    ),
                    "% de Selección": st.column_config.ProgressColumn(
                        format="%.1f%%",
                        min_value=0,
                        max_value=100,
                    ),
                },
                hide_index=True,
                use_container_width=True
            )

            # Agregar algunas métricas útiles
            if len(df_display) > 0:  # Solo si hay resultados después de filtrar
                filtered_total = df_display['USD'].sum()
                total_portfolio = df['usd'].sum()
                filtered_percent = (filtered_total / total_portfolio) * 100

                st.subheader("Métricas de la Selección")
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Posiciones", f"{len(df_display)}")
                with col2:
                    st.metric("Valor Total", f"${filtered_total:.2f}")
                with col3:
                    st.metric("% del Portfolio", f"{filtered_percent:.1f}%")
                with col4:
                    if len(df_display) > 0:
                        st.metric("Promedio", f"${df_display['USD'].mean():.2f}")

                # Añadir resumen descriptivo para las posiciones filtradas
                st.subheader("Análisis de la Selección")

                # Calcular información para el resumen
                top_position = df_display.loc[df_display['USD'].idxmax()]
                bottom_position = df_display.loc[df_display['USD'].idxmin()]

                # Calcular estadísticas y agregaciones
                chain_counts = df_display['Blockchain'].value_counts()
                top_chain = chain_counts.index[0] if len(chain_counts) > 0 else "ninguna"
                chain_diversity = len(chain_counts)

                wallet_distribution = df_display.groupby('Wallet')['USD'].sum()
                top_wallet = wallet_distribution.idxmax() if not wallet_distribution.empty else "ninguna"
                top_wallet_value = wallet_distribution.max() if not wallet_distribution.empty else 0
                top_wallet_percent = (top_wallet_value/filtered_total*100).round(2) if filtered_total > 0 else 0

                # Calcular estadísticas descriptivas
                value_range = df_display['USD'].max() - df_display['USD'].min()
                std_dev = df_display['USD'].std()
                median_value = df_display['USD'].median()
                cv = (std_dev / df_display['USD'].mean() * 100).round(1) if df_display['USD'].mean() > 0 else 0

                # Calcular índice de concentración
                wallet_hhi = ((wallet_distribution / filtered_total) ** 2).sum() * 100 if not df_display.empty and filtered_total > 0 else 0

                # Crear resumen descriptivo con datos importantes en negrita
                st.markdown(f"""
                ### Estadísticas de la Selección

                En esta selección de **{len(df_display)} posiciones** con valor total de **${filtered_total:.2f}**:

                #### Distribución de Valor:
                - **Posición máxima:** ${top_position['USD']:.2f} ({top_position['Token']} en {top_position['Protocolo']})
                - **Posición mínima:** ${bottom_position['USD']:.2f} ({bottom_position['Token']})
                - **Valor mediano:** ${median_value:.2f}
                - **Desviación estándar:** ${std_dev:.2f}
                - **Coeficiente de variación:** {cv}%
                - **Rango de valores:** ${value_range:.2f}

                #### Concentración y Diversificación:
                - **Índice de concentración por wallet:** {wallet_hhi:.1f}/100
                - **Blockchains representadas:** {chain_diversity} cadenas
                - **Principal blockchain:** {top_chain} ({chain_counts[top_chain]} posiciones)
                - **Principal wallet:** {top_wallet} (${top_wallet_value:.2f}, {top_wallet_percent}% del total seleccionado)

                Esta selección representa el **{filtered_percent:.1f}%** del valor total del portafolio.
                """)

                # Añadir datos adicionales si hay suficientes posiciones
                if len(df_display) > 1:
                    # Agregaciones adicionales
                    protocol_counts = df_display['Protocolo'].value_counts()
                    top_protocol = protocol_counts.index[0] if not protocol_counts.empty else "ninguno"
                    category_distribution = df_display.groupby('Categoría')['USD'].sum()
                    category_percents = ((category_distribution / filtered_total) * 100).round(1)

                    # Mostrar datos adicionales de forma neutral
                    st.markdown("### Agregaciones Adicionales")
                    st.markdown(f"""
                    #### Distribución por Categoría:
                    {', '.join([f"**{cat}:** **{val}%**" for cat, val in category_percents.items()])}

                    #### Distribución por Protocolo:
                    - **Protocolos utilizados:** {len(protocol_counts)}
                    - **Principal protocolo:** {top_protocol} ({protocol_counts[top_protocol]} posiciones)

                    #### Distribución Estadística:
                    - **Media vs. Mediana:** La media (${df_display['USD'].mean():.2f}) es {
                        "mayor que" if df_display['USD'].mean() > median_value else
                        "menor que" if df_display['USD'].mean() < median_value else
                        "igual a"} la mediana (${median_value:.2f}), lo que indica una distribución {
                        "con sesgo hacia valores altos" if df_display['USD'].mean() > median_value else
                        "con sesgo hacia valores bajos" if df_display['USD'].mean() < median_value else
                        "simétrica"}
                    """)

# Entrada de usuario
prompt = st.chat_input("Escribe tu consulta...")

if prompt:
    # Añadir mensaje del usuario
    st.session_state.messages.append({"role": "user", "content": prompt})

    # Analizar la consulta para determinar si es de visualización
    query_lower = prompt.lower()
    viz_terms = ["gráfico", "grafico", "visualiza", "visualizar", "mostrar", "ver", "distribución", "distribucion", "dashboard", "posiciones"]
    is_viz_query = any(term in query_lower for term in viz_terms)

    # Determinar el tipo de consulta
    if "posiciones" in query_lower or "positions" in query_lower:
        viz_type = 'positions'
        group_by = None
        asst_response = get_conversational_response('positions')
    else:
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
            if any(term in query_lower for term in ["dashboard", "completo", "general", "todos"]):
                viz_type = 'dashboard'
                group_by = None
                asst_response = get_conversational_response('dashboard')
            elif group_by:
                viz_type = 'specific'
                asst_response = get_conversational_response(group_by if group_by in ['wallet', 'chain', 'category'] else 'specific')
            else:
                # Si pide visualización pero no especifica variable ni dashboard
                viz_type = 'dashboard'
                group_by = None
                asst_response = get_conversational_response('dashboard')
        else:
            # No es una consulta de visualización
            viz_type = None
            if "total" in query_lower or "valor" in query_lower:
                total_value = df['usd'].sum()
                asst_response = f"{get_conversational_response('total')} ${total_value:.2f} USD"
            else:
                # Usar el agente para otras consultas
                if agent:
                    try:
                        asst_response = agent.run(prompt)
                    except Exception as e:
                        asst_response = f"Error al procesar tu consulta: {str(e)}"
                else:
                    asst_response = "No puedo responder sin una API key válida. Por favor, configura la API key en la barra lateral."

    # Actualizar estado de visualización
    if is_viz_query:
        st.session_state.show_visualization = {
            'show': True,
            'type': viz_type,
            'group_by': group_by
        }
    else:
        st.session_state.show_visualization = {
            'show': False,
            'type': None,
            'group_by': None
        }

    # Añadir respuesta del asistente
    st.session_state.messages.append({"role": "assistant", "content": asst_response})

    # Recargar para mostrar la respuesta completa
    st.rerun()
