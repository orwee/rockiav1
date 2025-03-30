import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import warnings
warnings.filterwarnings('ignore')

# Importar las librerías de LangChain necesarias
from langchain_openai import ChatOpenAI
from langchain_experimental.agents import create_pandas_dataframe_agent
from langchain.agents.agent_types import AgentType

# Configuración de la página
st.set_page_config(
    page_title="Analizador de Portafolio Cripto",
    page_icon="💰",
    layout="wide"
)

# Título de la aplicación
st.title("🚀 Analizador de Portafolio Cripto")
st.markdown("---")

# Sidebar para opciones
st.sidebar.header("Opciones")

# Opción para introducir API Key
openai_api_key = st.sidebar.text_input("OpenAI API Key (opcional)", type="password")

# Datos del portafolio
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

# Clasificar tokens en categorías (stablecoins, bluechips, altcoins)
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

# Opción para cargar datos propios
st.sidebar.header("Datos del Portafolio")
use_sample = st.sidebar.checkbox("Usar datos de ejemplo", value=True)

if not use_sample:
    uploaded_file = st.sidebar.file_uploader("Cargar CSV con datos del portafolio", type="csv")
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            if 'token' in df.columns and 'category' not in df.columns:
                df['category'] = df['token'].apply(classify_token)
            st.sidebar.success("¡Datos cargados correctamente!")
        except Exception as e:
            st.sidebar.error(f"Error al cargar el archivo: {e}")

# Mostrar los datos
st.subheader("📊 Datos del Portafolio")
st.dataframe(df, use_container_width=True)

# Función estandarizada de visualización
def plot_portfolio(
    group_by=None,         # Variable para agrupar (wallet, chain, category, etc.)
    measure='usd',         # Variable a medir (generalmente usd)
    agg_func='sum',        # Función de agregación (sum, mean, count, etc.)
    chart_type='bar',      # Tipo de gráfico (bar, pie, line, etc.)
    sort_values=True,      # Ordenar valores
    ascending=False,       # Orden ascendente o descendente
    title=None,            # Título personalizado
    figsize=(10, 6),       # Tamaño del gráfico
    **kwargs               # Parámetros adicionales para personalización
):
    """
    Función genérica para visualizar datos del portafolio con diferentes parámetros.
    """
    fig, ax = plt.subplots(figsize=figsize)

    # Si no se especifica grupo, mostrar datos sin agrupar
    if group_by is None:
        data = df
        x = kwargs.get('x', df.columns[0])  # Por defecto, primera columna como x
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
            # Usar getattr para ejecutar métodos dinámicamente es más seguro que eval
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
        plt.axis('equal')
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
    elif chart_type == 'area':
        data.plot(kind='area', title=title, ax=ax, **kwargs)
        ax.set_xlabel(group_by)
        ax.set_ylabel(measure)
    elif chart_type == 'heatmap' and kwargs.get('second_group'):
        # Para heatmaps necesitamos dos variables para agrupar
        second_group = kwargs.get('second_group')
        pivot_data = df.pivot_table(
            values=measure,
            index=group_by,
            columns=second_group,
            aggfunc=agg_func
        )
        sns.heatmap(pivot_data, annot=True, cmap='viridis', ax=ax, **kwargs)
        ax.set_title(title)
    else:
        # Cualquier otro tipo soportado por pandas
        data.plot(kind=chart_type, title=title, ax=ax, **kwargs)

    ax.tick_params(axis='x', rotation=kwargs.get('rotation', 45))
    plt.tight_layout()

    return fig, data  # Devolver la figura y los datos agregados

# Función para mostrar múltiples gráficos en subplots
def plot_portfolio_dashboard(
    plots_config,   # Lista de diccionarios con configuración para cada gráfico
    figsize=(15, 10),
    grid=(2, 2),    # Filas, columnas para el grid de subplots
    **kwargs
):
    """
    Genera un dashboard con múltiples visualizaciones configurables.
    """
    fig, axes = plt.subplots(grid[0], grid[1], figsize=figsize)
    axes = axes.flatten() if hasattr(axes, 'flatten') else [axes]  # Manejar caso 1x1

    results = {}

    for i, config in enumerate(plots_config):
        position = config.pop("position", i+1) - 1  # Obtener posición (1-indexed, convertir a 0-indexed)
        if position < len(axes):
            plt.sca(axes[position])  # Establecer el axes actual

            # Extraer parámetros de configuración
            group_by = config.pop("group_by", None)
            measure = config.pop("measure", "usd")
            agg_func = config.pop("agg_func", "sum")
            chart_type = config.pop("chart_type", "bar")
            title = config.pop("title", None)

            # Generar el gráfico con la configuración especificada
            # Modificamos para no crear una nueva figura
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

                # Ordenar valores si se solicita
                if config.get('sort_values', True):
                    data = data.sort_values(ascending=config.get('ascending', False))

            # Configurar título
            if title is None:
                agg_name = {'sum': 'Suma', 'mean': 'Promedio', 'count': 'Conteo'}.get(agg_func, agg_func)
                measure_name = measure.upper()
                group_name = group_by.capitalize() if group_by else "Sin agrupar"
                title = f"{agg_name} de {measure_name} por {group_name}"

            # Generar gráfico en el subplot actual
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
            elif chart_type == 'heatmap' and config.get('second_group'):
                second_group = config.get('second_group')
                pivot_data = df.pivot_table(
                    values=measure,
                    index=group_by,
                    columns=second_group,
                    aggfunc=agg_func
                )
                sns.heatmap(pivot_data, annot=True, cmap='viridis', ax=axes[position], **config)
                axes[position].set_title(title)
            else:
                data.plot(kind=chart_type, title=title, ax=axes[position], **config)

            axes[position].tick_params(axis='x', rotation=config.get('rotation', 45))
            results[f"plot_{position+1}"] = data

    plt.tight_layout()
    return fig, results

def generate_complete_analysis():
    """Genera un análisis completo del portafolio con agregaciones y visualizaciones"""
    # Valor total del portafolio
    total_value = df['usd'].sum()

    # Métricas generales
    avg_value = df['usd'].mean()
    max_value = df['usd'].max()
    min_value = df['usd'].min()

    # Agregaciones por wallet, chain y categoría
    wallet_totals = df.groupby('wallet')['usd'].sum().sort_values(ascending=False)
    chain_totals = df.groupby('chain')['usd'].sum().sort_values(ascending=False)
    category_totals = df.groupby('category')['usd'].sum().sort_values(ascending=False)

    # Crear contenedor para el informe
    report = st.container()

    with report:
        st.header("📝 ANÁLISIS COMPLETO DEL PORTAFOLIO CRIPTO")
        st.markdown("---")

        # Resumen general en métricas de Streamlit
        st.subheader("1️⃣ RESUMEN GENERAL")
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric("Valor Total", f"${total_value:.2f}")
        with col2:
            st.metric("Inversión Promedio", f"${avg_value:.2f}")
        with col3:
            st.metric("Inversión Máxima", f"${max_value:.2f}")
        with col4:
            st.metric("Inversión Mínima", f"${min_value:.2f}")

        # Distribuciones en tablas
        st.subheader("2️⃣ DISTRIBUCIONES DEL PORTAFOLIO")

        tab1, tab2, tab3 = st.tabs(["Por Wallet", "Por Blockchain", "Por Categoría"])

        with tab1:
            wallet_df = pd.DataFrame({
                'Wallet': wallet_totals.index,
                'USD': wallet_totals.values,
                'Porcentaje': [(v/total_value)*100 for v in wallet_totals.values]
            })
            st.dataframe(wallet_df, use_container_width=True, hide_index=True)

        with tab2:
            chain_df = pd.DataFrame({
                'Blockchain': chain_totals.index,
                'USD': chain_totals.values,
                'Porcentaje': [(v/total_value)*100 for v in chain_totals.values]
            })
            st.dataframe(chain_df, use_container_width=True, hide_index=True)

        with tab3:
            category_df = pd.DataFrame({
                'Categoría': category_totals.index,
                'USD': category_totals.values,
                'Porcentaje': [(v/total_value)*100 for v in category_totals.values]
            })
            st.dataframe(category_df, use_container_width=True, hide_index=True)

        # Visualizaciones
        st.subheader("3️⃣ VISUALIZACIONES")

        plots_config = [
            {"group_by": "wallet", "chart_type": "pie", "position": 1, "title": "Distribución por Wallet"},
            {"group_by": "chain", "chart_type": "bar", "position": 2, "title": "USD por Blockchain"},
            {"group_by": "category", "chart_type": "pie", "position": 3, "title": "Distribución por Categoría"},
            {"group_by": "token", "chart_type": "bar", "position": 4, "title": "USD por Token"},
        ]

        fig, _ = plot_portfolio_dashboard(plots_config, figsize=(12, 10))
        st.pyplot(fig)

        # Análisis de riesgo y recomendaciones
        st.subheader("4️⃣ ANÁLISIS Y RECOMENDACIONES")

        # Calcular porcentaje de stablecoins
        stablecoin_pct = category_totals.get('Stablecoin', 0) / total_value * 100 if 'Stablecoin' in category_totals else 0

        # Calcular diversificación por blockchain
        chain_diversity = len(chain_totals)

        # Calcular concentración (usando el índice HHI simplificado)
        hhi = sum((value/total_value)**2 for value in wallet_totals.values) * 10000

        col1, col2 = st.columns(2)

        with col1:
            st.metric("Porcentaje en Stablecoins", f"{stablecoin_pct:.1f}%")
            st.metric("Diversificación en Blockchains", f"{chain_diversity} cadenas")

        with col2:
            # Evaluación de riesgo
            risk_score = 0
            if stablecoin_pct > 50:
                risk_score += 1  # Bajo riesgo
            elif stablecoin_pct < 10:
                risk_score += 3  # Alto riesgo
            else:
                risk_score += 2  # Riesgo moderado

            if chain_diversity > 3:
                risk_score += 1  # Buena diversificación
            else:
                risk_score += 2  # Poca diversificación

            if hhi > 5000:
                risk_score += 3  # Alta concentración
            else:
                risk_score += 1  # Buena distribución

            # Escala de 3 a 8, donde 3 es bajo riesgo y 8 es alto riesgo
            risk_level = "Bajo" if risk_score <= 4 else "Moderado" if risk_score <= 6 else "Alto"
            st.metric("Nivel de Riesgo", risk_level)

            # Visualizar puntaje de riesgo
            st.progress((risk_score - 3) / 5)  # Normalizado de 0 a 1

        # Recomendaciones
        st.markdown("#### Recomendaciones:")

        recommendations = []

        if stablecoin_pct > 50:
            recommendations.append("📉 Tu portafolio tiene una alta concentración en stablecoins, lo que indica un enfoque conservador. Considera aumentar exposición a bluechips para mayor potencial de crecimiento.")
        elif stablecoin_pct < 10:
            recommendations.append("📈 Tu portafolio tiene poca exposición a stablecoins, lo que indica mayor riesgo. Considera aumentar tu posición en stablecoins para reducir la volatilidad.")

        if hhi > 5000:
            recommendations.append("⚠️ Tu portafolio está muy concentrado en pocos activos o wallets. Considera diversificar para reducir el riesgo.")

        if chain_diversity < 3:
            recommendations.append("🔗 Tu exposición a diferentes blockchains es limitada. Considera expandir a otras redes para diversificar el riesgo tecnológico.")

        if not recommendations:
            recommendations.append("✅ Tu portafolio parece estar bien balanceado en términos de riesgo y diversificación.")

        for rec in recommendations:
            st.markdown(f"- {rec}")

    return True

# Configuración del agente de LangChain
@st.cache_resource
def setup_agent(_df, api_key=None):
    if api_key:
        llm = ChatOpenAI(temperature=0, model="gpt-4o-mini", api_key=api_key)
    else:
        # Usar valor predeterminado de la variable de entorno si existe
        llm = ChatOpenAI(temperature=0, model="gpt-4o-mini")

    try:
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
        st.error(f"Error al configurar el agente LangChain: {e}")
        return None

# Inicializar agente si hay una API Key
agent = None
if openai_api_key:
    with st.spinner("Configurando asistente de LangChain..."):
        agent = setup_agent(df, openai_api_key)
        if agent:
            st.sidebar.success("¡Asistente LangChain listo!")

# Pestañas principales de la aplicación
tab1, tab2, tab3 = st.tabs(["Análisis Rápido", "Visualizaciones Personalizadas", "Asistente AI"])

with tab1:
    st.header("🚀 Análisis Rápido")

    if st.button("Generar Análisis Completo", key="generate_analysis"):
        with st.spinner("Generando análisis completo..."):
            generate_complete_analysis()

with tab2:
    st.header("📊 Visualizaciones Personalizadas")

    col1, col2 = st.columns(2)

    with col1:
        # Opciones para el gráfico
        group_by = st.selectbox(
            "Agrupar por:",
            options=["wallet", "chain", "category", "protocol", "token"],
            index=0
        )

        chart_type = st.selectbox(
            "Tipo de gráfico:",
            options=["bar", "pie", "line", "area"],
            index=0
        )

    with col2:
        agg_func = st.selectbox(
            "Función de agregación:",
            options=["sum", "mean", "count", "max", "min"],
            index=0
        )

        color_map = st.selectbox(
            "Esquema de colores:",
            options=["viridis", "plasma", "inferno", "magma", "cividis", "Blues", "Greens", "Reds"],
            index=0
        )

    # Generar visualización
    if st.button("Generar Visualización", key="generate_viz"):
        with st.spinner("Creando visualización..."):
            fig, data = plot_portfolio(
                group_by=group_by,
                chart_type=chart_type,
                agg_func=agg_func,
                cmap=color_map,
                figsize=(10, 6)
            )
            st.pyplot(fig)

            # Mostrar datos en formato de tabla
            st.subheader("Datos del Gráfico")

            if isinstance(data, pd.Series):
                # Convertir Series a DataFrame
                data_df = pd.DataFrame({
                    group_by: data.index,
                    f"{agg_func.capitalize()} de USD": data.values
                })
                st.dataframe(data_df, use_container_width=True, hide_index=True)
            else:
                st.dataframe(data, use_container_width=True)

    # Opción para dashboard personalizado
    st.subheader("Dashboard Personalizado")

    st.markdown("Selecciona los gráficos para tu dashboard:")

    dashboard_config = []

    col1, col2 = st.columns(2)

    with col1:
        if st.checkbox("Incluir distribución por Wallet", value=True):
            chart_type_wallet = st.radio("Tipo de gráfico para Wallet:", ["pie", "bar"], horizontal=True)
            dashboard_config.append({
                "group_by": "wallet",
                "chart_type": chart_type_wallet,
                "position": 1,
                "title": "Distribución por Wallet"
            })

        if st.checkbox("Incluir distribución por Categoría", value=True):
            chart_type_category = st.radio("Tipo de gráfico para Categoría:", ["pie", "bar"], horizontal=True)
            dashboard_config.append({
                "group_by": "category",
                "chart_type": chart_type_category,
                "position": 3,
                "title": "Distribución por Categoría"
            })

    with col2:
        if st.checkbox("Incluir distribución por Blockchain", value=True):
            chart_type_chain = st.radio("Tipo de gráfico para Blockchain:", ["bar", "pie"], horizontal=True)
            dashboard_config.append({
                "group_by": "chain",
                "chart_type": chart_type_chain,
                "position": 2,
                "title": "USD por Blockchain"
            })

        if st.checkbox("Incluir distribución por Protocolo", value=True):
            chart_type_protocol = st.radio("Tipo de gráfico para Protocolo:", ["bar", "pie"], horizontal=True)
            dashboard_config.append({
                "group_by": "protocol",
                "chart_type": chart_type_protocol,
                "position": 4,
                "title": "USD por Protocolo"
            })

    if st.button("Generar Dashboard Personalizado", key="generate_dashboard"):
        if dashboard_config:
            with st.spinner("Creando dashboard personalizado..."):
                fig, _ = plot_portfolio_dashboard(dashboard_config, figsize=(12, 10))
                st.pyplot(fig)
        else:
            st.warning("Selecciona al menos un gráfico para generar el dashboard.")

with tab3:
    st.header("🤖 Asistente AI para Análisis")

    if not agent and not openai_api_key:
        st.warning("Introduce tu OpenAI API Key en el panel lateral para activar el asistente AI.")
    else:
        st.markdown("""
        Puedes hacer preguntas en lenguaje natural sobre tu portafolio. Por ejemplo:
        - ¿Cuál es el valor total de mi portafolio en USD?
        - ¿Cuál es el valor promedio por inversión?
        - ¿Cuánto tengo invertido en la blockchain base?
        - ¿Qué porcentaje de mi portafolio está en stablecoins?
        - ¿Cuál es la distribución de mis activos por categoría?
        """)

        user_query = st.text_input("Escribe tu consulta:")

        if user_query and agent:
            with st.spinner("Procesando tu consulta..."):
                try:
                    # Procesar la consulta
                    if any(term in user_query.lower() for term in [
                        "gráfico", "grafico", "visualiza", "visualizar",
                        "mostrar", "dibujar", "generar gráfico"
                    ]):
                        # Determinar variable de agrupación
                        group_by = None
                        for term, var in {
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
                        }.items():
                            if term in user_query.lower():
                                group_by = var
                                break

                        # Determinar tipo de gráfico
                        chart_type = "bar"  # Valor predeterminado
                        for term, type_val in {
                            "torta": "pie",
                            "circular": "pie",
                            "pie": "pie",
                            "barras": "bar",
                            "bar": "bar",
                            "línea": "line",
                            "linea": "line",
                            "line": "line",
                            "área": "area",
                            "area": "area"
                        }.items():
                            if term in user_query.lower():
                                chart_type = type_val
                                break

                        # Determinar medida y agregación
                        agg_func = "sum"  # Valor predeterminado
                        for term, func in {
                            "promedio": "mean",
                            "media": "mean",
                            "mean": "mean",
                            "avg": "mean",
                            "contar": "count",
                            "count": "count",
                            "cantidad": "count",
                            "máximo": "max",
                            "maximo": "max",
                            "max": "max",
                            "mínimo": "min",
                            "minimo": "min",
                            "min": "min"
                        }.items():
                            if term in user_query.lower():
                                agg_func = func
                                break

                        st.markdown(f"#### Visualización: {chart_type.capitalize()} de {agg_func} USD por {group_by}")

                        if group_by:
                            fig, data = plot_portfolio(
                                group_by=group_by,
                                chart_type=chart_type,
                                agg_func=agg_func
                            )
                            st.pyplot(fig)
                        else:
                            # Si no se específica agrupación, mostrar dashboard múltiple
                            st.markdown("#### Dashboard con múltiples visualizaciones")
                            plots_config = [
                                {"group_by": "wallet", "chart_type": "pie", "position": 1},
                                {"group_by": "chain", "chart_type": "bar", "position": 2},
                                {"group_by": "category", "chart_type": "pie", "position": 3}
                            ]
                            fig, _ = plot_portfolio_dashboard(plots_config, figsize=(12, 8))
                            st.pyplot(fig)
                    else:
                        # Usar el agente para consultas generales
                        response = agent.run(user_query)
                        st.markdown(f"**Respuesta:**\n{response}")
                except Exception as e:
                    st.error(f"Error al procesar la consulta: {str(e)}")

# Información adicional en el sidebar
st.sidebar.markdown("---")
st.sidebar.header("Información")
st.sidebar.info("""
Esta aplicación te permite analizar tu portafolio de criptomonedas,
visualizar su distribución y obtener recomendaciones basadas en
análisis de datos y técnicas de IA.

Desarrollado con Streamlit, LangChain y OpenAI.
""")
