import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import random
from langchain_openai import ChatOpenAI
from langchain_experimental.agents import create_pandas_dataframe_agent
from langchain.agents.agent_types import AgentType
import os
import matplotlib as mpl
from PIL import Image
from io import BytesIO
import requests

# Custom color palette
PRIMARY_COLOR = "#A199DA"
SECONDARY_COLOR = "#403680"
BG_COLOR = "#000000"#"#2B314E"
ACCENT_COLOR = "#A199DA"
LOGO_URL = "https://corp.orwee.io/wp-content/uploads/2023/07/cropped-imageonline-co-transparentimage-23-e1689783905238.webp"

# Función para cargar imágenes desde URL
@st.cache_data
def load_image(url):
    try:
        response = requests.get(url)
        img = Image.open(BytesIO(response.content))
        return img
    except Exception as e:
        st.error(f"Error loading image: {e}")
        return None

def load_avatar_image(url):
    try:
        response = requests.get(url)
        img = Image.open(BytesIO(response.content))
        return img
    except Exception as e:
        st.error(f"Error loading avatar image: {e}")
        return None

# Cargar avatares
assistant_avatar = load_avatar_image(LOGO_URL)
user_avatar = None  

# Create custom sequential color palette for charts
def create_custom_cmap():
    return mpl.colors.LinearSegmentedColormap.from_list("Rocky", [PRIMARY_COLOR, SECONDARY_COLOR])

# Apply custom branding
def apply_custom_branding():
    # Custom CSS with Rocky branding
    css = f"""
    <style>
        /* Main background and text */
        .stApp {{
            background-color: {BG_COLOR};
            color: white;
        }}

        /* Header styling */
        h1, h2, h3, h4, h5, h6 {{
            font-family: 'IBM Plex Mono', monospace !important;
            color: {PRIMARY_COLOR};
        }}

        /* Custom button styling */
        .stButton > button {{
            background-color: {PRIMARY_COLOR} !important;
            color: white !important;
            border: none !important;
            font-family: 'IBM Plex Mono', monospace !important;
            padding: 10px 15px !important;
            border-radius: 4px !important;
            width: 100% !important;
            text-align: left !important;
            font-size: 14px !important;
            margin-bottom: 8px !important;
        }}

        .stButton > button:hover {{
            background-color: {SECONDARY_COLOR} !important;
        }}

        /* Sidebar styling */
        section[data-testid="stSidebar"] {{
            background-color: {BG_COLOR};
            border-right: 1px solid {PRIMARY_COLOR};
        }}

        /* Add your logo */
        .logo-container {{
            display: flex;
            align-items: center;
            margin-bottom: 20px;
        }}

        .logo-container img {{
            height: 50px;
            margin-right: 10px;
        }}

        .app-title {{
            font-family: 'IBM Plex Mono', monospace;
            font-weight: bold;
            font-size: 1.5em;
            color: {PRIMARY_COLOR};
        }}

        /* Import IBM Plex Mono font */
        @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;700&display=swap');

        /* Metrics and key figures */
        .metric-value {{
            color: {PRIMARY_COLOR};
            font-weight: bold;
        }}

        /* Custom chart background */
        .chart-container {{
            background-color: rgba(43, 49, 78, 0.7);
            padding: 15px;
            border-radius: 5px;
            border: 1px solid {PRIMARY_COLOR};
        }}
    </style>
    """
    st.markdown(css, unsafe_allow_html=True)

    # Logo and title
    st.markdown(
        f"""
        <div class="logo-container">
            <img src="{LOGO_URL}" alt="Rocky Logo">
            <div class="app-title">Rocky - DeFi Portfolio Intelligence</div>
        </div>
        """,
        unsafe_allow_html=True
    )

# Page configuration
st.set_page_config(
    page_title="Rocky - DeFi Portfolio",
    page_icon="📊",
    layout="wide"
)

# Apply branding
apply_custom_branding()

# Initialize session states
if 'messages' not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Hello! I'm your DeFi portfolio assistant. How can I help you today?"}
    ]

if 'show_visualization' not in st.session_state:
    st.session_state.show_visualization = {
        'show': False,
        'type': None,      # can be 'dashboard', 'specific', 'positions'
        'group_by': None   # wallet, chain, etc. (None for dashboard)
    }

# Bilingual keyword mapping
BILINGUAL_KEYWORDS = {
    'wallet': ['wallet', 'billetera', 'cartera', 'wallets'],
    'chain': ['chain', 'blockchain', 'cadena', 'blockchains', 'chains'],
    'category': ['category', 'categoría', 'categoria', 'tipo', 'categories'],
    'protocol': ['protocol', 'protocolo', 'protocols'],
    'dashboard': ['dashboard', 'tablero', 'completo', 'general', 'overview', 'summary'],
    'positions': ['positions', 'posiciones', 'activos', 'assets', 'holdings'],
    'total': ['total', 'valor', 'value', 'balance', 'worth']
}

# Conversational responses for each query type
def get_conversational_response(query_type):
    responses = {
        'wallet': [
            "Here's the distribution of your funds by wallet. There's an interesting concentration pattern:",
            "Analyzing your wallets... This is interesting. Here's how your funds are distributed across different wallets:",
            "I've reviewed your data and here's the wallet distribution. There are clear concentration patterns."
        ],
        'chain': [
            "I've analyzed your exposure to different blockchains. Here's a breakdown of how your funds are distributed:",
            "Blockchain diversification analysis: This data shows which chains you're currently invested in and how value is distributed:",
            "Here's the blockchain analysis. It's interesting to see the distribution across different ecosystems:"
        ],
        'category': [
            "I've categorized your tokens and here's the distribution. This shows your balance between stablecoins, bluechips, and altcoins:",
            "Let's look at the distribution by token category... This is interesting. The proportion between different asset types is notable:",
            "Here's the category analysis. The distribution reflects certain investment patterns:"
        ],
        'dashboard': [
            "Here's a dashboard with the main metrics and visualizations of your portfolio:",
            "An overview is always useful. I've generated this dashboard with different perspectives of your portfolio to visualize the distributions:",
            "Presenting a complete summary of your portfolio with different visualizations to better understand the current position:"
        ],
        'total': [
            "I've calculated the total value of your portfolio. You currently have invested:",
            "According to my calculations, the total value of your portfolio at this moment is:",
            "Reviewing your positions, the total value of your portfolio is:"
        ],
        'positions': [
            "Here are the details of all your positions. You can filter by any criteria and by value range. Percentages are calculated based on the current selection:",
            "I've prepared an interactive table with all your positions. Use the filters to find exactly what you're looking for and the value range that interests you:",
            "These are all your current positions. The filters allow you to analyze specific segments of your portfolio. The percentage column shows the proportion within your selection:"
        ]
    }

    return random.choice(responses.get(query_type, ["Here's what you asked for:"]))

# Load portfolio data
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

# Load data
df = load_portfolio_data()

# Classify tokens
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

# Configure the LangChain agent
@st.cache_resource
def setup_agent(_df):
    try:
        # Try to read from secrets.toml first
        api_key = st.secrets.get("openai", {}).get("api_key", None)
    except:
        # If not in secrets, read from environment variables
        api_key = os.environ.get("OPENAI_API_KEY", None)

    if not api_key:
        st.warning("OpenAI API key not found. Smart assistant functionality will be limited.")
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
        st.error(f"Error configuring the agent: {e}")
        return None

# Configure agent
agent = setup_agent(df)

# Configure plot style for all visualizations
plt.style.use('dark_background')
custom_cmap = create_custom_cmap()

def style_plot(ax):
    """Apply Rocky's branding style to matplotlib plots"""
    ax.set_facecolor(BG_COLOR)
    ax.figure.set_facecolor(BG_COLOR)
    ax.tick_params(colors='white')
    ax.xaxis.label.set_color('white')
    ax.yaxis.label.set_color('white')
    ax.title.set_color(PRIMARY_COLOR)
    for spine in ax.spines.values():
        spine.set_color(ACCENT_COLOR)
    return ax

# Sidebar for configuration
with st.sidebar:
    st.header("Configuration")

    st.subheader("Quick Queries")

    # Standard buttons with custom styling from CSS
    if st.button("Wallet Distribution", key="wallet_dist", use_container_width=True):
        response = get_conversational_response('wallet')
        st.session_state.messages.append({"role": "user", "content": "Show wallet distribution"})
        st.session_state.messages.append({"role": "assistant", "content": response})
        st.session_state.show_visualization = {
            'show': True,
            'type': 'specific',
            'group_by': 'wallet'
        }

    if st.button("Blockchain Analysis", key="blockchain_analysis", use_container_width=True):
        response = get_conversational_response('chain')
        st.session_state.messages.append({"role": "user", "content": "Visualize my blockchain exposure"})
        st.session_state.messages.append({"role": "assistant", "content": response})
        st.session_state.show_visualization = {
            'show': True,
            'type': 'specific',
            'group_by': 'chain'
        }

    if st.button("Token Categories", key="token_categories", use_container_width=True):
        response = get_conversational_response('category')
        st.session_state.messages.append({"role": "user", "content": "Distribution by token categories"})
        st.session_state.messages.append({"role": "assistant", "content": response})
        st.session_state.show_visualization = {
            'show': True,
            'type': 'specific',
            'group_by': 'category'
        }

    if st.button("Complete Dashboard", key="complete_dashboard", use_container_width=True):
        response = get_conversational_response('dashboard')
        st.session_state.messages.append({"role": "user", "content": "Show a complete dashboard"})
        st.session_state.messages.append({"role": "assistant", "content": response})
        st.session_state.show_visualization = {
            'show': True,
            'type': 'dashboard',
            'group_by': None
        }

    if st.button("Show Positions", key="show_positions", use_container_width=True):
        response = get_conversational_response('positions')
        st.session_state.messages.append({"role": "user", "content": "Show all my positions"})
        st.session_state.messages.append({"role": "assistant", "content": response})
        st.session_state.show_visualization = {
            'show': True,
            'type': 'positions',
            'group_by': None
        }

    if st.button("Total Value", key="total_value", use_container_width=True):
        response = get_conversational_response('total')
        total_value = df['usd'].sum()
        st.session_state.messages.append({"role": "user", "content": "What's the total value of my portfolio?"})
        st.session_state.messages.append({"role": "assistant", "content": f"{response} ${total_value:.2f} USD"})
        st.session_state.show_visualization = {
            'show': False,
            'type': None,
            'group_by': None
        }

    st.markdown("---")
    st.caption("This assistant analyzes your cryptocurrency portfolio and generates visualizations.")

# Display chat history with standard avatars
# Luego, en la parte donde muestras los mensajes:
for msg in st.session_state.messages:
    avatar = assistant_avatar if msg["role"] == "assistant" else user_avatar
    with st.chat_message(msg["role"], avatar=avatar):
        st.write(msg["content"])
        
# Visualization area (if activated)
if st.session_state.show_visualization['show']:
    viz_type = st.session_state.show_visualization['type']
    group_by = st.session_state.show_visualization.get('group_by', None)

    viz_container = st.container()

    with viz_container:
        if viz_type == 'specific' and group_by:
            st.subheader(f"Visualization by {group_by.capitalize()}")

            # Aggregate data
            grouped_data = df.groupby(group_by)['usd'].sum()
            total = grouped_data.sum()

            # Charts
            col1, col2 = st.columns(2)

            with col1:
                fig, ax = plt.subplots(figsize=(8, 5))
                grouped_data.plot(kind='bar', ax=ax, color=PRIMARY_COLOR)
                style_plot(ax)
                ax.set_title(f"USD by {group_by.capitalize()}")
                ax.set_xlabel(group_by)
                ax.set_ylabel("USD")
                st.pyplot(fig)

            with col2:
                fig, ax = plt.subplots(figsize=(8, 5))
                colors = plt.cm.get_cmap(custom_cmap)(np.linspace(0, 1, len(grouped_data)))
                grouped_data.plot(kind='pie', autopct='%1.1f%%', ax=ax, colors=colors)
                style_plot(ax)
                ax.set_title(f"Distribution by {group_by.capitalize()}")
                ax.axis('equal')
                st.pyplot(fig)

            # Data table
            data_df = pd.DataFrame({
                group_by.capitalize(): grouped_data.index,
                "USD": grouped_data.values.round(2),
                "Percentage (%)": [(v/total*100).round(2) for v in grouped_data.values]
            })
            st.dataframe(data_df, hide_index=True)

            # Add descriptive summary
            st.subheader("Analysis Summary")

            # Prepare information for the summary
            top_item = grouped_data.idxmax()
            top_value = grouped_data.max()
            top_percent = (top_value/total*100).round(2)

            # Calculate concentration index (simplified Herfindahl-Hirschman)
            hhi = ((grouped_data / total) ** 2).sum() * 100

            # Formatted text
            st.markdown(f"""
            ### Distribution Analysis by {group_by}

            - **Total value:** ${total:.2f} USD
            - **Number of {group_by}s:** {len(grouped_data)}
            - **Highest concentration:** {top_item} with ${top_value:.2f} ({top_percent}% of total)
            - **Average value per {group_by}:** ${(total/len(grouped_data)).round(2)} USD
            - **Concentration index:** {hhi:.1f}/100 (higher values indicate greater concentration)
            - **Percentage distribution:** {', '.join([f"**{idx}:** {(val/total*100).round(1)}%" for idx, val in grouped_data.items()])}
            """)

        elif viz_type == 'dashboard':
            st.subheader("Portfolio Dashboard")

            # First row: general metrics
            total_value = df['usd'].sum()
            avg_value = df['usd'].mean()
            unique_chains = df['chain'].nunique()

            col1, col2, col3 = st.columns(3)
            col1.metric("Total Value", f"${total_value:.2f}")
            col2.metric("Average per Position", f"${avg_value:.2f}")
            col3.metric("Blockchains", f"{unique_chains}")

            # Wallet Distribution
            st.subheader("Wallet Distribution")
            wallet_data = df.groupby('wallet')['usd'].sum().sort_values(ascending=False)

            col1, col2 = st.columns(2)

            with col1:
                # Wallet chart (pie)
                fig, ax = plt.subplots(figsize=(8, 5))
                colors = plt.cm.get_cmap(custom_cmap)(np.linspace(0, 1, len(wallet_data)))
                wallet_data.plot(kind='pie', autopct='%1.1f%%', ax=ax, colors=colors)
                style_plot(ax)
                ax.set_title("Distribution by Wallet")
                ax.axis('equal')
                st.pyplot(fig)

            with col2:
                # Wallet chart (bar)
                fig, ax = plt.subplots(figsize=(8, 5))
                wallet_data.plot(kind='bar', ax=ax, color=colors)
                style_plot(ax)
                ax.set_title("USD by Wallet")
                ax.set_xlabel("Wallet")
                ax.set_ylabel("USD")
                st.pyplot(fig)

            # Blockchain Distribution
            st.subheader("Blockchain Distribution")
            chain_data = df.groupby('chain')['usd'].sum().sort_values(ascending=False)

            col1, col2 = st.columns(2)

            with col1:
                # Chain chart (pie)
                fig, ax = plt.subplots(figsize=(8, 5))
                colors = plt.cm.get_cmap(custom_cmap)(np.linspace(0, 1, len(chain_data)))
                chain_data.plot(kind='pie', autopct='%1.1f%%', ax=ax, colors=colors)
                style_plot(ax)
                ax.set_title("Distribution by Blockchain")
                ax.axis('equal')
                st.pyplot(fig)

            with col2:
                # Chain chart (bar)
                fig, ax = plt.subplots(figsize=(8, 5))
                chain_data.plot(kind='bar', ax=ax, color=colors)
                style_plot(ax)
                ax.set_title("USD by Blockchain")
                ax.set_xlabel("Blockchain")
                ax.set_ylabel("USD")
                st.pyplot(fig)

            # Category Distribution
            st.subheader("Category Distribution")
            cat_data = df.groupby('category')['usd'].sum().sort_values(ascending=False)

            col1, col2 = st.columns(2)

            with col1:
                # Category chart (pie)
                fig, ax = plt.subplots(figsize=(8, 5))
                colors = plt.cm.get_cmap(custom_cmap)(np.linspace(0, 1, len(cat_data)))
                cat_data.plot(kind='pie', autopct='%1.1f%%', ax=ax, colors=colors)
                style_plot(ax)
                ax.set_title("Distribution by Category")
                ax.axis('equal')
                st.pyplot(fig)

            with col2:
                # Category chart (bar)
                fig, ax = plt.subplots(figsize=(8, 5))
                cat_data.plot(kind='bar', ax=ax, color=colors)
                style_plot(ax)
                ax.set_title("USD by Category")
                ax.set_xlabel("Category")
                ax.set_ylabel("USD")
                st.pyplot(fig)

            # Protocol Distribution
            st.subheader("Protocol Distribution")
            protocol_data = df.groupby('protocol')['usd'].sum().sort_values(ascending=False)

            fig, ax = plt.subplots(figsize=(10, 6))
            colors = plt.cm.get_cmap(custom_cmap)(np.linspace(0, 1, len(protocol_data)))
            protocol_data.plot(kind='barh', ax=ax, color=colors)
            style_plot(ax)
            ax.invert_yaxis()  # Make it display in descending order visually
            ax.set_title("USD by Protocol (Descending)")
            ax.set_xlabel("USD")
            ax.set_ylabel("Protocol")
            st.pyplot(fig)

            # Position Ranking
            st.subheader("Position Ranking")
            positions_df = df.copy()
            positions_df['position_name'] = positions_df['token'] + ' (' + positions_df['protocol'] + ')'

            # Sort by USD in descending order
            top_positions = positions_df.sort_values('usd', ascending=False)

            if len(top_positions) > 10:
                top_positions = top_positions.head(10)

            # Position chart (horizontal bars)
            fig, ax = plt.subplots(figsize=(10, max(6, len(top_positions) * 0.4)))
            positions_plot = top_positions.set_index('position_name')['usd']
            colors = plt.cm.get_cmap(custom_cmap)(np.linspace(0, 1, len(positions_plot)))
            positions_plot.plot(kind='barh', ax=ax, color=colors)
            style_plot(ax)
            ax.invert_yaxis()  # Make it display in descending order visually
            ax.set_title("Top Positions by USD")
            ax.set_xlabel("USD")
            ax.set_ylabel("Position")
            st.pyplot(fig)

            # Add descriptive summary for the dashboard
            st.subheader("Portfolio Summary")

            # Calculate data for the summary
            top_wallet = wallet_data.idxmax()
            top_wallet_value = wallet_data.max()
            top_wallet_percent = (top_wallet_value/total_value*100).round(2)

            top_chain = chain_data.idxmax()
            top_chain_value = chain_data.max()
            top_chain_percent = (top_chain_value/total_value*100).round(2)

            top_category = cat_data.idxmax()
            top_category_value = cat_data.max()
            top_category_percent = (top_category_value/total_value*100).round(2)

            # Calculate concentration indices
            wallet_hhi = ((wallet_data / total_value) ** 2).sum() * 100
            chain_hhi = ((chain_data / total_value) ** 2).sum() * 100
            category_hhi = ((cat_data / total_value) ** 2).sum() * 100

            # Calculate diversification metrics
            coef_var = (df['usd'].std() / df['usd'].mean() * 100)  # Coefficient of variation
            positions_per_chain = round(len(df) / unique_chains, 1)  # Corrected

            # Create descriptive summary
            st.markdown(f"""
            ### Portfolio Statistics

            The portfolio has a total value of **${total_value:.2f}** distributed across **{len(df)}** positions through **{unique_chains}** different blockchains.

            #### Main Distribution:
            - **Wallet**: Highest concentration in **{top_wallet}** with **${top_wallet_value:.2f}** (**{top_wallet_percent}%** of total)
            - **Blockchain**: Predominance of **{top_chain}** with **${top_chain_value:.2f}** (**{top_chain_percent}%** of total)
            - **Category**: Highest presence of **{top_category}** with **${top_category_value:.2f}** (**{top_category_percent}%**)

            #### Diversification Metrics:
            - **Wallet concentration index**: **{wallet_hhi:.1f}**/100
            - **Blockchain concentration index**: **{chain_hhi:.1f}**/100
            - **Category concentration index**: **{category_hhi:.1f}**/100
            - **Coefficient of variation**: **{coef_var:.1f}%** (value dispersion)
            - **Positions per blockchain**: **{positions_per_chain}** (average)

            #### Blockchain Distribution:
            {', '.join([f"**{chain}**: **{(value/total_value*100).round(1)}%**" for chain, value in chain_data.items()])}
            """)

        elif viz_type == 'positions':
            st.subheader("All Positions")

            # Enrich DataFrame with data to display
            df_display = df.copy()

            # Create filters
            st.write("#### Filters")
            col1, col2, col3, col4 = st.columns(4)

            with col1:
                # Wallet filter
                wallet_options = ['All'] + sorted(df_display['wallet'].unique().tolist())
                wallet_filter = st.selectbox('Wallet', wallet_options)

            with col2:
                # Blockchain filter
                chain_options = ['All'] + sorted(df_display['chain'].unique().tolist())
                chain_filter = st.selectbox('Blockchain', chain_options)

            with col3:
                # Category filter
                category_options = ['All'] + sorted(df_display['category'].unique().tolist())
                category_filter = st.selectbox('Category', category_options)

            with col4:
                # Protocol filter
                protocol_options = ['All'] + sorted(df_display['protocol'].unique().tolist())
                protocol_filter = st.selectbox('Protocol', protocol_options)

            # Apply filters
            if wallet_filter != 'All':
                df_display = df_display[df_display['wallet'] == wallet_filter]

            if chain_filter != 'All':
                df_display = df_display[df_display['chain'] == chain_filter]

            if category_filter != 'All':
                df_display = df_display[df_display['category'] == category_filter]

            if protocol_filter != 'All':
                df_display = df_display[df_display['protocol'] == protocol_filter]

            # Range to filter by USD value (minimum and maximum)
            min_usd = float(df['usd'].min())
            max_usd = float(df['usd'].max())

            usd_range = st.slider(
                "Value Range (USD)",
                min_value=min_usd,
                max_value=max_usd,
                value=(max(min_usd, 5.0), max_usd),  # Default value: minimum $5
                step=1.0
            )

            # Apply USD range filter
            df_display = df_display[(df_display['usd'] >= usd_range[0]) & (df_display['usd'] <= usd_range[1])]

            # Show number of results
            st.write(f"Showing {len(df_display)} of {len(df)} positions")

            # Calculate percentages AFTER all filters, based on the filtered table
            if not df_display.empty:
                filtered_total = df_display['usd'].sum()
                df_display['% of Total'] = (df_display['usd'] / filtered_total * 100).round(2)
            else:
                df_display['% of Total'] = 0  # Handle empty case

            # Reorganize columns for better display
            df_display = df_display[['wallet', 'chain', 'protocol', 'token', 'category', 'usd', '% of Total']]

            # Rename columns for better presentation
            df_display.columns = ['Wallet', 'Blockchain', 'Protocol', 'Token', 'Category', 'USD', '% of Selection']

            # Interactive table with filtering and sorting
            st.dataframe(
                df_display,
                column_config={
                    "USD": st.column_config.NumberColumn(
                        format="$%.2f",
                    ),
                    "% of Selection": st.column_config.ProgressColumn(
                        format="%.1f%%",
                        min_value=0,
                        max_value=100,
                    ),
                },
                hide_index=True,
                use_container_width=True
            )

            # Add useful metrics
            if len(df_display) > 0:  # Only if there are results after filtering
                filtered_total = df_display['USD'].sum()
                total_portfolio = df['usd'].sum()
                filtered_percent = (filtered_total / total_portfolio) * 100

                st.subheader("Selection Metrics")
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Positions", f"{len(df_display)}")
                with col2:
                    st.metric("Total Value", f"${filtered_total:.2f}")
                with col3:
                    st.metric("% of Portfolio", f"{filtered_percent:.1f}%")
                with col4:
                    if len(df_display) > 0:
                        st.metric("Average", f"${df_display['USD'].mean():.2f}")

                # Add descriptive summary for filtered positions
                st.subheader("Selection Analysis")

                # Calculate information for the summary
                top_position = df_display.loc[df_display['USD'].idxmax()]
                bottom_position = df_display.loc[df_display['USD'].idxmin()]

                # Calculate statistics and aggregations
                chain_counts = df_display['Blockchain'].value_counts()
                top_chain = chain_counts.index[0] if len(chain_counts) > 0 else "none"
                chain_diversity = len(chain_counts)

                wallet_distribution = df_display.groupby('Wallet')['USD'].sum()
                top_wallet = wallet_distribution.idxmax() if not wallet_distribution.empty else "none"
                top_wallet_value = wallet_distribution.max() if not wallet_distribution.empty else 0
                top_wallet_percent = (top_wallet_value/filtered_total*100).round(2) if filtered_total > 0 else 0

                # Calculate descriptive statistics
                value_range = df_display['USD'].max() - df_display['USD'].min()
                std_dev = df_display['USD'].std()
                median_value = df_display['USD'].median()
                cv = (std_dev / df_display['USD'].mean() * 100).round(1) if df_display['USD'].mean() > 0 else 0

                # Calculate concentration index
                wallet_hhi = ((wallet_distribution / filtered_total) ** 2).sum() * 100 if not df_display.empty and filtered_total > 0 else 0

                # Create descriptive summary with important data in bold
                st.markdown(f"""
                ### Selection Statistics

                In this selection of **{len(df_display)} positions** with total value of **${filtered_total:.2f}**:

                #### Value Distribution:
                - **Maximum position:** ${top_position['USD']:.2f} ({top_position['Token']} in {top_position['Protocol']})
                - **Minimum position:** ${bottom_position['USD']:.2f} ({bottom_position['Token']})
                - **Median value:** ${median_value:.2f}
                - **Standard deviation:** ${std_dev:.2f}
                - **Coefficient of variation:** {cv}%
                - **Value range:** ${value_range:.2f}

                #### Concentration and Diversification:
                - **Wallet concentration index:** {wallet_hhi:.1f}/100
                - **Blockchains represented:** {chain_diversity} chains
                - **Main blockchain:** {top_chain} ({chain_counts[top_chain]} positions)
                - **Main wallet:** {top_wallet} (${top_wallet_value:.2f}, {top_wallet_percent}% of selected total)

                This selection represents **{filtered_percent:.1f}%** of the total portfolio value.
                """)

                # Add additional data if there are enough positions
                if len(df_display) > 1:
                    # Additional aggregations
                    protocol_counts = df_display['Protocol'].value_counts()
                    top_protocol = protocol_counts.index[0] if not protocol_counts.empty else "none"
                    category_distribution = df_display.groupby('Category')['USD'].sum()
                    category_percents = ((category_distribution / filtered_total) * 100).round(1)

                    # Display additional data neutrally
                    st.markdown("### Additional Aggregations")
                    st.markdown(f"""
                    #### Distribution by Category:
                    {', '.join([f"**{cat}:** **{val}%**" for cat, val in category_percents.items()])}

                    #### Distribution by Protocol:
                    - **Protocols used:** {len(protocol_counts)}
                    - **Main protocol:** {top_protocol} ({protocol_counts[top_protocol]} positions)

                    #### Statistical Distribution:
                    - **Mean vs. Median:** The mean (${df_display['USD'].mean():.2f}) is {
                        "higher than" if df_display['USD'].mean() > median_value else
                        "lower than" if df_display['USD'].mean() < median_value else
                        "equal to"} the median (${median_value:.2f}), indicating a distribution {
                        "with bias towards high values" if df_display['USD'].mean() > median_value else
                        "with bias towards low values" if df_display['USD'].mean() < median_value else
                        "that is symmetric"}
                    """)

# User input
prompt = st.chat_input("Type your query...")

if prompt:
    # Add user message
    st.session_state.messages.append({"role": "user", "content": prompt})

    # Function to detect query intent (supporting both English and Spanish)
    def detect_query_intent(query_text):
        query_lower = query_text.lower()

        for intent, keywords in BILINGUAL_KEYWORDS.items():
            if any(keyword in query_lower for keyword in keywords):
                return intent

        return None

    # Analyze the query
    intent = detect_query_intent(prompt.lower())

    # Handle visualization queries
    if intent == 'positions':
        viz_type = 'positions'
        group_by = None
        asst_response = get_conversational_response('positions')
    elif intent:
        # For visualization intents
        if intent == 'dashboard':
            viz_type = 'dashboard'
            group_by = None
            asst_response = get_conversational_response('dashboard')
        elif intent in ['wallet', 'chain', 'category', 'protocol']:
            viz_type = 'specific'
            group_by = 'category' if intent == 'category' else intent
            asst_response = get_conversational_response(intent)
        elif intent == 'total':
            total_value = df['usd'].sum()
            asst_response = f"{get_conversational_response('total')} ${total_value:.2f} USD"
            viz_type = None
            group_by = None
        else:
            # Default to dashboard if intent detected but not specific
            viz_type = 'dashboard'
            group_by = None
            asst_response = get_conversational_response('dashboard')
    else:
        # Not a visualization query
        viz_type = None
        group_by = None
        if agent:
            try:
                asst_response = agent.run(prompt)
            except Exception as e:
                asst_response = f"Error processing your query: {str(e)}"
        else:
            asst_response = "I can't respond without a valid API key. Please ensure your OpenAI API key is configured."

    # Update visualization state
    if intent and intent != 'total':
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

    # Add assistant response
    st.session_state.messages.append({"role": "assistant", "content": asst_response})

    # Reload to show the complete response
    st.rerun()
