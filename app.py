import streamlit as st
import pandas as pd
import json
import os
import requests
import logging
from typing import Dict, List

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class LangchainAssistant:
    def __init__(self):
        # Initialize your LangChain components here
        pass

    def generate_response(self, prompt: str, portfolio: List[Dict], opportunities: List[Dict], user_preferences: Dict) -> str:
        # Implement your LangChain logic here
        # This is a placeholder implementation
        return f"I understand you're asking about: {prompt}\nLet me analyze your portfolio and opportunities..."

def get_clavis_portfolio(wallet_address: str) -> Dict:
    """Get portfolio data from Clavis API."""
    api_key = os.environ.get("CLAVIS_API_KEY")
    if not api_key:
        logger.error("Clavis API key not found")
        return {"error": "API key not found"}

    url = f"https://api.clavis.com/portfolio/{wallet_address}"  # Replace with actual endpoint
    headers = {"Authorization": f"Bearer {api_key}"}

    try:
        response = requests.get(url, headers=headers)
        if response.status_code == 200:
            data = response.json()
            processed_data = {
                'wallet': [],
                'chain': [],
                'common_name': [],
                'module': [],
                'token_symbol': [],
                'balance_usd': []
            }

            for position in data.get('positions', []):
                processed_data['wallet'].append(wallet_address)
                processed_data['chain'].append(position.get('chain', 'Unknown'))
                processed_data['common_name'].append(position.get('protocol', 'Unknown'))
                processed_data['module'].append(position.get('type', 'Unknown'))
                processed_data['token_symbol'].append(position.get('symbol', 'Unknown'))
                processed_data['balance_usd'].append(f"${position.get('balanceUsd', 0)}")

            return processed_data
        else:
            logger.error(f"Error {response.status_code}: {response.text}")
            return {"error": f"Error {response.status_code}: {response.text}"}
    except Exception as e:
        logger.error(f"Exception occurred: {str(e)}")
        return {"error": f"Exception occurred: {str(e)}"}

def get_defi_opportunities(min_tvl: float, max_tvl: float, min_apy: float, max_apy: float,
                         blockchains: List[str], tokens: List[str], use_mock_data: bool = True) -> List[Dict]:
    """Get DeFi opportunities based on user preferences."""
    if use_mock_data:
        # Return mock data for testing
        return [
            {
                "protocol": "Uniswap V3",
                "chain": "Ethereum",
                "apy": 15.5,
                "tvl": 1000000,
                "tokens": ["ETH", "USDC"]
            },
            # Add more mock opportunities as needed
        ]
    else:
        # Implement real API call to DeFiLlama or other data source
        pass

# Set page configuration
st.set_page_config(page_title="DeFi Portfolio Assistant", layout="wide")

# Initialize session state variables
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

if 'user_preferences' not in st.session_state:
    st.session_state.user_preferences = {
        'min_tvl': 1000000,
        'max_tvl': 1000000000,
        'min_apy': 1,
        'max_apy': 100,
        'blockchains': [],
        'tokens': [],
        'use_mock_data': True
    }

if 'portfolio_df' not in st.session_state:
    data = {
        'wallet': ['Wallet #1', 'Wallet #1'],
        'chain': ['base', 'mnt'],
        'common_name': ['Uniswap V3', 'Pendle V2'],
        'module': ['Liquidity Pool', 'Liquidity Pool'],
        'token_symbol': ['ODOS', 'cmETH/PT-cmETH-13FEB2025'],
        'balance_usd': ['$21.914496', '$554.812326']
    }
    st.session_state.portfolio_df = pd.DataFrame(data)

if 'assistant' not in st.session_state:
    st.session_state.assistant = LangchainAssistant()

# Main layout
st.title("DeFi Portfolio Assistant")

# Sidebar
with st.sidebar:
    st.header("Settings")

    # API Configuration
    st.subheader("API Configuration")
    use_mock = st.toggle(
        "Use Mock Data",
        value=st.session_state.user_preferences['use_mock_data'],
        help="Toggle to use mock data instead of real API calls"
    )
    st.session_state.user_preferences['use_mock_data'] = use_mock

    if not use_mock:
        clavis_api_key = st.text_input(
            "Clavis API Key",
            value=os.environ.get("CLAVIS_API_KEY", ""),
            type="password"
        )
        if clavis_api_key:
            os.environ["CLAVIS_API_KEY"] = clavis_api_key

        wallet_address = st.text_input("Wallet Address")
        if wallet_address and st.button("Fetch Portfolio"):
            with st.spinner("Fetching portfolio data..."):
                portfolio_data = get_clavis_portfolio(wallet_address)
                st.session_state.portfolio_df = pd.DataFrame(portfolio_data)
                st.success("Portfolio data fetched successfully!")

    # User Preferences
    st.header("User Preferences")

    # TVL Range
    st.subheader("TVL Range (USD)")
    min_tvl, max_tvl = st.slider(
        "TVL Range",
        min_value=0,
        max_value=10000000000,
        value=(st.session_state.user_preferences['min_tvl'], st.session_state.user_preferences['max_tvl']),
        format="$%d"
    )
    st.session_state.user_preferences['min_tvl'] = min_tvl
    st.session_state.user_preferences['max_tvl'] = max_tvl

    # APY Range
    st.subheader("APY Range (%)")
    min_apy, max_apy = st.slider(
        "APY Range",
        min_value=0.0,
        max_value=1000.0,
        value=(st.session_state.user_preferences['min_apy'], st.session_state.user_preferences['max_apy']),
        format="%.2f%%"
    )
    st.session_state.user_preferences['min_apy'] = min_apy
    st.session_state.user_preferences['max_apy'] = max_apy

    # Blockchain Selection
    st.subheader("Blockchains")
    available_chains = ['Ethereum', 'Polygon', 'Arbitrum', 'Optimism', 'Base', 'Mantle', 'Avalanche',
                       'BSC', 'Fantom', 'Solana', 'zkSync Era', 'Linea', 'Scroll', 'Celo']
    selected_chains = st.multiselect(
        "Select blockchains",
        options=available_chains,
        default=st.session_state.user_preferences['blockchains']
    )
    st.session_state.user_preferences['blockchains'] = selected_chains

    # Token Selection
    st.subheader("Tokens")
    available_tokens = ['ETH', 'USDC', 'USDT', 'DAI', 'WBTC', 'ODOS', 'WETH', 'MATIC', 'OP', 'ARB', 'GLP', 'cmETH']
    selected_tokens = st.multiselect(
        "Select tokens",
        options=available_tokens,
        default=st.session_state.user_preferences['tokens']
    )
    st.session_state.user_preferences['tokens'] = selected_tokens

# Main content
st.header("Your DeFi Portfolio")
st.dataframe(st.session_state.portfolio_df, use_container_width=True)

# Chat interface
st.header("Chat with your DeFi Assistant")

# Display chat history
for message in st.session_state.chat_history:
    if message["role"] == "user":
        st.chat_message("user").write(message["content"])
    else:
        st.chat_message("assistant").write(message["content"])

# Chat input
if prompt := st.chat_input("Ask about your portfolio or DeFi opportunities..."):
    st.session_state.chat_history.append({"role": "user", "content": prompt})
    st.chat_message("user").write(prompt)

    with st.spinner("Fetching DeFi opportunities..."):
        opportunities = get_defi_opportunities(
            min_tvl=st.session_state.user_preferences['min_tvl'],
            max_tvl=st.session_state.user_preferences['max_tvl'],
            min_apy=st.session_state.user_preferences['min_apy'],
            max_apy=st.session_state.user_preferences['max_apy'],
            blockchains=st.session_state.user_preferences['blockchains'],
            tokens=st.session_state.user_preferences['tokens'],
            use_mock_data=st.session_state.user_preferences['use_mock_data']
        )

    with st.spinner("Generating response..."):
        portfolio_context = st.session_state.portfolio_df.to_dict(orient='records')
        response = st.session_state.assistant.generate_response(
            prompt=prompt,
            portfolio=portfolio_context,
            opportunities=opportunities,
            user_preferences=st.session_state.user_preferences
        )

    st.session_state.chat_history.append({"role": "assistant", "content": response})
    st.chat_message("assistant").write(response)

# Display current preferences and opportunities
st.header("Current User Preferences")
st.json(st.session_state.user_preferences)

if 'opportunities' in locals() and opportunities:
    st.header("Latest DeFi Opportunities")
    opp_df = pd.DataFrame(opportunities)

    if not opp_df.empty:
        if 'apy' in opp_df.columns:
            opp_df['apy'] = opp_df['apy'].apply(lambda x: f"{x:.2f}%")
        if 'tvl' in opp_df.columns:
            opp_df['tvl'] = opp_df['tvl'].apply(lambda x: f"${x:,.2f}")

    st.dataframe(opp_df, use_container_width=True)
