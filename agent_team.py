import streamlit as st
from phi.agent import Agent, RunResponse
from phi.model.openai import OpenAIChat
from phi.tools.duckduckgo import DuckDuckGo
from phi.tools.yfinance import YFinanceTools
import pandas as pd
import os
from dotenv import load_dotenv
import re
from thefuzz import fuzz
import requests
from bs4 import BeautifulSoup
import json
import yfinance as yf
import altair as alt

# --------------------------------------------------------------------------------
# 1. Load Environment Variables and Data
# --------------------------------------------------------------------------------

load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")
if openai_api_key is None:
    raise ValueError("OpenAI API key not found. Please set the OPENAI_API_KEY environment variable.")

input_file_path = "data/cleaned_file.xlsx"
try:
    peer_group_data = pd.read_excel(input_file_path)
    print("Peer group data loaded successfully.")
except Exception as e:
    raise ValueError(f"Error loading Excel file: {e}")

# --------------------------------------------------------------------------------
# 2. Define Synonyms and Known Tickers Dictionaries
# --------------------------------------------------------------------------------

SYNONYMS = {
    "google": "alphabet inc.",
    "goog": "alphabet inc.",
    "facebook": "meta platforms inc.",
    "fb": "meta platforms inc.",
    "apple": "apple inc.",
    "microsoft": "microsoft corp.",
    "msft": "microsoft corp.",
    "amazon": "amazon.com inc.",
    "amzn": "amazon.com inc.",
    "tesla": "tesla inc.",
    "tsla": "tesla inc.",
    "meta": "meta platforms inc.",
    "nvidia": "nvidia corp.",
    "nvda": "nvidia corp.",
    # Add more synonyms as needed
}

KNOWN_TICKERS = {
    "google": "GOOG",
    "alphabet inc.": "GOOGL",
    "apple inc.": "AAPL",
    "microsoft corp.": "MSFT",
    "amazon.com inc.": "AMZN",
    "tesla inc.": "TSLA",
    "meta platforms inc.": "META",
    "facebook": "META",
    "nvidia corp.": "NVDA",
}

def apply_synonym(company_name: str) -> str:
    return SYNONYMS.get(company_name.lower(), company_name)

def get_known_ticker(company_name: str) -> str:
    return KNOWN_TICKERS.get(company_name.lower(), None)

# --------------------------------------------------------------------------------
# 3. Memory Agent for Continuous Interaction
# --------------------------------------------------------------------------------

class MemoryAgent:
    def __init__(self):
        if "memory" not in st.session_state:
            st.session_state["memory"] = []

    def add_message(self, role: str, content: str):
        st.session_state["memory"].append({"role": role, "content": content})

    def get_memory(self) -> str:
        memory = st.session_state["memory"]
        formatted_memory = ""
        for msg in memory:
            if msg["role"] == "user":
                formatted_memory += f"<div style='text-align: right; margin-bottom: 10px;'><strong>You:</strong> {msg['content']}</div>"
            else:
                formatted_memory += f"<div style='text-align: left; margin-bottom: 10px;'><strong>Assistant:</strong> {msg['content']}</div>"
        return formatted_memory

memory_agent = MemoryAgent()

# --------------------------------------------------------------------------------
# 4. Rendering Agent
# --------------------------------------------------------------------------------

class RenderingAgent(Agent):
    def render_table(self, df: pd.DataFrame, title: str = "Data Table") -> str:
        markdown_header = f"### {title}\n\n"
        table_md = df.to_markdown(index=False)
        return markdown_header + table_md

    def render_explanation(self, matched_company: str, matched_ticker: str) -> str:
        return (
            f"### Peer Group Analysis for **{matched_company}** ({matched_ticker})\n\n"
            f"This analysis provides a comprehensive comparison of **{matched_company}** with its peer companies "
            f"based on key financial metrics. The table below outlines essential financial indicators, "
            f"and the accompanying chart visualizes the Market Capitalization across these entities."
        )

rendering_agent = RenderingAgent(
    name="Rendering Agent",
    role="Format data into markdown tables or charts",
    model=OpenAIChat(id="gpt-4"),
    instructions=[
        "You are responsible for formatting data into visual or textual Markdown output.",
        "When asked, generate tables or charts in Markdown."
    ],
    show_tool_calls=False,
    markdown=True
)

# --------------------------------------------------------------------------------
# 5. PeerGroupAnalysisTool
# --------------------------------------------------------------------------------

class PeerGroupAnalysisTool:
    def analyze(self, query: str, data: pd.DataFrame, matched_company: str, matched_ticker: str, rendering_agent: RenderingAgent) -> dict:
        row = data[
            (data["Company Name"].str.lower() == matched_company.lower()) |
            (data["Ticker"].str.lower() == matched_ticker.lower())
        ].iloc[0]

        # Extract peer tickers
        if "Tickers" in row and isinstance(row["Tickers"], str):
            peer_tickers = [t.strip() for t in row["Tickers"].split(",")]
        elif "Tickers" in row and isinstance(row["Tickers"], list):
            peer_tickers = row["Tickers"]
        else:
            peer_tickers = []

        all_tickers = [matched_ticker] + peer_tickers

        fundamentals_list = []
        for t in all_tickers:
            if not t:
                continue
            try:
                stock = yf.Ticker(t)
                info = stock.info
                company_short_name = info.get("shortName", "N/A")
                market_cap = info.get("marketCap", "N/A")
                revenue = info.get("totalRevenue", "N/A")
                pe_ratio = info.get("trailingPE", "N/A")

                fundamentals_list.append({
                    "Ticker": t,
                    "Company": company_short_name,
                    "MarketCap": market_cap,
                    "Revenue": revenue,
                    "PE Ratio": pe_ratio,
                })
            except Exception as e:
                fundamentals_list.append({
                    "Ticker": t,
                    "Company": "N/A",
                    "MarketCap": "N/A",
                    "Revenue": "N/A",
                    "PE Ratio": "N/A",
                })

        df_fundamentals = pd.DataFrame(fundamentals_list)

        ceo_list = []
        for t in all_tickers:
            ceo_info = data[data["Ticker"].str.lower() == t.lower()]
            if not ceo_info.empty:
                ceo_name = ceo_info.iloc[0].get("CEO", "N/A")
                ceo_salary = ceo_info.iloc[0].get("CEO Salary", "N/A")
            else:
                ceo_name = "N/A"
                ceo_salary = "N/A"
            ceo_list.append({
                "Ticker": t,
                "CEO": ceo_name,
                "CEO Salary": ceo_salary
            })

        df_ceo = pd.DataFrame(ceo_list)
        df_combined = pd.merge(df_fundamentals, df_ceo, on="Ticker", how="left")

        explanation_md = rendering_agent.render_explanation(matched_company, matched_ticker)
        fundamentals_md = rendering_agent.render_table(df_combined, title="Fundamentals Overview")

        df_combined_clean = df_combined[df_combined["MarketCap"] != "N/A"].copy()
        df_combined_clean["MarketCap"] = pd.to_numeric(df_combined_clean["MarketCap"], errors='coerce')

        if not df_combined_clean.empty:
            chart = alt.Chart(df_combined_clean).mark_bar().encode(
                x=alt.X('Ticker:N', title='Ticker'),
                y=alt.Y('MarketCap:Q', title='Market Capitalization (USD)', scale=alt.Scale(zero=True)),
                tooltip=['Company', 'MarketCap']
            ).properties(
                title='Market Capitalization Comparison'
            ).interactive()
        else:
            chart = None

        final_markdown = explanation_md + "\n\n" + fundamentals_md
        return {
            "markdown": final_markdown,
            "chart": chart
        }

peer_group_tool = PeerGroupAnalysisTool()

# --------------------------------------------------------------------------------
# 6. Utility Functions
# --------------------------------------------------------------------------------

def normalize(value: str) -> str:
    if isinstance(value, str):
        return re.sub(r'\s+', ' ', value.strip().lower())
    return ''

def fallback_company_ticker_extraction(query: str) -> tuple:
    pattern = re.compile(r"(?:of|for|about)\s+([A-Za-z0-9&.,'\s]+)", re.IGNORECASE)
    match = pattern.search(query)
    if match:
        extracted_name = match.group(1).strip(" ?!.,")
        ticker_guess = get_known_ticker(extracted_name)
        return extracted_name, ticker_guess
    else:
        return query.strip(" ?!.,"), None

####################################################################
# >>>>>>>>>>>>>>>>>>>>> CRITICAL FIX HERE <<<<<<<<<<<<<<<<<<<<<<<< #
####################################################################
def extract_company_name_and_ticker(query: str) -> tuple:
    """
    Extracts company name and ticker from the user's query via an LLM call.
    Returns (company_name, ticker).
    """
    llm_agent = Agent(
        name="LLM Company Identification Agent",
        role="Identify the company and ticker from the user's query using LLM",
        model=OpenAIChat(id="gpt-4"),
        instructions=[
            "You are a specialized agent that extracts the company name and ticker from the user's query.",
            "Return valid JSON ONLY with the keys 'company_name' and 'ticker'.",
            "If you cannot determine the ticker, set it to null.",
            "Example: {\"company_name\": \"Google\", \"ticker\": \"GOOG\"}. No additional text."
        ],
        show_tool_calls=True,
        markdown=True,
    )
    response = llm_agent.run(query)

    # Universal fallback pattern: convert RunResponse -> string
    if hasattr(response, "content") and isinstance(response.content, str):
        response_text = response.content
    else:
        response_text = str(response)

    # Now safe to strip
    response_text = response_text.strip()
    print("DEBUG: (extract_company_name_and_ticker) response_text =>", response_text)

    try:
        parsed_json = json.loads(response_text)
        company_name = parsed_json.get("company_name", "").strip()
        ticker = parsed_json.get("ticker", "").strip().upper() or None
    except (json.JSONDecodeError, AttributeError) as e:
        print(f"JSON parsing failed in extract_company_name_and_ticker: {e}")
        company_name, ticker = fallback_company_ticker_extraction(query)

    company_name_mapped = apply_synonym(company_name)
    if not ticker and company_name_mapped:
        ticker = get_known_ticker(company_name_mapped)

    return company_name_mapped, ticker

# --------------------------------------------------------------------------------
# 7. SEC Edgar Scraping Function
# --------------------------------------------------------------------------------

def scrape_sec_edgar_data(company_name: str) -> str:
    company_name_encoded = requests.utils.quote(company_name)
    url = f"https://www.sec.gov/cgi-bin/browse-edgar?company={company_name_encoded}&CIK=&type=def+14a&owner=include&count=40&action=getcurrent"
    headers = {
        "User-Agent": "Your Name your.email@example.com"
    }
    response = requests.get(url, headers=headers)
    soup = BeautifulSoup(response.text, 'html.parser')

    logs = [f"Scraping SEC Edgar for company: {company_name}"]
    table = soup.find('table', class_='tableFile2')
    if not table:
        logs.append("No filings found for the company.")
        return "\n".join(logs)

    rows = table.find_all('tr')[1:]
    peer_companies = set()
    for row in rows:
        cols = row.find_all('td')
        if len(cols) < 3:
            continue
        filing_type = cols[0].text.strip()
        comp = cols[1].text.strip()
        if filing_type in ['10-K', '10-Q', '8-K', 'DEF 14A']:
            peer_companies.add(comp)

    if not peer_companies:
        logs.append("No peer companies found in SEC Edgar filings.")
    else:
        logs.append(f"Found {len(peer_companies)} peer companies in SEC Edgar filings.")

    peer_companies_formatted = "\n".join(peer_companies)
    return "\n".join(logs) + "\n\n### Peer Companies:\n" + peer_companies_formatted

# --------------------------------------------------------------------------------
# 8. Agents and Tools
# --------------------------------------------------------------------------------

web_agent = Agent(
    name="Web Agent",
    role="Search the web for information",
    model=OpenAIChat(id="gpt-4"),
    tools=[DuckDuckGo()],
    instructions=["Always include sources"],
    show_tool_calls=True,
    markdown=True,
)

finance_agent = Agent(
    name="Finance Agent",
    role="Get financial data",
    model=OpenAIChat(id="gpt-4"),
    tools=[YFinanceTools(stock_price=True, analyst_recommendations=True, company_info=True)],
    instructions=["Use tables to display data"],
    show_tool_calls=True,
    markdown=True,
)

agent_team = Agent(
    team=[web_agent, finance_agent],
    instructions=["Always include sources", "Use tables to display data"],
    show_tool_calls=True,
    markdown=True,
)

# --------------------------------------------------------------------------------
# 9. Peer Group Analysis Function
# --------------------------------------------------------------------------------

def peer_group_analysis_tool_function(query: str, company_name: str, ticker: str = None) -> dict:
    if not company_name:
        peers = scrape_sec_edgar_data(query)
        return {"markdown": f"Company '{query}' not found in the dataset. Fetched peers:\n{peers}", "chart": None}

    match_found = False
    matched_company = None
    matched_ticker = None

    company_name_normalized = normalize(company_name)
    ticker_normalized = normalize(ticker) if ticker else ''

    for idx, row in peer_group_data.iterrows():
        company = row.get("Company Name", "")
        ticker_symbol = row.get("Ticker", "")

        company_norm = normalize(company)
        ticker_normed = normalize(ticker_symbol)

        # exact match
        if (company_norm == company_name_normalized) or (ticker_normed == ticker_normalized):
            matched_company = company
            matched_ticker = ticker_symbol
            match_found = True
            break

        # fuzzy match
        similarity_company = fuzz.ratio(company_norm, company_name_normalized)
        similarity_ticker = fuzz.ratio(ticker_normed, ticker_normalized) if ticker else 0
        if similarity_company > 85 or similarity_ticker > 85:
            matched_company = company
            matched_ticker = ticker_symbol
            match_found = True
            break

    if match_found and matched_company:
        if hasattr(peer_group_tool, "analyze"):
            return peer_group_tool.analyze(query, peer_group_data, matched_company, matched_ticker, rendering_agent)
        else:
            raise AttributeError("The tool does not have an 'analyze' method.")
    else:
        peers = scrape_sec_edgar_data(company_name)
        return {"markdown": f"Company '{company_name}' not found in dataset. Fetched peers:\n{peers}", "chart": None}

# --------------------------------------------------------------------------------
# 10. Query Classification Agent (Critical Fix)
# --------------------------------------------------------------------------------

class QueryClassificationAgent(Agent):
    def classify_query(self, query: str) -> str:
        classification_prompt = (
            "Classify the following user query into one of the categories: "
            "'peer_group_analysis', 'company_info', 'stock_price', 'ceo_info', 'web_search', 'unknown'. "
            "Respond with only the category name.\n\n"
            f"Query: {query}\nCategory:"
        )

        raw_response = self.run(classification_prompt)

        # Universal fallback again:
        if hasattr(raw_response, "content") and isinstance(raw_response.content, str):
            category_text = raw_response.content
        else:
            category_text = str(raw_response)

        category_text = category_text.strip().lower()  # safe now
        print(f"Classified Query Category => {category_text}")

        return category_text

query_classification_agent = QueryClassificationAgent()

# --------------------------------------------------------------------------------
# 11. Additional Tools and Functions
# --------------------------------------------------------------------------------

def company_info_tool(company_name: str, ticker: str) -> str:
    try:
        stock = yf.Ticker(ticker)
        info = stock.info
        sname = info.get("shortName", "N/A")
        sector = info.get("sector", "N/A")
        industry = info.get("industry", "N/A")
        website = info.get("website", "N/A")

        return f"""
### Company Information for **{sname}** ({ticker})

- **Sector**: {sector}
- **Industry**: {industry}
- **Website**: [{website}]({website})
"""
    except Exception as e:
        print(f"Error fetching company info: {e}")
        return f"### Company Information for **{company_name}** ({ticker})\nN/A"

def stock_price_tool(ticker: str) -> str:
    try:
        stock = yf.Ticker(ticker)
        info = stock.info
        current_price = info.get("regularMarketPrice", "N/A")
        previous_close = info.get("regularMarketPreviousClose", "N/A")
        price_change = info.get("regularMarketChange", "N/A")
        percent_change = info.get("regularMarketChangePercent", "N/A")

        return f"""
### Stock Price for **{ticker}**

- **Current Price**: ${current_price}
- **Previous Close**: ${previous_close}
- **Change**: {price_change} ({percent_change}%)
"""
    except Exception as e:
        print(f"Error fetching stock price for {ticker}: {e}")
        return f"### Stock Price for **{ticker}**\nN/A"

def ceo_info_tool(ticker: str) -> str:
    try:
        ceo_name = peer_group_data[peer_group_data["Ticker"].str.lower() == ticker.lower()]["CEO"].values[0]
        ceo_salary = peer_group_data[peer_group_data["Ticker"].str.lower() == ticker.lower()]["CEO Salary"].values[0]
        return f"""
### CEO Information for **{ticker}**

- **CEO Name**: {ceo_name}
- **CEO Salary**: {ceo_salary}
"""
    except Exception as e:
        print(f"Error fetching CEO info for {ticker}: {e}")
        return f"### CEO Information for **{ticker}**\nN/A"

# --------------------------------------------------------------------------------
# 12. Streamlit App
# --------------------------------------------------------------------------------

st.set_page_config(page_title="GPT-like Peer Group Analysis", layout="wide")

st.markdown("<h1 style='text-align: center;'>Peer Group Analysis Chat Interface</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center;'>Interact with your agents to analyze peer group data, search the web, or fetch financial data.</p>", unsafe_allow_html=True)

query = st.text_input("Ask a question:", placeholder="Type your query here and press Enter")

if query:
    try:
        category = query_classification_agent.classify_query(query)

        if category == 'peer_group_analysis':
            company_name, ticker = extract_company_name_and_ticker(query)
            analysis_result = peer_group_analysis_tool_function(query, company_name, ticker)
            memory_agent.add_message("user", query)
            memory_agent.add_message("assistant", analysis_result["markdown"])
            st.markdown(memory_agent.get_memory(), unsafe_allow_html=True)
            if analysis_result["chart"]:
                st.altair_chart(analysis_result["chart"], use_container_width=True)

        elif category == 'company_info':
            company_name, ticker = extract_company_name_and_ticker(query)
            info_md = company_info_tool(company_name, ticker)
            memory_agent.add_message("user", query)
            memory_agent.add_message("assistant", info_md)
            st.markdown(memory_agent.get_memory(), unsafe_allow_html=True)

        elif category == 'stock_price':
            company_name, ticker = extract_company_name_and_ticker(query)
            price_md = stock_price_tool(ticker)
            memory_agent.add_message("user", query)
            memory_agent.add_message("assistant", price_md)
            st.markdown(memory_agent.get_memory(), unsafe_allow_html=True)

        elif category == 'ceo_info':
            company_name, ticker = extract_company_name_and_ticker(query)
            ceo_md = ceo_info_tool(ticker)
            memory_agent.add_message("user", query)
            memory_agent.add_message("assistant", ceo_md)
            st.markdown(memory_agent.get_memory(), unsafe_allow_html=True)

        elif category == 'web_search':
            web_search_result = web_agent.run(query)
            memory_agent.add_message("user", query)
            memory_agent.add_message("assistant", web_search_result.content)
            st.markdown(memory_agent.get_memory(), unsafe_allow_html=True)

        else:
            # unknown or general
            general_response = web_agent.run(query).content
            memory_agent.add_message("user", query)
            memory_agent.add_message("assistant", general_response)
            st.markdown(memory_agent.get_memory(), unsafe_allow_html=True)

    except Exception as e:
        st.error(f"An error occurred: {e}")
