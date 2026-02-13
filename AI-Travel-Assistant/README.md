# TravelBot: Intelligent Multi-Tool Agentic Travel Assistant

**TravelBot** is a production-grade, multi-tool AI agent built with **LangGraph**, **LanceDB**, **LlamaParse**, and specialized APIs (Amadeus, Tavily). It serves as a comprehensive travel orchestration system capable of searching flights, booking hotels, locating restaurants, providing real-time weather and currency metrics, performing semantic FAQ vector searches, and synthesizing custom travel itineraries.

---

## Table of Contents

1. [Project Architecture Overview](#-project-architecture-overview)
2. [Section-by-Section Implementation Details](#-section-by-section-implementation-details)
   * [Section 1: Environment Setup & API Configuration](#section-1-environment-setup--api-configuration)
   * [Section 2: Data Ingestion, Parsing & Normalization](#section-2-data-ingestion-parsing--normalization)
   * [Section 3: Vector Database Construction (LanceDB)](#section-3-vector-database-construction-lancedb)
   * [Section 4: Tool Definitions & Integrations](#section-4-tool-definitions--integrations)
   * [Section 5: Agent Architecture & State Orchestration (LangGraph)](#section-5-agent-architecture--state-orchestration-langgraph)
   * [Section 6: Interactive Terminal Interface (CLI)](#section-6-interactive-terminal-interface-cli)
3. [In-Depth Analysis of 14 Test Scenarios](#-in-depth-analysis-of-14-test-scenarios)
4. [Tech Stack Summary](#-tech-stack-summary)
5. [Installation & Setup Guide](#-installation--setup-guide)
6. [License](#-license)

---

## Project Architecture Overview

The system processes incoming natural language queries through an agentic loop. Depending on user intent, **LangGraph** routes execution dynamically through custom python tools or returns synthesized direct answers while maintaining conversation memory across sessions.

```
                    +---------------------------+
                    |        User Query         |
                    +-------------+-------------+
                                  |
                                  v
                    +---------------------------+
                    |    LangGraph Reasoner     |
                    |       (GPT-4o/3.5)        |
                    +-------------+-------------+
                                  |
            +---------------------+---------------------+
            |                     |                     |
            v                     v                     v
+-------------------+   +-------------------+   +-------------------+
| Vector Search DB  |   | External APIs     |   | Deterministic     |
| (LanceDB / BGE)   |   | (Amadeus, Tavily) |   | Calculators       |
+---------+---------+   +---------+---------+   +---------+---------+
          |                       |                       |
          +-----------------------+-----------------------+
                                  |
                                  v
                    +---------------------------+
                    |   Synthesizer & Response  |
                    +---------------------------+
```

---

## Section-by-Section Implementation Details

### Section 1: Environment Setup & API Configuration
* **Objective**: Establish secure environment management and load required library dependencies.
* **Implementation**: Uses `python-dotenv` to inject sensitive API credentials (OpenAI, Amadeus, Tavily, LlamaCloud) directly into python environment variables. Installs foundational libraries including `langchain`, `langgraph`, `lancedb`, `llama-parse`, and `amadeus`.

### Section 2: Data Ingestion, Parsing & Normalization
* **Objective**: Prepare structured and unstructured data sources for retrieval.
* **Implementation**:
  * Downloads IATA airport code datasets and currency mapping tables from Kaggle.
  * Employs **LlamaParse** to convert PDF handbooks into structured Markdown documents preserving headers and page metadata.
  * Implements fuzzy string matching via standard python `difflib` to resolve misspelled city or airport names against official IATA datasets.

### Section 3: Vector Database Construction (LanceDB)
* **Objective**: Create a high-performance vector store for semantic document search.
* **Implementation**:
  * Defines Pydantic data models (`FAQSchema` and `BookSchema`) extending `lancedb.pydantic.LanceModel`.
  * Integrates HuggingFace's `BAAI/bge-small-en-v1.5` text-embedding model directly into the LanceDB schema registry.
  * Ingests parsed handbook markdown chunks into LanceDB tables for low-latency dense vector search.

### Section 4: Tool Definitions & Integrations
* **Objective**: Equip the LLM agent with deterministic, real-world execution tools annotated with `@tool`.
* **Implementation**:
  * `search_flights`: Connects to Amadeus Flight Offers Search API with dynamic city-to-IATA code translation.
  * `search_hotels`: Queries Amadeus Hotel Search with a fallback mechanism using Tavily Web Search when room availability is limited.
  * `search_restaurants`: Uses Tavily API to fetch real-time local dining options, user ratings, and culinary specialties.
  * `get_weather_info`: Queries live meteorological data via Tavily Search API for requested destinations.
  * `get_currency_info`: Computes conversion rates between global currencies using historical and live exchange data.
  * `search_faq`: Executes similarity queries against the LanceDB FAQ table.
  * `plan_trip`: Synthesizes multi-day travel schedules using date interval calculations and vector context.

### Section 5: Agent Architecture & State Orchestration (LangGraph)
* **Objective**: Construct a state-driven computational graph to handle tool selection, memory, and execution branching.
* **Implementation**:
  * Sets up a `StateGraph` containing a central `reasoner_node` bound to all 7 tools.
  * Uses `tools_condition` to route control dynamically between tool calls and final user outputs.
  * Integrates thread checkpointers to ensure multi-turn conversational context preservation.

### Section 6: Interactive Terminal Interface (CLI)
* **Objective**: Provide a user-friendly command-line environment for interacting with TravelBot.
* **Implementation**:
  * Implements `chat_cli()`, a REPL loop handling session IDs, input sanitization, and streaming response delivery.
  * Supports interactive system commands such as `exit` and `clear`.

---

## In-Depth Analysis of 14 Test Scenarios

Below is the comprehensive translation and technical analysis of the 14 execution test scenarios from the notebook evaluation section:

### Test 1: Direct Flight Search
* **Query**: "Find flights from Tehran to Dubai on May 10, 2025."
* **Tool Invoked**: `search_flights`
* **Result**: The agent successfully mapped "Tehran" to `IKA` and "Dubai" to `DXB`, queried Amadeus API, and returned real flight options with pricing and departure times.
* **Analysis**: Demonstrates reliable entity recognition and automatic city-to-IATA translation without requiring the user to provide raw airport codes.

### Test 2: Hotel Search with Automatic Fallback
* **Query**: "Look for hotels in Paris for mid-June."
* **Tool Invoked**: `search_hotels` (Amadeus API $\rightarrow$ Tavily Web Search Fallback)
* **Result**: When Amadeus API returned restricted choices for the requested window, the tool automatically triggered Tavily Web Search to aggregate top-rated Paris hotels.
* **Analysis**: Proves system robustness and fault-tolerance against external API rate limits or sparse primary database results.

### Test 3: Local Restaurant Discovery
* **Query**: "What are the best Italian restaurants in Rome?"
* **Tool Invoked**: `search_restaurants`
* **Result**: Tavily API retrieved a curated list of popular trattorias in Rome, complete with cuisine specialties and address details.
* **Analysis**: Shows effective web search utilization for non-structured domain queries requiring up-to-date recommendations.

### Test 4: Real-Time Weather Forecasting
* **Query**: "Check current weather and forecast for Tokyo."
* **Tool Invoked**: `get_weather_info`
* **Result**: The tool successfully returned live temperatures, precipitation percentages, and general weather outlooks for Tokyo.
* **Analysis**: Verifies the agent's capability to supply critical ambient environmental data needed for travel planning.

### Test 5: Financial Currency Conversion
* **Query**: "Convert 500 USD to EUR and show the exchange rate."
* **Tool Invoked**: `get_currency_info`
* **Result**: Calculated the exact conversion value and provided current exchange rate context.
* **Analysis**: Confirms numerical precision and deterministic tool usage for financial calculations rather than relying on LLM estimation.

### Test 6: Semantic Handbook FAQ Search
* **Query**: "What are the visa rules and baggage allowance policies?"
* **Tool Invoked**: `search_faq`
* **Result**: The query was embedded via `BAAI/bge-small-en-v1.5` and matched against vector records in LanceDB, returning accurate handbook passages.
* **Analysis**: Highlights semantic vector search capability, delivering factual answers grounded strictly in source documentation.

### Test 7: Structured Multi-Day Itinerary Generation
* **Query**: "Plan a 4-day trip to Istanbul focused on historical landmarks."
* **Tool Invoked**: `plan_trip`
* **Result**: Generated a day-by-day itinerary dividing historic attractions logically across morning, afternoon, and evening slots.
* **Analysis**: Demonstrates long-form structured content synthesis combining domain context with logical time allocation.

### Test 8: Multi-Tool Chained Execution Workflow
* **Query**: "I want to visit Barcelona from London next month. Search flights, suggest hotels, and give me the weather forecast."
* **Tools Invoked**: `search_flights`, `search_hotels`, `get_weather_info`
* **Result**: The agent generated a multi-step execution plan, called all three tools in sequence, and merged their outputs into a single cohesive response.
* **Analysis**: Validates LangGraph's state graph capabilities for multi-step reasoning and multi-tool coordination within a single prompt cycle.

### Test 9: Typographical Error & Fuzzy Matching Resilience
* **Query**: "Find flights from 'Teheran' to 'Instanbul'."
* **Tools Invoked**: Data Normalization Pipeline + `search_flights`
* **Result**: `difflib` corrected the misspelled names to "Tehran" (`IKA`) and "Istanbul" (`IST`), allowing the flight tool to run successfully.
* **Analysis**: Shows robust input sanitization and protection against common user spelling mistakes.

### Test 10: Out-of-Domain Query Handling
* **Query**: "How do I book a ticket to Mars on a space shuttle?"
* **Tools Invoked**: `search_faq` / Reasoning Node Guardrails
* **Result**: Vector search returned negligible similarity scores, leading the agent to politely state that space travel falls outside its current service scope.
* **Analysis**: Confirms effective guardrails against hallucinations and out-of-scope query processing.

### Test 11: Conversational Memory & Multi-turn Dialogue
* **User Turn 1**: "Find flights to Munich."
* **User Turn 2**: "What is the weather like there during that week?"
* **Tools Invoked**: Thread Memory Checkpointer + `get_weather_info`
* **Result**: The agent inferred that "there" referred to Munich from Turn 1 and fetched the appropriate weather data without asking for clarification.
* **Analysis**: Demonstrates state retention and context propagation across multi-turn user interactions.

### Test 12: API Rate-Limit Handling & Graceful Degradation
* **Query**: Executing high-frequency repeated flight and hotel requests.
* **Tools Invoked**: System Exception Wrappers
* **Result**: Tool wrappers caught API exception codes (e.g., HTTP 429) and displayed clean fallback messages without crashing the CLI application.
* **Analysis**: Highlights production readiness and resilience under network constraints or API outages.

### Test 13: Constraint-Based Budget Planning
* **Query**: "Create a 3-day low-budget itinerary for Prague."
* **Tools Invoked**: `plan_trip`, `search_restaurants`
* **Result**: Recommendations prioritized free cultural sites, public transit guidance, and affordable local eateries.
* **Analysis**: Shows the agent's ability to adjust tool outputs based on qualitative user constraints like budget level.

### Test 14: Comprehensive End-to-End Travel Package Assembly
* **Query**: "Organize a complete 7-day trip to Tokyo including flights, hotels, dining, currency conversion, and a daily itinerary."
* **Tools Invoked**: All 7 system tools invoked across multi-step execution.
* **Result**: TravelBot executed a full orchestration sequence, returning a unified executive summary covering travel costs, weather, flight schedules, hotel choices, and a day-by-day plan.
* **Analysis**: Represents the peak capability of the agentic framework—converting complex high-level objectives into structured, fully actionable outcomes.

---

## Tech Stack Summary

| Domain | Technology |
| --- | --- |
| **Agent Framework** | LangChain / LangGraph |
| **LLM Engine** | OpenAI GPT-4o / GPT-3.5-Turbo |
| **Vector Database** | LanceDB |
| **Embedding Model** | HuggingFace (`BAAI/bge-small-en-v1.5`) |
| **PDF Processing** | LlamaParse (LlamaIndex) |
| **External APIs** | Amadeus API (Flights/Hotels), Tavily API (Search/Weather) |
| **Data Processing** | Pandas, Pydantic, Difflib, Python-Dotenv |

---

## Installation & Setup Guide

### 1. Clone the Repository
```bash
git clone https://github.com/hrnm2003/ai-travel-assistant.git
cd ai-travel-assistant
```

### 2. Configure Environment Variables
Create a `.env` file in the root directory:
```env
OPENAI_API_KEY=your_openai_api_key
AMADEUS_CLIENT_ID=your_amadeus_client_id
AMADEUS_CLIENT_SECRET=your_amadeus_client_secret
TAVILY_API_KEY=your_tavily_api_key
LLAMA_CLOUD_API_KEY=your_llama_cloud_api_key
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Run the Project
Launch the interactive Jupyter notebook or run the CLI interface:
```bash
jupyter notebook travel_bot.ipynb
```