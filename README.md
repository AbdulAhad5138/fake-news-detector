# 🛡️ GuardianAI: Fake News Detection System

GuardianAI is a professional-grade Fake News Detection platform leveraging **Graph RAG (Retrieval-Augmented Generation)**. It combines the power of **Neo4j Knowledge Graphs**, **Vector Embeddings**, and **LLMs (Groq)** to provide deeply contextualized news analysis.

## ✨ Features
*   **📊 Dataset Dashboard**: Interactive visualizations of news distributions and subjects.
*   **🛡️ RAG-Powered Detector**: Analyzes news credibility based on historical data and entity relationships.
*   **🕸️ Knowledge Graph**: Interactive 3D visualization of connections between articles and entities.
*   **🚨 Real-time Alerts**: Prominent visual indicators for "FAKE" and "Authentic" news.

## 🚀 Getting Started

### 1. Installation
Install the required dependencies:
```bash
pip install -r requirements.txt
```

### 2. Configuration
Create a `.env` file in the root directory with your credentials:
```env
NEO4J_URI=bolt://your-neo4j-uri:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=your-password
GROQ_API_KEY=your-groq-api-key
```

### 3. Usage
Run the Streamlit application:
```bash
streamlit run app.py
```

## 📁 Project Structure
*   `app.py`: Main Streamlit web application.
*   `config.py`: Centralized configuration management.
*   `scripts/`: Backend pipeline scripts for data preparation and cleaning.
*   `data/`: Storage for datasets, processed graph files, and visualizations.

## 🛠️ Technology Stack
*   **Frontend**: Streamlit
*   **Graph Database**: Neo4j
*   **LLM**: Groq (Llama 3.3 70B)
*   **Logic**: Python (Pandas, NumPy, Scikit-learn)
*   **Visualization**: Matplotlib, Seaborn, Pyvis

---
*GuardianAI © 2025 | Built for Precision News Analysis*
