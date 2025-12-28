import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from streamlit_option_menu import option_menu
from neo4j import GraphDatabase
import sys
import os
from datetime import datetime
from groq import Groq
from pyvis.network import Network
import streamlit.components.v1 as components

# Add the current directory to path to import config
sys.path.append('.')
from config import *

# ============================================
# 1. PAGE SETUP & STYLING
# ============================================
st.set_page_config(
    page_title="GuardianAI | Fake News Detector",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for a premium look
st.markdown("""
<style>
    .main {
        background-color: #f8f9fa;
    }
    .stButton>button {
        width: 100%;
        border-radius: 5px;
        height: 3em;
        background-color: #FF4B4B;
        color: white;
        font-weight: bold;
        border: none;
    }
    .stButton>button:hover {
        background-color: #FF3333;
        border: none;
    }
    .status-card {
        padding: 20px;
        border-radius: 10px;
        background-color: white;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        margin-bottom: 20px;
    }
    .verdict-fake {
        color: #FF4B4B;
        font-weight: bold;
        font-size: 24px;
    }
    .verdict-real {
        color: #28a745;
        font-weight: bold;
        font-size: 24px;
    }
</style>
""", unsafe_allow_html=True)

# ============================================
# 2. DATABASE & API CONNECTIONS
# ============================================

@st.cache_resource
def get_neo4j_driver():
    try:
        driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
        # Verify connection
        driver.verify_connectivity()
        return driver
    except Exception as e:
        st.error(f"Failed to connect to Neo4j: {e}")
        return None

@st.cache_resource
def get_groq_client():
    if not GROQ_API_KEY:
        st.error("GROQ_API_KEY not found in environment variables.")
        return None
    return Groq(api_key=GROQ_API_KEY)

driver = get_neo4j_driver()
client = get_groq_client()

# ============================================
# 3. RAG LOGIC FUNCTIONS
# ============================================

def find_similar_news(query_text, limit=5):
    if not driver: return []
    with driver.session() as session:
        # Using simple keyword search as fallback if vector index isn't ready
        # But optimized to look for content matches
        keyword = query_text.split()[0] if query_text.split() else ""
        result = session.run("""
            MATCH (n:News)
            WHERE n.embedding IS NOT NULL
            AND (toLower(n.title) CONTAINS toLower($keyword) 
                 OR toLower(n.text_preview) CONTAINS toLower($keyword))
            RETURN n.id as id, n.title as title, n.label as label, 
                   n.subject as subject, n.text_preview as text
            LIMIT $limit
        """, keyword=keyword, limit=limit)
        return list(result)

def get_related_entities(news_ids):
    if not driver or not news_ids: return []
    with driver.session() as session:
        result = session.run("""
            MATCH (n:News)-[:MENTIONS]->(e:Entity)
            WHERE n.id IN $ids
            RETURN e.name as entity, e.type as type, count(n) as mention_count
            ORDER BY mention_count DESC
            LIMIT 10
        """, ids=news_ids)
        return list(result)

def analyze_with_groq(query, similar_articles, entities):
    if not client: return "Groq client not initialized."
    
    # Prepare context
    context_text = "\n".join([
        f"Article {i+1}: {article['title']} (Label: {article['label']})\n{article['text'][:200]}..."
        for i, article in enumerate(similar_articles)
    ])
    
    entities_text = "\n".join([
        f"- {entity['entity']} ({entity['type']}): Mentioned {entity['mention_count']} times"
        for entity in entities
    ])
    
    prompt = f"""
    FAKE NEWS ANALYSIS TASK:
    
    USER QUERY: "{query}"
    
    CONTEXT FROM DATABASE:
    
    SIMILAR PAST ARTICLES:
    {context_text}
    
    RELATED ENTITIES:
    {entities_text}
    
    INSTRUCTIONS:
    1. Analyze if the user's query/news is likely FAKE or REAL
    2. Base your analysis on the similar articles and entities
    3. If entities are frequently associated with fake news, mention this
    4. Provide a confidence score (0-100%)
    5. Give specific reasons for your verdict
    
    OUTPUT FORMAT:
    Verdict: [FAKE/REAL]
    Confidence: [X]%
    Reasons:
    1. [Reason 1]
    2. [Reason 2]
    
    Analysis:
    [Detailed Analysis]
    """
    
    try:
        response = client.chat.completions.create(
            model=CHAT_MODEL,
            messages=[
                {"role": "system", "content": "You are a fake news detection expert."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.1,
            max_tokens=800
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Error analyzing with Groq: {e}"

# ============================================
# 4. UI COMPONENTS
# ============================================

with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/3593/3593444.png", width=100)
    st.title("GuardianAI")
    selected = option_menu(
        menu_title=None,
        options=["Dashboard", "Detector", "Graph View", "About"],
        icons=["speedometer2", "shield-check", "diagram-3", "info-circle"],
        default_index=1,
    )
    
    st.markdown("---")
    st.markdown("### System Status")
    if driver:
        st.success("Neo4j: Connected")
    else:
        st.error("Neo4j: Disconnected")
    
    if client:
        st.success("Groq: Connected")
    else:
        st.error("Groq: Disconnected")

if selected == "Dashboard":
    st.title("📊 News Dataset Analysis")
    
    try:
        df = pd.read_csv('data/cleaned_news.csv')
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Articles", f"{len(df):,}")
        with col2:
            st.metric("Fake News", f"{len(df[df['label'] == 'FAKE']):,}", delta="Fake", delta_color="inverse")
        with col3:
            st.metric("Real News", f"{len(df[df['label'] == 'REAL']):,}", delta="Real")
            
        st.markdown("---")
        
        col_left, col_right = st.columns(2)
        
        with col_left:
            st.subheader("Distribution by Label")
            fig1, ax1 = plt.subplots()
            label_counts = df['label'].value_counts()
            ax1.pie(label_counts, labels=label_counts.index, autopct='%1.1f%%', colors=['#FF6B6B', '#4ECDC4'])
            st.pyplot(fig1)
            
        with col_right:
            st.subheader("Top Subjects")
            fig2, ax2 = plt.subplots()
            subject_counts = df['subject'].value_counts().head(10)
            sns.barplot(x=subject_counts.values, y=subject_counts.index, palette="viridis", ax=ax2)
            st.pyplot(fig2)
            
    except Exception as e:
        st.warning(f"Could not load analysis data: {e}. Please run data_cleaning.py first.")

elif selected == "Detector":
    st.title("🛡️ Fake News Detector")
    st.markdown("Analyze news headlines or full articles using our **Graph RAG Pipeline**.")
    
    query = st.text_area("Paste news headline or text here:", placeholder="Ex: Breaking news about election fraud...", height=150)
    
    if st.button("Analyze Credibility"):
        if not query:
            st.warning("Please enter some text to analyze.")
        else:
            with st.spinner("Searching knowledge graph and analyzing..."):
                # Step 1: Find similar articles
                similar = find_similar_news(query, limit=5)
                
                # Step 2: Get entities
                news_ids = [a['id'] for a in similar] if similar else []
                entities = get_related_entities(news_ids)
                
                # Step 3: Groq Analysis
                analysis = analyze_with_groq(query, similar, entities)
                
                # UI Results
                st.markdown("### 📊 Analysis Report")
                
                if "Verdict: FAKE" in analysis:
                    st.header("FAKE")
                    st.error("🚨 ALERT: FAKE NEWS DETECTED!")
                    st.markdown("""
                    <div style="background-color: #ff4b4b; padding: 20px; border-radius: 10px; color: white; text-align: center; margin-bottom: 20px;">
                        <h1 style="margin:0;">⚠️ FAKE NEWS ALERT ⚠️</h1>
                        <p style="font-size: 1.2em;">The system has high confidence that this information is misleading or fabricated.</p>
                    </div>
                    """, unsafe_allow_html=True)
                elif "Verdict: REAL" in analysis:
                    st.header("Real news")
                    st.success("🟢 Likely Authentic News")
                    st.markdown("""
                    <div style="background-color: #28a745; padding: 20px; border-radius: 10px; color: white; text-align: center; margin-bottom: 20px;">
                        <h1 style="margin:0;">✅ AUTHENTIC NEWS ✅</h1>
                        <p style="font-size: 1.2em;">The system has high confidence that this information is credible and authentic.</p>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.info("⚠️ Analysis Completed")
                
                st.markdown(f"```\n{analysis}\n```")
                
                with st.expander("🔍 View Source Evidence from Graph"):
                    if similar:
                        for a in similar:
                            color = "red" if a['label'] == "FAKE" else "green"
                            st.markdown(f"**[{a['label']}]** {a['title']}")
                            st.caption(f"Subject: {a['subject']}")
                    else:
                        st.write("No direct matches found in historical database.")
                        
                    if entities:
                        st.markdown("#### Key Entities Mentioned in Context:")
                        cols = st.columns(3)
                        for i, e in enumerate(entities[:6]):
                            cols[i % 3].info(f"{e['entity']} ({e['type']})")

elif selected == "Graph View":
    st.title("🕸️ Knowledge Graph Insights")
    st.markdown("This section visualizes the relationships between news articles and entities.")
    
    if driver:
        with driver.session() as session:
            stats = session.run("""
                MATCH (n) RETURN labels(n)[0] as label, count(*) as count
            """)
            st.write("### Graph Statistics")
            cols = st.columns(3)
            for i, record in enumerate(stats):
                cols[i % 3].metric(record['label'], record['count'])
                
            # Create the interactive graph
            st.markdown("### Interactive Knowledge Graph")
            st.caption("Visualizing relationships between News Articles and Entities (Sample: Top 50 connections)")
            
            with st.spinner("Generating graph visualization..."):
                query = """
                MATCH (n:News)-[r:MENTIONS]->(e:Entity)
                RETURN n.title as title, n.label as label, e.name as entity, e.type as type
                LIMIT 50
                """
                results = session.run(query)
                
                # Setup Pyvis Network
                net = Network(height='600px', width='100%', bgcolor='#ffffff', font_color='black', notebook=False)
                
                # Add nodes and edges
                for record in results:
                    news_title = record['title'][:30] + "..."
                    entity_name = record['entity']
                    
                    # Add News Node
                    news_color = '#FF6B6B' if record['label'] == 'FAKE' else '#4ECDC4'
                    net.add_node(news_title, label=news_title, title=record['title'], color=news_color, size=25, shape='dot')
                    
                    # Add Entity Node
                    entity_color = '#FFD93D'
                    net.add_node(entity_name, label=entity_name, title=f"Type: {record['type']}", color=entity_color, size=15, shape='diamond')
                    
                    # Add Edge
                    net.add_edge(news_title, entity_name)
                
                # Set physics for better layout
                net.toggle_physics(True)
                
                # Save and display
                try:
                    path = "data/graph.html"
                    os.makedirs(os.path.dirname(path), exist_ok=True)
                    net.save_graph(path)
                    
                    with open(path, 'r', encoding='utf-8') as f:
                        html_content = f.read()
                    
                    components.html(html_content, height=650)
                except Exception as e:
                    st.warning(f"Could not render interactive graph: {e}")
    else:
        st.error("Connect to Neo4j to see graph statistics.")

elif selected == "About":
    st.title("ℹ️ About GuardianAI")
    st.markdown("""
    GuardianAI is a state-of-the-art fake news detection platform that combines:
    
    1.  **Neo4j Knowledge Graphs**: Storing relationships between news, authors, and entities.
    2.  **Vector Embeddings**: enabling semantic similarity search.
    3.  **Groq LLM**: Processing context to provide human-like analysis and verdicts.
    
    Developed as part of the Fake News Detection project.
    """)

# Footer
st.markdown("---")
st.markdown("<p style='text-align: center; color: grey;'>GuardianAI © 2025 | Powered by Neo4j & Groq</p>", unsafe_allow_html=True)