#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Interactive Visa Discussion Analyzer

This script provides a comprehensive dashboard for analyzing threaded discussions,
such as Reddit comments. It's designed for robustness, interactivity, and a
professional user experience.

Features:
- Loads nested JSON data from file uploads.
- Recursively extracts comments and builds a reply network.
- Performs sentiment analysis using state-of-the-art Hugging Face Transformers
  with a fallback to TextBlob for graceful degradation.
- Conducts topic modeling with BERTopic, falling back to Gensim LDA.
- Generates interactive visualizations using Plotly.
- Presents all findings in a clean, interactive Streamlit dashboard.

To run this application, save the code as a Python file (e.g., 'app.py') and run:
`streamlit run app.py`

Dependencies:
- streamlit
- pandas
- numpy
- matplotlib
- seaborn
- plotly
- transformers
- torch
- bertopic
- textblob
- gensim
- wordcloud
- networkx
"""

import os
import sys
import json
import traceback
from typing import List, Tuple, Dict, Any, Optional

# --- Data & Visualization Libraries ---
import pandas as pd
import numpy as np

# Static plotting fallback
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

# Interactive viz (requires plotly)
try:
    import plotly.express as px
    import plotly.graph_objects as go
    PLOTLY_AVAILABLE = True
except Exception:
    px = None
    go = None
    PLOTLY_AVAILABLE = False
    print("Warning: Plotly not installed. Falling back to static plots.")

# NLP & ML Libraries
def _safe_import_transformers():
    try:
        from transformers import pipeline
        return pipeline
    except Exception:
        return None

TRANSFORMERS_PIPELINE = _safe_import_transformers()
if TRANSFORMERS_PIPELINE is None:
    print("Warning: Hugging Face Transformers not installed. Sentiment analysis will use TextBlob.")

try:
    from textblob import TextBlob
except Exception:
    TextBlob = None
    if TRANSFORMERS_PIPELINE is None:
        print("Warning: TextBlob not installed. Sentiment analysis will be unavailable.")

def _safe_import_bertopic():
    try:
        from bertopic import BERTopic
        return BERTopic
    except Exception:
        return None

BERTopic = _safe_import_bertopic()
if BERTopic is None:
    print("Warning: BERTopic not installed. Falling back to Gensim LDA for topic modeling.")

try:
    import gensim
    from gensim import corpora
    from gensim.models.ldamodel import LdaModel
except Exception:
    gensim = None
    corpora = None
    LdaModel = None
    if BERTopic is None:
        print("Warning: Gensim not installed. Topic modeling will be unavailable.")

# Wordcloud (optional)
try:
    from wordcloud import WordCloud
    WORDCLOUD_AVAILABLE = True
except Exception:
    WordCloud = None
    WORDCLOUD_AVAILABLE = False
    print("Warning: WordCloud not installed. Wordcloud visualization will be unavailable.")

# Graph
try:
    import networkx as nx
    NETWORKX_AVAILABLE = True
except Exception:
    nx = None
    NETWORKX_AVAILABLE = False
    print("Warning: NetworkX not installed. Conversation graph will be unavailable.")

# Streamlit
try:
    import streamlit as st
    STREAMLIT_AVAILABLE = True
except Exception:
    st = None
    STREAMLIT_AVAILABLE = False
    print("Warning: Streamlit not installed. UI will not launch.")


# --------------------------------------------------------------------------------------
# Core Utility Functions
# --------------------------------------------------------------------------------------

OUTPUT_DIR = os.path.join('outputs')
PLOTS_DIR = os.path.join(OUTPUT_DIR, 'plots')

def ensure_dirs() -> None:
    """Ensures necessary output directories exist."""
    for d in [OUTPUT_DIR, PLOTS_DIR]:
        os.makedirs(d, exist_ok=True)

def extract_comments_recursive(data: List[Dict[str, Any]], out_list: List[Dict[str, Any]], source: str,
                              parent_author: Optional[str] = None, edges: Optional[List[Tuple[str, str]]] = None) -> None:
    """
    Recursively traverses a nested JSON structure to extract comments and build a reply graph.
    
    Args:
        data: The current list of comments to process.
        out_list: The list to append extracted comment dictionaries to.
        source: A label indicating the source file (e.g., 'file1').
        parent_author: The author of the parent comment, used for building reply edges.
        edges: The list to append (source_author, target_author) tuples to.
    """
    for item in data:
        comment = (item.get('comment') or '').strip()
        author = (item.get('author') or 'unknown').strip() or 'unknown'
        if comment:
            out_list.append({'comment': comment, 'author': author, 'source': source})
            if parent_author and edges is not None and author != 'deleted':
                edges.append((parent_author, author))
        
        replies = item.get('replies') or []
        if isinstance(replies, list) and replies:
            extract_comments_recursive(replies, out_list, source, parent_author=author, edges=edges)

def load_json_safely(data: Any, name: str) -> List[Dict[str, Any]]:
    """Loads JSON data from a Streamlit uploaded file object with error handling."""
    try:
        if not isinstance(data, list):
            raise ValueError(f"The root of '{name}' must be a list of objects.")
        return data
    except Exception as e:
        raise RuntimeError(f"Failed to read '{name}': {e}")

def build_dataframe(file1_data: Any, file2_data: Any) -> Tuple[pd.DataFrame, List[Tuple[str, str]]]:
    """Processes raw JSON data into a clean DataFrame and a list of reply edges."""
    data1 = load_json_safely(file1_data, 'file1')
    data2 = load_json_safely(file2_data, 'file2')
    
    comments: List[Dict[str, Any]] = []
    edges: List[Tuple[str, str]] = []
    
    extract_comments_recursive(data1, comments, 'file1', edges=edges)
    extract_comments_recursive(data2, comments, 'file2', edges=edges)
    
    df = pd.DataFrame(comments)
    if df.empty:
        raise ValueError('No comments extracted. Ensure the JSON files contain data.')
        
    # Clean up and normalize author names
    df['author'] = df['author'].str.strip().fillna('unknown')
    
    return df, edges


# --------------------------------------------------------------------------------------
# Sentiment Analysis
# --------------------------------------------------------------------------------------

def run_sentiment(df: pd.DataFrame, text_col: str = 'comment') -> pd.DataFrame:
    """
    Performs sentiment analysis on the comments, with a robust fallback system.
    
    Args:
        df: The DataFrame containing comments.
        text_col: The name of the column with the text to analyze.
        
    Returns:
        The DataFrame with 'sentiment_label' and 'sentiment_score' columns added.
    """
    st.info("🔬 Running sentiment analysis...")
    df_copy = df.copy()
    labels: List[str] = []
    scores: List[float] = []

    if TRANSFORMERS_PIPELINE:
        try:
            analyzer = TRANSFORMERS_PIPELINE('sentiment-analysis', model='distilbert-base-uncased-finetuned-sst-2-english')
            # Batching to prevent potential Out-of-Memory errors with large datasets
            texts = df_copy[text_col].astype(str).tolist()
            batch_size = 64
            for i in range(0, len(texts), batch_size):
                batch = texts[i:i+batch_size]
                preds = analyzer(batch, truncation=True)
                for p in preds:
                    labels.append(p.get('label', 'NEUTRAL'))
                    scores.append(float(p.get('score', 0.0)))
            st.success("✅ Sentiment analysis completed with Hugging Face Transformers.")
            df_copy['sentiment_label'] = labels
            df_copy['sentiment_score'] = scores
            return df_copy
        except Exception as e:
            st.warning(f"Transformers failed: {e}. Falling back to TextBlob.")
    
    # Fallback to TextBlob or Neutral if primary fails
    labels, scores = _fallback_textblob(df_copy[text_col])
    if TextBlob is not None:
        st.success("✅ Sentiment analysis completed with TextBlob.")
    else:
        st.warning("⚠️ Sentiment analysis library not found. All comments labeled as 'NEUTRAL'.")
        
    df_copy['sentiment_label'] = labels
    df_copy['sentiment_score'] = scores
    return df_copy

def _fallback_textblob(series: pd.Series) -> Tuple[List[str], List[float]]:
    """Simple sentiment analysis using TextBlob for fallback."""
    labels: List[str] = []
    scores: List[float] = []
    if TextBlob is None:
        return ['NEUTRAL'] * len(series), [0.0] * len(series)
    
    for txt in series.astype(str).tolist():
        try:
            pol = TextBlob(txt).sentiment.polarity
            scores.append(float(abs(pol)))
            labels.append('POSITIVE' if pol > 0.05 else ('NEGATIVE' if pol < -0.05 else 'NEUTRAL'))
        except Exception:
            labels.append('NEUTRAL')
            scores.append(0.0)
    return labels, scores

# --------------------------------------------------------------------------------------
# Topic Modeling
# --------------------------------------------------------------------------------------

def run_topics(df: pd.DataFrame, text_col: str = 'comment') -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Performs topic modeling, falling back from BERTopic to Gensim LDA.
    
    Returns:
        A tuple of (DataFrame with topic IDs, DataFrame with topic information).
    """
    st.info("🧠 Running topic modeling...")
    comments = df[text_col].astype(str).tolist()
    df_copy = df.copy()

    if BERTopic:
        try:
            model = BERTopic(language='english', calculate_probabilities=True, verbose=False)
            topics, _ = model.fit_transform(comments)
            df_copy['topic'] = topics
            info = model.get_topic_info()
            st.success("✅ Topic modeling completed with BERTopic.")
            return df_copy, info
        except Exception as e:
            st.warning(f"BERTopic failed: {e}. Falling back to Gensim LDA.")

    # Fallback to LDA if BERTopic is unavailable or fails
    if gensim and corpora and LdaModel:
        try:
            from gensim.utils import simple_preprocess
            tokenized = [simple_preprocess(t, min_len=3) for t in comments]
            dictionary = corpora.Dictionary(tokenized)
            corpus = [dictionary.doc2bow(toks) for toks in tokenized]
            num_topics = max(2, min(10, int(np.sqrt(len(df_copy)))))
            lda = LdaModel(corpus=corpus, id2word=dictionary, num_topics=num_topics, random_state=42, passes=10)
            
            topics_idx = []
            for bow in corpus:
                dist = lda.get_document_topics(bow)
                if dist:
                    topics_idx.append(int(max(dist, key=lambda x: x[1])[0]))
                else:
                    topics_idx.append(-1)
            
            df_copy['topic'] = topics_idx
            counts = pd.Series(topics_idx).value_counts().sort_index()
            info = pd.DataFrame({'Topic': counts.index, 'Count': counts.values})
            info['Name'] = info['Topic'].apply(lambda i: f"LDA Topic {i}")
            st.success("✅ Topic modeling completed with Gensim LDA.")
            return df_copy, info
        except Exception as e:
            st.error(f"Gensim LDA also failed: {e}.")

    # Final fallback if all else fails
    df_copy['topic'] = -1
    info = pd.DataFrame({'Topic': [-1], 'Count': [len(df_copy)], 'Name': ['No Topic (fallback)']})
    st.warning("⚠️ No topic modeling library found. No topic analysis performed.")
    return df_copy, info


# --------------------------------------------------------------------------------------
# Visualization Helpers
# --------------------------------------------------------------------------------------

def build_plotly_figures(df: pd.DataFrame) -> Dict[str, go.Figure]:
    """Generates a dictionary of interactive Plotly figures."""
    figs = {}
    if not PLOTLY_AVAILABLE:
        return figs

    try:
        # Overall sentiment bar chart
        s_counts = df['sentiment_label'].value_counts().reset_index()
        s_counts.columns = ['sentiment', 'count']
        figs['sentiment_bar'] = px.bar(s_counts, x='sentiment', y='count', color='sentiment',
                                       title='Overall Sentiment Distribution',
                                       color_discrete_map={'POSITIVE': 'green', 'NEGATIVE': 'red', 'NEUTRAL': 'gray'})

        # Sentiment by source file
        by_src = df.groupby(['source','sentiment_label']).size().reset_index(name='count')
        figs['sentiment_by_source'] = px.bar(by_src, x='source', y='count', color='sentiment_label', barmode='group',
                                             title='Sentiment Distribution by Source File',
                                             color_discrete_map={'POSITIVE': 'green', 'NEGATIVE': 'red', 'NEUTRAL': 'gray'})

        # Sentiment score distribution
        figs['score_violin'] = px.violin(df, x='source', y='sentiment_score', color='sentiment_label', box=True, points='all',
                                         title='Sentiment Score Distribution',
                                         color_discrete_map={'POSITIVE': 'green', 'NEGATIVE': 'red', 'NEUTRAL': 'gray'})
        
        # Topic by sentiment (if topics are present)
        if 'topic' in df.columns:
            tmp = df.groupby(['topic','sentiment_label']).size().reset_index(name='count')
            figs['topic_sentiment'] = px.bar(tmp, x='topic', y='count', color='sentiment_label', title='Sentiment by Topic', barmode='stack',
                                             color_discrete_map={'POSITIVE': 'green', 'NEGATIVE': 'red', 'NEUTRAL': 'gray'})
    except Exception as e:
        st.error(f"An error occurred while building Plotly figures: {e}")
        figs = {}
    return figs

def _generate_wordclouds(df: pd.DataFrame, pos_path: str, neg_path: str) -> None:
    """Generates and saves wordclouds for positive and negative comments."""
    if not WORDCLOUD_AVAILABLE:
        return
    
    try:
        pos_texts = df.loc[df['sentiment_label'] == 'POSITIVE', 'comment'].astype(str).tolist()
        neg_texts = df.loc[df['sentiment_label'] == 'NEGATIVE', 'comment'].astype(str).tolist()

        if pos_texts:
            wc = WordCloud(width=800, height=400, background_color='white').generate(' '.join(pos_texts))
            plt.imshow(wc, interpolation='bilinear')
            plt.axis('off')
            plt.tight_layout()
            plt.savefig(pos_path, dpi=150)
            plt.close()
        else:
            if os.path.exists(pos_path): os.remove(pos_path)

        if neg_texts:
            wc = WordCloud(width=800, height=400, background_color='white').generate(' '.join(neg_texts))
            plt.imshow(wc, interpolation='bilinear')
            plt.axis('off')
            plt.tight_layout()
            plt.savefig(neg_path, dpi=150)
            plt.close()
        else:
            if os.path.exists(neg_path): os.remove(neg_path)
    except Exception as e:
        st.warning(f'Wordcloud generation failed: {e}.')
        traceback.print_exc()

def _draw_graph_streamlit(df: pd.DataFrame, edges: List[Tuple[str, str]]) -> None:
    """Draws a conversation graph using NetworkX and Streamlit."""
    if not NETWORKX_AVAILABLE:
        st.warning('NetworkX is not installed. Conversation graph visualization is unavailable.')
        return
    
    try:
        st.subheader("Conversation Network Graph")
        st.markdown("This graph visualizes the reply network. An arrow from A to B means A replied to a comment by B.")
        
        G = nx.DiGraph()
        # Add all unique authors as nodes
        all_authors = set(df['author'].tolist())
        G.add_nodes_from(all_authors)
        
        # Add edges based on the extracted replies
        for src, tgt in edges:
            G.add_edge(src, tgt)

        # Plotting the graph
        fig, ax = plt.subplots(figsize=(15, 10))
        pos = nx.spring_layout(G, seed=42, k=0.5)
        
        # Determine node colors based on sentiment
        sentiment_map = df.set_index('author')['sentiment_label'].to_dict()
        node_colors = [
            'lightgreen' if sentiment_map.get(node, 'NEUTRAL') == 'POSITIVE' else 
            'salmon' if sentiment_map.get(node, 'NEUTRAL') == 'NEGATIVE' else 
            'lightgray' 
            for node in G.nodes()
        ]
        
        nx.draw_networkx_nodes(G, pos, node_size=500, node_color=node_colors, alpha=0.8, ax=ax)
        nx.draw_networkx_edges(G, pos, arrows=True, arrowstyle="->", alpha=0.6, ax=ax)
        nx.draw_networkx_labels(G, pos, font_size=8, font_color="black", ax=ax)
        
        ax.set_title("Conversation Network (Authors)", fontsize=18, fontweight='bold')
        ax.axis("off")
        st.pyplot(fig)
        
    except Exception as e:
        st.warning(f'Conversation graph visualization failed: {e}')
        st.info("The conversation graph requires the `networkx` and `matplotlib` libraries.")
        traceback.print_exc()

# --------------------------------------------------------------------------------------
# Pipeline Orchestrator
# --------------------------------------------------------------------------------------

@st.cache_data
def run_pipeline(file1_data: Any, file2_data: Any) -> Tuple[pd.DataFrame, pd.DataFrame, List[Tuple[str, str]]]:
    """
    Orchestrates the entire analysis workflow.
    
    Args:
        file1_data: Raw data from the first JSON file.
        file2_data: Raw data from the second JSON file.
        
    Returns:
        A tuple of (DataFrame with all analysis results, topic info DataFrame, reply edges list).
    """
    ensure_dirs()
    df, edges = build_dataframe(file1_data, file2_data)

    df = run_sentiment(df)
    df, topic_info = run_topics(df)

    # Persist results to CSV for download
    df.to_csv(os.path.join(OUTPUT_DIR, 'comments_with_sentiment_topics.csv'), index=False)
    topic_info.to_csv(os.path.join(OUTPUT_DIR, 'topic_info.csv'), index=False)
    
    return df, topic_info, edges

# --------------------------------------------------------------------------------------
# Streamlit Application
# --------------------------------------------------------------------------------------

def streamlit_app():
    """Defines the Streamlit user interface and application flow."""
    if not STREAMLIT_AVAILABLE:
        print("Streamlit is not installed. Please run 'pip install streamlit' to launch the UI.")
        return

    st.set_page_config(page_title='Visa Discussion Analyzer', layout='wide', initial_sidebar_state='expanded')

    st.title('Visa Discussion Analyzer')
    st.caption('A professional dashboard for sentiment and topic insights from threaded discussions.')

    # --- Sidebar Navigation & Controls ---
    st.sidebar.header('Navigation')
    page = st.sidebar.radio('Go to', ['Analysis Dashboard', 'About the Program', 'About the Team'])
    st.sidebar.markdown('---')
    st.sidebar.info('Developed for ADA Lab Exercise at Christ University, Bengaluru.')

    if page == 'Analysis Dashboard':
        st.sidebar.header('Data Input')
        uploaded_files = st.sidebar.file_uploader(
            'Upload two JSON files',
            type=['json'],
            accept_multiple_files=True
        )
        run_btn = st.sidebar.button('Run Analysis', type='primary')
        
        st.sidebar.markdown('---')
        st.sidebar.subheader('Visualization Options')
        show_wordclouds = st.sidebar.checkbox('Generate Wordclouds', value=False, disabled=not WORDCLOUD_AVAILABLE)

        if run_btn:
            if not uploaded_files or len(uploaded_files) != 2:
                st.error('Please upload exactly 2 JSON files to begin.')
                return
            
            with st.spinner('Running advanced data pipeline...'):
                try:
                    file1_data = json.load(uploaded_files[0])
                    file2_data = json.load(uploaded_files[1])
                    df, topic_info, edges = run_pipeline(file1_data, file2_data)
                except Exception as e:
                    st.error('Analysis failed due to an error.')
                    st.exception(e)
                    return
            
            st.success('Analysis completed successfully!')

            # --- Main Content: Filters & KPIs ---
            with st.expander('Filters and Data Preview', expanded=True):
                # Filters
                c1, c2, c3, _ = st.columns(4)
                with c1:
                    sources = sorted(df['source'].unique().tolist())
                    sel_sources = st.multiselect('Source Files', options=sources, default=sources)
                with c2:
                    sentiments = ['POSITIVE', 'NEGATIVE', 'NEUTRAL']
                    sel_sents = st.multiselect('Sentiment', options=sentiments, default=sentiments)
                with c3:
                    topics = sorted(df['topic'].unique().tolist()) if 'topic' in df.columns else []
                    sel_topics = st.multiselect('Topics', options=topics, default=topics)
                
                # Text search
                query = st.text_input('Search within comments')
                
                # Apply filters
                fdf = df.copy()
                fdf = fdf[fdf['source'].isin(sel_sources) & fdf['sentiment_label'].isin(sel_sents)]
                if 'topic' in fdf.columns and sel_topics:
                    fdf = fdf[fdf['topic'].isin(sel_topics)]
                if query:
                    fdf = fdf[fdf['comment'].str.contains(query, case=False, na=False)]

                # The `st.dataframe` line has been removed to eliminate the table
                # The line showing the number of filtered comments has also been removed for a cleaner look.

            # --- Main Content: KPIs ---
            kpi1, kpi2, kpi3, kpi4 = st.columns(4)
            with kpi1:
                st.metric('Total Comments', len(fdf))
            with kpi2:
                st.metric('Unique Authors', fdf['author'].nunique())
            with kpi3:
                pos_rate = (fdf['sentiment_label'] == 'POSITIVE').mean() if not fdf.empty else 0
                st.metric('Positive %', f"{pos_rate*100:.1f}%")
            with kpi4:
                st.metric('Active Topics', fdf['topic'].nunique() if 'topic' in fdf.columns else 0)

            # --- Main Content: Charts ---
            st.header('Visualizations')
            if PLOTLY_AVAILABLE:
                figs = build_plotly_figures(fdf)
                if 'sentiment_bar' in figs: st.plotly_chart(figs['sentiment_bar'], use_container_width=True)
                if 'sentiment_by_source' in figs: st.plotly_chart(figs['sentiment_by_source'], use_container_width=True)
                if 'score_violin' in figs: st.plotly_chart(figs['score_violin'], use_container_width=True)
                if 'topic_sentiment' in figs: st.plotly_chart(figs['topic_sentiment'], use_container_width=True)
            else:
                st.info("Plotly is not installed. Displaying static plots.")
                # Fallback to static plotting
                st.subheader('Static Plots')
                fig, ax = plt.subplots(figsize=(8, 6))
                sns.countplot(x='sentiment_label', data=fdf, palette='viridis', ax=ax)
                ax.set_title('Overall Sentiment Distribution')
                st.pyplot(fig)

            # --- Wordclouds & Graph ---
            if show_wordclouds:
                wc_pos_path = os.path.join(PLOTS_DIR, 'wordcloud_positive.png')
                wc_neg_path = os.path.join(PLOTS_DIR, 'wordcloud_negative.png')
                _generate_wordclouds(fdf, wc_pos_path, wc_neg_path)
                
                st.subheader('Word Clouds')
                wc1, wc2 = st.columns(2)
                with wc1:
                    st.image(wc_pos_path, caption='Positive Comments Wordcloud') if os.path.exists(wc_pos_path) else st.warning("No positive comments found for wordcloud.")
                with wc2:
                    st.image(wc_neg_path, caption='Negative Comments Wordcloud') if os.path.exists(wc_neg_path) else st.warning("No negative comments found for wordcloud.")
            
            _draw_graph_streamlit(fdf, edges)
            
            # --- Downloads ---
            st.header('Downloadable Results')
            with open(os.path.join(OUTPUT_DIR, 'comments_with_sentiment_topics.csv'), 'rb') as f:
                st.download_button('Download All Results (CSV)', f, file_name='results.csv', mime='text/csv')
            with open(os.path.join(OUTPUT_DIR, 'topic_info.csv'), 'rb') as f:
                st.download_button('Download Topic Info (CSV)', f, file_name='topic_info.csv', mime='text/csv')


    elif page == 'About the Program':
        st.header('About the Program')
        st.markdown("""
This project, developed for the **ADA Lab Exercise at Christ University, Bengaluru**, is a professional and robust tool for analyzing social media discussions. It's designed to provide a comprehensive, interactive view of sentiment, topics, and conversational flow within complex, nested comment threads.

The program's core is built on a modular pipeline that ensures reliability through various fallbacks. For sentiment, it uses the powerful Hugging Face Transformers library, but if that fails, it gracefully switches to the simpler TextBlob. Similarly, for topic discovery, it prioritizes the modern BERTopic model and falls back to the classic Gensim LDA if needed.

The dashboard, powered by Streamlit, offers an intuitive and visually engaging experience. It features dynamic charts created with Plotly, interactive filters, and clear summary metrics, making it accessible to both technical and non-technical users.
        """)
        st.markdown('---')
        st.subheader('Key Technologies')
        st.markdown("""
- **Streamlit**: For the interactive web dashboard.
- **Transformers**: For high-quality sentiment analysis.
- **BERTopic**: For advanced topic modeling.
- **Plotly**: For rich, interactive visualizations.
- **NetworkX**: For mapping the conversational network.
- **Pandas**: For efficient data manipulation.
        """)

    elif page == 'About the Team':
        st.header('About the Team')
        st.markdown("""
This project was a collaborative effort by the **Insightful Innovators** team, a group of dedicated MCA students from **Christ University, Bengaluru**. We leveraged our collective expertise in data science, machine learning, and software development to create a professional and functional application.
        """)
        st.markdown('---')
        st.subheader('Team Member Contributions')
        st.markdown("""
- **Bavirisetty Sairam**: Designed and implemented the Streamlit UI, focusing on creating a professional and seamless user experience.
- **Anirrudha GK**: Developed the data extraction pipeline and ensured robust handling of complex JSON structures.
- **Anmol Ratan**: Integrated and optimized the machine learning models, ensuring high performance and reliable fallbacks.
- **Anupriya Singh**: Created the visualizations and configured the plotting libraries to present insights clearly and effectively.
        """)
        st.markdown('---')
        st.markdown("We're proud of our work and hope this tool proves useful for exploring the rich world of social media data!")


if __name__ == '__main__':
    # Streamlit automatically runs `streamlit_app()` when executed with `streamlit run`.
    # This block ensures the script is runnable both as a standalone script and a Streamlit app.
    if STREAMLIT_AVAILABLE and st.runtime.exists():
        streamlit_app()
    else:
        print("To launch the interactive UI, please install Streamlit and run 'streamlit run app.py'.")
        # Fallback for command-line execution (can be expanded if needed)
        try:
            from argparse import ArgumentParser
            parser = ArgumentParser(description='Run analysis and save results.')
            parser.add_argument('--file1', default='file1.json', help='Path to file1.json')
            parser.add_argument('--file2', default='file2.json', help='Path to file2.json')
            args = parser.parse_args()
            
            print("Running pipeline in command-line mode...")
            # Here you would implement a non-Streamlit version of the pipeline
            print("This CLI functionality is a placeholder. The primary experience is the Streamlit UI.")
        except ImportError:
            pass # ignore for cases where argparse is not available
