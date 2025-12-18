"""
Streamlit Application for RAG Pipeline
Two main features:
1. Q&A Interface
2. Evaluation Interface
"""

import streamlit as st
import pandas as pd
import os
from rag_functions import RAGPipeline, RAGEvaluator

# Page configuration
st.set_page_config(
    page_title="RAG Q&A System",
    page_icon="🤖",
    layout="wide"
)

# Initialize session state
if 'rag_pipeline' not in st.session_state:
    st.session_state.rag_pipeline = None
if 'evaluation_results' not in st.session_state:
    st.session_state.evaluation_results = None

# Sidebar for API Key and Data Upload
st.sidebar.title("⚙️ Configuration")

api_key = st.sidebar.text_input(
    "OpenAI API Key",
    type="password",
    help="Enter your OpenAI API key"
)

csv_path = st.sidebar.text_input(
    "Company Data CSV Path",
    value="Data Manipulation/companies_enriched_deepseek.csv",
    help="Path to the enriched companies CSV file"
)

chroma_dir = st.sidebar.text_input(
    "Vector DB Directory Path",
    value="./chroma_db",
    help="Path to the existing Chroma vector database directory"
)

# Initialize RAG Pipeline Button
if st.sidebar.button("🚀 Initialize RAG System", type="primary"):
    if not api_key:
        st.sidebar.error("Please provide an OpenAI API key")
    elif not os.path.exists(csv_path):
        st.sidebar.error(f"Company data CSV not found at: {csv_path}")
    elif not os.path.exists(chroma_dir):
        st.sidebar.error(f"Vector database not found at: {chroma_dir}")
    else:
        with st.sidebar.status("Initializing RAG system...") as status:
            try:
                # Initialize pipeline with existing files
                st.session_state.rag_pipeline = RAGPipeline(api_key, csv_path, chroma_dir)
                status.update(label="✅ RAG System Ready!", state="complete")
                st.sidebar.success("RAG Pipeline initialized successfully!")
                
            except Exception as e:
                status.update(label="❌ Initialization failed", state="error")
                st.sidebar.error(f"Error: {str(e)}")

# Show system status
if st.session_state.rag_pipeline:
    st.sidebar.success("✅ System Status: Active")
    st.sidebar.info(f"📊 Companies Loaded: {len(st.session_state.rag_pipeline.VALID_COMPANIES)}")
else:
    st.sidebar.warning("⚠️ System Status: Not Initialized")

st.sidebar.markdown("---")
st.sidebar.markdown("### 📖 Instructions")
st.sidebar.markdown("""
1. Enter your OpenAI API key
2. Verify CSV and Vector DB paths
3. Click 'Initialize RAG System'
4. Use the Q&A or Evaluation tabs
""")

# Main Application Tabs
tab1, tab2 = st.tabs(["💬 Q&A Interface", "📊 Evaluation"])

# Tab 1: Q&A Interface
with tab1:
    st.title("🤖 RAG Question Answering System")
    st.markdown("Ask questions about companies in the database")
    
    if not st.session_state.rag_pipeline:
        st.warning("⚠️ Please initialize the RAG system first using the sidebar")
    else:
        # Question input
        user_question = st.text_area(
            "Enter your question:",
            height=100,
            placeholder="e.g., What are the main products of Siemens Energy AG?"
        )
        
        col1, col2 = st.columns([1, 5])
        with col1:
            ask_button = st.button("🔍 Ask", type="primary", use_container_width=True)
        with col2:
            clear_button = st.button("🗑️ Clear", use_container_width=True)
        
        if clear_button:
            st.rerun()
        
        if ask_button and user_question:
            with st.spinner("🔍 Searching knowledge base..."):
                try:
                    result = st.session_state.rag_pipeline.query(user_question)
                    
                    # Display answer
                    st.markdown("### 💡 Answer")
                    st.success(result['answer'])
                    
                    # Display metadata in expandable sections
                    with st.expander("📋 Query Details"):
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Documents Retrieved", result['num_docs'])
                        with col2:
                            st.metric("Context Length", f"{result['context_length']} chars")
                        with col3:
                            st.metric("Companies Found", len(result['companies']))
                        
                        st.markdown("**Companies Referenced:**")
                        for company in result['companies']:
                            st.markdown(f"- {company}")
                    
                    with st.expander("🔄 Expanded Queries"):
                        for i, query in enumerate(result['expanded_queries'], 1):
                            st.markdown(f"{i}. {query}")
                
                except Exception as e:
                    st.error(f"Error processing question: {str(e)}")

# Tab 2: Evaluation Interface
with tab2:
    st.title("📊 RAG System Evaluation")
    st.markdown("Upload a test CSV file to evaluate the RAG system's performance")
    
    if not st.session_state.rag_pipeline:
        st.warning("⚠️ Please initialize the RAG system first using the sidebar")
    else:
        # File uploader for test data
        test_file = st.file_uploader(
            "Upload Test CSV (must contain 'question' and 'expected_answer' columns)",
            type=['csv'],
            key="test_uploader"
        )
        
        if test_file:
            try:
                test_df = pd.read_csv(test_file)
                
                # Validate columns
                if 'question' not in test_df.columns or 'expected_answer' not in test_df.columns:
                    st.error("❌ CSV must contain 'question' and 'expected_answer' columns")
                else:
                    st.success(f"✅ Loaded {len(test_df)} test cases")
                    
                    # Show preview
                    with st.expander("👀 Preview Test Data"):
                        st.dataframe(test_df.head())
                    
                    # Run evaluation
                    if st.button("▶️ Run Evaluation", type="primary"):
                        progress_bar = st.progress(0)
                        status_text = st.empty()
                        
                        try:
                            status_text.text("Initializing evaluator...")
                            evaluator = RAGEvaluator(st.session_state.rag_pipeline)
                            
                            status_text.text("Running evaluation (this may take a few minutes)...")
                            progress_bar.progress(50)
                            
                            eval_results = evaluator.evaluate(test_df)
                            st.session_state.evaluation_results = eval_results
                            
                            progress_bar.progress(100)
                            status_text.text("✅ Evaluation complete!")
                            
                        except Exception as e:
                            st.error(f"Error during evaluation: {str(e)}")
                            progress_bar.empty()
                            status_text.empty()
            except Exception as e:
                st.error(f"❌ Failed to load CSV file: {str(e)}")

        # Display results if available
        if st.session_state.evaluation_results:
            st.markdown("---")
            st.markdown("## 📈 Evaluation Results")
            
            results = st.session_state.evaluation_results
            
            # Metrics
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Total Tests", results['total'])
            with col2:
                st.metric("Correct", results['correct'], delta=None)
            with col3:
                st.metric("Incorrect", results['total'] - results['correct'], delta=None)
            with col4:
                st.metric("Accuracy", f"{results['accuracy']:.1%}")
            
            # Results table
            st.markdown("### 📋 Detailed Results")
            
            # Add filtering
            filter_option = st.selectbox(
                "Filter by:",
                ["All", "Correct Only", "Incorrect Only"]
            )
            
            filtered_df = results['results_df'].copy()
            if filter_option == "Correct Only":
                filtered_df = filtered_df[filtered_df['decision'] == True]
            elif filter_option == "Incorrect Only":
                filtered_df = filtered_df[filtered_df['decision'] == False]
            
            # Display dataframe with color coding
            def highlight_decision(row):
                if row['decision']:
                    return ['background-color: #d4edda'] * len(row)
                else:
                    return ['background-color: #f8d7da'] * len(row)
            
            styled_df = filtered_df.style.apply(highlight_decision, axis=1)
            st.dataframe(styled_df, use_container_width=True)
            
            # Download button
            csv = results['results_df'].to_csv(index=False)
            st.download_button(
                label="📥 Download Evaluation Results",
                data=csv,
                file_name="rag_evaluation_results.csv",
                mime="text/csv",
                use_container_width=True
            )

# Footer
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: gray;'>"
    "RAG Q&A System | Built with Streamlit & LangChain"
    "</div>",
    unsafe_allow_html=True
)