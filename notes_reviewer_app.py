import streamlit as st
import pandas as pd
from notes_reviewer_agent import NotesReviewerAgent
import os

# Page configuration
st.set_page_config(
    page_title="Notes Reviewer",
    page_icon="📝",
    layout="wide"
)

# Model configuration
@st.cache_resource
def get_model_config():
    """Get model configuration from session state or defaults"""
    if 'model_name' not in st.session_state:
        st.session_state.model_name = "gpt-3.5-turbo"
    
    return st.session_state.model_name

# Initialize the notes agent
@st.cache_resource
def get_notes_agent():
    model_name = get_model_config()
    
    try:
        # Check if environment variable is set
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            return None
        
        return NotesReviewerAgent(
            model_name=model_name
        )
    except Exception as e:
        st.error(f"Error initializing agent: {e}")
        return None

# Cache claim summaries
@st.cache_data
def get_cached_claim_summary(_agent, clm_num: str) -> str:
    """
    Get cached claim summary. This will only regenerate if the claim number changes.
    """
    try:
        return _agent.generate_claim_summary(clm_num)
    except Exception as e:
        st.warning(f"LLM summary failed: {e}")
        # Fallback to basic summary
        claim_notes = _agent.get_notes_by_claim(clm_num)
        return _agent._generate_basic_summary(claim_notes)

# Main app
def main():
    st.title("📝 Notes Reviewer")
    st.markdown("Select a claim number to view notes in timeline format")
    
    # Model selection sidebar
    with st.sidebar:
        st.header("🤖 OpenAI Model Settings")
        
        # Model name selection
        model_name = st.selectbox(
            "OpenAI Model:",
            ["gpt-3.5-turbo", "gpt-4o-mini", "gpt-4o"],
            index=0 if st.session_state.get('model_name', 'gpt-3.5-turbo') == 'gpt-3.5-turbo' else 1,
            help="Select OpenAI model (gpt-3.5-turbo is most cost-effective)"
        )
        
        if model_name != st.session_state.get('model_name'):
            st.session_state.model_name = model_name
            get_notes_agent.clear()
            st.rerun()
        
        # Show API key status
        api_key = os.getenv("OPENAI_API_KEY")
        if api_key:
            st.success("✅ OpenAI API key found in environment")
        else:
            st.error("❌ OPENAI_API_KEY environment variable not set")
            st.info("💡 Set it with: export OPENAI_API_KEY='your_key_here'")
        
        # Show cost info for OpenAI
        st.info("💡 OpenAI models incur costs per token. gpt-3.5-turbo is most cost-effective.")
        
        # Show current model info
        st.markdown("---")
        st.markdown(f"**Current Model:** OpenAI")
        st.markdown(f"**Model Name:** {model_name}")
        
        # Test connection button
        if st.button("🧪 Test Model Connection"):
            with st.spinner("Testing connection..."):
                try:
                    test_agent = NotesReviewerAgent(
                        model_name=model_name
                    )
                    st.success("✅ OpenAI connection successful!")
                except Exception as e:
                    st.error(f"❌ Connection failed: {e}")
    
    # Initialize agent
    agent = get_notes_agent()
    
    if agent is None:
        st.error("❌ OpenAI API key not found!")
        st.info("💡 Please set the OPENAI_API_KEY environment variable:")
        st.code("export OPENAI_API_KEY='your_api_key_here'")
        st.info("Then restart the app.")
        return
    
    try:
        # Import notes data
        notes_df = agent.import_notes()
        
        # Get unique claim numbers for dropdown
        claim_numbers = sorted(notes_df['clmNum'].unique())
        
        # Claim selection
        col1, col2 = st.columns([1, 3])
        
        with col1:
            selected_claim = st.selectbox(
                "Select Claim Number:",
                options=claim_numbers,
                index=0 if len(claim_numbers) > 0 else None
            )
        
        with col2:
            if selected_claim:
                # Get claim summary
                claim_notes = agent.get_notes_by_claim(selected_claim)
                st.metric("Total Notes", len(claim_notes))
        
        # Display claim summary at the top
        if selected_claim:
            st.markdown("---")
            
            # Model info and summary header
            col1, col2, col3 = st.columns([2, 2, 1])
            with col1:
                st.subheader(f"📋 Claim Summary for {selected_claim}")
            with col2:
                # Show current model info
                model_name, _ = get_model_config()
                st.markdown(f"**🤖 Using:** OpenAI - {model_name}")
                st.caption("💰 Cloud model (costs apply)")
            with col3:
                if st.button("🔄 Refresh Summary", key=f"refresh_summary_{selected_claim}"):
                    # Clear the cache for this specific claim
                    get_cached_claim_summary.clear()
                    st.rerun()
            
            # Generate and display cached summary
            with st.spinner("Loading summary..."):
                try:
                    summary = get_cached_claim_summary(agent, selected_claim)
                    st.info(summary)
                except Exception as e:
                    st.error(f"Error loading summary: {e}")
                    # Fallback to basic summary
                    claim_notes = agent.get_notes_by_claim(selected_claim)
                    basic_summary = agent._generate_basic_summary(claim_notes)
                    st.info(basic_summary)
            
            # Chat interface for asking questions
            st.markdown("---")
            st.subheader(f"💬 Ask Questions About {selected_claim}")
            
            # Initialize chat history in session state
            if 'chat_history' not in st.session_state:
                st.session_state.chat_history = {}
            
            if selected_claim not in st.session_state.chat_history:
                st.session_state.chat_history[selected_claim] = []
            
            # Chat input
            col1, col2 = st.columns([4, 1])
            with col1:
                user_question = st.text_input(
                    "Ask a question about this claim:",
                    placeholder="e.g., What caused the delay? When was the last payment?",
                    key=f"question_{selected_claim}"
                )
            
            with col2:
                if st.button("Ask", key=f"ask_{selected_claim}"):
                    if user_question.strip():
                        # Add user question to chat history
                        st.session_state.chat_history[selected_claim].append({
                            "role": "user",
                            "content": user_question,
                            "timestamp": pd.Timestamp.now()
                        })
                        
                        # Generate answer
                        with st.spinner("Thinking..."):
                            try:
                                answer = agent.answer_claim_question(selected_claim, user_question)
                                st.session_state.chat_history[selected_claim].append({
                                    "role": "assistant",
                                    "content": answer,
                                    "timestamp": pd.Timestamp.now()
                                })
                            except Exception as e:
                                error_msg = f"Sorry, I encountered an error: {str(e)}"
                                st.session_state.chat_history[selected_claim].append({
                                    "role": "assistant",
                                    "content": error_msg,
                                    "timestamp": pd.Timestamp.now()
                                })
                        
                        # Clear the input
                        st.rerun()
            
            # Display chat history
            if st.session_state.chat_history[selected_claim]:
                st.markdown("**Chat History:**")
                for i, message in enumerate(st.session_state.chat_history[selected_claim]):
                    if message["role"] == "user":
                        st.markdown(f"**👤 You ({message['timestamp'].strftime('%H:%M')}):** {message['content']}")
                    else:
                        st.markdown(f"**🤖 Assistant ({message['timestamp'].strftime('%H:%M')}):** {message['content']}")
                    
                    if i < len(st.session_state.chat_history[selected_claim]) - 1:
                        st.markdown("---")
            
            # Display notes timeline
            st.markdown("---")
            st.subheader(f"📅 Notes Timeline for {selected_claim}")
            
            # Get timeline data
            timeline_df = agent.get_notes_timeline(selected_claim)
            
            if len(timeline_df) > 0:
                # Create timeline view
                for idx, row in timeline_df.iterrows():
                    # Create a card-like container for each note
                    with st.container():
                        col1, col2 = st.columns([1, 4])
                        
                        with col1:
                            # Date and time info
                            st.markdown(f"**{row['whenadded'].strftime('%Y-%m-%d')}**")
                            st.markdown(f"*{row['whenadded'].strftime('%H:%M')}*")
                            if row['days_since_first'] > 0:
                                st.caption(f"+{row['days_since_first']} days")
                        
                        with col2:
                            # Note content
                            st.markdown(f"**{row['note']}**")
                            
                            # Note metadata
                            col2a, col2b, col2c = st.columns(3)
                            with col2a:
                                st.caption(f"Length: {row['note_length']} chars")
                            with col2b:
                                st.caption(f"Words: {row['note_word_count']}")
                            with col2c:
                                st.caption(f"Note #{idx + 1}")
                        
                        # Separator line
                        st.markdown("---")
                        
                        # Add some spacing for better readability
                        st.markdown("&nbsp;")
            
            else:
                st.info("No notes found for this claim.")
        
        # Sidebar with additional info
        with st.sidebar:
            st.header("📊 Data Summary")
            
            summary = agent.get_notes_summary()
            st.metric("Total Notes", summary['total_notes'])
            st.metric("Unique Claims", summary['unique_claims'])
            
            st.markdown("---")
            st.markdown("**Date Range:**")
            st.write(f"From: {summary['date_range']['earliest'].strftime('%Y-%m-%d')}")
            st.write(f"To: {summary['date_range']['latest'].strftime('%Y-%m-%d')}")
            
            st.markdown("---")
            st.markdown("**Note Statistics:**")
            st.write(f"Avg Length: {summary['note_length_stats']['mean']:.1f} chars")
            st.write(f"Avg Words: {summary['word_count_stats']['mean']:.1f}")
            
            # Search functionality
            st.markdown("---")
            st.markdown("**🔍 Search Notes**")
            search_term = st.text_input("Search term:")
            if search_term:
                search_results = agent.search_notes(search_term)
                st.write(f"Found {len(search_results)} matching notes")
                
                if len(search_results) > 0:
                    st.markdown("**Sample results:**")
                    for _, result in search_results.head(3).iterrows():
                        st.caption(f"{result['clmNum']}: {result['note'][:50]}...")
            
            # Clear chat history button
            if selected_claim and st.button("Clear Chat History", key="clear_chat"):
                if selected_claim in st.session_state.chat_history:
                    st.session_state.chat_history[selected_claim] = []
                st.rerun()
    
    except Exception as e:
        st.error(f"Error loading notes: {str(e)}")
        st.info("Make sure the notes data is available. The app will create dummy data if needed.")

if __name__ == "__main__":
    main()
