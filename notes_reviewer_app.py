import streamlit as st
import pandas as pd
import plotly.graph_objects as go
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

def plot_single_claim_lifetime(df_txn, selected_claim):
    """Plot the lifetime development pattern for a single claim"""
    if selected_claim:
        # Get transaction data for the selected claim
        claim_transactions = df_txn[df_txn['clmNum'] == selected_claim].sort_values('datetxn')
        selected_cause = claim_transactions['clmCause'].unique()[0] if 'clmCause' in claim_transactions.columns else 'Unknown'
        
        if len(claim_transactions) > 0:
            # Create development pattern visualization
            fig_development = go.Figure()
            
            # Add traces for each metric
            metrics_to_plot = ['reserve_cumsum', 'paid_cumsum', 'incurred_cumsum', 'expense_cumsum']
            colors = ['red', 'green', 'blue', 'orange']
            
            for i, metric in enumerate(metrics_to_plot):
                if metric in claim_transactions.columns:
                    fig_development.add_trace(go.Scatter(
                        x=claim_transactions['datetxn'],
                        y=claim_transactions[metric],
                        mode='lines+markers',
                        name=metric.replace('_', ' ').title(),
                        line=dict(color=colors[i], width=3),
                        marker=dict(size=6, color=colors[i])
                    ))
            
            # Update layout
            fig_development.update_layout(
                title=f"Claim {selected_claim} Development Pattern - {selected_cause}",
                xaxis_title="Transaction Date",
                yaxis_title="Cumulative Amount ($)",
                showlegend=True,
                height=500,
                hovermode='x unified'
            )
            
            # Format x-axis to show dates nicely
            fig_development.update_xaxes(
                tickformat='%Y-%m-%d',
                tickangle=45
            )
            
            st.plotly_chart(fig_development, use_container_width=True)
            
            # Add claim summary information
            st.write("**Claim Summary Information:**")
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Status", claim_transactions['clmStatus'].iloc[-1])
            
            with col2:
                st.metric("Total Transactions", len(claim_transactions))
            
            with col3:
                st.metric("First Date", claim_transactions['datetxn'].min().strftime('%Y-%m-%d'))
            
            with col4:
                st.metric("Last Date", claim_transactions['datetxn'].max().strftime('%Y-%m-%d'))
            
            # Show transaction details table
            st.write("**Transaction Details:**")
            display_cols = ['datetxn', 'reserve_cumsum', 'paid_cumsum', 'incurred_cumsum', 'expense_cumsum', 'clmStatus']
            available_display_cols = [col for col in display_cols if col in claim_transactions.columns]
            
            transaction_summary = claim_transactions[available_display_cols].copy()
            transaction_summary.columns = ['Transaction Date', 'Reserve Cumsum', 'Paid Cumsum', 'Incurred Cumsum', 'Expense Cumsum', 'Status']
            
            # Format currency columns
            currency_cols = ['Reserve Cumsum', 'Paid Cumsum', 'Incurred Cumsum', 'Expense Cumsum']
            for col in currency_cols:
                if col in transaction_summary.columns:
                    transaction_summary[col] = transaction_summary[col].apply(lambda x: f"${x:,.2f}")
            
            st.dataframe(transaction_summary, use_container_width=True)
            
        else:
            st.warning(f"No transaction data found for claim {selected_claim}")

# Main app
def main():
    st.title("📝 Notes Reviewer For Open Claims")

    # add summary section to show the number of open claims at risk of being under-reserved
    # we have a risk_level column from 'Critical', 'high', 'medium', 'low'
    # 1. let's add the number claims at risk of being under-reserved ((critical high or medium
    # 2. let's add a summary table of the claims at risk of being under-reserved with a filter per level of risk
    
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
        
        # Claims Summary Table
        st.markdown("---")
        st.markdown("**📋 Claims Summary Table**")
        
        try:
            # Get all claims with their current status and amounts
            claims_summary = agent.get_claims_summary_table()
            
            if len(claims_summary) > 0:
                # Add a search/filter for claims
                search_claim = st.text_input("🔍 Search claims:", placeholder="Enter claim number...")
                
                if search_claim:
                    filtered_claims = claims_summary[
                        claims_summary['clmNum'].str.contains(search_claim, case=False, na=False)
                    ]
                    display_df = filtered_claims
                else:
                    display_df = claims_summary
                
                                # Show the claims table with clickable rows
                st.write("**Click on any claim row to view its lifetime development pattern:**")
                
                # Initialize selected claim in session state
                if 'selected_claim_from_table' not in st.session_state:
                    st.session_state.selected_claim_from_table = display_df['clmNum'].iloc[0] if len(display_df) > 0 else None
                
                # Create a true interactive table with row selection
                st.write("**💡 Tip: Click on any row in the table below to select a claim and see its lifetime analysis**")
                
                # Add a selection column to the dataframe
                interactive_df = display_df.copy()
                interactive_df['Select'] = False  # Initialize all as unselected
                # make select to the first column
                interactive_df = interactive_df[['Select', 'risk_level', 'clmNum', 'current_reserve', 'dateReceived', 'dateCompleted', 'dateReopened', 'total_incurred']]
                
                # Use data_editor for true row selection
                edited_df = st.data_editor(
                    interactive_df,
                    use_container_width=True,
                    hide_index=True,
                    column_config={
                        "Select": st.column_config.CheckboxColumn("Select", width="small", help="Click to select this claim"),
                        "risk_level": st.column_config.TextColumn("Risk Level", width="medium",disabled=True),
                        "clmNum": st.column_config.TextColumn("Claim #", width="medium",disabled=True),
                        "current_reserve": st.column_config.NumberColumn("Reserve ($)", format="$%.2f", width="medium",disabled=True),
                        "dateReceived": st.column_config.DateColumn("Opened", width="small",disabled=True),
                        "dateCompleted": st.column_config.DateColumn("Closed", width="small",disabled=True),
                        "dateReopened": st.column_config.DateColumn("Reopened", width="small",disabled=True),
                        "total_incurred": st.column_config.NumberColumn("Total Incurred ($)", format="$%.2f", width="medium",disabled=True),
                    },
                    key="interactive_claims_table"
                )
                
                # Detect which claim was selected
                if edited_df is not None and len(edited_df) > 0:
                    # Find the selected row
                    selected_rows = edited_df[edited_df['Select'] == True]
                    if len(selected_rows) > 0:
                        # Get the first selected claim
                        selected_claim_from_table = selected_rows.iloc[0]['clmNum']
                        # Update session state
                        if st.session_state.selected_claim_from_table != selected_claim_from_table:
                            st.session_state.selected_claim_from_table = selected_claim_from_table
                    else:
                        # No selection, use the first claim or previous selection
                        selected_claim_from_table = st.session_state.selected_claim_from_table
                else:
                    selected_claim_from_table = st.session_state.selected_claim_from_table
                
                # Show summary metrics
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Total Claims", len(display_df))
                with col2:
                    st.metric("Total Reserve", f"${display_df['current_reserve'].sum():,.0f}")
                
                # Claim Lifetime Visualization
                if selected_claim_from_table:
                    st.markdown("---")
                    st.subheader(f"📊 Claim Lifetime Analysis: {selected_claim_from_table}")
                    
                    try:
                        # Load the claims transaction data for visualization
                        claims_file = os.path.join("./_data", "critical_claims.csv")
                        if os.path.exists(claims_file):
                            df_raw_txn = pd.read_csv(claims_file)
                            df_raw_txn['datetxn'] = pd.to_datetime(df_raw_txn['datetxn'])
                            
                            # Plot the claim lifetime
                            plot_single_claim_lifetime(df_raw_txn, selected_claim_from_table)
                        else:
                            st.warning("Claims transaction data not found. Please ensure critical_claims.csv exists.")
                    except Exception as e:
                        st.error(f"Error loading claim lifetime data: {e}")
                        
            else:
                st.info("No claims data available")
                    
        except Exception as e:
            st.warning(f"Could not load claims summary: {e}")
            st.info("Claims summary will be available once data is loaded")
        
        # st.markdown("---")
        # st.markdown("**Select a claim number to view notes in timeline format**")

        # # Get unique claim numbers for dropdown
        # claim_numbers = sorted(notes_df['clmNum'].unique())
        
        # # Claim selection
        # col1, col2 = st.columns([1, 3])
        
        # with col1:
        #     selected_claim = st.selectbox(
        #         "Select Claim Number:",
        #         options=claim_numbers,
        #         key="claim_selector"
        #     )
        
        # with col2:
        #     if selected_claim:
        #         st.info(f"Selected: **{selected_claim}**")
        
        # Initialize chat history in session state
        if 'chat_history' not in st.session_state:
            st.session_state.chat_history = {}
        
        if selected_claim_from_table not in st.session_state.chat_history:
            st.session_state.chat_history[selected_claim_from_table] = []
        
        # Chat interface
        if selected_claim_from_table:
            st.markdown("---")
            st.subheader(f"💬 Chat with AI about {selected_claim_from_table}")
            
            # Claim Summary Box
            with st.container():
                st.markdown("**📋 Claim Summary & Analysis**")
                
                # Create a placeholder for the summary
                summary_placeholder = st.empty()
                
                # Check if we already have a summary in session state
                summary_key = f"claim_summary_{selected_claim_from_table}"
                if summary_key not in st.session_state:
                    # Generate summary using OpenAI
                    with st.spinner("🤖 Generating claim summary..."):
                        try:
                            # Get claim data for context
                            claims_file = os.path.join("./_data", "critical_claims.csv")
                            if os.path.exists(claims_file):
                                df_raw_txn = pd.read_csv(claims_file)
                                df_raw_txn['datetxn'] = pd.to_datetime(df_raw_txn['datetxn'])
                                
                                # Filter for selected claim
                                claim_data = df_raw_txn[df_raw_txn['clmNum'] == selected_claim_from_table]
                                
                                if len(claim_data) > 0:
                                    # Get claim notes for additional context
                                    claim_notes = agent.get_notes_by_claim(selected_claim_from_table)
                                    
                                    # Create context for OpenAI
                                    context = f"""
Claim Number: {selected_claim_from_table}
Status: {claim_data['clmStatus'].iloc[-1]}
Total Transactions: {len(claim_data)}
Date Range: {claim_data['datetxn'].min().strftime('%Y-%m-%d')} to {claim_data['datetxn'].max().strftime('%Y-%m-%d')}
Total Paid: ${claim_data['paid'].sum():,.2f}
Total Expense: ${claim_data['expense'].sum():,.2f}
Total Reserve: ${claim_data['reserve'].sum():,.2f}
Total Incurred: ${claim_data['incurred'].sum():,.2f}
Claim Cause: {claim_data['clmCause'].iloc[0] if 'clmCause' in claim_data.columns else 'Unknown'}
"""
                                    
                                    if len(claim_notes) > 0:
                                        context += f"\nNumber of Notes: {len(claim_notes)}"
                                        context += f"\nFirst Note: {claim_notes['whenadded'].min().strftime('%Y-%m-%d')}"
                                        context += f"\nLast Note: {claim_notes['whenadded'].max().strftime('%Y-%m-%d')}"
                                    
                                    # Create prompt for OpenAI
                                    prompt = f"""Act as a top notch claim rep and summarize the claim for me with the key amount paid (expense and payment) as well as critical dates. Also summarize the next step and why the claim is still opened?? Please don't invent, and tell me when you don't know.

Claim Context:
{context}

Please provide a clear, professional summary focusing on:
1. Key financial amounts (paid, expense, reserve, incurred)
2. Critical dates (opening, transactions, notes)
3. Current status and why it's still open
4. Next steps based on available information
5. Any red flags or concerns

If you don't have enough information for any part, clearly state "I don't have enough information to determine [specific detail]." """
                                    
                                    # Generate summary using OpenAI
                                    response = agent.openai_client.chat.completions.create(
                                        model=agent.model_name,
                                        messages=[{"role": "user", "content": prompt}],
                                        max_tokens=400,
                                        temperature=0.3
                                    )
                                    
                                    summary = response.choices[0].message.content.strip()
                                    st.session_state[summary_key] = summary
                                    
                                else:
                                    st.session_state[summary_key] = "❌ No claim data found for this claim number."
                            else:
                                st.session_state[summary_key] = "❌ Claims data file not found."
                                
                        except Exception as e:
                            st.session_state[summary_key] = f"❌ Error generating summary: {str(e)}"
                
                # Display the summary
                summary_content = st.session_state.get(summary_key, "Loading...")
                
                # Create an expandable box for the summary
                with st.expander("📊 **Claim Summary & Analysis**", expanded=True):
                    st.markdown(summary_content)
                    
                    # Add a refresh button
                    if st.button("🔄 Refresh Summary", key=f"refresh_summary_{selected_claim_from_table}"):
                        # Remove the summary from session state to regenerate
                        if summary_key in st.session_state:
                            del st.session_state[summary_key]
                        st.rerun()
            
            # Chat input
            col1, col2 = st.columns([3, 1])
            
            with col1:
                user_question = st.text_input(
                    "Ask a question about this claim:",
                    placeholder="e.g., What caused the delay? When was the last payment?",
                    key=f"question_{selected_claim_from_table}"
                )
            
            with col2:
                if st.button("Ask", key=f"ask_{selected_claim_from_table}"):
                    if user_question.strip():
                        # Add user question to chat history
                        st.session_state.chat_history[selected_claim_from_table].append({
                            "role": "user",
                            "content": user_question,
                            "timestamp": pd.Timestamp.now()
                        })
                        
                        # Generate answer
                        with st.spinner("Thinking..."):
                            try:
                                answer = agent.answer_claim_question(selected_claim_from_table, user_question)
                                st.session_state.chat_history[selected_claim_from_table].append({
                                    "role": "assistant",
                                    "content": answer,
                                    "timestamp": pd.Timestamp.now()
                                })
                            except Exception as e:
                                error_msg = f"Sorry, I encountered an error: {str(e)}"
                                st.session_state.chat_history[selected_claim_from_table].append({
                                    "role": "assistant",
                                    "content": error_msg,
                                    "timestamp": pd.Timestamp.now()
                                })
                        
                        # Clear the input
                        st.rerun()
            
            # Display chat history
            if st.session_state.chat_history[selected_claim_from_table]:
                st.markdown("**Chat History:**")
                for i, message in enumerate(st.session_state.chat_history[selected_claim_from_table]):
                    if message["role"] == "user":
                        st.markdown(f"**👤 You ({message['timestamp'].strftime('%H:%M')}):** {message['content']}")
                    else:
                        st.markdown(f"**🤖 Assistant ({message['timestamp'].strftime('%H:%M')}):** {message['content']}")
                    
                    if i < len(st.session_state.chat_history[selected_claim_from_table]) - 1:
                        st.markdown("---")
            
            # Display comprehensive claim information
            st.markdown("---")
            st.subheader(f"📅 Communication Timeline & Notes for {selected_claim_from_table}")
            
            # Load claim notes data
            try:
                notes_file = os.path.join("./_data", "claim_notes.csv")
                if os.path.exists(notes_file):
                    claim_notes_df = pd.read_csv(notes_file)
                    claim_notes_df['dateNote'] = pd.to_datetime(claim_notes_df['dateNote'])
                    
                    # Filter notes for selected claim
                    claim_notes = claim_notes_df[claim_notes_df['clmNum'] == selected_claim_from_table].sort_values('dateNote')
                    
                    if len(claim_notes) > 0:
                        # Show notes summary
                        col1, col2, col3, col4 = st.columns(4)
                        with col1:
                            st.metric("Total Notes", len(claim_notes))
                        with col2:
                            st.metric("Emails", len(claim_notes[claim_notes['note_type'] == 'email']))
                        with col3:
                            st.metric("High Priority", len(claim_notes[claim_notes['priority'] == 'High']))
                        with col4:
                            st.metric("Date Range", f"{(claim_notes['dateNote'].max() - claim_notes['dateNote'].min()).days} days")
                        
                        # Add filters for notes
                        st.markdown("**🔍 Filter Notes:**")
                        col1, col2, col3 = st.columns(3)
                        
                        with col1:
                            note_type_filter = st.selectbox(
                                "Note Type:",
                                ["All", "email", "internal_note"],
                                key=f"note_type_filter_{selected_claim_from_table}"
                            )
                        
                        with col2:
                            priority_filter = st.selectbox(
                                "Priority:",
                                ["All", "High", "Medium", "Low"],
                                key=f"priority_filter_{selected_claim_from_table}"
                            )
                        
                        with col3:
                            author_filter = st.selectbox(
                                "Author:",
                                ["All"] + sorted(claim_notes['author'].unique().tolist()),
                                key=f"author_filter_{selected_claim_from_table}"
                            )
                        
                        # Apply filters
                        filtered_notes = claim_notes.copy()
                        if note_type_filter != "All":
                            filtered_notes = filtered_notes[filtered_notes['note_type'] == note_type_filter]
                        if priority_filter != "All":
                            filtered_notes = filtered_notes[filtered_notes['priority'] == priority_filter]
                        if author_filter != "All":
                            filtered_notes = filtered_notes[filtered_notes['author'] == author_filter]
                        
                        # Display filtered notes
                        st.markdown(f"**📋 Showing {len(filtered_notes)} filtered notes:**")
                        
                        for idx, row in filtered_notes.iterrows():
                            # Create a card-like container for each note
                            with st.container():
                                col1, col2 = st.columns([1, 4])
                                
                                with col1:
                                    # Date and priority info
                                    st.markdown(f"**{row['dateNote'].strftime('%Y-%m-%d')}**")
                                    st.markdown(f"*{row['dateNote'].strftime('%H:%M')}*")
                                    
                                    # Priority indicator
                                    if row['priority'] == 'High':
                                        st.markdown("🔴 **HIGH PRIORITY**")
                                    elif row['priority'] == 'Medium':
                                        st.markdown("🟡 **MEDIUM PRIORITY**")
                                    else:
                                        st.markdown("🟢 **LOW PRIORITY**")
                                    
                                    # Author
                                    st.caption(f"From: {row['author']}")
                                
                                with col2:
                                    # Note content
                                    st.markdown(f"**{row['note']}**")
                                    
                                    # Note metadata
                                    col2a, col2b, col2c = st.columns(3)
                                    with col2a:
                                        st.caption(f"Type: {row['note_type'].replace('_', ' ').title()}")
                                    with col2b:
                                        st.caption(f"Priority: {row['priority']}")
                                    with col2c:
                                        st.caption(f"Note #{idx + 1}")
                                
                                # Separator line
                                st.markdown("---")
                                
                                # Add some spacing for better readability
                                st.markdown("&nbsp;")
                        
                    else:
                        st.info("No claim-specific notes found. Showing general notes timeline...")
                        timeline_df = agent.get_notes_timeline(selected_claim_from_table)
                        
                        if len(timeline_df) > 0:
                            for idx, row in timeline_df.iterrows():
                                with st.container():
                                    col1, col2 = st.columns([1, 4])
                                    
                                    with col1:
                                        st.markdown(f"**{row['whenadded'].strftime('%Y-%m-%d')}**")
                                        st.markdown(f"*{row['whenadded'].strftime('%H:%M')}*")
                                        if row['days_since_first'] > 0:
                                            st.caption(f"+{row['days_since_first']} days")
                                    
                                    with col2:
                                        st.markdown(f"**{row['note']}**")
                                        
                                        col2a, col2b, col2c = st.columns(3)
                                        with col2a:
                                            st.caption(f"Length: {row['note_length']} chars")
                                        with col2b:
                                            st.caption(f"Words: {row['note_word_count']}")
                                        with col2c:
                                            st.caption(f"Note #{idx + 1}")
                                    
                                    st.markdown("---")
                                    st.markdown("&nbsp;")
                        else:
                            st.info("No notes found for this claim.")
                            
                else:
                    st.warning("Claim notes data not found. Please ensure claim_notes.csv exists.")
                    st.info("Showing general notes timeline instead...")
                    
                    # Fallback to original notes
                    timeline_df = agent.get_notes_timeline(selected_claim_from_table)
                    
                    if len(timeline_df) > 0:
                        for idx, row in timeline_df.iterrows():
                            with st.container():
                                col1, col2 = st.columns([1, 4])
                                
                                with col1:
                                    st.markdown(f"**{row['whenadded'].strftime('%Y-%m-%d')}**")
                                    st.markdown(f"*{row['whenadded'].strftime('%H:%M')}*")
                                    if row['days_since_first'] > 0:
                                        st.caption(f"+{row['days_since_first']} days")
                                
                                with col2:
                                    st.markdown(f"**{row['note']}**")
                                    
                                    col2a, col2b, col2c = st.columns(3)
                                    with col2a:
                                        st.caption(f"Length: {row['note_length']} chars")
                                    with col2b:
                                        st.caption(f"Words: {row['note_word_count']}")
                                    with col2c:
                                        st.caption(f"Note #{idx + 1}")
                                
                                st.markdown("---")
                                st.markdown("&nbsp;")
                    else:
                        st.info("No notes found for this claim.")
                        
            except Exception as e:
                st.error(f"Error loading claim notes: {e}")
                st.info("Falling back to general notes timeline...")
                
                # Fallback to original notes
                timeline_df = agent.get_notes_timeline(selected_claim_from_table)
                
                if len(timeline_df) > 0:
                    for idx, row in timeline_df.iterrows():
                        with st.container():
                            col1, col2 = st.columns([1, 4])
                            
                            with col1:
                                st.markdown(f"**{row['whenadded'].strftime('%Y-%m-%d')}**")
                                st.markdown(f"*{row['whenadded'].strftime('%H:%M')}*")
                                if row['days_since_first'] > 0:
                                    st.caption(f"+{row['days_since_first']} days")
                            
                            with col2:
                                st.markdown(f"**{row['note']}**")
                                
                                col2a, col2b, col2c = st.columns(3)
                                with col2a:
                                    st.caption(f"Length: {row['note_length']} chars")
                                with col2b:
                                    st.caption(f"Words: {row['note_word_count']}")
                                with col2c:
                                    st.caption(f"Note #{idx + 1}")
                            
                            st.markdown("---")
                            st.markdown("&nbsp;")
                else:
                    st.info("No notes found for this claim.")
        
    except Exception as e:
        st.error(f"Error loading notes: {str(e)}")
        st.info("Make sure the notes data is available. The app will create dummy data if needed.")

if __name__ == "__main__":
    main()
