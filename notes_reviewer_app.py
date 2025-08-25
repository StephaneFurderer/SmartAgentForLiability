import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from notes_reviewer_agent import NotesReviewerAgent
from claim_status_predictor import ClaimStatusPredictor
from feedback_storage import FeedbackStorage
from general_claims_agent import GeneralClaimsAgent
import os
from datetime import datetime

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

# Initialize the claim status predictor
@st.cache_resource
def get_claim_predictor():
    """Initialize the claim status predictor"""
    try:
        return ClaimStatusPredictor()
    except Exception as e:
        st.error(f"Error initializing claim predictor: {e}")
        return None

# Initialize the feedback storage
@st.cache_resource
def get_feedback_storage():
    """Initialize the feedback storage system"""
    try:
        return FeedbackStorage()
    except Exception as e:
        st.error(f"Error initializing feedback storage: {e}")
        return None

# Initialize the general claims agent
@st.cache_resource
def get_general_agent():
    """Initialize the general claims agent for portfolio-wide analysis"""
    try:
        return GeneralClaimsAgent()
    except Exception as e:
        st.error(f"Error initializing general agent: {e}")
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
        
        # Feedback Summary
        st.markdown("---")
        st.markdown("**🔮 Feedback Summary**")
        
        feedback_storage = get_feedback_storage()
        if feedback_storage:
            feedback_summary = feedback_storage.get_feedback_summary()
            
            st.metric("Total Feedback", feedback_summary['total_feedback'])
            st.metric("Claims Assessed", feedback_summary['unique_claims'])
            st.metric("Representatives", feedback_summary['unique_reps'])
            st.metric("Avg Confidence", f"{feedback_summary['avg_confidence']:.1%}")
            
            if feedback_summary['feedback_by_status']:
                st.markdown("**Status Distribution:**")
                for status, count in feedback_summary['feedback_by_status'].items():
                    st.write(f"• {status}: {count}")
        else:
            st.info("Feedback storage not available")
    
    # Claims Summary Table
    st.markdown("---")
    st.markdown("**📋 Claims Summary Table**")
    
       
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
        
        # Portfolio Analysis Section
        st.markdown("---")
        st.subheader("🔍 Portfolio Analysis & Insights")
        
        # Initialize general agent
        general_agent = get_general_agent()
        
        if general_agent:
            # Portfolio Summary
            portfolio_summary = general_agent.get_portfolio_summary()
            
            if 'error' not in portfolio_summary:
                # Display portfolio metrics
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric(
                        "Portfolio Claims", 
                        portfolio_summary['total_claims'],
                        help="Total unique claims across all risk levels"
                    )
                
                with col2:
                    st.metric(
                        "Total Reserve", 
                        f"${portfolio_summary['total_reserve']:,.0f}",
                        help="Total current reserve across all claims"
                    )
                
                with col3:
                    st.metric(
                        "Total Paid", 
                        f"${portfolio_summary['total_paid']:,.0f}",
                        help="Total payments made across all claims"
                    )
                
                with col4:
                    st.metric(
                        "Total Incurred", 
                        f"${portfolio_summary['total_incurred']:,.0f}",
                        help="Total incurred costs across all claims"
                    )
                
                # Risk and Status Breakdown
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("**📊 Risk Level Breakdown:**")
                    for risk, count in portfolio_summary['risk_breakdown'].items():
                        st.write(f"• {risk}: {count} claims")
                
                with col2:
                    st.markdown("**📊 Status Breakdown:**")
                    for status, count in portfolio_summary['status_breakdown'].items():
                        st.write(f"• {status}: {count} claims")
                
                # Natural Language Query Interface
                st.markdown("---")
                st.subheader("💬 Ask Portfolio Questions")
                
                # Query input
                query = st.text_input(
                    "Ask a question about your portfolio:",
                    placeholder="e.g., 'How many claims have payments over $50k?' or 'Show claims with no payment and no expense'",
                    help="Ask natural language questions about your claims portfolio"
                )
                
                if query:
                    with st.spinner("🔍 Analyzing portfolio..."):
                        # Get query results
                        result = general_agent.answer_query(query)
                        
                        if result['type'] == 'error':
                            st.error(f"❌ Error processing query: {result['error']}")
                        else:
                            # Display query results
                            st.markdown(f"**📋 Query Results for: '{query}'**")
                            
                            if result['type'] == 'portfolio_summary':
                                st.info("Showing portfolio summary as requested.")
                            
                            elif result['type'] == 'payment_threshold':
                                st.success(f"Found {len(result['data'])} claims with payments ≥ ${result['threshold']:,.2f}")
                                
                                if len(result['data']) > 0:
                                    # Display results in a table
                                    display_data = result['data'][['clmNum', 'risk_level', 'clmStatus', 'paid_cumsum', 'reserve_cumsum', 'clmCause']].copy()
                                    display_data['paid_cumsum'] = display_data['paid_cumsum'].apply(lambda x: f"${x:,.2f}")
                                    display_data['reserve_cumsum'] = display_data['reserve_cumsum'].apply(lambda x: f"${x:,.2f}")
                                    
                                    st.dataframe(
                                        display_data,
                                        use_container_width=True,
                                        hide_index=True,
                                        column_config={
                                            "clmNum": st.column_config.TextColumn("Claim #", width="medium"),
                                            "risk_level": st.column_config.TextColumn("Risk", width="small"),
                                            "clmStatus": st.column_config.TextColumn("Status", width="small"),
                                            "paid_cumsum": st.column_config.TextColumn("Total Paid", width="medium"),
                                            "reserve_cumsum": st.column_config.TextColumn("Reserve", width="medium"),
                                            "clmCause": st.column_config.TextColumn("Cause", width="medium")
                                        }
                                    )
                            
                            elif result['type'] == 'expense_threshold':
                                st.success(f"Found {len(result['data'])} claims with expenses ≥ ${result['threshold']:,.2f}")
                                
                                if len(result['data']) > 0:
                                    # Display results in a table
                                    display_data = result['data'][['clmNum', 'risk_level', 'clmStatus', 'expense_cumsum', 'reserve_cumsum', 'clmCause']].copy()
                                    display_data['expense_cumsum'] = display_data['expense_cumsum'].apply(lambda x: f"${x:,.2f}")
                                    display_data['reserve_cumsum'] = display_data['reserve_cumsum'].apply(lambda x: f"${x:,.2f}")
                                    
                                    st.dataframe(
                                        display_data,
                                        use_container_width=True,
                                        hide_index=True,
                                        column_config={
                                            "clmNum": st.column_config.TextColumn("Claim #", width="medium"),
                                            "risk_level": st.column_config.TextColumn("Risk", width="small"),
                                            "clmStatus": st.column_config.TextColumn("Status", width="small"),
                                            "expense_cumsum": st.column_config.TextColumn("Total Expenses", width="medium"),
                                            "reserve_cumsum": st.column_config.TextColumn("Reserve", width="medium"),
                                            "clmCause": st.column_config.TextColumn("Cause", width="medium")
                                        }
                                    )
                            
                            elif result['type'] == 'no_activity':
                                st.success(f"Found {len(result['data'])} claims with no payment and no expense")
                                
                                if len(result['data']) > 0:
                                    display_data = result['data'][['clmNum', 'risk_level', 'clmStatus', 'reserve_cumsum', 'clmCause']].copy()
                                    display_data['reserve_cumsum'] = display_data['reserve_cumsum'].apply(lambda x: f"${x:,.2f}")
                                    
                                    st.dataframe(
                                        display_data,
                                        use_container_width=True,
                                        hide_index=True,
                                        column_config={
                                            "clmNum": st.column_config.TextColumn("Claim #", width="medium"),
                                            "risk_level": st.column_config.TextColumn("Risk", width="small"),
                                            "clmStatus": st.column_config.TextColumn("Status", width="small"),
                                            "reserve_cumsum": st.column_config.TextColumn("Reserve", width="medium"),
                                            "clmCause": st.column_config.TextColumn("Cause", width="medium")
                                        }
                                    )
                            
                            elif result['type'] == 'law_firm_search':
                                st.success(f"Found {len(result['data'])} notes mentioning law firm '{result['firm_name']}'")
                                
                                if len(result['data']) > 0:
                                    # Show unique claims
                                    unique_claims = result['data']['clmNum'].unique()
                                    st.write(f"**Claims involved:** {', '.join(unique_claims)}")
                                    
                                    # Show sample notes
                                    st.markdown("**Sample notes:**")
                                    for _, note in result['data'].head(3).iterrows():
                                        st.write(f"**{note['clmNum']}** ({note['dateNote'].strftime('%Y-%m-%d')}): {note['note'][:200]}...")
                            
                            elif result['type'] in ['risk_level', 'status', 'cause']:
                                st.success(f"Found {len(result['data'])} claims matching your criteria")
                                
                                if len(result['data']) > 0:
                                    display_data = result['data'][['clmNum', 'risk_level', 'clmStatus', 'incurred_cumsum', 'reserve_cumsum', 'clmCause']].copy()
                                    display_data['incurred_cumsum'] = display_data['incurred_cumsum'].apply(lambda x: f"${x:,.2f}")
                                    display_data['reserve_cumsum'] = display_data['reserve_cumsum'].apply(lambda x: f"${x:,.2f}")
                                    
                                    st.dataframe(
                                        display_data,
                                        use_container_width=True,
                                        hide_index=True,
                                        column_config={
                                            "clmNum": st.column_config.TextColumn("Claim #", width="medium"),
                                            "risk_level": st.column_config.TextColumn("Risk", width="small"),
                                            "clmStatus": st.column_config.TextColumn("Status", width="small"),
                                            "incurred_cumsum": st.column_config.TextColumn("Total Incurred", width="medium"),
                                            "reserve_cumsum": st.column_config.TextColumn("Reserve", width="medium"),
                                            "clmCause": st.column_config.TextColumn("Cause", width="medium")
                                        }
                                    )
                            
                            elif result['type'] == 'feedback':
                                st.success(f"Found {len(result['data'])} claims with representative feedback")
                                
                                if len(result['data']) > 0:
                                    display_data = result['data'][['clmNum', 'risk_level', 'clmStatus', 'rep_name', 'rep_confidence', 'timestamp']].copy()
                                    display_data['rep_confidence'] = display_data['rep_confidence'].apply(lambda x: f"{x:.1%}")
                                    
                                    st.dataframe(
                                        display_data,
                                        use_container_width=True,
                                        hide_index=True,
                                        column_config={
                                            "clmNum": st.column_config.TextColumn("Claim #", width="medium"),
                                            "risk_level": st.column_config.TextColumn("Risk", width="small"),
                                            "clmStatus": st.column_config.TextColumn("Status", width="small"),
                                            "rep_name": st.column_config.TextColumn("Rep Name", width="medium"),
                                            "rep_confidence": st.column_config.TextColumn("Confidence", width="medium"),
                                            "timestamp": st.column_config.TextColumn("Assessment Date", width="medium")
                                        }
                                    )
                            
                            # Show query type for debugging
                            st.caption(f"Query type: {result['type']}")
                            
                            # Export results
                            if 'data' in result and hasattr(result['data'], 'to_csv'):
                                # Only show export for DataFrame results
                                if len(result['data']) > 0:
                                    st.markdown("**📤 Export Results**")
                                    csv_data = result['data'].to_csv(index=False)
                                    st.download_button(
                                        label="📥 Download Results as CSV",
                                        data=csv_data,
                                        file_name=f"portfolio_query_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                                        mime="text/csv"
                                    )
                            elif result['type'] == 'portfolio_summary':
                                # For portfolio summary, show summary metrics instead of export
                                st.markdown("**📊 Portfolio Summary Metrics**")
                                summary = result['data']
                                col1, col2, col3 = st.columns(3)
                                with col1:
                                    st.metric("High Value Claims (>$100k)", summary['high_value_claims_count'])
                                with col2:
                                    st.metric("No Payment Claims", summary['no_payment_claims_count'])
                                with col3:
                                    st.metric("No Expense Claims", summary['no_expense_claims_count'])
            else:
                st.error(f"❌ Error loading portfolio data: {portfolio_summary['error']}")
        else:
            st.warning("General claims agent not available")
        
        # Claim Lifetime Visualization
        if selected_claim_from_table:
            st.markdown("---")
            st.subheader(f"📊 Claim Lifetime Analysis: {selected_claim_from_table}")
            
            try:
                # Load both CRITICAL and HIGH risk claims data for visualization
                all_transactions = []
                
                # Load CRITICAL claims
                critical_claims_file = os.path.join("./_data", "critical_claims.csv")
                if os.path.exists(critical_claims_file):
                    critical_df = pd.read_csv(critical_claims_file)
                    critical_df['datetxn'] = pd.to_datetime(critical_df['datetxn'])
                    all_transactions.append(critical_df)
                
                # Load HIGH risk claims
                high_risk_claims_file = os.path.join("./_data", "high_risk_claims.csv")
                if os.path.exists(high_risk_claims_file):
                    high_risk_df = pd.read_csv(high_risk_claims_file)
                    high_risk_df['datetxn'] = pd.to_datetime(high_risk_df['datetxn'])
                    all_transactions.append(high_risk_df)
                
                if len(all_transactions) > 0:
                    # Combine all transactions
                    df_raw_txn = pd.concat(all_transactions, ignore_index=True)
                    df_raw_txn = df_raw_txn.sort_values(['risk_level', 'clmNum', 'datetxn'])
                    
                    # Plot the claim lifetime
                    plot_single_claim_lifetime(df_raw_txn, selected_claim_from_table)
                else:
                    st.warning("No claims transaction data found. Please ensure critical_claims.csv or high_risk_claims.csv exists.")
            except Exception as e:
                st.error(f"Error loading claim lifetime data: {e}")
        
        # Bayesian Status Prediction Section
        st.markdown("---")
        st.subheader(f"🔮 Bayesian Final Status Prediction for {selected_claim_from_table}")
                
              
        predictor = get_claim_predictor()
        
        if predictor:
            # Get prediction for the selected claim
            prediction = predictor.predict_final_status_probabilities(selected_claim_from_table)
            
            # Display prediction results
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric(
                    "PAID Probability", 
                    f"{prediction['probabilities']['PAID']:.1%}",
                    help="Probability this claim will end with payments"
                )
            
            with col2:
                st.metric(
                    "CLOSED Probability", 
                    f"{prediction['probabilities']['CLOSED']:.1%}",
                    help="Probability this claim will close without payments"
                )
            
            with col3:
                st.metric(
                    "DENIED Probability", 
                    f"{prediction['probabilities']['DENIED']:.1%}",
                    help="Probability this claim will be denied"
                )
            
            with col4:
                st.metric(
                    "Confidence", 
                    f"{prediction['confidence']:.1%}",
                    help="Confidence level in this prediction"
                )
            
                # Display key factors
            st.markdown("**🔍 Key Factors Influencing Prediction:**")
            factors_col1, factors_col2 = st.columns(2)
            
            with factors_col1:
                for factor in prediction['factors'][:len(prediction['factors'])//2]:
                    st.markdown(f"• {factor}")
            
            with factors_col2:
                for factor in prediction['factors'][len(prediction['factors'])//2:]:
                    st.markdown(f"• {factor}")
            
            # Prediction metadata
            st.caption(f"**Prediction Date:** {prediction['prediction_date']} | **Last Updated:** {prediction['last_updated']}")
            
            # Add a note about the placeholder nature
            st.info("""
            **ℹ️ Note:** These are placeholder probabilities for demonstration purposes. 
            In production, this would use actual Bayesian calculations based on historical data, 
            claim characteristics, and machine learning models.
            """)
        
        else:
            st.warning("Claim status predictor not available")
         
                 # Claim Representative Feedback Section
        st.markdown("---")
        st.subheader(f"👤 Claim Representative Expert Assessment")
        
        # Get feedback storage and load existing feedback
        feedback_storage = get_feedback_storage()
        existing_feedback = None
        
        if feedback_storage:
            existing_feedback = feedback_storage.load_feedback(selected_claim_from_table)
        
        # Initialize feedback data
        if existing_feedback:
            # Use existing feedback data
            feedback_data = {
                'rep_paid_prob': existing_feedback['rep_paid_prob'],
                'rep_closed_prob': existing_feedback['rep_closed_prob'],
                'rep_denied_prob': existing_feedback['rep_denied_prob'],
                'rep_confidence': existing_feedback['rep_confidence'],
                'thought_process': existing_feedback['thought_process'],
                'timestamp': existing_feedback['timestamp'],
                'rep_name': existing_feedback['rep_name'],
                'custom_factors': existing_feedback.get('custom_factors', '')
            }
        else:
            # Initialize with default values
            feedback_data = {
                'rep_paid_prob': 0.0,
                'rep_closed_prob': 0.0,
                'rep_denied_prob': 0.0,
                'rep_confidence': 0.0,
                'thought_process': '',
                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'rep_name': ''
            }
         
         # Create two columns for input
        col1, col2 = st.columns([1, 1])
         
        with col1:
            st.markdown("**📊 Your Probability Assessment**")
            
            # Representative's probability inputs
            rep_paid = st.slider(
                "PAID Probability (%)", 
                0, 100, 
                int(feedback_data['rep_paid_prob'] * 100),
                help="Your assessment of the probability this claim will end with payments"
            )
            
            rep_closed = st.slider(
                "CLOSED Probability (%)", 
                0, 100, 
                int(feedback_data['rep_closed_prob'] * 100),
                help="Your assessment of the probability this claim will close without payments"
            )
            
            rep_denied = st.slider(
                "DENIED Probability (%)", 
                0, 100, 
                int(feedback_data['rep_denied_prob'] * 100),
                help="Your assessment of the probability this claim will be denied"
            )
            
            # Auto-calculate the third probability to ensure they sum to 100%
            total_prob = rep_paid + rep_closed + rep_denied
            if total_prob != 100:
                st.warning(f"⚠️ Probabilities sum to {total_prob}%. They should equal 100%.")
                
                # Auto-adjust the largest probability
                if total_prob > 100:
                    excess = total_prob - 100
                    if rep_paid >= rep_closed and rep_paid >= rep_denied:
                        rep_paid -= excess
                    elif rep_closed >= rep_paid and rep_closed >= rep_denied:
                        rep_closed -= excess
                    else:
                        rep_denied -= excess
                else:
                    deficit = 100 - total_prob
                    if rep_paid <= rep_closed and rep_paid <= rep_denied:
                        rep_paid += deficit
                    elif rep_closed <= rep_paid and rep_closed <= rep_denied:
                        rep_closed += deficit
                    else:
                        rep_denied += deficit
            
            # Representative's confidence
            rep_confidence = st.slider(
                "Your Confidence Level (%)", 
                0, 100, 
                int(feedback_data['rep_confidence'] * 100),
                help="How confident are you in your assessment?"
            )
            
            # Representative name
            rep_name = st.text_input(
                "Your Name/ID",
                value=feedback_data['rep_name'],
                placeholder="Enter your name or ID",
                help="For tracking and audit purposes"
            )
        
        with col2:
            st.markdown("**🔍 Your Thought Process & Insights**")
            
            # Text area for thought process
            thought_process = st.text_area(
                "Explain your reasoning:",
                value=feedback_data['thought_process'],
                height=200,
                placeholder="Share your insights, additional information, or reasoning that the model might not have considered...",
                help="This helps document your decision rationale and can improve future predictions"
            )
            
            # Key factors that influenced decision
            st.markdown("**🎯 Key Factors You Considered:**")
            
            # Checkboxes for common factors
            col2a, col2b = st.columns(2)
            
            with col2a:
                st.markdown("**Financial Factors:**")
                st.checkbox("Payment history", key=f"factor_payment_{selected_claim_from_table}")
                st.checkbox("Reserve adequacy", key=f"factor_reserve_{selected_claim_from_table}")
                st.checkbox("Expense patterns", key=f"factor_expense_{selected_claim_from_table}")
                st.checkbox("Claim size", key=f"factor_size_{selected_claim_from_table}")
            
            with col2b:
                st.markdown("**Operational Factors:**")
                st.checkbox("Legal complexity", key=f"factor_legal_{selected_claim_from_table}")
                st.checkbox("Documentation quality", key=f"factor_docs_{selected_claim_from_table}")
                st.checkbox("Third party involvement", key=f"factor_third_party_{selected_claim_from_table}")
                st.checkbox("Timeline factors", key=f"factor_timeline_{selected_claim_from_table}")
            
            # Additional custom factors
            custom_factors = st.text_input(
                "Other factors:",
                placeholder="e.g., Special circumstances, industry knowledge, etc.",
                help="Any other factors not listed above"
            )
        
        # Save button
        if st.button("💾 Save Expert Assessment", type="primary"):
            if feedback_storage:
                # Prepare feedback data for storage
                feedback_data_to_save = {
                    'claim_number': selected_claim_from_table,
                    'model_paid_prob': prediction['probabilities']['PAID'],
                    'model_closed_prob': prediction['probabilities']['CLOSED'],
                    'model_denied_prob': prediction['probabilities']['DENIED'],
                    'model_confidence': prediction['confidence'],
                    'rep_paid_prob': rep_paid / 100.0,
                    'rep_closed_prob': rep_closed / 100.0,
                    'rep_denied_prob': rep_denied / 100.0,
                    'rep_confidence': rep_confidence / 100.0,
                    'thought_process': thought_process,
                    'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                    'rep_name': rep_name,
                    'custom_factors': custom_factors
                }
                
                # Save to persistent storage
                success = feedback_storage.save_feedback(feedback_data_to_save)
                
                if success:
                    st.success("✅ Expert assessment saved successfully! It will persist across sessions.")
                    st.rerun()
                else:
                    st.error("❌ Failed to save assessment. Please try again.")
            else:
                st.error("❌ Feedback storage not available. Cannot save assessment.")
        
        # Display comparison if feedback exists
        if existing_feedback and existing_feedback['rep_name']:
            st.markdown("---")
            st.subheader("📊 Model vs. Expert Comparison")
            
            # Create comparison metrics
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric(
                    "PAID Probability",
                    f"{existing_feedback['rep_paid_prob']:.1%}",
                    delta=f"{(existing_feedback['rep_paid_prob'] - prediction['probabilities']['PAID']):+.1%}",
                    delta_color="normal"
                )
                st.caption("Expert vs Model")
            
            with col2:
                st.metric(
                    "CLOSED Probability",
                    f"{existing_feedback['rep_closed_prob']:.1%}",
                    delta=f"{(existing_feedback['rep_closed_prob'] - prediction['probabilities']['CLOSED']):+.1%}",
                    delta_color="normal"
                )
                st.caption("Expert vs Model")
            
            with col3:
                st.metric(
                    "DENIED Probability",
                    f"{existing_feedback['rep_denied_prob']:.1%}",
                    delta=f"{(existing_feedback['rep_denied_prob'] - prediction['probabilities']['DENIED']):+.1%}",
                    delta_color="normal"
                )
                st.caption("Expert vs Model")
            
            with col4:
                st.metric(
                    "Confidence Level",
                    f"{existing_feedback['rep_confidence']:.1%}",
                    delta=f"{(existing_feedback['rep_confidence'] - prediction['confidence']):+.1%}",
                    delta_color="normal"
                )
                st.caption("Expert vs Model")
            
            # Display thought process
            st.markdown("**💭 Expert Reasoning:**")
            st.info(existing_feedback['thought_process'])
            
            # Display metadata
            st.caption(f"**Assessed by:** {existing_feedback['rep_name']} | **Timestamp:** {existing_feedback['timestamp']}")
        
        # Export feedback data
        if existing_feedback and existing_feedback['rep_name']:
            st.markdown("---")
            st.markdown("**📤 Export Feedback Data**")
            
            # Create feedback data for export
            feedback_data = {
                'claim_number': selected_claim_from_table,
                'model_paid_prob': prediction['probabilities']['PAID'],
                'model_closed_prob': prediction['probabilities']['CLOSED'],
                'model_denied_prob': prediction['probabilities']['DENIED'],
                'model_confidence': prediction['confidence'],
                'rep_paid_prob': existing_feedback['rep_paid_prob'],
                'rep_closed_prob': existing_feedback['rep_closed_prob'],
                'rep_denied_prob': existing_feedback['rep_denied_prob'],
                'rep_confidence': existing_feedback['rep_confidence'],
                'thought_process': existing_feedback['thought_process'],
                'rep_name': existing_feedback['rep_name'],
                'timestamp': existing_feedback['timestamp'],
                'custom_factors': existing_feedback.get('custom_factors', '')
            }
            
            # Convert to DataFrame for download
            feedback_df = pd.DataFrame([feedback_data])
            
            st.download_button(
                label="📥 Download Feedback as CSV",
                data=feedback_df.to_csv(index=False),
                file_name=f"claim_feedback_{selected_claim_from_table}_{datetime.now().strftime('%Y%m%d')}.csv",
                mime="text/csv"
            )
                    
           
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
                            # Get claim data for context from both CRITICAL and HIGH risk claims
                            all_transactions = []
                            
                            # Load CRITICAL claims
                            critical_claims_file = os.path.join("./_data", "critical_claims.csv")
                            if os.path.exists(critical_claims_file):
                                critical_df = pd.read_csv(critical_claims_file)
                                critical_df['datetxn'] = pd.to_datetime(critical_df['datetxn'])
                                all_transactions.append(critical_df)
                            
                            # Load HIGH risk claims
                            high_risk_claims_file = os.path.join("./_data", "high_risk_claims.csv")
                            if os.path.exists(high_risk_claims_file):
                                high_risk_df = pd.read_csv(high_risk_claims_file)
                                high_risk_df['datetxn'] = pd.to_datetime(high_risk_df['datetxn'])
                                all_transactions.append(high_risk_df)
                            
                            if len(all_transactions) > 0:
                                # Combine all transactions
                                df_raw_txn = pd.concat(all_transactions, ignore_index=True)
                                df_raw_txn = df_raw_txn.sort_values(['risk_level', 'clmNum', 'datetxn'])
                                
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
            
            # Load claim notes data using the agent (combines CRITICAL and HIGH risk notes)
            try:
                # Get notes for the selected claim from both sources
                claim_notes = agent.get_notes_by_claim(selected_claim_from_table)
                
                if len(claim_notes) > 0:
                    # Convert whenadded to dateNote for compatibility with existing code
                    claim_notes = claim_notes.copy()
                    claim_notes['dateNote'] = claim_notes['whenadded']
                    claim_notes = claim_notes.sort_values('dateNote')
                    
                    if len(claim_notes) > 0:
                        # Show notes summary
                        col1, col2, col3, col4, col5 = st.columns(5)
                        with col1:
                            st.metric("Total Notes", len(claim_notes))
                        with col2:
                            st.metric("Emails", len(claim_notes[claim_notes['note_type'] == 'email']))
                        with col3:
                            st.metric("High Priority", len(claim_notes[claim_notes['priority'] == 'High']))
                        with col4:
                            st.metric("Date Range", f"{(claim_notes['dateNote'].max() - claim_notes['dateNote'].min()).days} days")
                        with col5:
                            # Show notes source breakdown
                            if 'note_source' in claim_notes.columns:
                                critical_notes = len(claim_notes[claim_notes['note_source'] == 'CRITICAL'])
                                high_notes = len(claim_notes[claim_notes['note_source'] == 'HIGH'])
                                st.metric("CRITICAL Notes", critical_notes)
                                st.caption(f"HIGH: {high_notes}")
                            else:
                                st.metric("Source", "Mixed")
                        
                        # Add filters for notes
                        st.markdown("**🔍 Filter Notes:**")
                        col1, col2, col3, col4 = st.columns(4)
                        
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
                        
                        with col4:
                            # Add source filter if available
                            if 'note_source' in claim_notes.columns:
                                source_filter = st.selectbox(
                                    "Source:",
                                    ["All", "CRITICAL", "HIGH"],
                                    key=f"source_filter_{selected_claim_from_table}"
                                )
                            else:
                                source_filter = "All"
                        
                        # Apply filters
                        filtered_notes = claim_notes.copy()
                        if note_type_filter != "All":
                            filtered_notes = filtered_notes[filtered_notes['note_type'] == note_type_filter]
                        if priority_filter != "All":
                            filtered_notes = filtered_notes[filtered_notes['priority'] == priority_filter]
                        if author_filter != "All":
                            filtered_notes = filtered_notes[filtered_notes['author'] == author_filter]
                        if source_filter != "All" and 'note_source' in filtered_notes.columns:
                            filtered_notes = filtered_notes[filtered_notes['note_source'] == source_filter]
                        
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
                                    col2a, col2b, col2c, col2d = st.columns(4)
                                    with col2a:
                                        st.caption(f"Type: {row['note_type'].replace('_', ' ').title()}")
                                    with col2b:
                                        st.caption(f"Priority: {row['priority']}")
                                    with col2c:
                                        st.caption(f"Note #{idx + 1}")
                                    with col2d:
                                        # Show note source if available
                                        if 'note_source' in row:
                                            source_emoji = "🔴" if row['note_source'] == 'CRITICAL' else "🟡"
                                            st.caption(f"{source_emoji} {row['note_source']}")
                                        else:
                                            st.caption("📝 General")
                                
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
        
    
if __name__ == "__main__":
    main()
