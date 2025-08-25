#!/usr/bin/env python3
"""
General Claims Agent

A comprehensive agent that can analyze ALL claims and notes across the entire portfolio.
Provides portfolio-level insights and can answer complex queries about claims, payments, expenses, and notes.
"""

import pandas as pd
import numpy as np
import os
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
import re

class GeneralClaimsAgent:
    """
    General agent for portfolio-wide claims analysis and insights.
    """
    
    def __init__(self, data_dir: str = "./_data"):
        self.data_dir = data_dir
        self.claims_data = None
        self.notes_data = None
        self.feedback_data = None
        self.load_all_data()
    
    def load_all_data(self):
        """Load all available data sources."""
        try:
            all_claims = []
            all_notes = []
            
            # Load CRITICAL claims
            critical_claims_file = os.path.join(self.data_dir, "critical_claims.csv")
            if os.path.exists(critical_claims_file):
                critical_df = pd.read_csv(critical_claims_file)
                critical_df['datetxn'] = pd.to_datetime(critical_df['datetxn'])
                critical_df['source'] = 'CRITICAL'
                all_claims.append(critical_df)
                print(f"Loaded {len(critical_df)} CRITICAL claim transactions")
            
            # Load HIGH risk claims
            high_risk_claims_file = os.path.join(self.data_dir, "high_risk_claims.csv")
            if os.path.exists(high_risk_claims_file):
                high_risk_df = pd.read_csv(high_risk_claims_file)
                high_risk_df['datetxn'] = pd.to_datetime(high_risk_df['datetxn'])
                high_risk_df['source'] = 'HIGH'
                all_claims.append(high_risk_df)
                print(f"Loaded {len(high_risk_df)} HIGH risk claim transactions")
            
            # Load CRITICAL claim notes
            critical_notes_file = os.path.join(self.data_dir, "claim_notes.csv")
            if os.path.exists(critical_notes_file):
                critical_notes_df = pd.read_csv(critical_notes_file)
                critical_notes_df['dateNote'] = pd.to_datetime(critical_notes_df['dateNote'])
                critical_notes_df['source'] = 'CRITICAL'
                all_notes.append(critical_notes_df)
                print(f"Loaded {len(critical_notes_df)} CRITICAL claim notes")
            
            # Load HIGH risk claim notes
            high_risk_notes_file = os.path.join(self.data_dir, "high_risk_claim_notes.csv")
            if os.path.exists(high_risk_notes_file):
                high_risk_notes_df = pd.read_csv(high_risk_notes_file)
                high_risk_notes_df['dateNote'] = pd.to_datetime(high_risk_notes_df['dateNote'])
                high_risk_notes_df['source'] = 'HIGH'
                all_notes.append(high_risk_notes_df)
                print(f"Loaded {len(high_risk_notes_df)} HIGH risk claim notes")
            
            # Load general notes (if available)
            general_notes_file = os.path.join(self.data_dir, "notes.parquet")
            if os.path.exists(general_notes_file):
                try:
                    general_notes_df = pd.read_parquet(general_notes_file)
                    general_notes_df['source'] = 'GENERAL'
                    all_notes.append(general_notes_df)
                    print(f"Loaded {len(general_notes_df)} general notes")
                except Exception as e:
                    print(f"Could not load general notes: {e}")
            
            # Load feedback data
            feedback_file = os.path.join(self.data_dir, "claim_feedback.csv")
            if os.path.exists(feedback_file):
                self.feedback_data = pd.read_csv(feedback_file)
                print(f"Loaded {len(self.feedback_data)} feedback records")
            
            # Combine all data
            if len(all_claims) > 0:
                self.claims_data = pd.concat(all_claims, ignore_index=True)
                print(f"Total claims transactions: {len(self.claims_data)}")
            
            if len(all_notes) > 0:
                self.notes_data = pd.concat(all_notes, ignore_index=True)
                print(f"Total notes: {len(self.notes_data)}")
                
        except Exception as e:
            print(f"Error loading data: {e}")
    
    def get_portfolio_summary(self) -> Dict:
        """Get comprehensive portfolio summary."""
        if self.claims_data is None:
            return {"error": "No claims data available"}
        
        try:
            # Get latest transaction for each claim
            latest_claims = self.claims_data.groupby('clmNum').agg({
                'risk_level': 'last',
                'clmStatus': 'last',
                'reserve_cumsum': 'last',
                'paid_cumsum': 'last',
                'expense_cumsum': 'last',
                'incurred_cumsum': 'last',
                'datetxn': 'last',
                'clmCause': 'last',
                'source': 'last'
            }).reset_index()
            
            # Calculate portfolio metrics
            total_claims = len(latest_claims)
            total_reserve = latest_claims['reserve_cumsum'].sum()
            total_paid = latest_claims['paid_cumsum'].sum()
            total_expense = latest_claims['expense_cumsum'].sum()
            total_incurred = latest_claims['incurred_cumsum'].sum()
            
            # Risk level breakdown
            risk_breakdown = latest_claims['risk_level'].value_counts().to_dict()
            
            # Status breakdown
            status_breakdown = latest_claims['clmStatus'].value_counts().to_dict()
            
            # Cause breakdown
            cause_breakdown = latest_claims['clmCause'].value_counts().to_dict()
            
            # High-value claims (>$100k)
            high_value_claims = latest_claims[latest_claims['incurred_cumsum'] > 100000]
            
            # Claims with no payments
            no_payment_claims = latest_claims[latest_claims['paid_cumsum'] == 0]
            
            # Claims with no expenses
            no_expense_claims = latest_claims[latest_claims['expense_cumsum'] == 0]
            
            summary = {
                'total_claims': total_claims,
                'total_reserve': total_reserve,
                'total_paid': total_paid,
                'total_expense': total_expense,
                'total_incurred': total_incurred,
                'risk_breakdown': risk_breakdown,
                'status_breakdown': status_breakdown,
                'cause_breakdown': cause_breakdown,
                'high_value_claims_count': len(high_value_claims),
                'no_payment_claims_count': len(no_payment_claims),
                'no_expense_claims_count': len(no_expense_claims),
                'avg_claim_value': total_incurred / total_claims if total_claims > 0 else 0,
                'avg_reserve': total_reserve / total_claims if total_claims > 0 else 0
            }
            
            return summary
            
        except Exception as e:
            return {"error": f"Error generating portfolio summary: {e}"}
    
    def query_claims_by_payment_threshold(self, threshold: float) -> pd.DataFrame:
        """Find claims with cumulative payments above a threshold."""
        if self.claims_data is None:
            return pd.DataFrame()
        
        try:
            # Get latest transaction for each claim
            latest_claims = self.claims_data.groupby('clmNum').agg({
                'risk_level': 'last',
                'clmStatus': 'last',
                'reserve_cumsum': 'last',
                'paid_cumsum': 'last',
                'expense_cumsum': 'last',
                'incurred_cumsum': 'last',
                'datetxn': 'last',
                'clmCause': 'last',
                'source': 'last'
            }).reset_index()
            
            # Filter by payment threshold
            high_payment_claims = latest_claims[latest_claims['paid_cumsum'] >= threshold]
            
            return high_payment_claims.sort_values('paid_cumsum', ascending=False)
            
        except Exception as e:
            print(f"Error querying claims by payment threshold: {e}")
            return pd.DataFrame()
    
    def query_claims_by_expense_threshold(self, threshold: float) -> pd.DataFrame:
        """Find claims with cumulative expenses above a threshold."""
        if self.claims_data is None:
            return pd.DataFrame()
        
        try:
            # Get latest transaction for each claim
            latest_claims = self.claims_data.groupby('clmNum').agg({
                'risk_level': 'last',
                'clmStatus': 'last',
                'reserve_cumsum': 'last',
                'paid_cumsum': 'last',
                'expense_cumsum': 'last',
                'incurred_cumsum': 'last',
                'datetxn': 'last',
                'clmCause': 'last',
                'source': 'last'
            }).reset_index()
            
            # Filter by expense threshold
            high_expense_claims = latest_claims[latest_claims['expense_cumsum'] >= threshold]
            
            return high_expense_claims.sort_values('expense_cumsum', ascending=False)
            
        except Exception as e:
            print(f"Error querying claims by expense threshold: {e}")
            return pd.DataFrame()
    
    def query_claims_no_payment_no_expense(self) -> pd.DataFrame:
        """Find claims with no payments and no expenses."""
        if self.claims_data is None:
            return pd.DataFrame()
        
        try:
            # Get latest transaction for each claim
            latest_claims = self.claims_data.groupby('clmNum').agg({
                'risk_level': 'last',
                'clmStatus': 'last',
                'reserve_cumsum': 'last',
                'paid_cumsum': 'last',
                'expense_cumsum': 'last',
                'incurred_cumsum': 'last',
                'datetxn': 'last',
                'clmCause': 'last',
                'source': 'last'
            }).reset_index()
            
            # Filter claims with no payments and no expenses
            no_activity_claims = latest_claims[
                (latest_claims['paid_cumsum'] == 0) & 
                (latest_claims['expense_cumsum'] == 0)
            ]
            
            return no_activity_claims.sort_values('reserve_cumsum', ascending=False)
            
        except Exception as e:
            print(f"Error querying claims with no activity: {e}")
            return pd.DataFrame()
    
    def search_notes_by_content(self, search_terms: List[str], case_sensitive: bool = False) -> pd.DataFrame:
        """Search notes by content using multiple search terms."""
        if self.notes_data is None:
            return pd.DataFrame()
        
        try:
            # Convert search terms to regex pattern
            if case_sensitive:
                pattern = '|'.join(search_terms)
            else:
                pattern = '|'.join([f'(?i){term}' for term in search_terms])
            
            # Search in note content
            mask = self.notes_data['note'].str.contains(pattern, regex=True, na=False)
            matching_notes = self.notes_data[mask].copy()
            
            # Add claim summary info
            if len(matching_notes) > 0 and self.claims_data is not None:
                # Get latest claim info for each matching note
                latest_claims = self.claims_data.groupby('clmNum').agg({
                    'risk_level': 'last',
                    'clmStatus': 'last',
                    'reserve_cumsum': 'last',
                    'paid_cumsum': 'last',
                    'expense_cumsum': 'last'
                }).reset_index()
                
                # Merge with notes
                matching_notes = matching_notes.merge(
                    latest_claims, 
                    on='clmNum', 
                    how='left'
                )
            
            return matching_notes.sort_values('dateNote', ascending=False)
            
        except Exception as e:
            print(f"Error searching notes: {e}")
            return pd.DataFrame()
    
    def search_notes_by_law_firm(self, law_firm_name: str) -> pd.DataFrame:
        """Search notes for mentions of a specific law firm."""
        if self.notes_data is None:
            return pd.DataFrame()
        
        try:
            # Search for law firm mentions (case insensitive)
            law_firm_pattern = f'(?i){re.escape(law_firm_name)}'
            mask = self.notes_data['note'].str.contains(law_firm_pattern, regex=True, na=False)
            law_firm_notes = self.notes_data[mask].copy()
            
            # Get unique claims mentioned
            unique_claims = law_firm_notes['clmNum'].unique()
            
            # Get claim summary info
            if len(law_firm_notes) > 0 and self.claims_data is not None:
                latest_claims = self.claims_data.groupby('clmNum').agg({
                    'risk_level': 'last',
                    'clmStatus': 'last',
                    'reserve_cumsum': 'last',
                    'paid_cumsum': 'last',
                    'expense_cumsum': 'last',
                    'clmCause': 'last'
                }).reset_index()
                
                # Merge with notes
                law_firm_notes = law_firm_notes.merge(
                    latest_claims, 
                    on='clmNum', 
                    how='left'
                )
            
            return law_firm_notes.sort_values('dateNote', ascending=False)
            
        except Exception as e:
            print(f"Error searching notes by law firm: {e}")
            return pd.DataFrame()
    
    def get_claims_by_risk_level(self, risk_level: str) -> pd.DataFrame:
        """Get all claims for a specific risk level."""
        if self.claims_data is None:
            return pd.DataFrame()
        
        try:
            # Get latest transaction for each claim
            latest_claims = self.claims_data.groupby('clmNum').agg({
                'risk_level': 'last',
                'clmStatus': 'last',
                'reserve_cumsum': 'last',
                'paid_cumsum': 'last',
                'expense_cumsum': 'last',
                'incurred_cumsum': 'last',
                'datetxn': 'last',
                'clmCause': 'last',
                'source': 'last'
            }).reset_index()
            
            # Filter by risk level
            filtered_claims = latest_claims[latest_claims['risk_level'] == risk_level.upper()]
            
            return filtered_claims.sort_values('incurred_cumsum', ascending=False)
            
        except Exception as e:
            print(f"Error getting claims by risk level: {e}")
            return pd.DataFrame()
    
    def get_claims_by_status(self, status: str) -> pd.DataFrame:
        """Get all claims for a specific status."""
        if self.claims_data is None:
            return pd.DataFrame()
        
        try:
            # Get latest transaction for each claim
            latest_claims = self.claims_data.groupby('clmNum').agg({
                'risk_level': 'last',
                'clmStatus': 'last',
                'reserve_cumsum': 'last',
                'paid_cumsum': 'last',
                'expense_cumsum': 'last',
                'incurred_cumsum': 'last',
                'datetxn': 'last',
                'clmCause': 'last',
                'source': 'last'
            }).reset_index()
            
            # Filter by status
            filtered_claims = latest_claims[latest_claims['clmStatus'] == status.upper()]
            
            return filtered_claims.sort_values('incurred_cumsum', ascending=False)
            
        except Exception as e:
            print(f"Error getting claims by status: {e}")
            return pd.DataFrame()
    
    def get_claims_by_cause(self, cause: str) -> pd.DataFrame:
        """Get all claims for a specific cause."""
        if self.claims_data is None:
            return pd.DataFrame()
        
        try:
            # Get latest transaction for each claim
            latest_claims = self.claims_data.groupby('clmNum').agg({
                'risk_level': 'last',
                'clmStatus': 'last',
                'reserve_cumsum': 'last',
                'paid_cumsum': 'last',
                'expense_cumsum': 'last',
                'incurred_cumsum': 'last',
                'datetxn': 'last',
                'clmCause': 'last',
                'source': 'last'
            }).reset_index()
            
            # Filter by cause (case insensitive)
            mask = latest_claims['clmCause'].str.contains(cause, case=False, na=False)
            filtered_claims = latest_claims[mask]
            
            return filtered_claims.sort_values('incurred_cumsum', ascending=False)
            
        except Exception as e:
            print(f"Error getting claims by cause: {e}")
            return pd.DataFrame()
    
    def get_claims_by_date_range(self, start_date: str, end_date: str) -> pd.DataFrame:
        """Get claims within a specific date range."""
        if self.claims_data is None:
            return pd.DataFrame()
        
        try:
            # Convert dates
            start_dt = pd.to_datetime(start_date)
            end_dt = pd.to_datetime(end_date)
            
            # Get latest transaction for each claim
            latest_claims = self.claims_data.groupby('clmNum').agg({
                'risk_level': 'last',
                'clmStatus': 'last',
                'reserve_cumsum': 'last',
                'paid_cumsum': 'last',
                'expense_cumsum': 'last',
                'incurred_cumsum': 'last',
                'datetxn': 'last',
                'clmCause': 'last',
                'source': 'last'
            }).reset_index()
            
            # Filter by date range
            mask = (latest_claims['datetxn'] >= start_dt) & (latest_claims['datetxn'] <= end_dt)
            filtered_claims = latest_claims[mask]
            
            return filtered_claims.sort_values('datetxn', ascending=False)
            
        except Exception as e:
            print(f"Error getting claims by date range: {e}")
            return pd.DataFrame()
    
    def get_high_risk_claims(self, reserve_threshold: float = 100000) -> pd.DataFrame:
        """Get claims with high reserve amounts."""
        if self.claims_data is None:
            return pd.DataFrame()
        
        try:
            # Get latest transaction for each claim
            latest_claims = self.claims_data.groupby('clmNum').agg({
                'risk_level': 'last',
                'clmStatus': 'last',
                'reserve_cumsum': 'last',
                'paid_cumsum': 'last',
                'expense_cumsum': 'last',
                'incurred_cumsum': 'last',
                'datetxn': 'last',
                'clmCause': 'last',
                'source': 'last'
            }).reset_index()
            
            # Filter by reserve threshold
            high_risk_claims = latest_claims[latest_claims['reserve_cumsum'] >= reserve_threshold]
            
            return high_risk_claims.sort_values('reserve_cumsum', ascending=False)
            
        except Exception as e:
            print(f"Error getting high risk claims: {e}")
            return pd.DataFrame()
    
    def get_claims_with_feedback(self) -> pd.DataFrame:
        """Get claims that have representative feedback."""
        if self.feedback_data is None or self.claims_data is None:
            return pd.DataFrame()
        
        try:
            # Get latest transaction for each claim
            latest_claims = self.claims_data.groupby('clmNum').agg({
                'risk_level': 'last',
                'clmStatus': 'last',
                'reserve_cumsum': 'last',
                'paid_cumsum': 'last',
                'expense_cumsum': 'last',
                'incurred_cumsum': 'last',
                'datetxn': 'last',
                'clmCause': 'last',
                'source': 'last'
            }).reset_index()
            
            # Get claims with feedback
            claims_with_feedback = latest_claims[
                latest_claims['clmNum'].isin(self.feedback_data['claim_number'])
            ]
            
            # Add feedback info
            feedback_summary = self.feedback_data.groupby('claim_number').agg({
                'rep_name': 'last',
                'timestamp': 'last',
                'rep_confidence': 'last'
            }).reset_index()
            
            # Merge with claims
            result = claims_with_feedback.merge(
                feedback_summary, 
                left_on='clmNum', 
                right_on='claim_number', 
                how='left'
            )
            
            return result.sort_values('incurred_cumsum', ascending=False)
            
        except Exception as e:
            print(f"Error getting claims with feedback: {e}")
            return pd.DataFrame()
    
    def answer_query(self, query: str) -> Dict:
        """
        Answer natural language queries about the portfolio.
        
        Args:
            query: Natural language query
            
        Returns:
            Dict: Query results and analysis
        """
        query_lower = query.lower()
        
        try:
            # Expense threshold queries (check first to avoid portfolio summary match)
            if 'expense' in query_lower and any(term in query_lower for term in ['over', 'above', '>', 'more than']):
                # Extract amount from query
                amount_match = re.search(r'[\$]?([\d,]+(?:\.\d+)?)[k]?', query)
                if amount_match:
                    amount_str = amount_match.group(1).replace(',', '')
                    amount = float(amount_str)
                    if 'k' in query_lower:
                        amount *= 1000
                    return {
                        'type': 'expense_threshold',
                        'data': self.query_claims_by_expense_threshold(amount),
                        'query': query,
                        'threshold': amount
                    }
            
            # Payment threshold queries
            if 'payment' in query_lower and any(term in query_lower for term in ['over', 'above', '>', 'more than']):
                # Extract amount from query
                amount_match = re.search(r'[\$]?([\d,]+(?:\.\d+)?)[k]?', query)
                if amount_match:
                    amount_str = amount_match.group(1).replace(',', '')
                    amount = float(amount_str)
                    if 'k' in query_lower:
                        amount *= 1000
                    return {
                        'type': 'payment_threshold',
                        'data': self.query_claims_by_payment_threshold(amount),
                        'query': query,
                        'threshold': amount
                    }
            
            # Portfolio summary queries
            if any(term in query_lower for term in ['summary', 'overview', 'portfolio', 'total']):
                return {
                    'type': 'portfolio_summary',
                    'data': self.get_portfolio_summary(),
                    'query': query
                }
            
            # Payment threshold queries
            if 'payment' in query_lower and any(term in query_lower for term in ['over', 'above', '>', 'more than']):
                # Extract amount from query
                amount_match = re.search(r'[\$]?([\d,]+(?:\.\d+)?)[k]?', query)
                if amount_match:
                    amount_str = amount_match.group(1).replace(',', '')
                    amount = float(amount_str)
                    if 'k' in query_lower:
                        amount *= 1000
                    return {
                        'type': 'payment_threshold',
                        'data': self.query_claims_by_payment_threshold(amount),
                        'query': query,
                        'threshold': amount
                    }
            
            # # Expense threshold queries
            # if 'expense' in query_lower and any(term in query_lower for term in ['over', 'above', '>', 'more than']):
            #     # Extract amount from query
            #     amount_match = re.search(r'[\$]?([\d,]+(?:\.\d+)?)[k]?', query)
            #     if amount_match:
            #         amount_str = amount_match.group(1).replace(',', '')
            #         amount = float(amount_str)
            #         if 'k' in query_lower:
            #             amount *= 1000
            #         return {
            #             'type': 'expense_threshold',
            #             'data': self.query_claims_by_expense_threshold(amount),
            #             'query': query,
            #             'threshold': amount
            #         }
            
            # No payment/expense queries
            if any(term in query_lower for term in ['no payment', 'no expense', 'no activity']):
                return {
                    'type': 'no_activity',
                    'data': self.query_claims_no_payment_no_expense(),
                    'query': query
                }
            
            # Law firm queries
            if any(term in query_lower for term in ['law firm', 'lawyer', 'attorney', 'legal']):
                # Extract firm name (simplified)
                words = query.split()
                for i, word in enumerate(words):
                    if word.lower() in ['firm', 'law', 'legal', 'abc']:
                        if i > 0:
                            firm_name = words[i-1]
                            return {
                                'type': 'law_firm_search',
                                'data': self.search_notes_by_law_firm(firm_name),
                                'query': query,
                                'firm_name': firm_name
                            }
            
            # Risk level queries
            if any(term in query_lower for term in ['critical', 'high risk', 'medium risk']):
                if 'critical' in query_lower:
                    return {
                        'type': 'risk_level',
                        'data': self.get_claims_by_risk_level('CRITICAL'),
                        'query': query,
                        'risk_level': 'CRITICAL'
                    }
                elif 'high' in query_lower:
                    return {
                        'type': 'risk_level',
                        'data': self.get_claims_by_risk_level('HIGH'),
                        'query': query,
                        'risk_level': 'HIGH'
                    }
            
            # Status queries
            if any(term in query_lower for term in ['open', 'closed', 'paid', 'denied']):
                if 'open' in query_lower:
                    return {
                        'type': 'status',
                        'data': self.get_claims_by_status('OPEN'),
                        'query': query,
                        'status': 'OPEN'
                    }
                elif 'closed' in query_lower:
                    return {
                        'type': 'status',
                        'data': self.get_claims_by_status('CLOSED'),
                        'query': query,
                        'status': 'CLOSED'
                    }
                elif 'paid' in query_lower:
                    return {
                        'type': 'status',
                        'data': self.get_claims_by_status('PAID'),
                        'query': query,
                        'status': 'PAID'
                    }
            
            # Cause queries
            if any(term in query_lower for term in ['auto', 'property', 'injury', 'theft', 'weather']):
                if 'auto' in query_lower:
                    return {
                        'type': 'cause',
                        'data': self.get_claims_by_cause('Auto Accident'),
                        'query': query,
                        'cause': 'Auto Accident'
                    }
                elif 'property' in query_lower:
                    return {
                        'type': 'cause',
                        'data': self.get_claims_by_cause('Property Damage'),
                        'query': query,
                        'cause': 'Property Damage'
                    }
                elif 'injury' in query_lower:
                    return {
                        'type': 'cause',
                        'data': self.get_claims_by_cause('Personal Injury'),
                        'query': query,
                        'cause': 'Personal Injury'
                    }
                elif 'theft' in query_lower:
                    return {
                        'type': 'cause',
                        'data': self.get_claims_by_cause('Theft'),
                        'query': query,
                        'cause': 'Theft'
                    }
                elif 'weather' in query_lower:
                    return {
                        'type': 'cause',
                        'data': self.get_claims_by_cause('Weather Damage'),
                        'query': query,
                        'cause': 'Weather Damage'
                    }
            
            # High risk claims queries
            if any(term in query_lower for term in ['high reserve', 'high risk', 'large reserve']):
                return {
                    'type': 'high_risk',
                    'data': self.get_high_risk_claims(),
                    'query': query
                }
            
            # Feedback queries
            if 'feedback' in query_lower:
                return {
                    'type': 'feedback',
                    'data': self.get_claims_with_feedback(),
                    'query': query
                }
            
            # Default: return portfolio summary
            return {
                'type': 'portfolio_summary',
                'data': self.get_portfolio_summary(),
                'query': query,
                'note': 'Query not specifically recognized, showing portfolio summary'
            }
            
        except Exception as e:
            return {
                'type': 'error',
                'error': str(e),
                'query': query
            }

def main():
    """Test the general claims agent."""
    print("Testing General Claims Agent...")
    
    agent = GeneralClaimsAgent()
    
    # Test portfolio summary
    print("\n1. Portfolio Summary:")
    summary = agent.get_portfolio_summary()
    if 'error' not in summary:
        print(f"Total Claims: {summary['total_claims']}")
        print(f"Total Reserve: ${summary['total_reserve']:,.2f}")
        print(f"Total Paid: ${summary['total_paid']:,.2f}")
        print(f"Risk Breakdown: {summary['risk_breakdown']}")
    else:
        print(f"Error: {summary['error']}")
    
    # Test payment threshold query
    print("\n2. Claims with payments > $50k:")
    high_payment_claims = agent.query_claims_by_payment_threshold(50000)
    if len(high_payment_claims) > 0:
        print(f"Found {len(high_payment_claims)} claims")
        for _, claim in high_payment_claims.head(3).iterrows():
            print(f"  {claim['clmNum']}: ${claim['paid_cumsum']:,.2f}")
    else:
        print("No claims found")
    
    # Test no activity query
    print("\n3. Claims with no payment and no expense:")
    no_activity_claims = agent.query_claims_no_payment_no_expense()
    if len(no_activity_claims) > 0:
        print(f"Found {len(no_activity_claims)} claims")
        for _, claim in no_activity_claims.head(3).iterrows():
            print(f"  {claim['clmNum']}: Reserve ${claim['reserve_cumsum']:,.2f}")
    else:
        print("No claims found")
    
    # Test natural language query
    print("\n4. Natural Language Query Test:")
    test_query = "how many claims have cumulated payments over the lifetime over $500k?"
    result = agent.answer_query(test_query)
    print(f"Query: {test_query}")
    print(f"Result Type: {result['type']}")
    if 'data' in result and len(result['data']) > 0:
        print(f"Found {len(result['data'])} claims")

if __name__ == "__main__":
    main()
