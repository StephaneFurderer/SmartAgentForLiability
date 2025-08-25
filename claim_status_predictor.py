#!/usr/bin/env python3
"""
Claim Status Predictor Module

This module provides Bayesian probability predictions for the final status of claims.
Currently uses placeholder values for demonstration purposes.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os

class ClaimStatusPredictor:
    """
    Predicts final claim status probabilities using Bayesian approach.
    Currently uses placeholder values for demonstration.
    """
    
    def __init__(self, data_dir: str = "./_data"):
        self.data_dir = data_dir
        self.claims_data = None
        self.load_claims_data()
    
    def load_claims_data(self):
        """Load claims data from CSV files."""
        try:
            all_claims = []
            
            # Load CRITICAL claims
            critical_file = os.path.join(self.data_dir, "critical_claims.csv")
            if os.path.exists(critical_file):
                critical_df = pd.read_csv(critical_file)
                critical_df['datetxn'] = pd.to_datetime(critical_df['datetxn'])
                all_claims.append(critical_df)
            
            # Load HIGH risk claims
            high_risk_file = os.path.join(self.data_dir, "high_risk_claims.csv")
            if os.path.exists(high_risk_file):
                high_risk_df = pd.read_csv(high_risk_file)
                high_risk_df['datetxn'] = pd.to_datetime(high_risk_df['datetxn'])
                all_claims.append(high_risk_df)
            
            if len(all_claims) > 0:
                self.claims_data = pd.concat(all_claims, ignore_index=True)
                print(f"Loaded {len(self.claims_data)} claim transactions")
            else:
                print("No claims data found")
                
        except Exception as e:
            print(f"Error loading claims data: {e}")
    
    def predict_final_status_probabilities(self, clm_num: str) -> dict:
        """
        Predict final status probabilities for a specific claim.
        
        Args:
            clm_num (str): Claim number to predict
            
        Returns:
            dict: Dictionary with status probabilities and confidence metrics
        """
        if self.claims_data is None:
            return self._get_placeholder_probabilities()
        
        # Get claim data
        claim_data = self.claims_data[self.claims_data['clmNum'] == clm_num]
        
        if len(claim_data) == 0:
            return self._get_placeholder_probabilities()
        
        # For now, return placeholder probabilities based on claim characteristics
        # In the future, this would use actual Bayesian calculations
        return self._calculate_placeholder_probabilities(claim_data)
    
    def _get_placeholder_probabilities(self) -> dict:
        """Return placeholder probabilities when no data is available."""
        return {
            'claim_number': 'Unknown',
            'prediction_date': datetime.now().strftime('%Y-%m-%d'),
            'probabilities': {
                'PAID': 0.33,
                'CLOSED': 0.33,
                'DENIED': 0.34
            },
            'confidence': 0.25,
            'factors': ['Insufficient data for prediction'],
            'last_updated': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
    
    def _calculate_placeholder_probabilities(self, claim_data: pd.DataFrame) -> dict:
        """
        Calculate placeholder probabilities based on claim characteristics.
        This is a simplified version for demonstration purposes.
        """
        # Get latest transaction
        latest_txn = claim_data.sort_values('datetxn').iloc[-1]
        
        # Extract key metrics
        total_paid = claim_data['paid_cumsum'].max() if 'paid_cumsum' in claim_data.columns else 0
        total_expense = claim_data['expense_cumsum'].max() if 'expense_cumsum' in claim_data.columns else 0
        current_reserve = claim_data['reserve_cumsum'].max() if 'reserve_cumsum' in claim_data.columns else 0
        risk_level = latest_txn.get('risk_level', 'UNKNOWN')
        claim_cause = latest_txn.get('clmCause', 'Unknown')
        
        # Simple heuristic-based probabilities (placeholder logic)
        if total_paid > 0:
            # Already has payments - likely to be PAID
            paid_prob = 0.75
            closed_prob = 0.20
            denied_prob = 0.05
        elif total_expense > 0 and current_reserve == 0:
            # Has expenses but no reserve - likely to be CLOSED
            paid_prob = 0.15
            closed_prob = 0.70
            denied_prob = 0.15
        elif total_expense == 0 and total_paid == 0:
            # No expenses or payments - could be DENIED
            paid_prob = 0.10
            closed_prob = 0.20
            denied_prob = 0.70
        else:
            # Default case - balanced probabilities
            paid_prob = 0.40
            closed_prob = 0.35
            denied_prob = 0.25
        
        # Adjust based on risk level
        if risk_level == 'CRITICAL':
            paid_prob *= 0.8  # Critical claims less likely to be paid
            closed_prob *= 1.2
            denied_prob *= 1.1
        elif risk_level == 'HIGH':
            paid_prob *= 0.9  # High risk claims slightly less likely to be paid
            closed_prob *= 1.1
            denied_prob *= 1.0
        
        # Normalize probabilities to sum to 1
        total_prob = paid_prob + closed_prob + denied_prob
        paid_prob /= total_prob
        closed_prob /= total_prob
        denied_prob /= total_prob
        
        # Calculate confidence based on data quality
        confidence = min(0.95, 0.3 + (len(claim_data) * 0.05) + (0.2 if total_paid > 0 or total_expense > 0 else 0))
        
        # Identify key factors
        factors = []
        if total_paid > 0:
            factors.append(f"Has payments: ${total_paid:,.2f}")
        if total_expense > 0:
            factors.append(f"Has expenses: ${total_expense:,.2f}")
        if current_reserve > 0:
            factors.append(f"Current reserve: ${current_reserve:,.2f}")
        factors.append(f"Risk level: {risk_level}")
        factors.append(f"Claim cause: {claim_cause}")
        factors.append(f"Transactions: {len(claim_data)}")
        
        return {
            'claim_number': latest_txn['clmNum'],
            'prediction_date': datetime.now().strftime('%Y-%m-%d'),
            'probabilities': {
                'PAID': round(paid_prob, 3),
                'CLOSED': round(closed_prob, 3),
                'DENIED': round(denied_prob, 3)
            },
            'confidence': round(confidence, 3),
            'factors': factors,
            'last_updated': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'current_metrics': {
                'total_paid': total_paid,
                'total_expense': total_expense,
                'current_reserve': current_reserve,
                'risk_level': risk_level,
                'claim_cause': claim_cause
            }
        }
    
    def get_all_claims_predictions(self) -> pd.DataFrame:
        """
        Get predictions for all claims.
        
        Returns:
            pd.DataFrame: DataFrame with predictions for all claims
        """
        if self.claims_data is None:
            return pd.DataFrame()
        
        # Get unique claims
        unique_claims = self.claims_data['clmNum'].unique()
        
        predictions = []
        for clm_num in unique_claims:
            pred = self.predict_final_status_probabilities(clm_num)
            predictions.append({
                'clmNum': clm_num,
                'paid_probability': pred['probabilities']['PAID'],
                'closed_probability': pred['probabilities']['CLOSED'],
                'denied_probability': pred['probabilities']['DENIED'],
                'confidence': pred['confidence'],
                'predicted_status': max(pred['probabilities'], key=pred['probabilities'].get),
                'last_updated': pred['last_updated']
            })
        
        return pd.DataFrame(predictions)
    
    def get_risk_summary(self) -> dict:
        """
        Get summary of risk predictions across all claims.
        
        Returns:
            dict: Summary statistics
        """
        predictions_df = self.get_all_claims_predictions()
        
        if len(predictions_df) == 0:
            return {
                'total_claims': 0,
                'avg_confidence': 0,
                'status_distribution': {},
                'high_risk_claims': 0
            }
        
        # Calculate summary statistics
        summary = {
            'total_claims': len(predictions_df),
            'avg_confidence': round(predictions_df['confidence'].mean(), 3),
            'status_distribution': predictions_df['predicted_status'].value_counts().to_dict(),
            'high_risk_claims': len(predictions_df[predictions_df['confidence'] < 0.5]),
            'avg_probabilities': {
                'PAID': round(predictions_df['paid_probability'].mean(), 3),
                'CLOSED': round(predictions_df['closed_probability'].mean(), 3),
                'DENIED': round(predictions_df['denied_probability'].mean(), 3)
            }
        }
        
        return summary

def main():
    """Test the claim status predictor."""
    print("Testing Claim Status Predictor...")
    
    predictor = ClaimStatusPredictor()
    
    # Test with a sample claim
    if predictor.claims_data is not None:
        sample_claim = predictor.claims_data['clmNum'].iloc[0]
        print(f"\nTesting prediction for claim: {sample_claim}")
        
        prediction = predictor.predict_final_status_probabilities(sample_claim)
        
        print(f"\nPrediction Results:")
        print(f"Claim: {prediction['claim_number']}")
        print(f"Date: {prediction['prediction_date']}")
        print(f"Confidence: {prediction['confidence']}")
        print(f"\nProbabilities:")
        for status, prob in prediction['probabilities'].items():
            print(f"  {status}: {prob:.1%}")
        
        print(f"\nKey Factors:")
        for factor in prediction['factors']:
            print(f"  • {factor}")
        
        # Test summary
        print(f"\nRisk Summary:")
        summary = predictor.get_risk_summary()
        print(f"Total Claims: {summary['total_claims']}")
        print(f"Average Confidence: {summary['avg_confidence']}")
        print(f"Status Distribution: {summary['status_distribution']}")
        print(f"High Risk Claims: {summary['high_risk_claims']}")
        
    else:
        print("No claims data available for testing")

if __name__ == "__main__":
    main()
