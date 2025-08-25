#!/usr/bin/env python3
"""
Create HIGH Risk Claims Dataset

This script generates 20 HIGH risk claims with low cumulative amounts:
- 10 claims with both paid and expense amounts (low cumulative)
- 10 claims with only expenses, no payments yet
- All claims are OPEN status (no completed/reopened dates)
- Risk level is determined by low cumulative amounts
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os

def create_high_risk_claims_data():
    """
    Create 20 HIGH risk claims with low cumulative amounts.
    
    Returns:
        pd.DataFrame: DataFrame containing the high risk claims data
    """
    
    # Create data directory if it doesn't exist
    os.makedirs("./_data", exist_ok=True)
    
    # Set random seed for reproducibility
    np.random.seed(42)
    
    # Generate 20 unique claim numbers
    claim_numbers = [f"CLM{np.random.randint(10000000, 99999999)}" for _ in range(20)]
    
    # Claim causes for variety
    claim_causes = [
        "Auto Accident", "Property Damage", "Personal Injury", 
        "Theft", "Weather Damage", "Slip and Fall", "Product Liability",
        "Professional Negligence", "Construction Defect", "Medical Malpractice"
    ]
    
    # Generate base dates (all claims start within last 2 years)
    base_date = datetime.now() - timedelta(days=730)
    
    all_transactions = []
    
    # Group 1: 10 claims with both paid and expense amounts (low cumulative)
    for i in range(10):
        clm_num = claim_numbers[i]
        clm_cause = np.random.choice(claim_causes)
        
        # Generate 3-6 transactions per claim
        num_transactions = np.random.randint(3, 7)
        
        # Start date for this claim (within last 2 years)
        claim_start = base_date + timedelta(days=np.random.randint(0, 730))
        
        for j in range(num_transactions):
            # Transaction date (spread over claim lifetime)
            transaction_date = claim_start + timedelta(
                days=np.random.randint(0, (datetime.now() - claim_start).days)
            )
            
            # Low amounts for HIGH risk (expenses: $100-$2000, payments: $200-$3000)
            expense = np.random.uniform(100, 2000)
            paid = np.random.uniform(200, 3000)
            reserve = np.random.uniform(500, 1500)  # Small reserve for open claims
            
            # Ensure total incurred is reasonable
            incurred = expense + reserve
            
            transaction = {
                'clmNum': clm_num,
                'datetxn': transaction_date,
                'clmStatus': 'OPEN',
                'paid': round(paid, 2),
                'expense': round(expense, 2),
                'reserve': round(reserve, 2),
                'incurred': round(incurred, 2),
                'clmCause': clm_cause,
                'risk_level': 'HIGH'
            }
            
            all_transactions.append(transaction)
    
    # Group 2: 10 claims with only expenses, no payments yet
    for i in range(10, 20):
        clm_num = claim_numbers[i]
        clm_cause = np.random.choice(claim_causes)
        
        # Generate 2-5 transactions per claim (fewer since no payments)
        num_transactions = np.random.randint(2, 6)
        
        # Start date for this claim (within last 2 years)
        claim_start = base_date + timedelta(days=np.random.randint(0, 730))
        
        for j in range(num_transactions):
            # Transaction date (spread over claim lifetime)
            transaction_date = claim_start + timedelta(
                days=np.random.randint(0, (datetime.now() - claim_start).days)
            )
            
            # Only expenses and reserves (no payments yet)
            expense = np.random.uniform(150, 2500)
            paid = 0.0  # No payments yet
            reserve = np.random.uniform(300, 2000)  # Reserve for future payments
            
            # Ensure total incurred is reasonable
            incurred = expense + reserve
            
            transaction = {
                'clmNum': clm_num,
                'datetxn': transaction_date,
                'clmStatus': 'OPEN',
                'paid': round(paid, 2),
                'expense': round(expense, 2),
                'reserve': round(reserve, 2),
                'incurred': round(incurred, 2),
                'clmCause': clm_cause,
                'risk_level': 'HIGH'
            }
            
            all_transactions.append(transaction)
    
    # Convert to DataFrame
    df = pd.DataFrame(all_transactions)
    
    # Sort by claim number and date
    df = df.sort_values(['clmNum', 'datetxn'])
    
    # Add cumulative columns for analysis
    df['paid_cumsum'] = df.groupby('clmNum')['paid'].cumsum()
    df['expense_cumsum'] = df.groupby('clmNum')['expense'].cumsum()
    df['reserve_cumsum'] = df.groupby('clmNum')['reserve'].cumsum()
    df['incurred_cumsum'] = df.groupby('clmNum')['incurred'].cumsum()
    
    return df

def main():
    """Main function to create and save the high risk claims data."""
    
    print("Creating 20 HIGH risk claims with low cumulative amounts...")
    print("- 10 claims with both paid and expense amounts")
    print("- 10 claims with only expenses (no payments yet)")
    print("- All claims are OPEN status")
    
    # Create the data
    df = create_high_risk_claims_data()
    
    # Save to CSV
    output_file = "./_data/high_risk_claims.csv"
    df.to_csv(output_file, index=False)
    
    # Display summary statistics
    print(f"\n✅ Saved {len(df)} transactions to {output_file}")
    print(f"📊 Dataset contains {df['clmNum'].nunique()} unique claims")
    print(f"💰 Total incurred: ${df['incurred'].sum():,.2f}")
    print(f"💸 Total paid: ${df['paid'].sum():,.2f}")
    print(f"🔴 Total reserve: ${df['reserve'].sum():,.2f}")
    print(f"📋 Total expenses: ${df['expense'].sum():,.2f}")
    
    # Group analysis
    claims_summary = df.groupby('clmNum').agg({
        'paid_cumsum': 'max',
        'expense_cumsum': 'max',
        'reserve_cumsum': 'max',
        'incurred_cumsum': 'max',
        'clmStatus': 'last',
        'clmCause': 'first',
        'risk_level': 'first'
    }).reset_index()
    
    print(f"\n📈 Claims Summary:")
    print(f"- Claims with payments: {len(claims_summary[claims_summary['paid_cumsum'] > 0])}")
    print(f"- Claims without payments: {len(claims_summary[claims_summary['paid_cumsum'] == 0])}")
    print(f"- Average incurred per claim: ${claims_summary['incurred_cumsum'].mean():,.2f}")
    print(f"- Average reserve per claim: ${claims_summary['reserve_cumsum'].mean():,.2f}")
    
    print(f"\n📋 Sample data:")
    print(df.head(10).to_string(index=False))
    
    # Save claims summary to separate file
    summary_file = "./_data/high_risk_claims_summary.csv"
    claims_summary.to_csv(summary_file, index=False)
    print(f"\n📊 Claims summary saved to {summary_file}")

if __name__ == "__main__":
    main()
