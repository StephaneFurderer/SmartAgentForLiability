#!/usr/bin/env python3
"""
Create dummy CRITICAL claims dataset for POC
CRITICAL = Reopened claims with $0 reserve
"""

import pandas as pd
import numpy as np
import os

def create_critical_claims_data():
    """Create 10 CRITICAL claims with 5 transactions each"""
    
    np.random.seed(42)  # For reproducibility
    
    # Generate 10 CRITICAL claims
    n_claims = 10
    transactions_per_claim = 5
    
    transactions = []
    
    for i in range(n_claims):
        # Generate claim-level data
        booknum = np.random.choice(['BK001', 'BK002', 'BK003'])
        cidpol = f'POL{np.random.randint(100000, 999999):06d}'
        clmNum = f'CLM{np.random.randint(10000000, 99999999):08d}'
        clmCause = np.random.choice(['Auto Accident', 'Property Damage', 'Personal Injury', 'Theft', 'Weather Damage'])
        
        # Generate dates for CRITICAL claims (reopened)
        dateReceived = pd.to_datetime(np.random.choice(pd.date_range('2020-01-01', '2022-12-31')))
        
        # For CRITICAL claims: first they were completed, then reopened
        # Complete date should be before reopen date
        dateCompleted = dateReceived + pd.Timedelta(days=np.random.randint(90, 365))  # 3 months to 1 year after opening
        dateReopened = dateCompleted + pd.Timedelta(days=np.random.randint(30, 180))  # 1-6 months after completion
        
        # Generate 5 transaction dates after reopening
        transaction_dates = pd.to_datetime(np.sort(np.random.choice(
            pd.date_range(dateReopened, dateReopened + pd.Timedelta(days=365), 
                         periods=transactions_per_claim), 
            transactions_per_claim, replace=False
        )))
        
        # Track cumulative amounts
        cumulative_paid = 0
        cumulative_expense = 0
        
        for j, datetxn in enumerate(transaction_dates):
            # Generate transaction amounts (exponential distribution for realistic skewed data)
            transaction_paid = np.random.exponential(8000)  # Higher amounts for critical claims
            transaction_expense = np.random.exponential(2000)
            
            # Cap cumulative paid at 500K per claim
            if cumulative_paid + transaction_paid > 500000:
                transaction_paid = max(0, 500000 - cumulative_paid)
            
            cumulative_paid += transaction_paid
            cumulative_expense += transaction_expense
            
            # Round amounts
            transaction_paid = round(transaction_paid, 2)
            transaction_expense = round(transaction_expense, 2)
            
            # CRITICAL claims have $0 reserve
            transaction_reserve = 0
            transaction_recovery = 0
            
            # Calculate incurred amount
            incurred_amount = transaction_paid + transaction_expense + transaction_recovery + transaction_reserve
            
            # Determine claim status - all transactions are REOPENED for CRITICAL claims
            clmStatus = 'REOPENED'
            
            # Add transaction record
            transaction = {
                'datetxn': datetxn,
                'dateReceived': dateReceived,
                'booknum': booknum,
                'cidpol': cidpol,
                'clmNum': clmNum,
                'clmStatus': clmStatus,
                'dateCompleted': dateCompleted,  # Date when claim was originally completed
                'dateReopened': dateReopened,    # Date when claim was reopened
                'paid': transaction_paid,
                'expense': transaction_expense,
                'recovery': transaction_recovery,
                'reserve': transaction_reserve,  # Always 0 for CRITICAL
                'incurred': incurred_amount,
                'clmCause': clmCause,
                'risk_level': 'CRITICAL'
            }
            
            transactions.append(transaction)
    
    # Create DataFrame
    df = pd.DataFrame(transactions)
    
    # Convert date columns
    df['dateReceived'] = pd.to_datetime(df['dateReceived'])
    df['dateReopened'] = pd.to_datetime(df['dateReopened'])
    df['datetxn'] = pd.to_datetime(df['datetxn'])
    
    # Calculate cumulative sums for each claim
    claim_group = df.groupby('clmNum')
    df['paid_cumsum'] = claim_group['paid'].cumsum()
    df['expense_cumsum'] = claim_group['expense'].cumsum()
    df['reserve_cumsum'] = claim_group['reserve'].cumsum()
    df['incurred_cumsum'] = claim_group['incurred'].cumsum()
    
    # Sort by claim number and transaction date
    df = df.sort_values(['clmNum', 'datetxn'])
    
    return df

def save_critical_claims():
    """Create and save the CRITICAL claims dataset"""
    
    # Create data directory if it doesn't exist
    os.makedirs('./_data', exist_ok=True)
    
    # Generate data
    print("Creating 10 CRITICAL claims with 5 transactions each...")
    df = create_critical_claims_data()
    
    # Save to CSV
    csv_file = './_data/critical_claims.csv'
    df.to_csv(csv_file, index=False)
    
    print(f"✅ Saved {len(df)} transactions to {csv_file}")
    print(f"📊 Dataset contains {df['clmNum'].nunique()} unique claims")
    print(f"💰 Total incurred: ${df['incurred'].sum():,.2f}")
    print(f"💸 Total paid: ${df['paid'].sum():,.2f}")
    print(f"🔴 Total reserve: ${df['reserve'].sum():,.2f}")
    
    # Show sample of the data
    print("\n📋 Sample data:")
    print(df[['clmNum', 'datetxn', 'clmStatus', 'paid', 'expense', 'reserve', 'incurred']].head(10))
    
    return df

if __name__ == "__main__":
    df = save_critical_claims()
