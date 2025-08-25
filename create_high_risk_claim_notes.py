#!/usr/bin/env python3
"""
Create Notes for HIGH Risk Claims

This script generates realistic notes for HIGH risk claims, following the same logic
as CRITICAL claims but adapted for open claims that are still in progress.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os

def create_high_risk_claim_notes():
    """
    Create realistic notes for HIGH risk claims.
    
    Returns:
        pd.DataFrame: DataFrame containing the claim notes
    """
    
    # Create data directory if it doesn't exist
    os.makedirs("./_data", exist_ok=True)
    
    # Set random seed for reproducibility
    np.random.seed(42)
    
    # Load the HIGH risk claims data
    high_risk_file = "./_data/high_risk_claims.csv"
    if not os.path.exists(high_risk_file):
        print(f"High risk claims file not found at: {high_risk_file}")
        print("Please run create_high_risk_claims.py first to generate the claims data.")
        return pd.DataFrame()
    
    # Read the high risk claims
    claims_df = pd.read_csv(high_risk_file)
    claims_df['datetxn'] = pd.to_datetime(claims_df['datetxn'])
    
    # Get unique claims and their summary
    claims_summary = claims_df.groupby('clmNum').agg({
        'clmCause': 'first',
        'risk_level': 'first',
        'datetxn': ['min', 'max'],
        'expense_cumsum': 'max',
        'paid_cumsum': 'max',
        'reserve_cumsum': 'max'
    }).reset_index()
    
    # Flatten column names
    claims_summary.columns = ['clmNum', 'clmCause', 'risk_level', 'dateReceived', 'lastActivity', 'total_expense', 'total_paid', 'current_reserve']
    
    # Email templates for different parties
    email_templates = {
        'Insurance_Opening': [
            "Dear {client_name},\n\nWe have received your claim for {cause_description}. Your claim number is {clm_num}.\n\nWe have assigned an initial reserve of ${reserve_amount:,.2f} to cover potential expenses. Our adjuster will be in touch within 24-48 hours to begin the investigation.\n\nIf you have any immediate questions, please don't hesitate to contact us.\n\nBest regards,\n{insurance_rep}",
            "Hello {client_name},\n\nThank you for submitting your claim for {cause_description}. Claim number: {clm_num}\n\nWe have reviewed the initial information and set a reserve of ${reserve_amount:,.2f}. Our team will begin processing this immediately.\n\nPlease expect a call from our adjuster within the next business day.\n\nRegards,\n{insurance_rep}"
        ],
        'Insurance_Investigation': [
            "Dear {client_name},\n\nWe are actively investigating your claim {clm_num} for {cause_description}. We have received additional documentation and are reviewing the details.\n\nCurrent reserve: ${reserve_amount:,.2f}\nExpenses to date: ${expense_amount:,.2f}\n\nWe will provide updates as the investigation progresses.\n\nBest regards,\n{insurance_rep}",
            "Hello {client_name},\n\nUpdate on claim {clm_num}: Our investigation is ongoing. We have gathered additional information and are assessing the situation.\n\nCurrent status: Under Investigation\nReserve: ${reserve_amount:,.2f}\n\nWe appreciate your patience during this process.\n\nRegards,\n{insurance_rep}"
        ],
        'Client_Response': [
            "Hi {insurance_rep},\n\nThank you for the update on claim {clm_num}. I have additional information that might be helpful:\n\n{additional_info}\n\nPlease let me know if you need anything else from me.\n\nBest regards,\n{client_name}",
            "Hello,\n\nRegarding claim {clm_num}, I wanted to follow up on the investigation status. I have some questions:\n\n{questions}\n\nLooking forward to hearing from you.\n\nThanks,\n{client_name}"
        ],
        'Adjuster_Internal': [
            "Internal Note - Claim {clm_num}:\n\nRisk Level: {risk_level}\nCurrent Reserve: ${reserve_amount:,.2f}\nTotal Expenses: ${expense_amount:,.2f}\n\nStatus: {status}\nNext Steps: {next_steps}\n\n{adjuster_name}",
            "Claim Review - {clm_num}:\n\nRisk Assessment: {risk_level}\nFinancial Summary:\n- Reserve: ${reserve_amount:,.2f}\n- Expenses: ${expense_amount:,.2f}\n- Payments: ${payment_amount:,.2f}\n\nRecommendation: {recommendation}\n\n{adjuster_name}"
        ],
        'Legal_Review': [
            "Legal Review Required - Claim {clm_num}:\n\nRisk Level: {risk_level}\nPotential Legal Issues: {legal_issues}\n\nRecommendation: {legal_recommendation}\n\n{lawyer_name}",
            "Claim {clm_num} - Legal Assessment:\n\nRisk Category: {risk_level}\nLegal Considerations: {legal_considerations}\n\nAction Required: {action_required}\n\n{lawyer_name}"
        ],
        'Medical_Provider': [
            "Medical Update - Claim {clm_num}:\n\nPatient: {client_name}\nTreatment: {treatment_description}\nCost: ${medical_cost:,.2f}\n\nNext Appointment: {next_appointment}\n\n{medical_provider}",
            "Claim {clm_num} - Medical Report:\n\nPatient: {client_name}\nDiagnosis: {diagnosis}\nTreatment Plan: {treatment_plan}\nEstimated Cost: ${estimated_cost:,.2f}\n\n{medical_provider}"
        ],
        'Contractor_Estimate': [
            "Repair Estimate - Claim {clm_num}:\n\nDamage Assessment: {damage_description}\nRepair Work Required: {repair_work}\nEstimated Cost: ${repair_cost:,.2f}\n\nTimeline: {timeline}\n\n{contractor_name}",
            "Claim {clm_num} - Contractor Quote:\n\nScope of Work: {scope_of_work}\nMaterials: ${materials_cost:,.2f}\nLabor: ${labor_cost:,.2f}\nTotal Estimate: ${total_estimate:,.2f}\n\n{contractor_name}"
        ]
    }
    
    # Party names for variety
    insurance_reps = ["Sarah Johnson", "Michael Chen", "Lisa Rodriguez", "David Thompson", "Jennifer Lee"]
    client_names = ["Robert Smith", "Maria Garcia", "James Wilson", "Patricia Brown", "John Davis"]
    adjusters = ["Alex Turner", "Rachel Green", "Kevin Martinez", "Amanda White", "Chris Anderson"]
    lawyers = ["Attorney Williams", "Legal Counsel Davis", "Law Firm Johnson", "Legal Services Brown"]
    medical_providers = ["Dr. Sarah Miller", "Medical Center", "Dr. Robert Taylor", "Healthcare Associates"]
    contractors = ["ABC Construction", "Quality Repairs Inc.", "Premier Contracting", "Reliable Builders"]
    
    all_notes = []
    
    # Generate notes for each claim
    for idx, claim in claims_summary.iterrows():
        clm_num = claim['clmNum']
        clm_cause = claim['clmCause']
        risk_level = claim['risk_level']
        date_received = claim['dateReceived']
        total_expense = claim['total_expense']
        current_reserve = claim['current_reserve']
        total_paid = claim['total_paid']
        
        # Determine number of notes based on claim complexity (20-50 notes)
        num_notes = np.random.randint(20, 51)
        
        # Generate notes over the claim timeline
        claim_duration = (datetime.now() - date_received).days
        note_dates = []
        
        # Create a realistic timeline of notes
        for i in range(num_notes):
            # Distribute notes over the claim duration with more activity early on
            if i < num_notes * 0.3:  # First 30% of notes - high activity
                days_offset = np.random.randint(0, claim_duration // 3)
            elif i < num_notes * 0.7:  # Next 40% - medium activity
                days_offset = np.random.randint(claim_duration // 3, claim_duration * 2 // 3)
            else:  # Last 30% - ongoing activity
                days_offset = np.random.randint(claim_duration * 2 // 3, claim_duration)
            
            note_date = date_received + timedelta(days=days_offset)
            note_dates.append(note_date)
        
        # Sort note dates chronologically
        note_dates.sort()
        
        # Generate notes for this claim
        for i, note_date in enumerate(note_dates):
            # Determine note type based on timeline and claim status
            if i == 0:  # First note - always opening
                note_type = 'Insurance_Opening'
                priority = 'High'
                author = np.random.choice(insurance_reps)
            elif i < num_notes * 0.2:  # Early notes - investigation
                note_type = np.random.choice(['Insurance_Investigation', 'Client_Response', 'Adjuster_Internal'])
                priority = 'High'
                author = np.random.choice(insurance_reps + adjusters)
            elif i < num_notes * 0.5:  # Middle notes - ongoing process
                note_type = np.random.choice(['Adjuster_Internal', 'Client_Response', 'Legal_Review', 'Medical_Provider', 'Contractor_Estimate'])
                priority = 'Medium'
                author = np.random.choice(adjusters + lawyers + medical_providers + contractors)
            else:  # Later notes - ongoing management
                note_type = np.random.choice(['Adjuster_Internal', 'Client_Response', 'Insurance_Investigation'])
                priority = 'Medium'
                author = np.random.choice(adjusters + insurance_reps)
            
            # Get appropriate template
            template = np.random.choice(email_templates[note_type])
             
             # Fill template with claim-specific information
            note_content = template.format(
                 clm_num=clm_num,
                 client_name=np.random.choice(client_names),
                 cause_description=clm_cause,
                 reserve_amount=current_reserve,
                 expense_amount=total_expense,
                 payment_amount=total_paid,
                 risk_level=risk_level,
                 status="Open - Under Investigation",
                 next_steps="Continue monitoring and investigation",
                 recommendation="Maintain current reserve level",
                 legal_issues="Standard claim processing",
                 legal_recommendation="Continue standard procedures",
                 legal_considerations="No immediate legal action required",
                 action_required="Continue investigation",
                 treatment_description="Initial assessment completed",
                 medical_cost=np.random.uniform(100, 2000),
                 next_appointment="TBD",
                 diagnosis="Under evaluation",
                 treatment_plan="Assessment in progress",
                 estimated_cost=np.random.uniform(500, 3000),
                 damage_description="Damage assessment ongoing",
                 repair_work="Evaluation in progress",
                 repair_cost=np.random.uniform(1000, 5000),
                 timeline="TBD",
                 scope_of_work="Assessment and evaluation in progress",
                 materials_cost=np.random.uniform(200, 1000),
                 labor_cost=np.random.uniform(300, 1500),
                 total_estimate=np.random.uniform(1000, 8000),
                 additional_info="Additional documentation provided",
                 questions="What is the next step in the process?",
                 insurance_rep=np.random.choice(insurance_reps),
                 adjuster_name=np.random.choice(adjusters),
                 lawyer_name=np.random.choice(lawyers),
                 medical_provider=np.random.choice(medical_providers),
                 contractor_name=np.random.choice(contractors)
             )
            
            # Create note record
            note_record = {
                'dateNote': note_date,
                'clmNum': clm_num,
                'note': note_content,
                'note_type': note_type,
                'priority': priority,
                'author': author
            }
            
            all_notes.append(note_record)
    
    # Convert to DataFrame
    notes_df = pd.DataFrame(all_notes)
    
    # Sort by claim number and date
    notes_df = notes_df.sort_values(['clmNum', 'dateNote'])
    
    return notes_df

def main():
    """Main function to create and save the high risk claim notes."""
    
    print("Creating notes for HIGH risk claims...")
    print("This will generate realistic email exchanges and notes for open claims.")
    
    # Create the notes
    notes_df = create_high_risk_claim_notes()
    
    if len(notes_df) > 0:
        # Save to CSV
        output_file = "./_data/high_risk_claim_notes.csv"
        notes_df.to_csv(output_file, index=False)
        
        # Display summary statistics
        print(f"\n✅ Saved {len(notes_df)} notes to {output_file}")
        print(f"📊 Notes generated for {notes_df['clmNum'].nunique()} unique claims")
        print(f"📅 Date range: {notes_df['dateNote'].min().strftime('%Y-%m-%d')} to {notes_df['dateNote'].max().strftime('%Y-%m-%d')}")
        
        # Breakdown by note type
        print(f"\n📋 Note Type Breakdown:")
        note_type_counts = notes_df['note_type'].value_counts()
        for note_type, count in note_type_counts.items():
            print(f"  - {note_type}: {count}")
        
        # Breakdown by priority
        print(f"\n🚨 Priority Breakdown:")
        priority_counts = notes_df['priority'].value_counts()
        for priority, count in priority_counts.items():
            print(f"  - {priority}: {count}")
        
        # Sample notes
        print(f"\n📝 Sample notes:")
        sample_notes = notes_df.head(3)
        for idx, note in sample_notes.iterrows():
            print(f"\nClaim: {note['clmNum']}")
            print(f"Date: {note['dateNote'].strftime('%Y-%m-%d')}")
            print(f"Type: {note['note_type']}")
            print(f"Priority: {note['priority']}")
            print(f"Author: {note['author']}")
            print(f"Note: {note['note'][:100]}...")
        
        # Save summary to separate file
        summary_file = "./_data/high_risk_claim_notes_summary.csv"
        notes_summary = notes_df.groupby('clmNum').agg({
            'dateNote': ['min', 'max', 'count'],
            'note_type': lambda x: list(x.unique()),
            'priority': lambda x: list(x.unique())
        }).reset_index()
        
        # Flatten column names
        notes_summary.columns = ['clmNum', 'first_note', 'last_note', 'total_notes', 'note_types', 'priorities']
        notes_summary.to_csv(summary_file, index=False)
        print(f"\n📊 Notes summary saved to {summary_file}")
        
    else:
        print("❌ No notes were generated. Please check if high_risk_claims.csv exists.")

if __name__ == "__main__":
    main()
