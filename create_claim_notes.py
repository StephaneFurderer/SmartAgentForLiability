#!/usr/bin/env python3
"""
Create realistic dummy notes for CRITICAL claims
Generates 20-50 notes per claim with emails and communications
"""

import pandas as pd
import numpy as np
import os
from datetime import datetime, timedelta

def create_claim_notes():
    """Create realistic notes for each CRITICAL claim"""
    
    # Load the existing CRITICAL claims data
    claims_file = os.path.join("./_data", "critical_claims.csv")
    if not os.path.exists(claims_file):
        print("❌ Critical claims data not found. Please run create_critical_claims.py first.")
        return
    
    claims_df = pd.read_csv(claims_file)
    claims_df['datetxn'] = pd.to_datetime(claims_df['datetxn'])
    claims_df['dateReceived'] = pd.to_datetime(claims_df['dateReceived'])
    claims_df['dateCompleted'] = pd.to_datetime(claims_df['dateCompleted'])
    claims_df['dateReopened'] = pd.to_datetime(claims_df['dateReopened'])
    
    # Get unique claims
    unique_claims = claims_df.groupby('clmNum').agg({
        'clmCause': 'first',
        'dateReceived': 'first',
        'dateCompleted': 'first',
        'dateReopened': 'first',
        'reserve_cumsum': 'last'
    }).reset_index()
    
    all_notes = []
    
    # Note templates based on claim cause
    note_templates = {
        'Auto Accident': {
            'opening': [
                "📧 EMAIL FROM INSURANCE TO CLIENT: Claim {clm_num} has been opened with initial reserve of ${reserve:,.2f}. Our adjuster will contact you within 24 hours to discuss the accident details and begin investigation.",
                "📧 EMAIL FROM INSURANCE TO CLIENT: We have received your auto accident claim {clm_num}. Initial reserve set at ${reserve:,.2f}. Please provide police report and any witness statements.",
                "📧 EMAIL FROM INSURANCE TO CLIENT: Auto claim {clm_num} opened with reserve of ${reserve:,.2f}. Please submit photos of vehicle damage and medical bills if applicable."
            ],
            'investigation': [
                "📧 EMAIL FROM ADJUSTER TO CLIENT: Following up on claim {clm_num}. Need additional photos of damage and repair estimates from 2-3 body shops.",
                "📧 EMAIL FROM ADJUSTER TO LEGAL: Requesting legal review of liability determination for auto claim {clm_num}. Multiple parties involved.",
                "📧 EMAIL FROM LEGAL TO ADJUSTER: Liability analysis complete for {clm_num}. Recommend 60/40 split based on witness statements.",
                "📧 EMAIL FROM ADJUSTER TO CLIENT: Liability investigation complete. We have determined 60% fault on other party. Reserve increased to ${reserve:,.2f}.",
                "📧 EMAIL FROM CLIENT TO ADJUSTER: Disagree with liability determination. My lawyer will be contacting you regarding claim {clm_num}.",
                "📧 EMAIL FROM LAWYER TO INSURANCE: Representing client on claim {clm_num}. Requesting full liability investigation file."
            ],
            'settlement': [
                "📧 EMAIL FROM ADJUSTER TO CLIENT: Settlement offer of ${amount:,.2f} for claim {clm_num}. Please review and respond within 30 days.",
                "📧 EMAIL FROM CLIENT TO ADJUSTER: Settlement offer rejected. My medical bills alone exceed ${amount:,.2f}. Need higher offer.",
                "📧 EMAIL FROM ADJUSTER TO LEGAL: Client rejecting settlement on {clm_num}. Requesting authority to increase offer to ${amount:,.2f}.",
                "📧 EMAIL FROM LEGAL TO ADJUSTER: Authorized to increase settlement to ${amount:,.2f} for {clm_num}. Final offer."
            ]
        },
        'Property Damage': {
            'opening': [
                "📧 EMAIL FROM INSURANCE TO CLIENT: Property damage claim {clm_num} opened with reserve of ${reserve:,.2f}. Inspector scheduled for {date}.",
                "📧 EMAIL FROM INSURANCE TO CLIENT: We have received your property damage claim {clm_num}. Initial reserve set at ${reserve:,.2f}. Please secure the property.",
                "📧 EMAIL FROM INSURANCE TO CLIENT: Property claim {clm_num} established with reserve of ${reserve:,.2f}. Contractor estimates needed."
            ],
            'investigation': [
                "📧 EMAIL FROM INSPECTOR TO ADJUSTER: Property inspection complete for {clm_num}. Damage estimate: ${amount:,.2f}. Photos attached.",
                "📧 EMAIL FROM ADJUSTER TO CONTRACTOR: Requesting detailed repair estimate for claim {clm_num}. Need breakdown of materials and labor.",
                "📧 EMAIL FROM CONTRACTOR TO ADJUSTER: Detailed estimate for {clm_num}: ${amount:,.2f}. Includes materials, labor, and permits.",
                "📧 EMAIL FROM ADJUSTER TO LEGAL: Property claim {clm_num} involves code upgrade requirements. Need legal opinion on coverage."
            ],
            'settlement': [
                "📧 EMAIL FROM ADJUSTER TO CLIENT: Property repair settlement: ${amount:,.2f} for claim {clm_num}. Contractor can begin work next week.",
                "📧 EMAIL FROM CLIENT TO ADJUSTER: Settlement accepted for {clm_num}. Please send payment to contractor directly."
            ]
        },
        'Personal Injury': {
            'opening': [
                "📧 EMAIL FROM INSURANCE TO CLIENT: Personal injury claim {clm_num} opened with medical reserve of ${reserve:,.2f}. Please provide all medical records.",
                "📧 EMAIL FROM INSURANCE TO CLIENT: Injury claim {clm_num} established. Reserve set at ${reserve:,.2f}. Medical evaluation required.",
                "📧 EMAIL FROM INSURANCE TO CLIENT: Personal injury claim {clm_num} received. Initial reserve: ${reserve:,.2f}. Need medical authorization."
            ],
            'investigation': [
                "📧 EMAIL FROM ADJUSTER TO MEDICAL PROVIDER: Requesting medical records for injury claim {clm_num}. Authorization attached.",
                "📧 EMAIL FROM MEDICAL PROVIDER TO ADJUSTER: Medical records for {clm_num} attached. Patient has permanent disability.",
                "📧 EMAIL FROM ADJUSTER TO LEGAL: Injury claim {clm_num} involves permanent disability. Need legal review of long-term exposure.",
                "📧 EMAIL FROM LEGAL TO ADJUSTER: Injury claim {clm_num} analysis complete. Recommend increasing reserve to ${amount:,.2f}.",
                "📧 EMAIL FROM ADJUSTER TO CLIENT: Medical evaluation complete. Reserve increased to ${amount:,.2f} for claim {clm_num}."
            ],
            'settlement': [
                "📧 EMAIL FROM ADJUSTER TO CLIENT: Injury settlement offer: ${amount:,.2f} for claim {clm_num}. Covers all medical and pain/suffering.",
                "📧 EMAIL FROM CLIENT TO ADJUSTER: Settlement offer too low. My medical bills are ${amount:,.2f} and I have permanent disability.",
                "📧 EMAIL FROM ADJUSTER TO LEGAL: Client rejecting injury settlement on {clm_num}. Requesting authority for higher offer."
            ]
        },
        'Theft': {
            'opening': [
                "📧 EMAIL FROM INSURANCE TO CLIENT: Theft claim {clm_num} opened with reserve of ${reserve:,.2f}. Police report required.",
                "📧 EMAIL FROM INSURANCE TO CLIENT: Theft claim {clm_num} established. Reserve: ${reserve:,.2f}. Need itemized list of stolen property.",
                "📧 EMAIL FROM INSURANCE TO CLIENT: Theft claim {clm_num} received. Initial reserve: ${reserve:,.2f}. Proof of ownership needed."
            ],
            'investigation': [
                "📧 EMAIL FROM ADJUSTER TO CLIENT: Need receipts and photos for stolen items in claim {clm_num}. Also need police report number.",
                "📧 EMAIL FROM CLIENT TO ADJUSTER: Receipts and police report attached for theft claim {clm_num}. Total value: ${amount:,.2f}.",
                "📧 EMAIL FROM ADJUSTER TO LEGAL: Theft claim {clm_num} involves high-value items. Need legal review of coverage limits.",
                "📧 EMAIL FROM LEGAL TO ADJUSTER: Theft coverage analysis complete for {clm_num}. Items covered under personal property endorsement."
            ],
            'settlement': [
                "📧 EMAIL FROM ADJUSTER TO CLIENT: Theft settlement: ${amount:,.2f} for claim {clm_num}. Payment will be issued within 5 business days.",
                "📧 EMAIL FROM CLIENT TO ADJUSTER: Settlement accepted for theft claim {clm_num}. Please send payment to my address."
            ]
        },
        'Weather Damage': {
            'opening': [
                "📧 EMAIL FROM INSURANCE TO CLIENT: Weather damage claim {clm_num} opened with reserve of ${reserve:,.2f}. Inspector scheduled.",
                "📧 EMAIL FROM INSURANCE TO CLIENT: Weather claim {clm_num} established. Reserve: ${reserve:,.2f}. Need photos of damage.",
                "📧 EMAIL FROM INSURANCE TO CLIENT: Weather damage claim {clm_num} received. Initial reserve: ${reserve:,.2f}. Emergency repairs authorized."
            ],
            'investigation': [
                "📧 EMAIL FROM INSPECTOR TO ADJUSTER: Weather damage inspection complete for {clm_num}. Damage estimate: ${amount:,.2f}.",
                "📧 EMAIL FROM ADJUSTER TO CONTRACTOR: Need emergency repair estimate for weather claim {clm_num}. Roof damage involved.",
                "📧 EMAIL FROM CONTRACTOR TO ADJUSTER: Emergency repair estimate for {clm_num}: ${amount:,.2f}. Can start immediately.",
                "📧 EMAIL FROM ADJUSTER TO LEGAL: Weather claim {clm_num} involves multiple properties. Need legal opinion on coverage."
            ],
            'settlement': [
                "📧 EMAIL FROM ADJUSTER TO CLIENT: Weather damage settlement: ${amount:,.2f} for claim {clm_num}. Contractor can begin repairs.",
                "📧 EMAIL FROM CLIENT TO ADJUSTER: Settlement accepted for weather claim {clm_num}. Please coordinate with contractor."
            ]
        }
    }
    
    # Generic note templates for all claim types
    generic_templates = [
        "📧 INTERNAL NOTE: Claim {clm_num} assigned to adjuster {adjuster_name}.",
        "📧 INTERNAL NOTE: Reserve review completed for {clm_num}. Current reserve: ${reserve:,.2f}.",
        "📧 INTERNAL NOTE: Legal review requested for {clm_num}. Issue: {issue}.",
        "📧 INTERNAL NOTE: Settlement authority requested for {clm_num}. Amount: ${amount:,.2f}.",
        "📧 INTERNAL NOTE: Claim {clm_num} escalated to supervisor for review.",
        "📧 INTERNAL NOTE: Quality review completed for {clm_num}. No issues found.",
        "📧 INTERNAL NOTE: Claim {clm_num} transferred to special investigations unit.",
        "📧 INTERNAL NOTE: Fraud investigation initiated for {clm_num}. Red flags identified.",
        "📧 INTERNAL NOTE: Subrogation opportunity identified for {clm_num}. Third party at fault.",
        "📧 INTERNAL NOTE: Claim {clm_num} closed with final payment of ${amount:,.2f}.",
        "📧 INTERNAL NOTE: Reopening request received for {clm_num}. New damages alleged.",
        "📧 INTERNAL NOTE: Claim {clm_num} reopened due to new information. Reserve increased.",
        "📧 INTERNAL NOTE: Litigation filed on {clm_num}. Case number: {case_number}.",
        "📧 INTERNAL NOTE: Mediation scheduled for {clm_num}. Date: {date}.",
        "📧 INTERNAL NOTE: Trial date set for {clm_num}. Court: {court}.",
        "📧 INTERNAL NOTE: Settlement conference for {clm_num}. Attorney attending.",
        "📧 INTERNAL NOTE: Expert witness retained for {clm_num}. Cost: ${amount:,.2f}.",
        "📧 INTERNAL NOTE: Independent medical exam scheduled for {clm_num}. Doctor: {doctor}.",
        "📧 INTERNAL NOTE: Surveillance completed for {clm_num}. No activity observed.",
        "📧 INTERNAL NOTE: Recorded statement taken for {clm_num}. Transcript attached."
    ]
    
    # Generate notes for each claim
    for _, claim in unique_claims.iterrows():
        clm_num = claim['clmNum']
        clm_cause = claim['clmCause']
        date_received = claim['dateReceived']
        date_completed = claim['dateCompleted']
        date_reopened = claim['dateReopened']
        current_reserve = claim['reserve_cumsum']
        
        # Determine number of notes (20-50 per claim)
        num_notes = np.random.randint(20, 51)
        
        # Generate timeline dates for notes
        timeline_dates = []
        
        # Phase 1: Initial claim (first 30 days)
        initial_dates = pd.date_range(date_received, date_received + timedelta(days=30), periods=8)
        timeline_dates.extend(initial_dates)
        
        # Phase 2: Investigation (next 60 days)
        investigation_dates = pd.date_range(date_received + timedelta(days=30), date_received + timedelta(days=90), periods=12)
        timeline_dates.extend(investigation_dates)
        
        # Phase 3: Settlement (next 30 days)
        settlement_dates = pd.date_range(date_received + timedelta(days=90), date_completed, periods=8)
        timeline_dates.extend(settlement_dates)
        
        # Phase 4: Reopening (after completion)
        if pd.notna(date_reopened):
            reopening_dates = pd.date_range(date_reopened, date_reopened + timedelta(days=60), periods=12)
            timeline_dates.extend(reopening_dates)
        
        # Fill remaining notes with random dates
        remaining_notes = num_notes - len(timeline_dates)
        if remaining_notes > 0:
            random_dates = pd.to_datetime(np.random.choice(
                pd.date_range(date_received, date_reopened if pd.notna(date_reopened) else date_completed + timedelta(days=30),
                             periods=remaining_notes),
                remaining_notes, replace=False
            ))
            timeline_dates.extend(random_dates)
        
        # Sort dates and ensure we have the right number
        timeline_dates = sorted(timeline_dates)[:num_notes]
        
        # Generate notes for each date
        for i, note_date in enumerate(timeline_dates):
            note_content = ""
            
            # Use cause-specific templates for early notes
            if i < 15 and clm_cause in note_templates:
                if i < 3:  # Opening phase
                    template = np.random.choice(note_templates[clm_cause]['opening'])
                elif i < 10:  # Investigation phase
                    template = np.random.choice(note_templates[clm_cause]['investigation'])
                elif i < 15:  # Settlement phase
                    template = np.random.choice(note_templates[clm_cause]['settlement'])
                else:
                    template = np.random.choice(generic_templates)
                
                # Fill template variables
                note_content = template.format(
                    clm_num=clm_num,
                    reserve=current_reserve,
                    amount=np.random.uniform(1000, current_reserve * 0.8),
                    date=note_date.strftime('%Y-%m-%d'),
                    adjuster_name=np.random.choice(['John Smith', 'Sarah Johnson', 'Mike Davis', 'Lisa Wilson']),
                    issue=np.random.choice(['coverage dispute', 'liability determination', 'damage assessment', 'settlement authority']),
                    case_number=f"CV-{np.random.randint(2020, 2025)}-{np.random.randint(1000, 9999)}",
                    court=np.random.choice(['Superior Court', 'District Court', 'Circuit Court']),
                    doctor=np.random.choice(['Dr. Brown', 'Dr. Garcia', 'Dr. Lee', 'Dr. Martinez'])
                )
            else:
                # Use generic templates for remaining notes
                template = np.random.choice(generic_templates)
                note_content = template.format(
                    clm_num=clm_num,
                    reserve=current_reserve,
                    amount=np.random.uniform(1000, current_reserve * 0.8),
                    date=note_date.strftime('%Y-%m-%d'),
                    adjuster_name=np.random.choice(['John Smith', 'Sarah Johnson', 'Mike Davis', 'Lisa Wilson']),
                    issue=np.random.choice(['coverage dispute', 'liability determination', 'damage assessment', 'settlement authority']),
                    case_number=f"CV-{np.random.randint(2020, 2025)}-{np.random.randint(1000, 9999)}",
                    court=np.random.choice(['Superior Court', 'District Court', 'Circuit Court']),
                    doctor=np.random.choice(['Dr. Brown', 'Dr. Garcia', 'Dr. Lee', 'Dr. Martinez'])
                )
            
            # Add some random variation
            if np.random.random() < 0.3:
                note_content += f" - Priority: {np.random.choice(['High', 'Medium', 'Low'])}"
            if np.random.random() < 0.2:
                note_content += f" - Follow-up required by {np.random.choice(['adjuster', 'legal', 'supervisor', 'client'])}"
            
            # Create note record
            note_record = {
                'dateNote': note_date,
                'clmNum': clm_num,
                'note': note_content,
                'note_type': 'email' if '📧' in note_content else 'internal_note',
                'priority': np.random.choice(['High', 'Medium', 'Low']),
                'author': np.random.choice(['Insurance Company', 'Adjuster', 'Legal Department', 'Client', 'Lawyer', 'Medical Provider', 'Contractor'])
            }
            
            all_notes.append(note_record)
    
    # Create DataFrame and sort by date
    notes_df = pd.DataFrame(all_notes)
    notes_df = notes_df.sort_values(['clmNum', 'dateNote'])
    
    # Save to CSV
    output_file = os.path.join("./_data", "claim_notes.csv")
    notes_df.to_csv(output_file, index=False)
    
    print(f"✅ Created {len(notes_df)} notes for {len(unique_claims)} claims")
    print(f"📁 Saved to: {output_file}")
    
    # Show sample notes
    print(f"\n📋 Sample notes for claim {unique_claims.iloc[0]['clmNum']}:")
    sample_notes = notes_df[notes_df['clmNum'] == unique_claims.iloc[0]['clmNum']].head(5)
    for _, note in sample_notes.iterrows():
        print(f"  {note['dateNote'].strftime('%Y-%m-%d')}: {note['note'][:80]}...")
    
    return notes_df

if __name__ == "__main__":
    create_claim_notes()
