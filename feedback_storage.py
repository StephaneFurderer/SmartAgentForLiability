#!/usr/bin/env python3
"""
Feedback Storage Module

Handles persistent storage of claim representative feedback data.
"""

import pandas as pd
import os
from datetime import datetime
from typing import Dict, List, Optional

class FeedbackStorage:
    """
    Manages persistent storage of claim representative feedback.
    """
    
    def __init__(self, data_dir: str = "./_data"):
        self.data_dir = data_dir
        self.feedback_file = os.path.join(data_dir, "claim_feedback.csv")
        self._ensure_data_dir()
        self._initialize_feedback_file()
    
    def _ensure_data_dir(self):
        """Ensure the data directory exists."""
        os.makedirs(self.data_dir, exist_ok=True)
    
    def _initialize_feedback_file(self):
        """Initialize the feedback CSV file if it doesn't exist."""
        if not os.path.exists(self.feedback_file):
            # Create empty feedback file with headers
            empty_df = pd.DataFrame(columns=[
                'claim_number',
                'model_paid_prob',
                'model_closed_prob', 
                'model_denied_prob',
                'model_confidence',
                'rep_paid_prob',
                'rep_closed_prob',
                'rep_denied_prob',
                'rep_confidence',
                'thought_process',
                'rep_name',
                'timestamp',
                'custom_factors'
            ])
            empty_df.to_csv(self.feedback_file, index=False)
            print(f"Initialized feedback file: {self.feedback_file}")
    
    def save_feedback(self, feedback_data: Dict) -> bool:
        """
        Save feedback data to CSV file.
        
        Args:
            feedback_data: Dictionary containing feedback information
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            # Load existing feedback
            if os.path.exists(self.feedback_file):
                existing_df = pd.read_csv(self.feedback_file)
            else:
                existing_df = pd.DataFrame()
            
            # Check if feedback for this claim already exists
            claim_number = feedback_data['claim_number']
            if len(existing_df) > 0:
                # Remove existing feedback for this claim
                existing_df = existing_df[existing_df['claim_number'] != claim_number]
            
            # Add new feedback
            new_row = pd.DataFrame([feedback_data])
            updated_df = pd.concat([existing_df, new_row], ignore_index=True)
            
            # Save to CSV
            updated_df.to_csv(self.feedback_file, index=False)
            
            print(f"Feedback saved for claim {claim_number}")
            return True
            
        except Exception as e:
            print(f"Error saving feedback: {e}")
            return False
    
    def load_feedback(self, claim_number: str) -> Optional[Dict]:
        """
        Load feedback for a specific claim.
        
        Args:
            claim_number: The claim number to load feedback for
            
        Returns:
            Dict: Feedback data if found, None otherwise
        """
        try:
            if not os.path.exists(self.feedback_file):
                return None
            
            df = pd.read_csv(self.feedback_file)
            if len(df) == 0:
                return None
            
            # Find feedback for this claim
            claim_feedback = df[df['claim_number'] == claim_number]
            
            if len(claim_feedback) > 0:
                # Get the most recent feedback (last row)
                latest_feedback = claim_feedback.iloc[-1].to_dict()
                
                # Convert numeric fields back to float
                numeric_fields = [
                    'model_paid_prob', 'model_closed_prob', 'model_denied_prob', 'model_confidence',
                    'rep_paid_prob', 'rep_closed_prob', 'rep_denied_prob', 'rep_confidence'
                ]
                
                for field in numeric_fields:
                    if field in latest_feedback and pd.notna(latest_feedback[field]):
                        try:
                            latest_feedback[field] = float(latest_feedback[field])
                        except (ValueError, TypeError):
                            latest_feedback[field] = 0.0
                    else:
                        latest_feedback[field] = 0.0
                
                return latest_feedback
            
            return None
            
        except Exception as e:
            print(f"Error loading feedback: {e}")
            return None
    
    def get_all_feedback(self) -> pd.DataFrame:
        """
        Get all feedback data.
        
        Returns:
            pd.DataFrame: All feedback data
        """
        try:
            if os.path.exists(self.feedback_file):
                return pd.read_csv(self.feedback_file)
            else:
                return pd.DataFrame()
        except Exception as e:
            print(f"Error loading all feedback: {e}")
            return pd.DataFrame()
    
    def delete_feedback(self, claim_number: str) -> bool:
        """
        Delete feedback for a specific claim.
        
        Args:
            claim_number: The claim number to delete feedback for
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            if not os.path.exists(self.feedback_file):
                return True
            
            df = pd.read_csv(self.feedback_file)
            if len(df) == 0:
                return True
            
            # Remove feedback for this claim
            updated_df = df[df['claim_number'] != claim_number]
            
            # Save updated data
            updated_df.to_csv(self.feedback_file, index=False)
            
            print(f"Feedback deleted for claim {claim_number}")
            return True
            
        except Exception as e:
            print(f"Error deleting feedback: {e}")
            return False
    
    def get_feedback_summary(self) -> Dict:
        """
        Get summary statistics of feedback data.
        
        Returns:
            Dict: Summary statistics
        """
        try:
            df = self.get_all_feedback()
            
            if len(df) == 0:
                return {
                    'total_feedback': 0,
                    'unique_claims': 0,
                    'unique_reps': 0,
                    'avg_confidence': 0.0,
                    'feedback_by_status': {}
                }
            
            # Calculate summary statistics
            summary = {
                'total_feedback': len(df),
                'unique_claims': df['claim_number'].nunique(),
                'unique_reps': df['rep_name'].nunique(),
                'avg_confidence': df['rep_confidence'].mean() if 'rep_confidence' in df.columns else 0.0,
                'feedback_by_status': {}
            }
            
            # Analyze feedback by predicted status
            if 'rep_paid_prob' in df.columns and 'rep_closed_prob' in df.columns and 'rep_denied_prob' in df.columns:
                # Determine predicted status based on highest probability
                df['predicted_status'] = df[['rep_paid_prob', 'rep_closed_prob', 'rep_denied_prob']].idxmax(axis=1)
                df['predicted_status'] = df['predicted_status'].str.replace('_prob', '').str.upper()
                
                status_counts = df['predicted_status'].value_counts().to_dict()
                summary['feedback_by_status'] = status_counts
            
            return summary
            
        except Exception as e:
            print(f"Error generating feedback summary: {e}")
            return {
                'total_feedback': 0,
                'unique_claims': 0,
                'unique_reps': 0,
                'avg_confidence': 0.0,
                'feedback_by_status': {}
            }

def main():
    """Test the feedback storage system."""
    print("Testing Feedback Storage System...")
    
    storage = FeedbackStorage()
    
    # Test data
    test_feedback = {
        'claim_number': 'TEST123',
        'model_paid_prob': 0.6,
        'model_closed_prob': 0.3,
        'model_denied_prob': 0.1,
        'model_confidence': 0.75,
        'rep_paid_prob': 0.8,
        'rep_closed_prob': 0.15,
        'rep_denied_prob': 0.05,
        'rep_confidence': 0.9,
        'thought_process': 'Test feedback for demonstration',
        'rep_name': 'Test User',
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'custom_factors': 'Test factors'
    }
    
    # Test save
    print("\n1. Testing save feedback...")
    success = storage.save_feedback(test_feedback)
    print(f"Save successful: {success}")
    
    # Test load
    print("\n2. Testing load feedback...")
    loaded_feedback = storage.load_feedback('TEST123')
    if loaded_feedback:
        print(f"Loaded feedback for claim: {loaded_feedback['claim_number']}")
        print(f"Rep name: {loaded_feedback['rep_name']}")
        print(f"Rep confidence: {loaded_feedback['rep_confidence']}")
    else:
        print("No feedback loaded")
    
    # Test summary
    print("\n3. Testing feedback summary...")
    summary = storage.get_feedback_summary()
    print(f"Total feedback: {summary['total_feedback']}")
    print(f"Unique claims: {summary['unique_claims']}")
    print(f"Unique reps: {summary['unique_reps']}")
    print(f"Average confidence: {summary['avg_confidence']:.2f}")
    
    # Test delete
    print("\n4. Testing delete feedback...")
    delete_success = storage.delete_feedback('TEST123')
    print(f"Delete successful: {delete_success}")
    
    # Verify deletion
    loaded_after_delete = storage.load_feedback('TEST123')
    print(f"Feedback after deletion: {loaded_after_delete is None}")

if __name__ == "__main__":
    main()
