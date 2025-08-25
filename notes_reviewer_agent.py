import pandas as pd
import numpy as np
import os
from typing import Optional, Dict, List, Any
from datetime import datetime
import json

# Import OpenAI
try:
    import openai
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False


class NotesReviewerAgent:
    """
    Agent for reviewing and analyzing notes from claims data.
    
    This agent handles importing notes from CSV files and provides functionality
    for analyzing note patterns, frequencies, and relationships with claims.
    Now includes LLM capabilities for intelligent summarization and Q&A using OpenAI models.
    """
    
    def __init__(self, data_dir: str = "./_data", model_name: str = "gpt-3.5-turbo", 
                 openai_api_key: Optional[str] = None):
        """
        Initialize the NotesReviewerAgent.
        
        Args:
            data_dir (str): Directory containing the notes data files
            model_name (str): Name of the OpenAI model to use
            openai_api_key (str, optional): OpenAI API key (will use environment variable if not provided)
        """
        self.data_dir = data_dir
        self.notes_df: Optional[pd.DataFrame] = None
        self.notes_parquet_file = os.path.join(data_dir, "notes.parquet")
        self.model_name = model_name
        
        # Initialize OpenAI client
        if not OPENAI_AVAILABLE:
            raise ImportError("OpenAI package not installed. Run: pip install openai")
        
        if not openai_api_key:
            openai_api_key = os.getenv("OPENAI_API_KEY")
            if not openai_api_key:
                raise ValueError("OpenAI API key required. Set OPENAI_API_KEY environment variable or pass openai_api_key parameter.")
        
        self.openai_client = openai.OpenAI(api_key=openai_api_key)
        print(f"Initialized NotesReviewerAgent with OpenAI model: {self.model_name}")
        
    def import_notes(self, reload: bool = False) -> pd.DataFrame:
        """
        Import notes from CSV file and optionally save as parquet for faster loading.
        
        Args:
            reload (bool): If True, reload from CSV even if parquet exists
            
        Returns:
            pd.DataFrame: DataFrame containing the notes data
        """
        if reload or (not os.path.exists(self.notes_parquet_file)):
            print(f"Creating parquet file: {self.notes_parquet_file}")
            
            # Read from CSV
            csv_file = os.path.join(self.data_dir, "notes.csv")
            if not os.path.exists(csv_file):
                raise FileNotFoundError(f"Notes CSV file not found at: {csv_file}")
            
            # Read the CSV file
            df = pd.read_csv(csv_file)
            
            # Validate required columns
            required_columns = ['note', 'clmNum', 'whenadded']
            missing_columns = [col for col in required_columns if col not in df.columns]
            if missing_columns:
                raise ValueError(f"Missing required columns: {missing_columns}")
            
            # Convert whenadded to datetime
            df['whenadded'] = pd.to_datetime(df['whenadded'], errors='coerce')
            
            # Clean and process the data
            df = self._clean_notes_data(df)
            
            # Ensure data directory exists
            os.makedirs(self.data_dir, exist_ok=True)
            
            # Save as parquet for faster future loading
            df.to_parquet(self.notes_parquet_file)
            
            self.notes_df = df
            print(f"Successfully imported {len(df)} notes from CSV")
            
        else:
            print(f"Loading parquet file: {self.notes_parquet_file}")
            self.notes_df = pd.read_parquet(self.notes_parquet_file)
            print(f"Successfully loaded {len(self.notes_df)} notes from parquet")
        
        return self.notes_df
    
    def _clean_notes_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean and preprocess the notes data.
        
        Args:
            df (pd.DataFrame): Raw notes DataFrame
            
        Returns:
            pd.DataFrame: Cleaned notes DataFrame
        """
        # Remove rows with missing critical data
        df = df.dropna(subset=['note', 'clmNum'])
        
        # Clean note text - remove extra whitespace
        df['note'] = df['note'].str.strip()
        
        # Remove empty notes
        df = df[df['note'].str.len() > 0]
        
        # Sort by whenadded for chronological analysis
        df = df.sort_values(['clmNum', 'whenadded'])
        
        # Add note length for analysis
        df['note_length'] = df['note'].str.len()
        
        # Add note word count
        df['note_word_count'] = df['note'].str.split().str.len()
        
        return df
    
    def generate_claim_summary(self, clm_num: str) -> str:
        """
        Generate an intelligent summary of a claim using the configured LLM.
        
        Args:
            clm_num (str): Claim number to summarize
            
        Returns:
            str: Generated summary of the claim
        """
        if self.notes_df is None:
            raise ValueError("No notes data loaded. Call import_notes() first.")
        
        # Get all notes for the claim
        claim_notes = self.get_notes_by_claim(clm_num)
        
        if len(claim_notes) == 0:
            return "No notes found for this claim."
        
        # Format notes for the LLM
        notes_text = self._format_notes_for_llm(claim_notes)
        
        # Create prompt for summary
        prompt = f"""You are an insurance claims analyst. Based on the following notes for claim {clm_num}, provide a concise summary of what happened with this claim.

Focus on:
- The main issue or incident
- Key events and timeline
- Current status
- Any significant amounts or payments
- Important milestones or delays

Notes:
{notes_text}

Please provide a clear, professional summary in 2-3 sentences:"""

        try:
            # Generate summary using OpenAI
            response = self.openai_client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=150,
                temperature=0.3
            )
            return response.choices[0].message.content.strip()
            
        except Exception as e:
            print(f"Error generating summary with OpenAI: {e}")
            # Fallback to basic summary
            return self._generate_basic_summary(claim_notes)
    
    def answer_claim_question(self, clm_num: str, question: str) -> str:
        """
        Answer a specific question about a claim using the configured LLM.
        
        Args:
            clm_num (str): Claim number to ask about
            question (str): Question to ask about the claim
            
        Returns:
            str: Answer to the question based on the notes
        """
        if self.notes_df is None:
            raise ValueError("No notes data loaded. Call import_notes() first.")
        
        # Get all notes for claim
        claim_notes = self.get_notes_by_claim(clm_num)
        
        if len(claim_notes) == 0:
            return f"No notes found for claim {clm_num}."
        
        # Format notes for the LLM
        notes_text = self._format_notes_for_llm(claim_notes)
        
        # Create prompt for Q&A
        prompt = f"""You are an insurance claims analyst. Answer the following question about claim {clm_num} based on the notes provided.

Question: {question}

Notes for claim {clm_num}:
{notes_text}

Please provide a clear, accurate answer based only on the information in these notes. If the information isn't available in the notes, say so. Be specific and reference dates and details when possible:"""

        try:
            # Generate answer using OpenAI
            response = self.openai_client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=200,
                temperature=0.3
            )
            return response.choices[0].message.content.strip()
            
        except Exception as e:
            print(f"Error generating answer with OpenAI: {e}")
            return f"Sorry, I encountered an error while processing your question. Please try again or rephrase your question."
    
    def _format_notes_for_llm(self, claim_notes: pd.DataFrame) -> str:
        """
        Format claim notes into a text format suitable for LLM processing.
        
        Args:
            claim_notes (pd.DataFrame): Notes for a specific claim
            
        Returns:
            str: Formatted notes text
        """
        formatted_notes = []
        
        for idx, row in claim_notes.iterrows():
            date_str = row['whenadded'].strftime('%Y-%m-%d %H:%M')
            note_text = row['note']
            formatted_notes.append(f"{date_str}: {note_text}")
        
        return "\n".join(formatted_notes)
    
    def _generate_basic_summary(self, claim_notes: pd.DataFrame) -> str:
        """
        Generate a basic summary without LLM (fallback method).
        
        Args:
            claim_notes (pd.DataFrame): Notes for a specific claim
            
        Returns:
            str: Basic summary
        """
        if len(claim_notes) == 0:
            return "No notes available for this claim."
        
        first_note = claim_notes.iloc[0]
        last_note = claim_notes.iloc[-1]
        
        total_notes = len(claim_notes)
        date_range = (last_note['whenadded'] - first_note['whenadded']).days
        
        summary = f"Claim has {total_notes} notes spanning {date_range} days. "
        summary += f"Started on {first_note['whenadded'].strftime('%Y-%m-%d')} with '{first_note['note'][:50]}...' "
        
        if total_notes > 1:
            summary += f"Latest note on {last_note['whenadded'].strftime('%Y-%m-%d')}: '{last_note['note'][:50]}...'"
        
        return summary
    
    def get_notes_summary(self) -> Dict[str, Any]:
        """
        Get a summary of the notes data.
        
        Returns:
            Dict[str, Any]: Summary statistics about the notes
        """
        if self.notes_df is None:
            raise ValueError("No notes data loaded. Call import_notes() first.")
        
        summary = {
            'total_notes': len(self.notes_df),
            'unique_claims': self.notes_df['clmNum'].nunique(),
            'date_range': {
                'earliest': self.notes_df['whenadded'].min(),
                'latest': self.notes_df['whenadded'].max()
            },
            'note_length_stats': {
                'mean': self.notes_df['note_length'].mean(),
                'median': self.notes_df['note_length'].median(),
                'min': self.notes_df['note_length'].min(),
                'max': self.notes_df['note_length'].max()
            },
            'word_count_stats': {
                'mean': self.notes_df['note_word_count'].mean(),
                'median': self.notes_df['note_word_count'].median(),
                'min': self.notes_df['note_word_count'].min(),
                'max': self.notes_df['note_word_count'].max()
            }
        }
        
        return summary
    
    def get_notes_by_claim(self, clm_num: str) -> pd.DataFrame:
        """
        Get all notes for a specific claim number.
        
        Args:
            clm_num (str): Claim number to search for
            
        Returns:
            pd.DataFrame: Notes for the specified claim
        """
        if self.notes_df is None:
            raise ValueError("No notes data loaded. Call import_notes() first.")
        
        claim_notes = self.notes_df[self.notes_df['clmNum'] == clm_num].copy()
        return claim_notes.sort_values('whenadded')
    
    def get_notes_by_date_range(self, start_date: str, end_date: str) -> pd.DataFrame:
        """
        Get notes within a specific date range.
        
        Args:
            start_date (str): Start date in 'YYYY-MM-DD' format
            end_date (str): End date in 'YYYY-MM-DD' format
            
        Returns:
            pd.DataFrame: Notes within the specified date range
        """
        if self.notes_df is None:
            raise ValueError("No notes data loaded. Call import_notes() first.")
        
        start_dt = pd.to_datetime(start_date)
        end_dt = pd.to_datetime(end_date)
        
        date_mask = (self.notes_df['whenadded'] >= start_dt) & (self.notes_df['whenadded'] <= end_dt)
        date_notes = self.notes_df[date_mask].copy()
        
        return date_notes.sort_values('whenadded')
    
    def search_notes(self, search_term: str, case_sensitive: bool = False) -> pd.DataFrame:
        """
        Search notes for specific text.
        
        Args:
            search_term (str): Text to search for in notes
            case_sensitive (bool): Whether the search should be case sensitive
            
        Returns:
            pd.DataFrame: Notes containing the search term
        """
        if self.notes_df is None:
            raise ValueError("No notes data loaded. Call import_notes() first.")
        
        if case_sensitive:
            mask = self.notes_df['note'].str.contains(search_term, na=False)
        else:
            mask = self.notes_df['note'].str.contains(search_term, case=False, na=False)
        
        matching_notes = self.notes_df[mask].copy()
        return matching_notes.sort_values(['clmNum', 'whenadded'])
    
    def get_claim_note_frequency(self) -> pd.DataFrame:
        """
        Get the frequency of notes per claim.
        
        Returns:
            pd.DataFrame: Claim numbers with their note counts
        """
        if self.notes_df is None:
            raise ValueError("No notes data loaded. Call import_notes() first.")
        
        claim_frequency = self.notes_df.groupby('clmNum').agg({
            'note': 'count',
            'whenadded': ['min', 'max'],
            'note_length': 'mean'
        }).reset_index()
        
        # Flatten column names
        claim_frequency.columns = ['clmNum', 'note_count', 'first_note_date', 'last_note_date', 'avg_note_length']
        
        return claim_frequency.sort_values('note_count', ascending=False)
    
    def get_notes_timeline(self, clm_num: str) -> pd.DataFrame:
        """
        Get a timeline of notes for a specific claim.
        
        Args:
            clm_num (str): Claim number to get timeline for
            
        Returns:
            pd.DataFrame: Timeline of notes with days since first note
        """
        if self.notes_df is None:
            raise ValueError("No notes data loaded. Call import_notes() first.")
        
        claim_notes = self.get_notes_by_claim(clm_num)
        
        if len(claim_notes) == 0:
            return pd.DataFrame()
        
        # Calculate days since first note
        first_note_date = claim_notes['whenadded'].min()
        claim_notes['days_since_first'] = (claim_notes['whenadded'] - first_note_date).dt.days
        
        return claim_notes[['whenadded', 'days_since_first', 'note', 'note_length', 'note_word_count']]


def create_dummy_notes_data(n_claims: int = 100, notes_per_claim: int = 5) -> pd.DataFrame:
    """
    Create dummy notes data for testing purposes.
    
    Args:
        n_claims (int): Number of unique claims to generate
        notes_per_claim (int): Average number of notes per claim
        
    Returns:
        pd.DataFrame: Dummy notes data
    """
    np.random.seed(42)
    
    # Sample note templates
    note_templates = [
        "Initial review completed for claim {clm_num}",
        "Follow-up investigation required for {clm_num}",
        "Documentation received for claim {clm_num}",
        "Payment processed for {clm_num}",
        "Claim {clm_num} status updated",
        "Additional information requested for {clm_num}",
        "Claim {clm_num} assigned to adjuster",
        "Review completed for {clm_num}",
        "Claim {clm_num} documentation verified",
        "Payment authorization for {clm_num}"
    ]
    
    notes_data = []
    
    for i in range(n_claims):
        clm_num = f"CLM{np.random.randint(10000000, 99999999):08d}"
        
        # Generate random number of notes for this claim (1 to 2*notes_per_claim)
        n_notes = np.random.poisson(notes_per_claim) + 1
        
        # Generate base date for this claim (within last 2 years)
        base_date = pd.to_datetime('2022-01-01') + pd.Timedelta(days=np.random.randint(0, 730))
        
        for j in range(n_notes):
            # Generate note date (within 6 months of base date)
            note_date = base_date + pd.Timedelta(days=np.random.randint(0, 180))
            
            # Select random note template and format it
            template = np.random.choice(note_templates)
            note_text = template.format(clm_num=clm_num)
            
            # Add some random variation to the note
            if np.random.random() < 0.3:
                note_text += f" - Priority: {np.random.choice(['High', 'Medium', 'Low'])}"
            if np.random.random() < 0.2:
                note_text += f" - Amount: ${np.random.randint(1000, 50000):,}"
            
            notes_data.append({
                'note': note_text,
                'clmNum': clm_num,
                'whenadded': note_date
            })
    
    # Create DataFrame
    df = pd.DataFrame(notes_data)
    
    # Sort by claim number and date
    df = df.sort_values(['clmNum', 'whenadded'])
    
    return df


if __name__ == "__main__":
    # Example usage
    agent = NotesReviewerAgent()
    
    # Create dummy data if notes.csv doesn't exist
    notes_csv_path = os.path.join("./_data", "notes.csv")
    if not os.path.exists(notes_csv_path):
        print("Creating dummy notes data...")
        os.makedirs("./_data", exist_ok=True)
        dummy_notes = create_dummy_notes_data()
        dummy_notes.to_csv(notes_csv_path, index=False)
        print(f"Created dummy notes data with {len(dummy_notes)} notes")
    
    # Import notes
    notes_df = agent.import_notes()
    
    # Display summary
    summary = agent.get_notes_summary()
    print("\nNotes Summary:")
    for key, value in summary.items():
        print(f"  {key}: {value}")
    
    # Example: Get notes for first claim
    if len(notes_df) > 0:
        first_claim = notes_df['clmNum'].iloc[0]
        print(f"\nNotes for claim {first_claim}:")
        claim_notes = agent.get_notes_by_claim(first_claim)
        print(claim_notes[['whenadded', 'note']].to_string(index=False))
        
        # Test OpenAI summary
        print(f"\nOpenAI Summary for {first_claim}:")
        try:
            summary = agent.generate_claim_summary(first_claim)
            print(summary)
        except Exception as e:
            print(f"OpenAI summary failed: {e}")
            print("Using basic summary instead")
            basic_summary = agent._generate_basic_summary(claim_notes)
            print(basic_summary)
