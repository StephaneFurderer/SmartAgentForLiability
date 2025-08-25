#!/usr/bin/env python3
"""
Test script to verify OpenAI-only NotesReviewerAgent
"""

import os
from notes_reviewer_agent import NotesReviewerAgent

def test_openai_agent():
    """Test OpenAI-only agent"""
    print("🧪 Testing OpenAI-Only NotesReviewerAgent")
    print("=" * 50)
    
    # Check if OpenAI API key is available
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("❌ OPENAI_API_KEY not found in environment variables")
        print("Please set it with: export OPENAI_API_KEY='your_key_here'")
        return False
    
    print("✅ OpenAI API Key found!")
    
    try:
        # Test OpenAI agent
        print("\n🔧 Initializing OpenAI agent...")
        agent = NotesReviewerAgent(
            model_name="gpt-3.5-turbo",
            openai_api_key=api_key
        )
        print("✅ OpenAI agent initialized successfully!")
        
        # Test with dummy data
        print("\n📊 Testing with dummy data...")
        
        # Create dummy notes
        import pandas as pd
        dummy_notes = pd.DataFrame({
            'note': [
                'Initial claim filed for car accident',
                'Police report received',
                'Insurance adjuster assigned',
                'Vehicle inspection completed',
                'Payment processed for repairs'
            ],
            'clmNum': ['CLM12345'] * 5,
            'whenadded': pd.date_range('2024-01-01', periods=5, freq='D')
        })
        
        # Test OpenAI summary
        print("\n🤖 Testing OpenAI summary...")
        try:
            # Mock the notes_df for testing
            agent.notes_df = dummy_notes
            
            summary = agent.generate_claim_summary("CLM12345")
            print(f"✅ OpenAI Summary: {summary}")
            
            # Test Q&A
            print("\n💬 Testing OpenAI Q&A...")
            answer = agent.answer_claim_question("CLM12345", "What was the initial issue?")
            print(f"✅ OpenAI Answer: {answer}")
            
        except Exception as e:
            print(f"❌ OpenAI test failed: {e}")
            return False
        
        print("\n" + "=" * 50)
        print("✅ All tests passed! Your OpenAI integration is working.")
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False

if __name__ == "__main__":
    success = test_openai_agent()
    if success:
        print("🎉 OpenAI integration is ready to use!")
    else:
        print("💥 Some tests failed. Check the errors above.")
