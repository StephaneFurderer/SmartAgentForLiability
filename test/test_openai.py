import os
import openai
from dotenv import load_dotenv

# Load environment variables from .env file (if it exists)
load_dotenv()

# Get API key from environment variable
api_key = os.getenv("OPENAI_API_KEY")

if not api_key:
    print("❌ OPENAI_API_KEY not found in environment variables")
    print("Please set it with: export OPENAI_API_KEY='your_key_here'")
    exit(1)

print("✅ OpenAI API Key found!")

# Configure OpenAI client (new format)
client = openai.OpenAI(api_key=api_key)

def test_openai_connection():
    """Test basic OpenAI API connection"""
    try:
        # Test with a simple completion (new format)
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "user", "content": "Say 'Hello from OpenAI!' in a fun way"}
            ],
            max_tokens=50
        )
        
        print("🎉 OpenAI API connection successful!")
        print(f"Response: {response.choices[0].message.content}")
        return True
        
    except Exception as e:
        print(f"❌ Error connecting to OpenAI: {e}")
        return False

def test_models():
    """List available models"""
    try:
        models = client.models.list()
        print(f"\n📚 Available models: {len(models.data)}")
        for model in models.data[:5]:  # Show first 5 models
            print(f"  - {model.id}")
        return True
    except Exception as e:
        print(f"❌ Error listing models: {e}")
        return False

if __name__ == "__main__":
    print("🚀 Testing OpenAI API Connection...")
    print("=" * 40)
    
    # Test connection
    if test_openai_connection():
        # Test model listing
        test_models()
    
    print("\n" + "=" * 40)
    print("Test complete!")
