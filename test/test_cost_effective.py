import os
import openai
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Get API key from environment variable
api_key = os.getenv("OPENAI_API_KEY")

if not api_key:
    print("❌ OPENAI_API_KEY not found in environment variables")
    exit(1)

print("✅ OpenAI API Key found!")

# Configure OpenAI client
client = openai.OpenAI(api_key=api_key)

def test_cost_effective_models():
    """Test different cost-effective models"""
    
    # Cost-effective models to test
    models_to_test = [
        "gpt-3.5-turbo",           # Fastest, cheapest
        "gpt-4o-mini",             # Good balance of cost/quality
        "gpt-3.5-turbo-0125"       # Latest 3.5 version
    ]
    
    test_prompt = "Explain quantum computing in 2 sentences like you're talking to a 10-year-old."
    
    for model in models_to_test:
        print(f"\n🧪 Testing model: {model}")
        print("-" * 50)
        
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "user", "content": test_prompt}
                ],
                max_tokens=100,
                temperature=0.7
            )
            
            print(f"✅ Success!")
            print(f"Response: {response.choices[0].message.content}")
            print(f"Usage: {response.usage}")
            
        except Exception as e:
            print(f"❌ Error: {e}")

def test_gpt_3_5_turbo_cheap():
    """Test the most cost-effective model with different settings"""
    print(f"\n💰 Testing GPT-3.5-turbo (cheapest option)")
    print("=" * 60)
    
    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "user", "content": "Write a haiku about coding"}
            ],
            max_tokens=50,          # Limit tokens to save cost
            temperature=0.3,        # Lower temperature = more focused
            presence_penalty=0,     # No penalties to save cost
            frequency_penalty=0
        )
        
        print(f"✅ Success!")
        print(f"Response: {response.choices[0].message.content}")
        print(f"Tokens used: {response.usage.total_tokens}")
        print(f"Cost estimate: ~${response.usage.total_tokens * 0.000002:.6f}")
        
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    print("🚀 Testing Cost-Effective OpenAI Models...")
    print("=" * 60)
    
    # Test different models
    test_cost_effective_models()
    
    # Test the cheapest option with cost optimization
    test_gpt_3_5_turbo_cheap()
    
    print("\n" + "=" * 60)
    print("💡 Cost-saving tips:")
    print("  - Use gpt-3.5-turbo for most tasks")
    print("  - Limit max_tokens to what you need")
    print("  - Use lower temperature for focused responses")
    print("  - Avoid presence/frequency penalties")
    print("Test complete!")
