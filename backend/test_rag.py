from app.chatbot.generator import generate_local_treatment

def run_test():
    print("🚀 Initializing Local RAG Pipeline...")
    print("Note: If this is the first run, it will take a moment to scrape the WHO site and build ChromaDB.\n")
    
    try:
        # Let's test a scenario that would trigger a corrective treatment
        parameter = "Ammonia"
        condition = "High"
        
        print(f"Asking Phi-3 for {condition} {parameter} treatments based on WHO guidelines...")
        
        # Trigger the engine
        response = generate_local_treatment(parameter=parameter, condition=condition)
        
        print("\n" + "="*50)
        print("✅ SUCCESS! HERE IS THE AI GENERATED RESPONSE:")
        print("="*50 + "\n")
        print(response)
        
    except Exception as e:
        print(f"\n❌ Error during execution: {e}")

if __name__ == "__main__":
    run_test()