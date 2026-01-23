#!/usr/bin/env python3
import argparse
import sys
import json
import os

# Adjust path to find backend modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


from llm.openai_provider import OpenAIProvider
from llm.ollama_provider import OllamaProvider
from scripts.ingestors.datasheet_ingestor import DatasheetIngestor

# Default Sample Text (Fallback if no file provided)
SAMPLE_OCR_TEXT = """
PRODUCT SPECIFICATION
Model: U8-II-KV100
Manufacturer: T-Motor
Type: Brushless DC Motor
Application: Heavy Lift Multirotor / UAV

ELECTRICAL CHARACTERISTICS
KV ............................ 100
Internal Resistance ........... 17mOhm
Max Continuous Power .......... 940W (180s)
Max Continuous Current ........ 32A
Idle Current @ 10V ............ 1.2A
Recommended Battery ........... 6-12S LiPo

DIMENSIONS
Stator: 85x20 mm
Shaft Diameter: 15 mm
Motor Dimensions: 90.2 x 36.5 mm
Weight: 240g (including cables)
Cable Length: 800mm

LIFECYCLE & RELIABILITY
Service Life: >1000 hours
MTBF: 5000H (at 50% throttle)
Bearings: EZO 6902Z
"""

def main():
    parser = argparse.ArgumentParser(description="Datasheet Extraction Agent Utility")
    parser.add_argument("--file", help="Path to text file containing OCR/Datasheet text")

    args = parser.parse_args()

    print("🤖 -- Datasheet Agent Utility --")
    
    # 1. Determine Input Text
    if args.file:
        try:
            with open(args.file, 'r') as f:
                input_text = f.read()
            print(f"📄 Loaded Input from: {args.file}")
        except Exception as e:
            print(f"❌ Error reading file: {e}")
            return
    else:
        print("📄 No input file provided, using INTERNAL SAMPLE TEXT.")
        input_text = SAMPLE_OCR_TEXT.strip()

    print("-" * 40)
    print(input_text[:500] + ("..." if len(input_text) > 500 else ""))
    print("-" * 40)
    
    # 2. Determine Provider
    provider = None
    
    if args.mock:
        print("\n🎭 Mock mode forced via --mock")
    else:
        # Try OpenAI
        if os.getenv("OPENAI_API_KEY"):
            print("\n🌐 Detected OPENAI_API_KEY. Using OpenAIProvider (gpt-4-turbo)...")
            provider = OpenAIProvider()
        
        # Try Ollama (Auto-detect model)
        if not provider:
            print("\n🦙 Checking Ollama availability...")
            try:
                import requests
                resp = requests.get("http://localhost:11434/api/tags", timeout=30)
                if resp.status_code == 200:
                    models = [m["name"] for m in resp.json().get("models", [])]
                    print(f"   Found models: {models}")
                    
                    # Heuristic selection
                    selected_model = None
                    for m in ["llama3.2:latest", "llama3.2", "llama3:latest", "llama3", "llama2:latest", "llama2"]:
                        if m in models:
                            selected_model = m
                            break
                    
                    if not selected_model and models:
                        selected_model = models[0] # Pick first available
                        
                    if selected_model:
                        print(f"   ✅ Auto-selected model: {selected_model}")
                        provider = OllamaProvider(model_name=selected_model)
                    else:
                        print("   ❌ Ollama running but no models found.")
                else:
                    print(f"   ❌ Ollama returned status {resp.status_code}")
            except Exception as e:
                print(f"   (Ollama check failed: {e})")

    # Fallback to Mock if all else fails
    if not provider:
                "specs": {"kv": 100, "max_power_w": 940}, 
                "geometry_def": {"params": {"stator_w": 85, "stator_h": 20}},
                "behavior_model": {"reliability": {"mtbf_hours": 5000}}
            }
            provider.generate_json = MagicMock(return_value=expected_response)
    
    # 3. Run Agent
    agent = DatasheetIngestor(provider=provider)
    print("\n🚀 Sending Text to LLM Provider...")
    results = agent.parse_text(input_text)
    
    if not results:
        print("❌ Extraction Failed (result empty). Check Provider logs.")
        return

    print(f"\n✅ Extraction Complete. Found {len(results)} component(s):")
    for comp in results:
        print("\n📦 Component JSON Structure:")
        print(json.dumps(comp, indent=2))

if __name__ == "__main__":
    main()
