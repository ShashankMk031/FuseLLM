"""
Test script for NLP integration in FuseLLM.

This script tests the NLP processor and its integration with the orchestrator.
"""
import sys
import os
import json
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

# Add the project root to the Python path
project_root = str(Path(__file__).parent.parent)
sys.path.insert(0, project_root)

from src.nlp.processor import NLPProcessor, nlp_processor
from src.core.orchestrator import run_fuse_pipeline

def test_nlp_processor():
    """Test the NLP processor with sample inputs."""
    print("\n" + "="*50)
    print("Testing NLP Processor...")
    print("="*50)
    
    # Initialize the processor
    nlp = NLPProcessor()
    
    # Test cases with expected intents
    test_cases = [
        ("Hello, how are you?", ["greeting"]),
        ("Hi there!", ["greeting"]),
        ("Tell me a joke", ["joke"]),
        ("Can you make me laugh?", ["joke"]),
        ("What's the weather like today?", ["weather"]),
        ("Will it rain tomorrow?", ["weather"]),
        ("Goodbye!", ["goodbye"]),
        ("See you later", ["goodbye"]),
        ("Help me with something", ["help"]),
        ("I need assistance", ["help"]),
        ("Tell me about AI", ["general"]),
        ("What can you do?", ["general"])
    ]
    
    for text, expected_intents in test_cases:
        print(f"\n{'='*80}")
        print(f"Input: {text}")
        print(f"Expected intents: {', '.join(expected_intents)}")
        
        # Test intent detection
        intents = nlp.detect_intents(text)
        print("\nDetected intents:")
        for intent in intents:
            print(f"- {intent['intent']} (score: {intent['score']:.4f})")
        
        # Test primary intent
        primary = nlp.get_primary_intent(text)
        if primary:
            print(f"\nPrimary intent: {primary['intent']} (score: {primary['score']:.4f})")
            
            # Verify primary intent is in expected intents
            if primary['intent'] in expected_intents:
                print("✅ Primary intent matches expected")
            else:
                print(f"❌ Primary intent '{primary['intent']}' not in expected: {expected_intents}")
        else:
            print("❌ No primary intent detected")

def test_orchestrator_integration():
    """Test the orchestrator with NLP integration."""
    print("\n" + "="*50)
    print("Testing Orchestrator with NLP Integration...")
    print("="*50)
    
    test_cases = [
        ("Hello!", "greeting"),
        ("Tell me a joke about programmers", "joke"),
        ("What's the weather in New York?", "weather"),
        ("Help me with something", "help"),
        ("Goodbye!", "goodbye"),
        ("Explain quantum computing", "general")
    ]
    
    for text, expected_intent in test_cases:
        print(f"\n{'='*80}")
        print(f"Input: {text}")
        print(f"Expected intent: {expected_intent}")
        
        try:
            response = run_fuse_pipeline(text)
            print(f"\nResponse metadata:")
            print(json.dumps({
                'intent': response.get('intent', 'unknown'),
                'model': response.get('model', 'unknown'),
                'confidence': response.get('confidence', 0.0)
            }, indent=2))
            
            print(f"\nResponse: {response.get('response', 'No response')[:200]}...")
            
            # Verify the detected intent
            if response.get('intent') == expected_intent:
                print(f"✅ Correct intent detected: {expected_intent}")
            else:
                print(f"❌ Expected intent '{expected_intent}', got '{response.get('intent', 'unknown')}'")
                
        except Exception as e:
            logger.error(f"Error processing input: {text}")
            logger.error(f"Error details: {str(e)}", exc_info=True)
            print(f"❌ Error: {str(e)}")

def test_custom_intents():
    """Test adding and detecting custom intents."""
    print("\n" + "="*50)
    print("Testing Custom Intents...")
    print("="*50)
    
    # Create a new processor instance for testing
    nlp = NLPProcessor()
    
    # Add a custom intent
    custom_intent = "pizza_order"
    examples = [
        "I'd like to order a pizza",
        "Can I get a large pepperoni pizza?",
        "I want to order pizza for delivery",
        "Pizza delivery please"
    ]
    
    print(f"\nAdding custom intent: {custom_intent}")
    print(f"Example phrases: {examples}")
    
    # Add a description for the custom intent
    description = "User wants to order pizza"
    nlp.add_custom_intent(custom_intent, examples, description)
    
    # Test the custom intent
    test_phrases = [
        ("I want to order a pizza", True),
        ("What's the weather like?", False),
        ("Can I get a large pepperoni pizza?", True),
        ("Tell me a joke", False)
    ]
    
    for phrase, should_match in test_phrases:
        print(f"\nTesting phrase: {phrase}")
        primary = nlp.get_primary_intent(phrase)
        
        if primary:
            print(f"Detected intent: {primary['intent']} (score: {primary['score']:.4f})")
            
            if should_match and primary['intent'] == custom_intent:
                print("✅ Correctly matched custom intent")
            elif not should_match and primary['intent'] != custom_intent:
                print("✅ Correctly did not match custom intent")
            else:
                print("❌ Incorrect intent match")
        else:
            print("❌ No intent detected")

if __name__ == "__main__":
    print("\n" + "="*80)
    print("=== FuseLLM NLP Integration Tests ===")
    print("="*80)
    
    # Test the NLP processor
    test_nlp_processor()
    
    # Test custom intents
    test_custom_intents()
    
    # Test orchestrator integration
    test_orchestrator_integration()
    
    print("\n" + "="*80)
    print("=== Testing Complete ===")
    print("="*80)
