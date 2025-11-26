#!/usr/bin/env python3
"""
Test script for generation pipeline
"""

import os
import sys
import asyncio

# Load .env first
from dotenv import load_dotenv
load_dotenv()

from llm_providers import get_provider, OpenAIProvider, OllamaProvider
from generator import DatasetGenerator
from domains import DOMAINS

def test_ollama_connection():
    """Test Ollama connection"""
    print("=" * 50)
    print("TEST 0: Ollama Connection")
    print("=" * 50)
    
    try:
        provider = get_provider("ollama")
        print(f"Provider type: {type(provider).__name__}")
        print(f"Provider available: {provider.is_available()}")
        
        if hasattr(provider, 'model'):
            print(f"Model: {provider.model}")
        
        # List available models
        models = provider.get_models()
        print(f"Available models: {models[:5]}..." if len(models) > 5 else f"Available models: {models}")
        
        return provider.is_available()
    except Exception as e:
        print(f"ERROR: {e}")
        return False


def test_ollama_generation():
    """Test Ollama text generation"""
    print("\n" + "=" * 50)
    print("TEST 0.5: Ollama Text Generation")
    print("=" * 50)
    
    try:
        provider = get_provider("ollama")
        
        if not provider.is_available():
            print("SKIP: Ollama not available")
            return False
        
        prompt = "Say 'Hello from Ollama!' and nothing else."
        print(f"Prompt: {prompt}")
        
        result = provider.generate_text(prompt, temperature=0.1)
        print(f"Result: {result[:200]}..." if len(result) > 200 else f"Result: {result}")
        
        return bool(result)
    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_openai_connection():
    """Test OpenAI API connection"""
    print("=" * 50)
    print("TEST 1: OpenAI Connection")
    print("=" * 50)
    
    api_key = os.getenv("OPENAI_API_KEY")
    print(f"API Key loaded: {'Yes' if api_key else 'No'}")
    if api_key:
        print(f"API Key prefix: {api_key[:20]}...")
    
    try:
        provider = get_provider("openai")
        print(f"Provider type: {type(provider).__name__}")
        print(f"Provider available: {provider.is_available()}")
        
        if hasattr(provider, 'model'):
            print(f"Model: {provider.model}")
        
        return provider.is_available()
    except Exception as e:
        print(f"ERROR: {e}")
        return False


def test_simple_generation():
    """Test simple text generation"""
    print("\n" + "=" * 50)
    print("TEST 2: Simple Text Generation")
    print("=" * 50)
    
    try:
        provider = get_provider("openai")
        
        if not provider.is_available():
            print("SKIP: Provider not available")
            return False
        
        prompt = "Say 'Hello, test successful!' and nothing else."
        print(f"Prompt: {prompt}")
        
        result = provider.generate_text(prompt, temperature=0.1)
        print(f"Result: {result}")
        
        return bool(result)
    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_json_generation():
    """Test JSON generation for dataset"""
    print("\n" + "=" * 50)
    print("TEST 3: JSON Generation")
    print("=" * 50)
    
    try:
        provider = get_provider("openai")
        
        if not provider.is_available():
            print("SKIP: Provider not available")
            return False
        
        prompt = """Generate a simple customer support conversation in JSON format:
{
    "messages": [
        {"role": "user", "content": "user question here"},
        {"role": "assistant", "content": "assistant response here"}
    ]
}
Return ONLY valid JSON, no markdown or explanations."""
        
        print(f"Prompt: {prompt[:100]}...")
        
        result = provider.generate_text(prompt, temperature=0.7)
        print(f"Raw result: {result[:500]}...")
        
        # Try to parse JSON
        import json
        start_idx = result.find('{')
        end_idx = result.rfind('}') + 1
        
        if start_idx != -1 and end_idx > start_idx:
            clean_json = result[start_idx:end_idx]
            data = json.loads(clean_json)
            print(f"Parsed JSON: {json.dumps(data, indent=2)[:300]}...")
            return True
        else:
            print("ERROR: Could not find JSON in response")
            return False
            
    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_dataset_generator():
    """Test full dataset generator"""
    print("\n" + "=" * 50)
    print("TEST 4: Dataset Generator")
    print("=" * 50)
    
    try:
        provider = get_provider("openai")
        
        if not provider.is_available():
            print("SKIP: Provider not available")
            return False
        
        generator = DatasetGenerator(provider)
        
        # Generate scenario (use None for random subdomain)
        scenario = generator.generate_scenario_parameters("support", None)
        print(f"Scenario: {scenario}")
        
        # Generate example
        example = generator.generate_example(scenario, "chat", "en")
        print(f"Generated example: {example}")
        
        return example is not None
        
    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_swarm_agents():
    """Test Swarm multi-agent generation"""
    print("\n" + "=" * 50)
    print("TEST 5: Swarm Multi-Agent Generation")
    print("=" * 50)
    
    try:
        from agents.base_agent import AgentEventBus, get_event_bus, reset_llm_limiter
        from agents.shared_memory import SharedMemory
        from agents.swarm import HybridSwarmSynthesis
        
        # Reset LLM limiter for this test
        reset_llm_limiter()
        
        # Create event bus and shared memory
        event_bus = AgentEventBus()
        shared_memory = SharedMemory()
        
        # Track events
        events_received = []
        async def event_listener(event):
            events_received.append(event)
            print(f"  [{event.agent_role.value}] {event.event_type.value}: {str(event.data)[:100]}...")
        
        event_bus.subscribe(event_listener)
        
        # Create swarm with minimal config for testing
        swarm = HybridSwarmSynthesis(
            event_bus=event_bus,
            shared_memory=shared_memory,
            num_scouts=1,  # Minimal for testing
            num_gatherers=1,
            num_mutators=1,
            iterations=1,
            llm_provider="openai",
            llm_model="gpt-5"
        )
        
        print("Swarm created successfully")
        print(f"Scouts: {swarm.num_scouts}, Gatherers: {swarm.num_gatherers}, Mutators: {swarm.num_mutators}")
        
        # Run a minimal task
        task = {
            "domain": "support",
            "subdomain": "tech_support",
            "count": 2,  # Generate only 2 examples for testing
            "format": "chat"
        }
        
        print(f"Running swarm with task: {task}")
        
        # Run async
        results = asyncio.run(swarm.run(task))
        
        print(f"Results: {len(results)} examples generated")
        print(f"Events received: {len(events_received)}")
        
        if results:
            print(f"First example preview: {str(results[0])[:200]}...")
        
        return len(results) > 0
        
    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_evolution_agents():
    """Test Evolution multi-agent generation"""
    print("\n" + "=" * 50)
    print("TEST 6: Evolution Multi-Agent Generation")
    print("=" * 50)
    
    try:
        from agents.base_agent import AgentEventBus, reset_llm_limiter
        from agents.shared_memory import SharedMemory
        from agents.evolution import EvolutionaryGeneFusion
        
        # Reset LLM limiter for this test
        reset_llm_limiter()
        
        # Create event bus and shared memory
        event_bus = AgentEventBus()
        shared_memory = SharedMemory()
        
        # Create evolution system
        evolution = EvolutionaryGeneFusion(
            event_bus=event_bus,
            shared_memory=shared_memory,
            population_size=2,  # Minimal for testing
            generations=1,
            llm_provider="openai",
            llm_model="gpt-5"
        )
        
        print("Evolution system created successfully")
        
        task = {
            "domain": "education",
            "subdomain": "tutoring",
            "count": 2,
            "format": "chat"
        }
        
        print(f"Running evolution with task: {task}")
        
        results = asyncio.run(evolution.run(task))
        
        print(f"Results: {len(results)} examples generated")
        
        return len(results) > 0
        
    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_cosmic_agents():
    """Test Cosmic multi-agent generation"""
    print("\n" + "=" * 50)
    print("TEST 7: Cosmic Multi-Agent Generation")
    print("=" * 50)
    
    try:
        from agents.base_agent import AgentEventBus, reset_llm_limiter
        from agents.shared_memory import SharedMemory
        from agents.cosmic import CosmicBurstSynthesis
        
        # Reset LLM limiter for this test
        reset_llm_limiter()
        
        # Create event bus and shared memory
        event_bus = AgentEventBus()
        shared_memory = SharedMemory()
        
        # Create cosmic system
        cosmic = CosmicBurstSynthesis(
            event_bus=event_bus,
            shared_memory=shared_memory,
            num_exploders=1,
            num_coolers=1,
            llm_provider="openai",
            llm_model="gpt-5"
        )
        
        print("Cosmic system created successfully")
        
        task = {
            "domain": "sales",
            "subdomain": "closing",
            "count": 2,
            "format": "chat"
        }
        
        print(f"Running cosmic with task: {task}")
        
        results = asyncio.run(cosmic.run(task))
        
        print(f"Results: {len(results)} examples generated")
        
        return len(results) > 0
        
    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_quantum_agents():
    """Test Quantum multi-agent generation"""
    print("\n" + "=" * 50)
    print("TEST 8: Quantum Multi-Agent Generation")
    print("=" * 50)
    
    try:
        from agents.base_agent import AgentEventBus, reset_llm_limiter
        from agents.shared_memory import SharedMemory
        from agents.quantum import QuantumFieldOrchestration
        
        # Reset LLM limiter for this test
        reset_llm_limiter()
        
        # Create event bus and shared memory
        event_bus = AgentEventBus()
        shared_memory = SharedMemory()
        
        # Create quantum system
        quantum = QuantumFieldOrchestration(
            event_bus=event_bus,
            shared_memory=shared_memory,
            llm_provider="openai",
            llm_model="gpt-5"
        )
        
        print("Quantum system created successfully")
        
        task = {
            "domain": "legal",
            "subdomain": "contracts",
            "count": 2,
            "format": "chat"
        }
        
        print(f"Running quantum with task: {task}")
        
        results = asyncio.run(quantum.run(task))
        
        print(f"Results: {len(results)} examples generated")
        
        return len(results) > 0
        
    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    print("Starting Generation Pipeline Tests")
    print("Working directory:", os.getcwd())
    print()
    
    results = {
        # Ollama tests
        "Ollama Connection": test_ollama_connection(),
        "Ollama Generation": test_ollama_generation(),
        # OpenAI tests
        "OpenAI Connection": test_openai_connection(),
        "Simple Generation": test_simple_generation(),
        "JSON Generation": test_json_generation(),
        "Dataset Generator": test_dataset_generator(),
        # Multi-agent tests
        "Swarm Agents": test_swarm_agents(),
        "Evolution Agents": test_evolution_agents(),
        "Cosmic Agents": test_cosmic_agents(),
        "Quantum Agents": test_quantum_agents(),
    }
    
    print("\n" + "=" * 50)
    print("RESULTS SUMMARY")
    print("=" * 50)
    
    for test_name, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{test_name}: {status}")
    
    all_passed = all(results.values())
    print()
    print(f"Overall: {'✅ ALL TESTS PASSED' if all_passed else '❌ SOME TESTS FAILED'}")
    
    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
