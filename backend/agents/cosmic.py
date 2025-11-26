"""
Cosmic Burst Synthesis
Big Bang-inspired rapid expansion from seed examples

Inspired by cosmic expansion and supernova nucleosynthesis
"""

import asyncio
import json
import random
import concurrent.futures
from typing import Dict, Any, List, Optional
from .base_agent import BaseAgent, AgentRole, AgentEventBus, set_llm_budget, clear_llm_budget, get_llm_limiter
from .shared_memory import SharedMemory

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from llm_providers import get_provider

class ExploderAgent(BaseAgent):
    """Exploder agent - uses LLM to create rapid variations from seed"""
    
    def __init__(self, agent_id: str, event_bus: AgentEventBus, **kwargs):
        super().__init__(agent_id, AgentRole.EXPLODER, event_bus, **kwargs)
        self._provider = None
        
    def _get_provider(self):
        if self._provider is None:
            self._provider = get_provider(self.llm_provider, model=self.llm_model)
        return self._provider
        
    async def run(self, task: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Explode seed into variations using LLM"""
        await self.start()
        
        seed = task.get("seed", {})
        expansion_factor = task.get("expansion_factor", 10)
        domain = task.get("domain", "general")
        
        await self.think(f"Exploding seed into {expansion_factor} LLM variations...")
        
        variations = []
        provider = self._get_provider()
        
        # Variation directions for explosion
        directions = [
            "more formal tone",
            "more casual tone", 
            "add technical details",
            "simplify for beginners",
            "add emotional context",
            "make more concise",
            "expand with examples",
            "change scenario context",
            "add urgency",
            "make more empathetic",
        ]
        
        for i in range(expansion_factor):
            # Check budget
            budget = get_llm_limiter()
            if not budget.use():
                await self.think("Budget exhausted")
                break
                
            await self.think(f"Creating variation {i+1}/{expansion_factor}")
            
            direction = directions[i % len(directions)]
            
            prompt = f"""Create a variation of this {domain} example.

Original:
{json.dumps(seed, indent=2, ensure_ascii=False)[:1500]}

Variation direction: {direction}

Return the variation as JSON:
{{"messages": [...], "metadata": {{...}}}}"""

            try:
                loop = asyncio.get_event_loop()
                with concurrent.futures.ThreadPoolExecutor() as executor:
                    result = await loop.run_in_executor(
                        executor,
                        lambda: provider.generate_text(prompt, temperature=0.9)  # High temp for diversity
                    )
                
                # Parse JSON
                json_str = result
                if "```json" in result:
                    json_str = result.split("```json")[1].split("```")[0]
                elif "```" in result:
                    json_str = result.split("```")[1].split("```")[0]
                
                start = json_str.find("{")
                end = json_str.rfind("}") + 1
                if start != -1 and end > start:
                    json_str = json_str[start:end]
                
                variation = json.loads(json_str.strip())
                variation["variation_id"] = i
                variation["variation_direction"] = direction
                variation["temperature"] = 1.0 - (i / expansion_factor)
                variation["energy"] = random.uniform(0.5, 1.0)
                
                variations.append(variation)
                
                await self.output(variation, {
                    "variation_index": i,
                    "direction": direction,
                    "temperature": variation["temperature"]
                })
                
            except Exception as e:
                await self.think(f"Variation {i+1} failed: {e}")
            
        await self.complete({
            "variations_created": len(variations),
            "expansion_factor": expansion_factor
        })
        
        return variations

class CoolerAgent(BaseAgent):
    """Cooler agent - uses LLM to refine and stabilize variations"""
    
    def __init__(self, agent_id: str, event_bus: AgentEventBus, **kwargs):
        super().__init__(agent_id, AgentRole.COOLER, event_bus, **kwargs)
        self._provider = None
        
    def _get_provider(self):
        if self._provider is None:
            self._provider = get_provider(self.llm_provider, model=self.llm_model)
        return self._provider
        
    async def run(self, task: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Cool down and stabilize examples using LLM refinement"""
        await self.start()
        
        examples = task.get("examples", [])
        temperature_threshold = task.get("temperature_threshold", 0.5)
        domain = task.get("domain", "general")
        
        await self.think(f"Cooling {len(examples)} examples with LLM refinement")
        
        stable = []
        provider = self._get_provider()
        
        for i, example in enumerate(examples):
            temp = example.get("temperature", 1.0)
            
            if temp <= temperature_threshold:
                # Already cool enough, just validate
                await self.think(f"Example {i+1} is stable (temp: {temp:.2f})")
                stable_example = {**example, "stable": True, "quality_score": 0.8}
                stable.append(stable_example)
            else:
                # Use LLM to refine/cool down
                budget = get_llm_limiter()
                if not budget.use():
                    await self.think("Budget exhausted, skipping refinement")
                    continue
                    
                await self.think(f"Refining hot example {i+1} (temp: {temp:.2f})")
                
                prompt = f"""Refine this {domain} example to improve quality and consistency.

Original (may be rough/inconsistent):
{json.dumps(example, indent=2, ensure_ascii=False)[:1500]}

Improvements needed:
- Fix any grammatical issues
- Ensure logical consistency
- Make responses more helpful
- Remove any artifacts or errors

Return refined example as JSON:
{{"messages": [...], "metadata": {{...}}}}"""

                try:
                    loop = asyncio.get_event_loop()
                    with concurrent.futures.ThreadPoolExecutor() as executor:
                        result = await loop.run_in_executor(
                            executor,
                            lambda: provider.generate_text(prompt, temperature=0.3)  # Low temp for stability
                        )
                    
                    # Parse JSON
                    json_str = result
                    if "```json" in result:
                        json_str = result.split("```json")[1].split("```")[0]
                    elif "```" in result:
                        json_str = result.split("```")[1].split("```")[0]
                    
                    start = json_str.find("{")
                    end = json_str.rfind("}") + 1
                    if start != -1 and end > start:
                        json_str = json_str[start:end]
                    
                    stable_example = json.loads(json_str.strip())
                    stable_example["stable"] = True
                    stable_example["refined"] = True
                    stable_example["quality_score"] = 0.85
                    
                    stable.append(stable_example)
                    
                    await self.output(stable_example, {
                        "index": i,
                        "original_temp": temp,
                        "refined": True
                    })
                    
                except Exception as e:
                    await self.think(f"Refinement failed for {i+1}: {e}")
                
        await self.complete({
            "stable_examples": len(stable),
            "discarded": len(examples) - len(stable)
        })
        
        return stable

class SynthesizerAgent(BaseAgent):
    """Synthesizer agent - uses LLM to combine and enhance final examples"""
    
    def __init__(self, agent_id: str, event_bus: AgentEventBus, **kwargs):
        super().__init__(agent_id, AgentRole.SYNTHESIZER, event_bus, **kwargs)
        self._provider = None
        
    def _get_provider(self):
        if self._provider is None:
            self._provider = get_provider(self.llm_provider, model=self.llm_model)
        return self._provider
        
    async def run(self, task: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Synthesize final examples with LLM enhancement"""
        await self.start()
        
        examples = task.get("examples", [])
        target_count = task.get("target_count", 50)
        domain = task.get("domain", "general")
        
        await self.think(f"Synthesizing {target_count} examples from {len(examples)} stable elements")
        
        # Sort by quality
        sorted_examples = sorted(
            examples,
            key=lambda x: x.get("quality_score", 0),
            reverse=True
        )
        
        # Take top candidates
        candidates = sorted_examples[:min(target_count * 2, len(sorted_examples))]
        
        # Use LLM to do final polish on top examples
        synthesized = []
        provider = self._get_provider()
        
        for i, example in enumerate(candidates[:target_count]):
            budget = get_llm_limiter()
            if not budget.use():
                # No budget, just add as-is
                synthesized.append(example)
                continue
                
            await self.think(f"Final polish on example {i+1}")
            
            prompt = f"""Do a final quality polish on this {domain} example.

Example:
{json.dumps(example, indent=2, ensure_ascii=False)[:1500]}

Make minimal improvements:
- Ensure natural flow
- Fix any remaining issues
- Enhance clarity if needed

Return polished example as JSON:
{{"messages": [...], "metadata": {{...}}}}"""

            try:
                loop = asyncio.get_event_loop()
                with concurrent.futures.ThreadPoolExecutor() as executor:
                    result = await loop.run_in_executor(
                        executor,
                        lambda: provider.generate_text(prompt, temperature=0.2)
                    )
                
                # Parse JSON
                json_str = result
                if "```json" in result:
                    json_str = result.split("```json")[1].split("```")[0]
                elif "```" in result:
                    json_str = result.split("```")[1].split("```")[0]
                
                start = json_str.find("{")
                end = json_str.rfind("}") + 1
                if start != -1 and end > start:
                    json_str = json_str[start:end]
                
                polished = json.loads(json_str.strip())
                polished["quality_score"] = example.get("quality_score", 0.8) + 0.05
                polished["synthesized"] = True
                
                synthesized.append(polished)
                
            except Exception as e:
                await self.think(f"Polish failed: {e}")
                synthesized.append(example)
        
        await self.think(f"Synthesized {len(synthesized)} final examples")
        
        await self.complete({
            "synthesized_count": len(synthesized),
            "avg_quality": sum(e.get("quality_score", 0) for e in synthesized) / len(synthesized) if synthesized else 0
        })
        
        return synthesized

class CosmicBurstSynthesis:
    """Main cosmic synthesis orchestrator"""
    
    def __init__(
        self,
        event_bus: AgentEventBus,
        shared_memory: SharedMemory,
        expansion_factor: int = 10,
        iterations: int = 3,
        temperature_threshold: float = 0.5,
        role_models: Optional[Dict[str, Dict[str, str]]] = None,
        **agent_kwargs
    ):
        """
        Args:
            role_models: Dict mapping role to LLM config, e.g.:
                {
                    "scout": {"llm_provider": "openai", "llm_model": "gpt-4"},
                    "exploder": {"llm_provider": "anthropic", "llm_model": "claude-3-sonnet"},
                    "cooler": {"llm_provider": "openai", "llm_model": "gpt-4o-mini"},
                }
        """
        self.event_bus = event_bus
        self.shared_memory = shared_memory
        self.expansion_factor = expansion_factor
        self.iterations = iterations
        self.temperature_threshold = temperature_threshold
        self.role_models = role_models or {}
        # Filter out orchestrator-specific kwargs that shouldn't be passed to agents
        self.agent_kwargs = {
            k: v for k, v in agent_kwargs.items() 
            if k not in ('num_exploders', 'num_coolers', 'num_synthesizers', 'role_models')
        }
    
    def _get_role_kwargs(self, role: str) -> Dict[str, Any]:
        """Get kwargs for a specific role, merging role-specific model config"""
        kwargs = {**self.agent_kwargs}
        if role in self.role_models:
            kwargs.update(self.role_models[role])
        return kwargs
        
    async def _generate_seed(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Generate initial seed"""
        from .swarm import ScoutAgent
        
        scout = ScoutAgent("cosmic_scout", self.event_bus, **self._get_role_kwargs("scout"))
        seeds = await scout.run({**task, "count": 1})
        
        return seeds[0] if seeds else {}
        
    async def run(self, task: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Run cosmic burst synthesis"""
        
        # Set LLM budget: scout(seed) + exploder(variations) + cooler(refine) + synthesizer(polish)
        # Budget = iterations * (1 seed + expansion_factor variations + refinements + polish)
        requested_count = task.get("count", 50)
        budget = set_llm_budget(
            self.iterations * (1 + self.expansion_factor * 3),  # seed + explode + cool + synthesize
            multiplier=2.0,
            max_absolute=500
        )
        print(f"[Cosmic] LLM budget: {budget.budget} calls for {requested_count} examples")
        
        try:
            # Create agents with role-specific LLM configs
            exploder = ExploderAgent("exploder", self.event_bus, **self._get_role_kwargs("exploder"))
            cooler = CoolerAgent("cooler", self.event_bus, **self._get_role_kwargs("cooler"))
            synthesizer = SynthesizerAgent("synthesizer", self.event_bus, **self._get_role_kwargs("synthesizer"))
            
            all_stable = []
            
            for iteration in range(self.iterations):
                print(f"\n=== Burst {iteration + 1}/{self.iterations} ===")
                
                # Generate seed
                seed = await self._generate_seed(task)
                
                # Explode
                variations = await exploder.run({
                    "seed": seed,
                    "expansion_factor": self.expansion_factor
                })
                
                # Cool down
                stable = await cooler.run({
                    "examples": variations,
                    "temperature_threshold": self.temperature_threshold
                })
                
                all_stable.extend(stable)
                
            # Synthesize final
            final = await synthesizer.run({
                "examples": all_stable,
                "target_count": task.get("count", 50)
            })
            
            print(f"[Cosmic] Completed. Budget used: {budget.status()}")
            return final
            
        finally:
            clear_llm_budget()
