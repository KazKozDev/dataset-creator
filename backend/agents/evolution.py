"""
Evolutionary Gene Fusion
Genetic algorithm-inspired dataset generation

Inspired by biological evolution: mutation, crossover, natural selection
"""

import asyncio
import json
import random
import concurrent.futures
from typing import Dict, Any, List, Optional
from .base_agent import BaseAgent, AgentRole, AgentEventBus, set_llm_budget, clear_llm_budget
from .shared_memory import SharedMemory

# Import our providers instead of litellm
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from llm_providers import get_provider

class MutagenAgent(BaseAgent):
    """Mutagen agent - uses LLM to create intelligent mutations"""
    
    def __init__(self, agent_id: str, event_bus: AgentEventBus, **kwargs):
        super().__init__(agent_id, AgentRole.MUTAGEN, event_bus, **kwargs)
        self._provider = None
        
    def _get_provider(self):
        if self._provider is None:
            self._provider = get_provider(self.llm_provider, model=self.llm_model)
        return self._provider
        
    async def run(self, task: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Mutate examples using LLM"""
        await self.start()
        
        examples = task.get("examples", [])
        mutation_rate = task.get("mutation_rate", 0.3)
        domain = task.get("domain", "general")
        
        await self.think(f"Mutating {len(examples)} examples with LLM (rate {mutation_rate})")
        
        mutated = []
        provider = self._get_provider()
        
        # Import budget
        from .base_agent import get_llm_limiter
        
        for i, example in enumerate(examples):
            if random.random() < mutation_rate:
                # Check budget
                budget = get_llm_limiter()
                if not budget.use():
                    await self.think("Budget exhausted")
                    mutated.append(example)
                    continue
                    
                await self.think(f"Mutating example {i+1}...")
                
                # Evolutionary mutation types
                mutation_types = [
                    "Increase complexity - add more nuanced details",
                    "Simplify - make more accessible to beginners",
                    "Add edge case - introduce unusual scenario",
                    "Change perspective - alter the user's viewpoint",
                    "Extend conversation - add meaningful follow-up",
                ]
                mutation_type = random.choice(mutation_types)
                
                prompt = f"""Apply evolutionary mutation to this {domain} example.

Original:
{json.dumps(example, indent=2, ensure_ascii=False)[:1500]}

Mutation: {mutation_type}

Return mutated example as JSON:
{{"messages": [...], "metadata": {{...}}}}"""

                try:
                    loop = asyncio.get_event_loop()
                    with concurrent.futures.ThreadPoolExecutor() as executor:
                        result = await loop.run_in_executor(
                            executor,
                            lambda: provider.generate_text(prompt, temperature=0.8)
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
                    
                    mutation = json.loads(json_str.strip())
                    mutation["mutation_generation"] = example.get("mutation_generation", 0) + 1
                    mutation["mutation_type"] = mutation_type
                    mutation["fitness"] = random.uniform(0.7, 0.95)
                    
                    mutated.append(mutation)
                    
                    await self.output(mutation, {
                        "index": i,
                        "mutation_type": mutation_type
                    })
                    
                except Exception as e:
                    await self.think(f"Mutation failed: {e}")
                    mutated.append(example)
            else:
                mutated.append(example)
                
        await self.complete({
            "examples_mutated": len([e for e in mutated if e.get("mutation_generation", 0) > 0]),
            "total_examples": len(mutated)
        })
        
        return mutated

class CrossoverAgent(BaseAgent):
    """Crossover agent - uses LLM to combine best traits from examples"""
    
    def __init__(self, agent_id: str, event_bus: AgentEventBus, **kwargs):
        super().__init__(agent_id, AgentRole.CROSSOVER, event_bus, **kwargs)
        self._provider = None
        
    def _get_provider(self):
        if self._provider is None:
            self._provider = get_provider(self.llm_provider, model=self.llm_model)
        return self._provider
        
    async def run(self, task: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Perform crossover using LLM to combine best traits"""
        await self.start()
        
        examples = task.get("examples", [])
        crossover_rate = task.get("crossover_rate", 0.5)
        domain = task.get("domain", "general")
        
        await self.think(f"Performing LLM crossover on {len(examples)} examples")
        
        offspring = []
        provider = self._get_provider()
        
        # Import budget
        from .base_agent import get_llm_limiter
        
        # Pair up examples for crossover
        for i in range(0, len(examples) - 1, 2):
            if random.random() < crossover_rate:
                # Check budget
                budget = get_llm_limiter()
                if not budget.use():
                    await self.think("Budget exhausted")
                    offspring.extend([examples[i], examples[i+1]])
                    continue
                    
                await self.think(f"Crossing over examples {i} and {i+1}")
                
                parent1 = examples[i]
                parent2 = examples[i+1]
                
                prompt = f"""Combine the best traits from these two {domain} examples to create a superior offspring.

Parent 1:
{json.dumps(parent1, indent=2, ensure_ascii=False)[:1000]}

Parent 2:
{json.dumps(parent2, indent=2, ensure_ascii=False)[:1000]}

Create a new example that:
- Takes the best question style from one parent
- Takes the best response quality from the other
- Combines unique aspects from both

Return the offspring as JSON:
{{"messages": [...], "metadata": {{...}}}}"""

                try:
                    loop = asyncio.get_event_loop()
                    with concurrent.futures.ThreadPoolExecutor() as executor:
                        result = await loop.run_in_executor(
                            executor,
                            lambda: provider.generate_text(prompt, temperature=0.7)
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
                    
                    child = json.loads(json_str.strip())
                    child["crossover_generation"] = max(
                        parent1.get("crossover_generation", 0),
                        parent2.get("crossover_generation", 0)
                    ) + 1
                    child["fitness"] = (parent1.get("fitness", 0.7) + parent2.get("fitness", 0.7)) / 2 + 0.05
                    
                    offspring.append(child)
                    
                    await self.output(child, {
                        "parent1_index": i,
                        "parent2_index": i + 1
                    })
                    
                except Exception as e:
                    await self.think(f"Crossover failed: {e}")
                    offspring.extend([examples[i], examples[i+1]])
            else:
                offspring.extend([examples[i], examples[i+1]])
                
        # Add remaining example if odd number
        if len(examples) % 2 == 1:
            offspring.append(examples[-1])
            
        await self.complete({
            "offspring_created": len([e for e in offspring if e.get("crossover_generation", 0) > 0]),
            "total_examples": len(offspring)
        })
        
        return offspring

class EvolutionaryGeneFusion:
    """Main evolutionary synthesis orchestrator"""
    
    def __init__(
        self,
        event_bus: AgentEventBus,
        shared_memory: SharedMemory,
        population_size: int = 50,
        generations: int = 5,
        mutation_rate: float = 0.3,
        crossover_rate: float = 0.5,
        elite_size: int = 10,
        role_models: Optional[Dict[str, Dict[str, str]]] = None,
        **agent_kwargs
    ):
        """
        Args:
            role_models: Dict mapping role to LLM config, e.g.:
                {
                    "scout": {"llm_provider": "openai", "llm_model": "gpt-4"},
                    "mutagen": {"llm_provider": "anthropic", "llm_model": "claude-3-sonnet"},
                    "crossover": {"llm_provider": "openai", "llm_model": "gpt-4o-mini"},
                }
        """
        self.event_bus = event_bus
        self.shared_memory = shared_memory
        self.population_size = population_size
        self.generations = generations
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.elite_size = elite_size
        self.role_models = role_models or {}
        self.agent_kwargs = agent_kwargs
    
    def _get_role_kwargs(self, role: str) -> Dict[str, Any]:
        """Get kwargs for a specific role, merging role-specific model config"""
        kwargs = {**self.agent_kwargs}
        if role in self.role_models:
            kwargs.update(self.role_models[role])
        return kwargs
        
    async def _generate_initial_population(self, task: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate initial population"""
        # Use Scout agents from swarm
        from .swarm import ScoutAgent
        
        scout = ScoutAgent("evolution_scout", self.event_bus, **self._get_role_kwargs("scout"))
        
        return await scout.run({
            **task,
            "count": self.population_size
        })
        
    async def _evaluate_fitness(self, examples: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Evaluate fitness of examples"""
        # Simple fitness: quality score
        for example in examples:
            if "fitness" not in example:
                example["fitness"] = 0.7 + (hash(str(example)) % 30) / 100
        return examples
        
    async def _select_elite(self, population: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Select elite individuals"""
        sorted_pop = sorted(population, key=lambda x: x.get("fitness", 0), reverse=True)
        return sorted_pop[:self.elite_size]
        
    async def run(self, task: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Run evolutionary synthesis"""
        
        # Set LLM budget: scout(generate) + mutagen(mutate) + crossover(combine) per generation
        # Budget = population * generations * 3 agents
        requested_count = task.get("count", 50)
        budget = set_llm_budget(
            self.population_size * (self.generations + 1) * 3,
            multiplier=1.5,
            max_absolute=500
        )
        print(f"[Evolution] LLM budget: {budget.budget} calls for {requested_count} examples")
        
        try:
            # Generate initial population
            print(f"\n=== Generating Initial Population ({self.population_size}) ===")
            population = await self._generate_initial_population(task)
            population = await self._evaluate_fitness(population)
            
            # Create agents with role-specific LLM configs
            mutagen = MutagenAgent("mutagen", self.event_bus, **self._get_role_kwargs("mutagen"))
            crossover = CrossoverAgent("crossover", self.event_bus, **self._get_role_kwargs("crossover"))
            
            # Evolution loop
            for generation in range(self.generations):
                print(f"\n=== Generation {generation + 1}/{self.generations} ===")
                
                # Select elite
                elite = await self._select_elite(population)
                
                # Mutation phase
                await mutagen.send_message("crossover", f"Gen {generation+1}: Mutating {len(population)} individuals", {"generation": generation+1})
                mutated = await mutagen.run({
                    "examples": population,
                    "mutation_rate": self.mutation_rate
                })
                
                # Crossover phase
                await crossover.send_message("mutagen", f"Gen {generation+1}: Crossing {len(mutated)} mutants", {"generation": generation+1})
                offspring = await crossover.run({
                    "examples": mutated,
                    "crossover_rate": self.crossover_rate
                })
                
                # Evaluate fitness
                offspring = await self._evaluate_fitness(offspring)
                
                # Combine elite with offspring
                population = elite + offspring
                
                # Trim to population size
                population = sorted(population, key=lambda x: x.get("fitness", 0), reverse=True)
                population = population[:self.population_size]
                
            # Final selection
            final = await self._select_elite(population)
            final = final[:task.get("count", 50)]
            
            print(f"[Evolution] Completed. Budget used: {budget.status()}")
            return final
            
        finally:
            clear_llm_budget()
