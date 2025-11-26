"""
Hybrid Swarm Synthesis (HSS)
Multi-agent swarm intelligence for dataset generation

Inspired by bee/ant colonies - agents compete and collaborate
"""

import asyncio
import json
import concurrent.futures
from typing import Dict, Any, List, Optional
from .base_agent import BaseAgent, AgentRole, AgentEventBus, get_llm_limiter, set_llm_budget, clear_llm_budget
from .shared_memory import SharedMemory

# Import our providers and generator
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from llm_providers import get_provider
from generator import DatasetGenerator
import database as db


class ScoutAgent(BaseAgent):
    """Scout agent - explores and generates initial examples"""
    
    def __init__(self, agent_id: str, event_bus: AgentEventBus, **kwargs):
        super().__init__(agent_id, AgentRole.SCOUT, event_bus, **kwargs)
        # Get LLM provider
        self._provider = None
        
    def _get_provider(self):
        """Lazy load provider"""
        if self._provider is None:
            self._provider = get_provider(self.llm_provider, model=self.llm_model)
        return self._provider
        
    async def generate(self, prompt: str, **kwargs) -> str:
        """Generate using LLM provider with budget tracking"""
        # Check LLM budget (if set)
        budget = get_llm_limiter()  # Returns budget or creates default
        if not budget.use():
            await self.error(f"LLM budget exhausted: {budget.status()}")
            raise RuntimeError(f"LLM budget exhausted: {budget.status()}")
        
        try:
            provider = self._get_provider()
            
            # Run sync generation in executor
            loop = asyncio.get_event_loop()
            with concurrent.futures.ThreadPoolExecutor() as executor:
                result = await loop.run_in_executor(
                    executor,
                    lambda: provider.generate_text(prompt, temperature=self.temperature)
                )
            
            # Stream the result
            await self.stream(result)
            return result
            
        except Exception as e:
            await self.error(f"Generation failed: {str(e)}")
            raise
            
    async def run(self, task: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate initial examples using DatasetGenerator for proper prompts"""
        await self.start()
        
        domain = task.get("domain")
        subdomain = task.get("subdomain")
        count = task.get("count", 10)
        format_type = task.get("format", "chat")
        language = task.get("language", "en")
        template_name = task.get("template")
        
        print(f"[Scout {self.agent_id}] Starting with domain={domain}, subdomain={subdomain}, count={count}")
        
        await self.think(f"Exploring {domain}/{subdomain} domain...")
        await self.think(f"Need to generate {count} diverse examples")
        
        # Load custom template if specified
        custom_template = None
        if template_name:
            try:
                prompts = db.get_prompts()
                for p in prompts:
                    if p['name'] == template_name:
                        custom_template = p['content']
                        await self.think(f"Using template: {template_name}")
                        break
            except Exception as e:
                await self.think(f"Could not load template: {e}")
        
        # Create generator for scenario parameters and prompts
        provider = self._get_provider()
        generator = DatasetGenerator(provider)
        
        examples = []
        
        for i in range(count):
            await self.think(f"Generating example {i+1}/{count}")
            
            # Generate scenario with proper domain-specific parameters
            scenario = generator.generate_scenario_parameters(domain, subdomain)
            
            # Create prompt using DatasetGenerator (includes template support)
            prompt = generator.get_prompt_in_language(scenario, format_type, language, custom_template)
            
            try:
                # Log the prompt being sent to LLM
                await self.log_llm_request(prompt, {"temperature": self.temperature})
                
                result = await self.generate(prompt)
                
                # Log the response
                await self.log_llm_response(result)
                
                # Parse JSON
                try:
                    # Extract JSON from response
                    json_str = result
                    if "```json" in result:
                        json_str = result.split("```json")[1].split("```")[0]
                    elif "```" in result:
                        json_str = result.split("```")[1].split("```")[0]
                    
                    start_idx = json_str.find('{')
                    end_idx = json_str.rfind('}') + 1
                    if start_idx != -1 and end_idx > start_idx:
                        json_str = json_str[start_idx:end_idx]
                    
                    example = json.loads(json_str.strip())
                    
                    # Add scenario metadata
                    if "metadata" not in example:
                        example["metadata"] = {}
                    example["metadata"].update({
                        "domain": domain,
                        "subdomain": subdomain,
                        "complexity_level": scenario.get("complexity_level"),
                        "communication_style": scenario.get("communication_style"),
                    })
                    
                    examples.append(example)
                    
                    await self.output(example, {
                        "index": i,
                        "quality_estimate": 0.8,
                        "scenario": scenario
                    })
                except json.JSONDecodeError as e:
                    await self.think(f"Failed to parse JSON: {e}")
                    
            except Exception as e:
                await self.error(f"Failed to generate example {i+1}: {str(e)}")
                
        await self.complete({
            "examples_generated": len(examples),
            "success_rate": len(examples) / count if count > 0 else 0
        })
        
        return examples

class GathererAgent(BaseAgent):
    """Gatherer agent - uses LLM to evaluate and critique examples"""
    
    def __init__(self, agent_id: str, event_bus: AgentEventBus, **kwargs):
        super().__init__(agent_id, AgentRole.GATHERER, event_bus, **kwargs)
        self._provider = None
        
    def _get_provider(self):
        if self._provider is None:
            self._provider = get_provider(self.llm_provider, model=self.llm_model)
        return self._provider
        
    async def run(self, task: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Evaluate examples using LLM as critic"""
        await self.start()
        
        examples = task.get("examples", [])
        threshold = task.get("threshold", 0.7)
        domain = task.get("domain", "general")
        
        await self.think(f"Evaluating {len(examples)} examples using LLM critic...")
        
        selected = []
        provider = self._get_provider()
        
        for i, example in enumerate(examples):
            await self.think(f"Analyzing example {i+1}/{len(examples)}...")
            
            # Check budget
            budget = get_llm_limiter()
            if not budget.use():
                await self.think(f"Budget exhausted, keeping remaining examples as-is")
                selected.extend(examples[i:])
                break
            
            # Use LLM to evaluate quality
            prompt = f"""Evaluate this {domain} conversation example for quality.

Example:
{json.dumps(example, indent=2, ensure_ascii=False)[:1500]}

Rate on these criteria (1-10 each):
1. Relevance to domain
2. Natural dialogue flow
3. Accuracy of information
4. Helpfulness of response
5. Grammar and clarity

Return JSON only:
{{"overall_score": 0.0-1.0, "issues": ["list of problems"], "strengths": ["list of strengths"]}}"""

            try:
                # Log the evaluation prompt
                await self.log_llm_request(prompt, {"temperature": 0.3})
                
                loop = asyncio.get_event_loop()
                with concurrent.futures.ThreadPoolExecutor() as executor:
                    result = await loop.run_in_executor(
                        executor,
                        lambda: provider.generate_text(prompt, temperature=0.3)
                    )
                
                # Log the response
                await self.log_llm_response(result)
                
                # Parse score
                json_str = result
                if "{" in result:
                    start = result.find("{")
                    end = result.rfind("}") + 1
                    json_str = result[start:end]
                
                evaluation = json.loads(json_str)
                quality_score = float(evaluation.get("overall_score", 0.7))
                
                await self.metadata({
                    "example_index": i,
                    "quality_score": quality_score,
                    "evaluation": evaluation
                })
                
                if quality_score >= threshold:
                    await self.think(f"✓ Example {i+1} passed (score: {quality_score:.2f})")
                    example_with_score = {**example, "quality_score": quality_score}
                    if "issues" in evaluation:
                        example_with_score["critic_issues"] = evaluation["issues"]
                    selected.append(example_with_score)
                else:
                    await self.think(f"✗ Example {i+1} rejected (score: {quality_score:.2f})")
                    
            except Exception as e:
                await self.think(f"Evaluation failed for example {i+1}: {e}, keeping it")
                selected.append({**example, "quality_score": 0.7})
                
        await self.complete({
            "examples_evaluated": len(examples),
            "examples_selected": len(selected),
            "selection_rate": len(selected) / len(examples) if examples else 0
        })
        
        return selected

class MutatorAgent(BaseAgent):
    """Mutator agent - uses LLM to create variations of examples"""
    
    def __init__(self, agent_id: str, event_bus: AgentEventBus, **kwargs):
        super().__init__(agent_id, AgentRole.MUTATOR, event_bus, **kwargs)
        self._provider = None
        
    def _get_provider(self):
        if self._provider is None:
            self._provider = get_provider(self.llm_provider, model=self.llm_model)
        return self._provider
        
    async def run(self, task: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Create mutations of examples using LLM"""
        await self.start()
        
        examples = task.get("examples", [])
        mutations_per_example = task.get("mutations_per_example", 2)
        domain = task.get("domain", "general")
        
        await self.think(f"Creating {mutations_per_example} LLM mutations for {len(examples)} examples...")
        
        mutated = []
        provider = self._get_provider()
        
        for i, example in enumerate(examples):
            await self.think(f"Mutating example {i+1}/{len(examples)}...")
            
            for m in range(mutations_per_example):
                # Check budget
                budget = get_llm_limiter()
                if not budget.use():
                    await self.think(f"Budget exhausted")
                    break
                
                await self.think(f"Creating mutation {m+1}/{mutations_per_example}")
                
                # Mutation types
                mutation_types = [
                    "Rephrase the user's question while keeping the same intent",
                    "Make the conversation more formal/professional",
                    "Make the conversation more casual/friendly", 
                    "Add more detail to the assistant's response",
                    "Simplify the language for a beginner audience",
                    "Add a follow-up question from the user",
                    "Change the specific scenario while keeping the same topic",
                ]
                mutation_type = mutation_types[(i + m) % len(mutation_types)]
                
                prompt = f"""Create a variation of this {domain} conversation example.

Original:
{json.dumps(example, indent=2, ensure_ascii=False)[:1500]}

Mutation instruction: {mutation_type}

Return the mutated example as JSON with the same structure:
{{"messages": [...], "metadata": {{...}}}}"""

                try:
                    # Log mutation prompt
                    await self.log_llm_request(prompt, {"temperature": 0.8, "mutation_type": mutation_type})
                    
                    loop = asyncio.get_event_loop()
                    with concurrent.futures.ThreadPoolExecutor() as executor:
                        result = await loop.run_in_executor(
                            executor,
                            lambda: provider.generate_text(prompt, temperature=0.8)
                        )
                    
                    # Log response
                    await self.log_llm_response(result)
                    
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
                    mutation["mutation_id"] = f"{i}_{m}"
                    mutation["mutation_type"] = mutation_type
                    mutation["parent_quality"] = example.get("quality_score", 0.8)
                    
                    mutated.append(mutation)
                    
                    await self.output(mutation, {
                        "parent_index": i,
                        "mutation_index": m,
                        "mutation_type": mutation_type
                    })
                    
                except Exception as e:
                    await self.think(f"Mutation failed: {e}")
                
        await self.complete({
            "examples_mutated": len(examples),
            "mutations_created": len(mutated)
        })
        
        return mutated

class SelectorAgent(BaseAgent):
    """Selector agent - uses LLM for final diversity-aware selection"""
    
    def __init__(self, agent_id: str, event_bus: AgentEventBus, **kwargs):
        super().__init__(agent_id, AgentRole.SELECTOR, event_bus, **kwargs)
        self._provider = None
        
    def _get_provider(self):
        if self._provider is None:
            self._provider = get_provider(self.llm_provider, model=self.llm_model)
        return self._provider
        
    async def run(self, task: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Select final examples using LLM for diversity check"""
        await self.start()
        
        examples = task.get("examples", [])
        target_count = task.get("target_count", 50)
        domain = task.get("domain", "general")
        
        await self.think(f"Selecting top {target_count} from {len(examples)} examples...")
        
        # First, sort by quality
        sorted_examples = sorted(
            examples,
            key=lambda x: x.get("quality_score", 0),
            reverse=True
        )
        
        # Take top candidates (more than needed for diversity filtering)
        candidates = sorted_examples[:min(target_count * 2, len(sorted_examples))]
        
        if len(candidates) <= target_count:
            selected = candidates
        else:
            # Use LLM to check for duplicates/similar examples
            budget = get_llm_limiter()
            if budget.use():
                await self.think("Using LLM to ensure diversity...")
                
                provider = self._get_provider()
                
                # Create summary of candidates for diversity check
                summaries = []
                for i, ex in enumerate(candidates[:20]):  # Limit to first 20 for prompt size
                    msgs = ex.get("messages", [])
                    if msgs:
                        summaries.append(f"{i}: {msgs[0].get('content', '')[:100]}")
                
                prompt = f"""Review these {domain} conversation examples and identify which ones are too similar or redundant.

Examples:
{chr(10).join(summaries)}

Return JSON with indices to KEEP (most diverse, high quality):
{{"keep_indices": [0, 2, 5, ...]}}

Select up to {target_count} diverse examples."""

                try:
                    # Log diversity check prompt
                    await self.log_llm_request(prompt, {"temperature": 0.3})
                    
                    loop = asyncio.get_event_loop()
                    with concurrent.futures.ThreadPoolExecutor() as executor:
                        result = await loop.run_in_executor(
                            executor,
                            lambda: provider.generate_text(prompt, temperature=0.3)
                        )
                    
                    # Log response
                    await self.log_llm_response(result)
                    
                    # Parse indices
                    if "{" in result:
                        start = result.find("{")
                        end = result.rfind("}") + 1
                        diversity_result = json.loads(result[start:end])
                        keep_indices = diversity_result.get("keep_indices", [])
                        
                        # Select by indices
                        selected = []
                        for idx in keep_indices:
                            if idx < len(candidates):
                                selected.append(candidates[idx])
                        
                        # Fill remaining from sorted list
                        for ex in candidates:
                            if len(selected) >= target_count:
                                break
                            if ex not in selected:
                                selected.append(ex)
                                
                        await self.think(f"Diversity filter kept {len(keep_indices)} unique examples")
                    else:
                        selected = candidates[:target_count]
                        
                except Exception as e:
                    await self.think(f"Diversity check failed: {e}, using quality sort")
                    selected = candidates[:target_count]
            else:
                selected = candidates[:target_count]
        
        avg_quality = sum(e.get('quality_score', 0) for e in selected) / len(selected) if selected else 0
        await self.think(f"Selected {len(selected)} examples, avg quality: {avg_quality:.2f}")
        
        await self.complete({
            "examples_selected": len(selected),
            "avg_quality": avg_quality
        })
        
        return selected

class HybridSwarmSynthesis:
    """Main swarm synthesis orchestrator"""
    
    def __init__(
        self,
        event_bus: AgentEventBus,
        shared_memory: SharedMemory,
        num_scouts: int = 3,
        num_gatherers: int = 2,
        num_mutators: int = 2,
        iterations: int = 3,
        role_models: Optional[Dict[str, Dict[str, str]]] = None,
        **agent_kwargs
    ):
        """
        Args:
            role_models: Dict mapping role to LLM config, e.g.:
                {
                    "scout": {"llm_provider": "openai", "llm_model": "gpt-4"},
                    "gatherer": {"llm_provider": "anthropic", "llm_model": "claude-3-sonnet"},
                    "mutator": {"llm_provider": "openai", "llm_model": "gpt-4o-mini"},
                }
        """
        self.event_bus = event_bus
        self.shared_memory = shared_memory
        self.num_scouts = num_scouts
        self.num_gatherers = num_gatherers
        self.num_mutators = num_mutators
        self.iterations = iterations
        self.role_models = role_models or {}
        self.agent_kwargs = agent_kwargs
    
    def _get_role_kwargs(self, role: str) -> Dict[str, Any]:
        """Get kwargs for a specific role, merging role-specific model config"""
        kwargs = {**self.agent_kwargs}
        if role in self.role_models:
            kwargs.update(self.role_models[role])
        return kwargs
        
    async def run(self, task: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Run swarm synthesis"""
        
        # Set LLM budget based on requested count
        # Now all agents use LLM: scout(generate) + gatherer(evaluate) + mutator(mutate) + selector(diversity)
        # Budget = count * iterations * 4 agents * 2 (for retries)
        requested_count = task.get("count", 50)
        budget = set_llm_budget(
            requested_count * self.iterations * 4,
            multiplier=2.0,
            max_absolute=500
        )
        print(f"[Swarm] LLM budget: {budget.budget} calls for {requested_count} examples")
        
        try:
            # Create agents with role-specific LLM configs
            scouts = [
                ScoutAgent(f"scout_{i}", self.event_bus, **self._get_role_kwargs("scout"))
                for i in range(self.num_scouts)
            ]
            
            gatherers = [
                GathererAgent(f"gatherer_{i}", self.event_bus, **self._get_role_kwargs("gatherer"))
                for i in range(self.num_gatherers)
            ]
            
            mutators = [
                MutatorAgent(f"mutator_{i}", self.event_bus, **self._get_role_kwargs("mutator"))
                for i in range(self.num_mutators)
            ]
            
            selector = SelectorAgent("selector", self.event_bus, **self._get_role_kwargs("selector"))
            
            all_examples = []
            
            for iteration in range(self.iterations):
                print(f"\n=== Iteration {iteration + 1}/{self.iterations} ===")
                
                # Phase 1: Scouts explore
                total_count = task.get("count", 50)
                # Ensure each scout generates at least 1 example
                count_per_scout = max(1, total_count // self.num_scouts)
                print(f"[Swarm] Each scout will generate {count_per_scout} examples")
                
                # Announce phase to UI
                for scout in scouts:
                    await scout.send_message("gatherer", f"Starting exploration, will generate {count_per_scout} examples", {"phase": "scout", "count": count_per_scout})
                
                scout_tasks = [
                    scout.run({
                        **task,
                        "count": count_per_scout
                    })
                    for scout in scouts
                ]
                scout_results = await asyncio.gather(*scout_tasks, return_exceptions=True)
                
                # Handle exceptions and collect results
                scout_examples = []
                for i, result in enumerate(scout_results):
                    if isinstance(result, Exception):
                        print(f"[Swarm] Scout {i} failed: {result}")
                    else:
                        scout_examples.extend(result)
                        print(f"[Swarm] Scout {i} generated {len(result)} examples")
                
                print(f"[Swarm] Total scout examples: {len(scout_examples)}")
                
                # Phase 2: Gatherers evaluate
                if scout_examples:
                    # Announce handoff
                    for gatherer in gatherers:
                        await gatherer.send_message("scout", f"Received {len(scout_examples)} examples for evaluation", {"phase": "gather", "count": len(scout_examples)})
                    gatherer_tasks = [
                        gatherer.run({
                            "examples": scout_examples[i::self.num_gatherers],
                            "threshold": 0.7
                        })
                        for i, gatherer in enumerate(gatherers)
                    ]
                    gatherer_results = await asyncio.gather(*gatherer_tasks, return_exceptions=True)
                    selected_examples = []
                    for i, result in enumerate(gatherer_results):
                        if isinstance(result, Exception):
                            print(f"[Swarm] Gatherer {i} failed: {result}")
                        else:
                            selected_examples.extend(result)
                    print(f"[Swarm] Gatherers selected {len(selected_examples)} examples")
                    
                    # Report to mutators
                    for gatherer in gatherers:
                        await gatherer.send_message("mutator", f"Passed {len(selected_examples)} quality examples", {"phase": "gather_done", "passed": len(selected_examples)})
                else:
                    selected_examples = []
                    print("[Swarm] No examples to evaluate")
                
                # Phase 3: Mutators create variations
                if iteration < self.iterations - 1 and selected_examples:
                    # Announce mutation phase
                    for mutator in mutators:
                        await mutator.send_message("gatherer", f"Creating variations of {len(selected_examples)} examples", {"phase": "mutate"})
                    mutator_tasks = [
                        mutator.run({
                            "examples": selected_examples[i::self.num_mutators],
                            "mutations_per_example": 2
                        })
                        for i, mutator in enumerate(mutators)
                    ]
                    mutator_results = await asyncio.gather(*mutator_tasks, return_exceptions=True)
                    mutated_examples = []
                    for i, result in enumerate(mutator_results):
                        if isinstance(result, Exception):
                            print(f"[Swarm] Mutator {i} failed: {result}")
                        else:
                            mutated_examples.extend(result)
                    print(f"[Swarm] Mutators created {len(mutated_examples)} variations")
                    
                    # Report to selector
                    for mutator in mutators:
                        await mutator.send_message("selector", f"Created {len(mutated_examples)} variations", {"phase": "mutate_done", "count": len(mutated_examples)})
                    all_examples.extend(mutated_examples)
                else:
                    all_examples.extend(selected_examples)
                    
            # Phase 4: Final selection
            await selector.send_message("all", f"Starting final selection from {len(all_examples)} candidates", {"phase": "select", "candidates": len(all_examples)})
            final_examples = await selector.run({
                "examples": all_examples,
                "target_count": task.get("count", 50)
            })
            
            print(f"[Swarm] Completed. Budget used: {budget.status()}")
            return final_examples
            
        finally:
            # Always clear budget when done
            clear_llm_budget()
