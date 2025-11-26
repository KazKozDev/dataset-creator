"""
Quantum Field Orchestration (QFO)
Physics-inspired generation based on Standard Model Lagrangian

Inspired by quantum field theory and particle interactions
"""

import asyncio
import json
import random
import concurrent.futures
from typing import Dict, Any, List, Optional
from .base_agent import BaseAgent, AgentRole, AgentEventBus, get_llm_limiter, set_llm_budget, clear_llm_budget
from .shared_memory import SharedMemory
from llm_providers import get_provider

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from generator import DatasetGenerator
import database as db

class GaugeAgent(BaseAgent):
    """Gauge agent - generates multiple APPROACHES to the same topic (superposition)"""
    
    def __init__(self, agent_id: str, event_bus: AgentEventBus, **kwargs):
        super().__init__(agent_id, AgentRole.GAUGE, event_bus, **kwargs)
        self._provider = None
        
    def _get_provider(self):
        if self._provider is None:
            self._provider = get_provider(self.llm_provider, model=self.llm_model)
        return self._provider
        
    async def run(self, task: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate N different approaches to the same topic (batch)"""
        await self.start()
        
        count = task.get("count", 3)  # Number of examples to generate
        superposition_n = task.get("superposition_n", 3)  # Variants per example
        domain = task.get("domain", "general")
        subdomain = task.get("subdomain", "conversation")
        
        await self.think(f"Creating superposition: {count} topics × {superposition_n} approaches...")
        
        all_fields = []
        provider = self._get_provider()
        
        # One LLM call per example (batch approaches)
        for example_idx in range(count):
            budget = get_llm_limiter()
            if not budget.use():
                await self.think("Budget exhausted")
                break
            
            await self.think(f"Generating {superposition_n} approaches for example {example_idx + 1}...")
            
            prompt = f"""For a {domain}/{subdomain} conversation, generate {superposition_n} DIFFERENT APPROACHES to ask about the same general topic.

Each approach should vary in:
- Angle/perspective (different way to frame the question)
- Specificity (broad vs narrow focus)
- User background (expert vs beginner)

Return JSON array with {superposition_n} approaches:
[
  {{"approach": "description", "angle": "perspective", "user_type": "who", "specificity": "level", "topic_seed": "core topic"}},
  ...
]

Make approaches genuinely different, not just rephrased."""

            try:
                loop = asyncio.get_event_loop()
                with concurrent.futures.ThreadPoolExecutor() as executor:
                    result = await loop.run_in_executor(
                        executor,
                        lambda: provider.generate_text(prompt, temperature=0.9)
                    )
                
                json_str = result
                if "[" in result:
                    start = result.find("[")
                    end = result.rfind("]") + 1
                    json_str = result[start:end]
                
                approaches = json.loads(json_str.strip())
                
                for i, approach in enumerate(approaches[:superposition_n]):
                    approach["example_id"] = example_idx
                    approach["variant_id"] = i
                    approach["domain"] = domain
                    approach["subdomain"] = subdomain
                    all_fields.append(approach)
                    
            except Exception as e:
                await self.think(f"Generation failed: {e}")
                # Fallback
                for i in range(superposition_n):
                    all_fields.append({
                        "example_id": example_idx,
                        "variant_id": i,
                        "approach": f"approach_{i}",
                        "domain": domain,
                        "subdomain": subdomain
                    })
            
        await self.complete({"approaches_generated": len(all_fields)})
        return all_fields

class FermionAgent(BaseAgent):
    """Fermion agent - generates Q&A pairs for each approach (batch per example)"""
    
    def __init__(self, agent_id: str, event_bus: AgentEventBus, **kwargs):
        super().__init__(agent_id, AgentRole.FERMION, event_bus, **kwargs)
        self._provider = None
        
    def _get_provider(self):
        if self._provider is None:
            self._provider = get_provider(self.llm_provider, model=self.llm_model)
        return self._provider
        
    async def run(self, task: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate Q&A pairs for all approaches of each example (batch)"""
        await self.start()
        
        approaches = task.get("fields", [])
        domain = task.get("domain", "general")
        
        # Group approaches by example_id
        examples = {}
        for approach in approaches:
            eid = approach.get("example_id", 0)
            if eid not in examples:
                examples[eid] = []
            examples[eid].append(approach)
        
        await self.think(f"Generating Q&A for {len(examples)} examples ({len(approaches)} total approaches)...")
        
        fermions = []
        provider = self._get_provider()
        
        # One LLM call per example (batch all approaches)
        for example_id, example_approaches in examples.items():
            budget = get_llm_limiter()
            if not budget.use():
                await self.think("Budget exhausted")
                break
            
            await self.think(f"Generating Q&A variants for example {example_id + 1}...")
            
            approaches_desc = "\n".join([
                f"{i+1}. {a.get('approach', 'approach')} (angle: {a.get('angle', 'general')}, user: {a.get('user_type', 'user')})"
                for i, a in enumerate(example_approaches)
            ])
            
            prompt = f"""For this {domain} topic, generate a question AND answer for each approach:

Approaches:
{approaches_desc}

For EACH approach, create a realistic user question and helpful assistant answer.

Return JSON array:
[
  {{"approach_id": 0, "user_question": "...", "assistant_answer": "...", "quality_notes": "strengths of this variant"}},
  ...
]

Make each Q&A genuinely reflect its approach's angle and user type."""

            try:
                loop = asyncio.get_event_loop()
                with concurrent.futures.ThreadPoolExecutor() as executor:
                    result = await loop.run_in_executor(
                        executor,
                        lambda: provider.generate_text(prompt, temperature=0.7)
                    )
                
                json_str = result
                if "[" in result:
                    start = result.find("[")
                    end = result.rfind("]") + 1
                    json_str = result[start:end]
                
                qa_pairs = json.loads(json_str.strip())
                
                for i, qa in enumerate(qa_pairs):
                    if i < len(example_approaches):
                        fermion = {
                            **example_approaches[i],
                            "user_question": qa.get("user_question", ""),
                            "assistant_answer": qa.get("assistant_answer", ""),
                            "quality_notes": qa.get("quality_notes", ""),
                        }
                        fermions.append(fermion)
                        
            except Exception as e:
                await self.think(f"Q&A generation failed: {e}")
                for approach in example_approaches:
                    fermions.append({**approach, "user_question": "Question?", "assistant_answer": "Answer."})
            
        await self.complete({"fermions_created": len(fermions)})
        return fermions

class YukawaAgent(BaseAgent):
    """Yukawa agent - combines best parts from all variants (interaction/coupling)"""
    
    def __init__(self, agent_id: str, event_bus: AgentEventBus, **kwargs):
        super().__init__(agent_id, AgentRole.YUKAWA, event_bus, **kwargs)
        self._provider = None
        
    def _get_provider(self):
        if self._provider is None:
            self._provider = get_provider(self.llm_provider, model=self.llm_model)
        return self._provider
        
    async def run(self, task: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Optionally combine best parts from variants (if improvements suggested)"""
        await self.start()
        
        selected = task.get("selected", [])
        all_variants = task.get("all_variants", [])
        domain = task.get("domain", "general")
        
        await self.think(f"Checking {len(selected)} selected examples for improvements...")
        
        enhanced = []
        provider = self._get_provider()
        
        # Group all variants by example_id for reference
        variants_by_example = {}
        for v in all_variants:
            eid = v.get("example_id", 0)
            if eid not in variants_by_example:
                variants_by_example[eid] = []
            variants_by_example[eid].append(v)
        
        for item in selected:
            improvements = item.get("improvements", "")
            example_id = item.get("example_id", 0)
            
            # If no improvements needed, keep as-is
            if not improvements or improvements.lower() in ["none", "n/a", ""]:
                enhanced.append(item)
                continue
            
            # Check budget for enhancement
            budget = get_llm_limiter()
            if not budget.use():
                await self.think("Budget exhausted, keeping original")
                enhanced.append(item)
                continue
            
            await self.think(f"Enhancing example {example_id + 1} based on suggestions...")
            
            # Get other variants for this example
            other_variants = variants_by_example.get(example_id, [])
            other_desc = "\n".join([
                f"- Variant: Q: {v.get('user_question', '')[:100]}... A: {v.get('assistant_answer', '')[:150]}..."
                for v in other_variants if v != item
            ][:3])  # Max 3 others
            
            prompt = f"""Improve this {domain} Q&A based on the suggested improvements.

Current best:
Q: {item.get('user_question', '')}
A: {item.get('assistant_answer', '')}

Suggested improvements: {improvements}

Other variants for reference (take good parts if helpful):
{other_desc}

Return improved version as JSON:
{{"user_question": "improved question", "assistant_answer": "improved answer"}}"""

            try:
                loop = asyncio.get_event_loop()
                with concurrent.futures.ThreadPoolExecutor() as executor:
                    result = await loop.run_in_executor(
                        executor,
                        lambda: provider.generate_text(prompt, temperature=0.4)
                    )
                
                json_str = result
                if "{" in result:
                    start = result.find("{")
                    end = result.rfind("}") + 1
                    json_str = result[start:end]
                
                improved = json.loads(json_str.strip())
                
                enhanced_item = item.copy()
                enhanced_item["user_question"] = improved.get("user_question", item.get("user_question", ""))
                enhanced_item["assistant_answer"] = improved.get("assistant_answer", item.get("assistant_answer", ""))
                enhanced_item["yukawa_enhanced"] = True
                enhanced_item["quality_score"] = 0.95
                
                enhanced.append(enhanced_item)
                
            except Exception as e:
                await self.think(f"Enhancement failed: {e}")
                enhanced.append(item)
        
        await self.complete({"enhanced_count": len(enhanced)})
        return enhanced

class HiggsAgent(BaseAgent):
    """Higgs agent - evaluates variants and selects the best (gives 'mass'/weight)"""
    
    def __init__(self, agent_id: str, event_bus: AgentEventBus, **kwargs):
        super().__init__(agent_id, AgentRole.HIGGS, event_bus, **kwargs)
        self._provider = None
        
    def _get_provider(self):
        if self._provider is None:
            self._provider = get_provider(self.llm_provider, model=self.llm_model)
        return self._provider
        
    async def run(self, task: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Evaluate all variants per example and select the best (batch)"""
        await self.start()
        
        fermions = task.get("fermions", [])
        domain = task.get("domain", "general")
        
        # Group by example_id
        examples = {}
        for f in fermions:
            eid = f.get("example_id", 0)
            if eid not in examples:
                examples[eid] = []
            examples[eid].append(f)
        
        await self.think(f"Evaluating {len(fermions)} variants across {len(examples)} examples...")
        
        selected = []
        provider = self._get_provider()
        
        # One LLM call per example to compare all variants
        for example_id, variants in examples.items():
            budget = get_llm_limiter()
            if not budget.use():
                await self.think("Budget exhausted, selecting first variant")
                if variants:
                    selected.append(variants[0])
                continue
            
            await self.think(f"Comparing {len(variants)} variants for example {example_id + 1}...")
            
            variants_desc = "\n\n".join([
                f"VARIANT {i+1}:\nQ: {v.get('user_question', '')[:200]}\nA: {v.get('assistant_answer', '')[:300]}"
                for i, v in enumerate(variants)
            ])
            
            prompt = f"""Compare these {len(variants)} Q&A variants for a {domain} conversation and select the BEST one.

{variants_desc}

Evaluate based on:
- Naturalness of the question
- Helpfulness of the answer
- Accuracy and completeness
- Overall quality

Return JSON:
{{"best_variant": 0-{len(variants)-1}, "reason": "why this is best", "improvements": "optional improvements to make it even better"}}"""

            try:
                loop = asyncio.get_event_loop()
                with concurrent.futures.ThreadPoolExecutor() as executor:
                    result = await loop.run_in_executor(
                        executor,
                        lambda: provider.generate_text(prompt, temperature=0.3)
                    )
                
                json_str = result
                if "{" in result:
                    start = result.find("{")
                    end = result.rfind("}") + 1
                    json_str = result[start:end]
                
                selection = json.loads(json_str.strip())
                best_idx = int(selection.get("best_variant", 0))
                best_idx = max(0, min(best_idx, len(variants) - 1))
                
                best = variants[best_idx].copy()
                best["higgs_selected"] = True
                best["selection_reason"] = selection.get("reason", "")
                best["improvements"] = selection.get("improvements", "")
                best["quality_score"] = 0.9  # Selected = high quality
                
                selected.append(best)
                await self.output(best, {"example_id": example_id, "selected_variant": best_idx})
                
            except Exception as e:
                await self.think(f"Selection failed: {e}, using first variant")
                if variants:
                    selected.append(variants[0])
        
        await self.complete({"selected_count": len(selected)})
        return selected

class PotentialAgent(BaseAgent):
    """Potential agent - final collapse: formats examples for output"""
    
    def __init__(self, agent_id: str, event_bus: AgentEventBus, **kwargs):
        super().__init__(agent_id, AgentRole.POTENTIAL, event_bus, **kwargs)
        self.llm_provider = kwargs.get("llm_provider", "openai")
        self.llm_model = kwargs.get("llm_model", "gpt-4o-mini")
        self._provider = None
        
    def _get_provider(self):
        if self._provider is None:
            self._provider = get_provider(self.llm_provider, model=self.llm_model)
        return self._provider
        
    async def run(self, task: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Final collapse: format enhanced examples into standard output format"""
        await self.start()
        
        enhanced = task.get("enhanced", [])
        domain = task.get("domain", "general")
        subdomain = task.get("subdomain", "conversation")
        
        await self.think(f"Collapsing {len(enhanced)} examples into final format...")
        
        examples = []
        
        for i, item in enumerate(enhanced):
            user_question = item.get("user_question", "")
            assistant_answer = item.get("assistant_answer", "")
            
            if not user_question or not assistant_answer:
                await self.think(f"Example {i+1} missing data, skipping")
                continue
            
            # Format as standard chat example
            example = {
                "messages": [
                    {"role": "user", "content": user_question},
                    {"role": "assistant", "content": assistant_answer}
                ],
                "metadata": {
                    "domain": domain,
                    "subdomain": subdomain,
                    "approach": item.get("approach", ""),
                    "angle": item.get("angle", ""),
                    "user_type": item.get("user_type", ""),
                    "selection_reason": item.get("selection_reason", ""),
                    "quantum_enhanced": item.get("yukawa_enhanced", False),
                },
                "quality_score": item.get("quality_score", 0.9)
            }
            
            examples.append(example)
            await self.output(example, {"index": i, "quality": example["quality_score"]})
                
        await self.complete({
            "examples_generated": len(examples),
            "collapsed": True
        })
        
        return examples

class QuantumFieldOrchestration:
    """Main quantum field orchestrator"""
    
    def __init__(
        self,
        event_bus: AgentEventBus,
        shared_memory: SharedMemory,
        iterations: int = 3,
        role_models: Optional[Dict[str, Dict[str, str]]] = None,
        **agent_kwargs
    ):
        """
        Args:
            role_models: Dict mapping role to LLM config, e.g.:
                {
                    "gauge": {"llm_provider": "openai", "llm_model": "gpt-4"},
                    "potential": {"llm_provider": "anthropic", "llm_model": "claude-3-sonnet"},
                }
        """
        self.event_bus = event_bus
        self.shared_memory = shared_memory
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
        """Run quantum field orchestration with superposition and collapse"""
        
        # Budget: count examples × (gauge + fermion + higgs + yukawa) = 4 LLM calls per example
        # Potential doesn't use LLM (just formatting)
        requested_count = task.get("count", 2)
        superposition_n = 3  # Number of variants per example
        budget = set_llm_budget(
            requested_count * 4,  # 4 LLM-using agents per example
            multiplier=1.5,
            max_absolute=200
        )
        print(f"[Quantum] LLM budget: {budget.budget} calls for {requested_count} examples (superposition={superposition_n})")
        
        domain = task.get("domain", "general")
        subdomain = task.get("subdomain", "conversation")
        
        try:
            # Create agents with role-specific LLM configs
            gauge = GaugeAgent("gauge", self.event_bus, **self._get_role_kwargs("gauge"))
            fermion = FermionAgent("fermion", self.event_bus, **self._get_role_kwargs("fermion"))
            higgs = HiggsAgent("higgs", self.event_bus, **self._get_role_kwargs("higgs"))
            yukawa = YukawaAgent("yukawa", self.event_bus, **self._get_role_kwargs("yukawa"))
            potential = PotentialAgent("potential", self.event_bus, **self._get_role_kwargs("potential"))
            
            # === STEP 1: GAUGE - Create superposition (multiple approaches per example) ===
            print("\n=== GAUGE: Creating Superposition ===")
            await gauge.send_message("fermion", f"Creating superposition: {requested_count} topics × {superposition_n} approaches", {"count": requested_count, "variants": superposition_n})
            approaches = await gauge.run({
                "count": requested_count,
                "superposition_n": superposition_n,
                "domain": domain,
                "subdomain": subdomain
            })
            print(f"  Created {len(approaches)} approaches ({requested_count} examples × {superposition_n} variants)")
            
            # === STEP 2: FERMION - Generate Q&A for each approach (batch per example) ===
            print("\n=== FERMION: Generating Q&A Variants ===")
            await fermion.send_message("gauge", f"Received {len(approaches)} approaches, generating Q&A pairs", {"approaches": len(approaches)})
            fermions = await fermion.run({
                "fields": approaches,
                "domain": domain
            })
            await fermion.send_message("higgs", f"Generated {len(fermions)} Q&A variants for evaluation", {"fermions": len(fermions)})
            print(f"  Generated {len(fermions)} Q&A pairs")
            
            # === STEP 3: HIGGS - Evaluate and select best variant per example ===
            print("\n=== HIGGS: Selecting Best Variants (Collapse) ===")
            await higgs.send_message("fermion", f"Evaluating {len(fermions)} variants, will collapse to best", {"variants": len(fermions)})
            selected = await higgs.run({
                "fermions": fermions,
                "domain": domain
            })
            await higgs.send_message("yukawa", f"Collapsed to {len(selected)} best variants", {"selected": len(selected)})
            print(f"  Selected {len(selected)} best variants")
            
            # === STEP 4: YUKAWA - Enhance selected with improvements (if needed) ===
            print("\n=== YUKAWA: Enhancing Selected ===")
            await yukawa.send_message("higgs", f"Enhancing {len(selected)} selected examples", {"count": len(selected)})
            enhanced = await yukawa.run({
                "selected": selected,
                "all_variants": fermions,
                "domain": domain
            })
            await yukawa.send_message("potential", f"Enhanced {len(enhanced)} examples, ready for formatting", {"enhanced": len(enhanced)})
            print(f"  Enhanced {len(enhanced)} examples")
            
            # === STEP 5: POTENTIAL - Final collapse to output format ===
            print("\n=== POTENTIAL: Final Collapse ===")
            await potential.send_message("all", f"Final collapse: formatting {len(enhanced)} examples", {"count": len(enhanced)})
            final = await potential.run({
                "enhanced": enhanced,
                "domain": domain,
                "subdomain": subdomain
            })
            
            print(f"[Quantum] Completed. Budget used: {budget.status()}")
            print(f"[Quantum] Generated {len(final)} high-quality examples")
            return final
            
        finally:
            clear_llm_budget()
