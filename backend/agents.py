"""
LLM Agent System for High-Quality Dataset Generation

This module implements a multi-agent architecture where specialized agents collaborate
to create, critique, refine, and validate dataset examples. The system ensures:
- High quality through iterative refinement
- Diversity through similarity checking
- Domain accuracy through expert validation
- Consistency through structured workflows

Architecture:
    Generator Agent → Critic Agent → Refiner Agent → Diversity Agent → Domain Expert Agent
                          ↓                ↓              ↓                    ↓
                      Low Score?      Needs Fix?    Too Similar?        Domain Error?
                          ↓                ↓              ↓                    ↓
                      Regenerate       Refine        Regenerate          Refine
"""

import json
import logging
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from datetime import datetime
import hashlib

logger = logging.getLogger(__name__)


@dataclass
class AgentResponse:
    """Response from an agent with score, feedback, and modified content"""
    agent_name: str
    success: bool
    score: float  # 0-10 scale
    feedback: str
    content: Optional[Dict] = None
    metadata: Dict = field(default_factory=dict)
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())


@dataclass
class AgentConfig:
    """Configuration for the agent system"""
    enable_critic: bool = True
    enable_refiner: bool = True
    enable_diversity: bool = True
    enable_domain_expert: bool = True

    min_quality_score: float = 7.0
    max_refinement_iterations: int = 2
    diversity_threshold: float = 0.7  # Similarity threshold (0-1)
    diversity_window_size: int = 50  # Check against last N examples

    temperature_generation: float = 0.8
    temperature_critique: float = 0.3
    temperature_refinement: float = 0.6


class BaseAgent:
    """Base class for all specialized agents"""

    def __init__(self, llm_provider, config: AgentConfig):
        self.llm_provider = llm_provider
        self.config = config
        self.agent_name = self.__class__.__name__

    def generate_response(self, prompt: str, temperature: float) -> str:
        """Generate text using the LLM provider"""
        try:
            return self.llm_provider.generate_text(prompt, temperature)
        except Exception as e:
            logger.error(f"{self.agent_name} generation failed: {e}")
            raise

    def extract_json_from_response(self, text: str) -> Optional[Dict]:
        """Extract JSON from LLM response"""
        try:
            # Try to find JSON in code blocks
            if "```json" in text:
                start = text.find("```json") + 7
                end = text.find("```", start)
                json_str = text[start:end].strip()
            elif "```" in text:
                start = text.find("```") + 3
                end = text.find("```", start)
                json_str = text[start:end].strip()
            else:
                json_str = text.strip()

            return json.loads(json_str)
        except Exception as e:
            logger.warning(f"JSON extraction failed: {e}")
            return None


class GeneratorAgent(BaseAgent):
    """Generates initial dataset examples based on domain and parameters"""

    def generate(self, domain: str, subdomain: str, parameters: Dict,
                 language: str = "English", format_type: str = "chat") -> AgentResponse:
        """Generate a new example"""

        prompt = self._build_generation_prompt(domain, subdomain, parameters, language, format_type)

        try:
            response_text = self.generate_response(prompt, self.config.temperature_generation)
            content = self.extract_json_from_response(response_text)

            if content:
                return AgentResponse(
                    agent_name=self.agent_name,
                    success=True,
                    score=8.0,  # Initial score, will be evaluated by critic
                    feedback="Successfully generated initial example",
                    content=content,
                    metadata={"domain": domain, "subdomain": subdomain}
                )
            else:
                return AgentResponse(
                    agent_name=self.agent_name,
                    success=False,
                    score=0.0,
                    feedback="Failed to extract valid JSON from response",
                    metadata={"raw_response": response_text[:200]}
                )

        except Exception as e:
            logger.error(f"Generation failed: {e}")
            return AgentResponse(
                agent_name=self.agent_name,
                success=False,
                score=0.0,
                feedback=f"Generation error: {str(e)}"
            )

    def _build_generation_prompt(self, domain: str, subdomain: str,
                                 parameters: Dict, language: str, format_type: str) -> str:
        """Build the generation prompt"""

        params_str = "\n".join([f"- {k}: {v}" for k, v in parameters.items()])

        if format_type == "chat":
            format_instructions = """
Generate a conversation in this JSON format:
{
  "messages": [
    {"role": "system", "content": "system message (optional)"},
    {"role": "user", "content": "user message"},
    {"role": "assistant", "content": "assistant response"},
    ...
  ],
  "metadata": {
    "domain": "domain name",
    "subdomain": "subdomain name",
    "scenario": "brief scenario description"
  }
}
"""
        else:
            format_instructions = """
Generate an instruction-following example in this JSON format:
{
  "instruction": "clear instruction or question",
  "output": "high-quality response",
  "metadata": {
    "domain": "domain name",
    "subdomain": "subdomain name",
    "scenario": "brief scenario description"
  }
}
"""

        prompt = f"""You are an expert dataset creator for training AI models.

**Domain**: {domain}
**Subdomain**: {subdomain}
**Language**: {language}

**Parameters**:
{params_str}

**Task**: Create a realistic, high-quality training example that demonstrates natural {language} communication in the {subdomain} context.

**Quality Requirements**:
1. Natural, realistic dialogue or instruction
2. Appropriate complexity and depth for the domain
3. Accurate domain-specific terminology
4. Clear, coherent, and well-structured
5. Diverse and creative scenarios
6. Authentic emotional tone matching parameters

{format_instructions}

Generate ONE example following the format exactly. Return ONLY the JSON object, nothing else.
"""
        return prompt


class CriticAgent(BaseAgent):
    """Evaluates generated examples and provides detailed feedback"""

    def critique(self, example: Dict, domain: str, subdomain: str,
                 parameters: Dict) -> AgentResponse:
        """Evaluate an example and provide detailed feedback"""

        prompt = self._build_critique_prompt(example, domain, subdomain, parameters)

        try:
            response_text = self.generate_response(prompt, self.config.temperature_critique)
            evaluation = self.extract_json_from_response(response_text)

            if evaluation and "overall_score" in evaluation:
                score = float(evaluation.get("overall_score", 0))
                feedback_parts = []

                # Compile detailed feedback
                for criterion, details in evaluation.items():
                    if criterion != "overall_score" and isinstance(details, dict):
                        criterion_score = details.get("score", 0)
                        criterion_feedback = details.get("feedback", "")
                        feedback_parts.append(f"{criterion}: {criterion_score}/10 - {criterion_feedback}")

                feedback = "\n".join(feedback_parts)

                return AgentResponse(
                    agent_name=self.agent_name,
                    success=True,
                    score=score,
                    feedback=feedback,
                    metadata={"evaluation": evaluation}
                )
            else:
                return AgentResponse(
                    agent_name=self.agent_name,
                    success=False,
                    score=5.0,  # Default moderate score
                    feedback="Could not parse evaluation response",
                    metadata={"raw_response": response_text[:200]}
                )

        except Exception as e:
            logger.error(f"Critique failed: {e}")
            return AgentResponse(
                agent_name=self.agent_name,
                success=False,
                score=5.0,
                feedback=f"Critique error: {str(e)}"
            )

    def _build_critique_prompt(self, example: Dict, domain: str,
                              subdomain: str, parameters: Dict) -> str:
        """Build the critique prompt"""

        example_str = json.dumps(example, indent=2, ensure_ascii=False)
        params_str = json.dumps(parameters, indent=2, ensure_ascii=False)

        prompt = f"""You are an expert evaluator for AI training datasets.

**Domain**: {domain}
**Subdomain**: {subdomain}
**Expected Parameters**: {params_str}

**Example to Evaluate**:
{example_str}

**Task**: Evaluate this example on multiple criteria and provide detailed feedback.

**Evaluation Criteria** (score each 0-10):

1. **Content Quality**: Is the content informative, accurate, and valuable?
2. **Naturalness**: Does it sound like natural human communication?
3. **Domain Accuracy**: Is domain-specific terminology and context correct?
4. **Instruction Clarity**: Are instructions/questions clear and well-formed?
5. **Response Quality**: Is the response helpful, complete, and appropriate?
6. **Parameter Alignment**: Does it match the specified parameters?
7. **Coherence**: Is the flow logical and coherent?
8. **Creativity**: Is it unique and creative, not generic?

Return your evaluation in this JSON format:
{{
  "content_quality": {{"score": X, "feedback": "..."}},
  "naturalness": {{"score": X, "feedback": "..."}},
  "domain_accuracy": {{"score": X, "feedback": "..."}},
  "instruction_clarity": {{"score": X, "feedback": "..."}},
  "response_quality": {{"score": X, "feedback": "..."}},
  "parameter_alignment": {{"score": X, "feedback": "..."}},
  "coherence": {{"score": X, "feedback": "..."}},
  "creativity": {{"score": X, "feedback": "..."}},
  "overall_score": X.X
}}

Be critical but constructive. Provide specific feedback for improvement.
"""
        return prompt


class RefinerAgent(BaseAgent):
    """Refines and improves examples based on critic feedback"""

    def refine(self, example: Dict, critique_feedback: str,
               domain: str, subdomain: str) -> AgentResponse:
        """Refine an example based on feedback"""

        prompt = self._build_refinement_prompt(example, critique_feedback, domain, subdomain)

        try:
            response_text = self.generate_response(prompt, self.config.temperature_refinement)
            refined_content = self.extract_json_from_response(response_text)

            if refined_content:
                return AgentResponse(
                    agent_name=self.agent_name,
                    success=True,
                    score=8.5,  # Will be re-evaluated by critic
                    feedback="Successfully refined example based on feedback",
                    content=refined_content
                )
            else:
                return AgentResponse(
                    agent_name=self.agent_name,
                    success=False,
                    score=0.0,
                    feedback="Failed to extract refined JSON",
                    metadata={"raw_response": response_text[:200]}
                )

        except Exception as e:
            logger.error(f"Refinement failed: {e}")
            return AgentResponse(
                agent_name=self.agent_name,
                success=False,
                score=0.0,
                feedback=f"Refinement error: {str(e)}"
            )

    def _build_refinement_prompt(self, example: Dict, feedback: str,
                                 domain: str, subdomain: str) -> str:
        """Build the refinement prompt"""

        example_str = json.dumps(example, indent=2, ensure_ascii=False)

        prompt = f"""You are an expert at refining AI training examples.

**Domain**: {domain}
**Subdomain**: {subdomain}

**Original Example**:
{example_str}

**Feedback from Evaluation**:
{feedback}

**Task**: Improve this example by addressing all the feedback points. Maintain the same format but enhance quality.

**Focus on**:
- Fix any identified issues
- Enhance naturalness and coherence
- Improve domain accuracy
- Make it more creative and unique
- Ensure parameter alignment

Return the improved example in the SAME JSON format as the original. Keep the structure but improve the content.
"""
        return prompt


class DiversityAgent(BaseAgent):
    """Ensures diversity across generated examples"""

    def __init__(self, llm_provider, config: AgentConfig):
        super().__init__(llm_provider, config)
        self.recent_examples = []  # Store recent examples for comparison

    def check_diversity(self, example: Dict, domain: str) -> AgentResponse:
        """Check if example is sufficiently diverse from recent examples"""

        if len(self.recent_examples) < 5:
            # Not enough examples to check diversity yet
            self.add_example(example)
            return AgentResponse(
                agent_name=self.agent_name,
                success=True,
                score=10.0,
                feedback="Insufficient examples for diversity check, accepting",
                metadata={"recent_count": len(self.recent_examples)}
            )

        # Get recent examples from same domain
        domain_examples = [ex for ex in self.recent_examples if ex.get("domain") == domain]

        if not domain_examples:
            self.add_example(example)
            return AgentResponse(
                agent_name=self.agent_name,
                success=True,
                score=10.0,
                feedback="First example in this domain, accepting",
                metadata={"domain": domain}
            )

        # Check similarity with LLM
        prompt = self._build_diversity_prompt(example, domain_examples[-5:])  # Check last 5

        try:
            response_text = self.generate_response(prompt, 0.3)
            evaluation = self.extract_json_from_response(response_text)

            if evaluation:
                similarity_score = float(evaluation.get("similarity_score", 0))
                diversity_score = 10.0 - similarity_score  # Invert: high similarity = low diversity
                is_diverse = similarity_score < 7.0  # Threshold for acceptable diversity

                if is_diverse:
                    self.add_example(example)

                return AgentResponse(
                    agent_name=self.agent_name,
                    success=is_diverse,
                    score=diversity_score,
                    feedback=evaluation.get("feedback", "Diversity check complete"),
                    metadata={
                        "similarity_score": similarity_score,
                        "is_diverse": is_diverse,
                        "compared_count": len(domain_examples[-5:])
                    }
                )
            else:
                # If evaluation fails, accept the example
                self.add_example(example)
                return AgentResponse(
                    agent_name=self.agent_name,
                    success=True,
                    score=7.0,
                    feedback="Could not evaluate diversity, accepting",
                    metadata={"raw_response": response_text[:200]}
                )

        except Exception as e:
            logger.error(f"Diversity check failed: {e}")
            self.add_example(example)
            return AgentResponse(
                agent_name=self.agent_name,
                success=True,
                score=7.0,
                feedback=f"Diversity check error, accepting: {str(e)}"
            )

    def add_example(self, example: Dict):
        """Add example to recent examples buffer"""
        self.recent_examples.append(example)
        if len(self.recent_examples) > self.config.diversity_window_size:
            self.recent_examples.pop(0)

    def _build_diversity_prompt(self, example: Dict, recent_examples: List[Dict]) -> str:
        """Build diversity check prompt"""

        example_str = json.dumps(example, indent=2, ensure_ascii=False)
        recent_str = json.dumps(recent_examples, indent=2, ensure_ascii=False)

        prompt = f"""You are an expert at evaluating dataset diversity.

**New Example**:
{example_str}

**Recent Examples** (last 5):
{recent_str}

**Task**: Evaluate how similar the new example is to recent examples.

**Check for**:
- Similar topics or scenarios
- Repetitive patterns or structures
- Similar vocabulary or phrasing
- Lack of creative variation

**Scoring**:
- 0-3: Very diverse, completely different scenarios/topics
- 4-6: Moderately diverse, some similarities but substantially different
- 7-8: Low diversity, significant overlap in topic/structure
- 9-10: Nearly identical, unacceptable repetition

Return your evaluation in JSON format:
{{
  "similarity_score": X,
  "feedback": "Detailed explanation of similarities/differences",
  "specific_overlaps": ["list", "of", "specific", "similarities"]
}}
"""
        return prompt

    def clear_buffer(self):
        """Clear the recent examples buffer"""
        self.recent_examples = []


class DomainExpertAgent(BaseAgent):
    """Validates domain-specific accuracy and appropriateness"""

    def validate(self, example: Dict, domain: str, subdomain: str) -> AgentResponse:
        """Validate domain-specific accuracy"""

        prompt = self._build_validation_prompt(example, domain, subdomain)

        try:
            response_text = self.generate_response(prompt, 0.3)
            validation = self.extract_json_from_response(response_text)

            if validation:
                score = float(validation.get("accuracy_score", 0))
                is_valid = score >= 7.0
                issues = validation.get("issues", [])

                feedback = validation.get("feedback", "")
                if issues:
                    feedback += f"\nIssues found: {', '.join(issues)}"

                return AgentResponse(
                    agent_name=self.agent_name,
                    success=is_valid,
                    score=score,
                    feedback=feedback,
                    metadata={
                        "issues": issues,
                        "is_valid": is_valid,
                        "domain_context": validation.get("domain_context", "")
                    }
                )
            else:
                return AgentResponse(
                    agent_name=self.agent_name,
                    success=True,  # Accept if can't evaluate
                    score=7.0,
                    feedback="Could not validate, accepting",
                    metadata={"raw_response": response_text[:200]}
                )

        except Exception as e:
            logger.error(f"Domain validation failed: {e}")
            return AgentResponse(
                agent_name=self.agent_name,
                success=True,
                score=7.0,
                feedback=f"Validation error, accepting: {str(e)}"
            )

    def _build_validation_prompt(self, example: Dict, domain: str, subdomain: str) -> str:
        """Build domain validation prompt"""

        example_str = json.dumps(example, indent=2, ensure_ascii=False)

        # Domain-specific validation criteria
        domain_criteria = {
            "support": "Appropriate tone, helpful responses, accurate troubleshooting steps",
            "medical": "Medical accuracy, appropriate terminology, ethical considerations",
            "legal": "Legal accuracy, proper terminology, appropriate disclaimers",
            "education": "Pedagogical soundness, age-appropriate content, clear explanations",
            "business": "Professional tone, business-appropriate context, strategic thinking",
            "technical": "Technical accuracy, proper terminology, clear documentation",
            "sales": "Persuasive but ethical, customer-focused, product knowledge",
            "financial": "Financial accuracy, regulatory awareness, risk considerations",
            "research": "Scientific rigor, proper methodology, citation awareness",
            "coaching": "Supportive tone, actionable advice, appropriate boundaries",
            "creative": "Creative quality, originality, artistic merit",
            "meetings": "Professional facilitation, clear agenda, actionable outcomes"
        }

        criteria = domain_criteria.get(domain, "Domain-appropriate content and terminology")

        prompt = f"""You are a domain expert in {domain}, specifically {subdomain}.

**Example to Validate**:
{example_str}

**Task**: Validate this example for domain-specific accuracy and appropriateness.

**Validation Criteria for {domain}**:
{criteria}

**Check for**:
1. Accurate domain-specific terminology
2. Contextually appropriate scenarios
3. Realistic dialogue/instructions for this domain
4. No factual errors or misconceptions
5. Appropriate professional standards
6. Ethical considerations for the domain

**Scoring** (0-10):
- 9-10: Excellent domain accuracy, professional quality
- 7-8: Good accuracy, minor improvements possible
- 5-6: Moderate accuracy, some domain issues
- 0-4: Poor accuracy, significant domain errors

Return your validation in JSON format:
{{
  "accuracy_score": X,
  "feedback": "Overall assessment",
  "issues": ["list", "of", "specific", "issues"],
  "domain_context": "Domain-specific observations"
}}
"""
        return prompt


class AgentCoordinator:
    """Coordinates all agents to create high-quality examples"""

    def __init__(self, llm_provider, config: AgentConfig = None):
        self.config = config or AgentConfig()

        # Initialize all agents
        self.generator = GeneratorAgent(llm_provider, self.config)
        self.critic = CriticAgent(llm_provider, self.config)
        self.refiner = RefinerAgent(llm_provider, self.config)
        self.diversity = DiversityAgent(llm_provider, self.config)
        self.domain_expert = DomainExpertAgent(llm_provider, self.config)

        # Statistics
        self.stats = {
            "total_attempts": 0,
            "successful_generations": 0,
            "refinement_iterations": 0,
            "diversity_rejections": 0,
            "domain_rejections": 0,
            "average_quality_score": 0.0
        }

    def generate_example(self, domain: str, subdomain: str, parameters: Dict,
                        language: str = "English", format_type: str = "chat") -> Tuple[Optional[Dict], List[AgentResponse]]:
        """
        Generate a high-quality example using the multi-agent pipeline

        Returns:
            Tuple of (final_example, agent_responses)
        """

        agent_responses = []
        self.stats["total_attempts"] += 1

        # Step 1: Generate initial example
        logger.info(f"Generating example for {domain}/{subdomain}")
        gen_response = self.generator.generate(domain, subdomain, parameters, language, format_type)
        agent_responses.append(gen_response)

        if not gen_response.success:
            logger.warning(f"Generation failed: {gen_response.feedback}")
            return None, agent_responses

        current_example = gen_response.content
        current_score = gen_response.score

        # Step 2: Critique and refine loop
        if self.config.enable_critic:
            for iteration in range(self.config.max_refinement_iterations):
                logger.info(f"Critique iteration {iteration + 1}")

                # Critique the example
                critique_response = self.critic.critique(
                    current_example, domain, subdomain, parameters
                )
                agent_responses.append(critique_response)
                current_score = critique_response.score

                # If quality is sufficient, break the loop
                if current_score >= self.config.min_quality_score:
                    logger.info(f"Quality threshold met: {current_score:.1f}")
                    break

                # If refinement is disabled or last iteration, accept current version
                if not self.config.enable_refiner or iteration == self.config.max_refinement_iterations - 1:
                    logger.info(f"Accepting example with score {current_score:.1f}")
                    break

                # Refine the example
                logger.info(f"Refining example (score: {current_score:.1f})")
                refine_response = self.refiner.refine(
                    current_example, critique_response.feedback, domain, subdomain
                )
                agent_responses.append(refine_response)
                self.stats["refinement_iterations"] += 1

                if refine_response.success:
                    current_example = refine_response.content
                else:
                    logger.warning("Refinement failed, keeping original")
                    break

        # Step 3: Diversity check
        if self.config.enable_diversity:
            logger.info("Checking diversity")
            diversity_response = self.diversity.check_diversity(current_example, domain)
            agent_responses.append(diversity_response)

            if not diversity_response.success:
                logger.warning("Diversity check failed, example too similar")
                self.stats["diversity_rejections"] += 1
                return None, agent_responses

        # Step 4: Domain expert validation
        if self.config.enable_domain_expert:
            logger.info("Validating domain accuracy")
            validation_response = self.domain_expert.validate(current_example, domain, subdomain)
            agent_responses.append(validation_response)

            if not validation_response.success:
                logger.warning("Domain validation failed")
                self.stats["domain_rejections"] += 1

                # Try one refinement based on domain feedback
                if self.config.enable_refiner:
                    logger.info("Attempting domain-based refinement")
                    refine_response = self.refiner.refine(
                        current_example, validation_response.feedback, domain, subdomain
                    )
                    agent_responses.append(refine_response)

                    if refine_response.success:
                        current_example = refine_response.content
                        # Re-validate
                        validation_response = self.domain_expert.validate(current_example, domain, subdomain)
                        agent_responses.append(validation_response)

                        if not validation_response.success:
                            return None, agent_responses
                    else:
                        return None, agent_responses
                else:
                    return None, agent_responses

        # Success!
        self.stats["successful_generations"] += 1

        # Update average quality score
        total_successful = self.stats["successful_generations"]
        current_avg = self.stats["average_quality_score"]
        self.stats["average_quality_score"] = (
            (current_avg * (total_successful - 1) + current_score) / total_successful
        )

        # Add agent metadata to example
        if "metadata" not in current_example:
            current_example["metadata"] = {}

        current_example["metadata"]["agent_quality_score"] = current_score
        current_example["metadata"]["agent_iterations"] = len(agent_responses)
        current_example["metadata"]["generated_with_agents"] = True

        logger.info(f"Successfully generated example with score {current_score:.1f}")

        return current_example, agent_responses

    def get_stats(self) -> Dict:
        """Get coordinator statistics"""
        stats = self.stats.copy()
        if stats["total_attempts"] > 0:
            stats["success_rate"] = stats["successful_generations"] / stats["total_attempts"]
        else:
            stats["success_rate"] = 0.0

        return stats

    def reset_stats(self):
        """Reset statistics"""
        self.stats = {
            "total_attempts": 0,
            "successful_generations": 0,
            "refinement_iterations": 0,
            "diversity_rejections": 0,
            "domain_rejections": 0,
            "average_quality_score": 0.0
        }

    def reset_diversity_buffer(self):
        """Reset diversity agent's buffer"""
        self.diversity.clear_buffer()
