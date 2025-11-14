"""
Dataset generation service for LLM Dataset Creator
Generates synthetic data for fine-tuning LLMs across various domains
"""

import os
import json
import random
import time
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime
from pathlib import Path
import asyncio
import concurrent.futures
from tqdm import tqdm

from domains import DOMAINS, COMMON_PARAMS
from llm_providers import LLMProvider, OllamaProvider
import database as db
from agents import AgentCoordinator, AgentConfig

class DatasetGenerator:
    """Generator for synthetic datasets"""

    def __init__(self, llm_provider: LLMProvider, use_agents: bool = False, agent_config: AgentConfig = None):
        self.llm_provider = llm_provider
        self.use_agents = use_agents
        self.agent_coordinator = None

        # Initialize agent coordinator if agents are enabled
        if use_agents:
            config = agent_config or AgentConfig()
            self.agent_coordinator = AgentCoordinator(llm_provider, config)
    
    def weighted_choice(self, options_dict: Dict[str, float]) -> str:
        """Choose an option based on weights"""
        items, weights = zip(*options_dict.items())
        total = sum(weights)
        normalized_weights = [w/total for w in weights]
        return random.choices(items, normalized_weights, k=1)[0]
    
    def generate_scenario_parameters(self, domain_key: str, subdomain_key: Optional[str] = None) -> Dict[str, Any]:
        """Generate parameters for a scenario in the selected domain and subdomain"""
        # Get domain info
        domain = DOMAINS[domain_key]
        specific_params = domain["specific_params"]
        
        # Randomly select a subdomain if not specified
        if subdomain_key is None:
            subdomain_key = random.choice(list(domain['subdomains'].keys()))
        
        subdomain = domain['subdomains'][subdomain_key]
        
        # Common parameters for all domains
        scenario = {
            "domain": domain_key,
            "domain_name": domain["name"],
            "subdomain": subdomain_key,
            "subdomain_name": subdomain["name"],
            "question_structure": self.weighted_choice(COMMON_PARAMS["question_structures"]),
            "communication_style": self.weighted_choice(COMMON_PARAMS["communication_styles"]),
            "emotional_tone": self.weighted_choice(COMMON_PARAMS["emotional_tones"]),
            "complexity_level": self.weighted_choice(COMMON_PARAMS["complexity_levels"])
        }
        
        # Domain-specific parameters
        if domain_key == "support":
            scenario.update({
                "conversation_type": self.weighted_choice(specific_params["conversation_types"]),
                "support_scenario": self.weighted_choice(specific_params["scenarios"]),
                "tech_knowledge": self.weighted_choice(specific_params["customer_knowledge"]),
                "channel": self.weighted_choice(specific_params["channels"])
            })
            scenario["exchanges"] = 1 if scenario["conversation_type"] == "question_answer" else random.randint(2, 5)
        
        elif domain_key == "medical":
            scenario.update({
                "conversation_type": self.weighted_choice(specific_params["conversation_types"]),
                "medical_scenario": self.weighted_choice(specific_params["scenarios"]),
                "medical_specialty": self.weighted_choice(specific_params["medical_specialties"]),
                "patient_demographic": self.weighted_choice(specific_params["patient_demographics"])
            })
            scenario["exchanges"] = 1 if scenario["conversation_type"] == "explanation" else random.randint(2, 5)
        
        elif domain_key == "legal":
            scenario.update({
                "document_type": self.weighted_choice(specific_params["document_types"]),
                "practice_area": self.weighted_choice(specific_params["practice_areas"]),
                "client_knowledge": self.weighted_choice(specific_params["client_knowledge"])
            })
            scenario["exchanges"] = 1 if scenario["document_type"] in ["legal_memo", "brief", "contract"] else random.randint(2, 4)
        
        elif domain_key == "education":
            scenario.update({
                "content_type": self.weighted_choice(specific_params["content_types"]),
                "subject_area": self.weighted_choice(specific_params["subject_areas"]),
                "education_level": self.weighted_choice(specific_params["education_levels"]),
                "instruction_approach": self.weighted_choice(specific_params["instruction_approaches"])
            })
            scenario["exchanges"] = 1 if scenario["content_type"] in ["explanation", "assessment"] else random.randint(2, 6)
        
        elif domain_key == "business":
            scenario.update({
                "document_type": self.weighted_choice(specific_params["document_types"]),
                "business_context": self.weighted_choice(specific_params["business_contexts"]),
                "business_sector": self.weighted_choice(specific_params["business_sectors"]),
                "formality_level": self.weighted_choice(specific_params["formality_levels"])
            })
            scenario["exchanges"] = 1 if scenario["document_type"] in ["report", "proposal", "presentation"] else random.randint(2, 4)
        
        elif domain_key == "technical":
            scenario.update({
                "document_type": self.weighted_choice(specific_params["document_types"]),
                "technical_domain": self.weighted_choice(specific_params["technical_domains"]),
                "audience_expertise": self.weighted_choice(specific_params["audience_expertise"]),
                "documentation_style": self.weighted_choice(specific_params["documentation_styles"])
            })
            scenario["exchanges"] = 1 if scenario["document_type"] in ["reference", "specification"] else random.randint(1, 3)
        
        elif domain_key == "sales":
            scenario.update({
                "interaction_type": self.weighted_choice(specific_params["interaction_types"]),
                "customer_type": self.weighted_choice(specific_params["customer_types"]),
                "sale_complexity": self.weighted_choice(specific_params["sale_complexity"]),
                "objection_type": self.weighted_choice(specific_params["objection_types"])
            })
            scenario["exchanges"] = random.randint(3, 6)
        
        elif domain_key == "financial":
            scenario.update({
                "report_type": self.weighted_choice(specific_params["report_types"]),
                "financial_sector": self.weighted_choice(specific_params["financial_sectors"]),
                "time_horizon": self.weighted_choice(specific_params["time_horizons"]),
                "audience_type": self.weighted_choice(specific_params["audience_types"])
            })
            scenario["exchanges"] = 1 if scenario["report_type"] in ["earnings_report", "analysis_report"] else random.randint(1, 3)
        
        elif domain_key == "research":
            scenario.update({
                "research_type": self.weighted_choice(specific_params["research_types"]),
                "academic_field": self.weighted_choice(specific_params["academic_fields"]),
                "publication_type": self.weighted_choice(specific_params["publication_types"])
            })
            scenario["exchanges"] = 1
        
        elif domain_key == "coaching":
            scenario.update({
                "coaching_type": self.weighted_choice(specific_params["coaching_types"]),
                "client_stage": self.weighted_choice(specific_params["client_stages"]),
                "coaching_approach": self.weighted_choice(specific_params["coaching_approaches"]),
                "session_type": self.weighted_choice(specific_params["session_types"])
            })
            scenario["exchanges"] = random.randint(4, 8)
        
        elif domain_key == "creative":
            scenario.update({
                "writing_form": self.weighted_choice(specific_params["writing_forms"]),
                "genre": self.weighted_choice(specific_params["genres"]),
                "tone_style": self.weighted_choice(specific_params["tone_styles"]),
                "perspective": self.weighted_choice(specific_params["perspective"])
            })
            scenario["exchanges"] = 1
        
        elif domain_key == "meetings":
            scenario.update({
                "meeting_type": self.weighted_choice(specific_params["meeting_types"]),
                "meeting_context": self.weighted_choice(specific_params["meeting_contexts"]),
                "summary_style": self.weighted_choice(specific_params["summary_styles"]),
                "participation_level": self.weighted_choice(specific_params["participation_levels"])
            })
            scenario["exchanges"] = 1
        
        return scenario
    
    def get_prompt_in_language(self, scenario: Dict[str, Any], format_type: str, language: str = "en") -> str:
        """Create a prompt for the LLM based on scenario parameters in the specified language"""
        domain_key = scenario["domain"]
        
        # Base prompt in English
        if language == "en":
            user_prompt = f"""Generate a realistic {scenario['domain_name']} content for subdomain "{scenario['subdomain_name']}" with these characteristics:

DOMAIN: {scenario['domain_name']}
SUBDOMAIN: {scenario['subdomain_name']}
COMPLEXITY LEVEL: {scenario['complexity_level']}
COMMUNICATION STYLE: {scenario['communication_style']}
EMOTIONAL TONE: {scenario['emotional_tone']}
"""
        # Base prompt in Russian
        else:
            domain_ru = DOMAINS.get("translations", {}).get("domains", {}).get(domain_key, {}).get("ru", scenario['domain_name'])
            user_prompt = f"""Generate realistic content for domain "{domain_ru}", subdomain "{scenario['subdomain_name']}" with the following characteristics:

DOMAIN: {domain_ru}
SUBDOMAIN: {scenario['subdomain_name']}
COMPLEXITY LEVEL: {scenario['complexity_level']}
COMMUNICATION STYLE: {scenario['communication_style']}
EMOTIONAL TONE: {scenario['emotional_tone']}
"""

        # Add domain-specific parameters to the prompt
        if domain_key == "support":
            if language == "en":
                user_prompt += f"""INTERACTION TYPE: {scenario['conversation_type']} ({'single question and answer' if scenario['conversation_type'] == 'question_answer' else 'multiple exchanges'})
SUPPORT SCENARIO: {scenario['support_scenario']}
QUESTION STRUCTURE: {scenario['question_structure']}
CUSTOMER KNOWLEDGE LEVEL: {scenario['tech_knowledge']}
CHANNEL: {scenario['channel']}
NUMBER OF EXCHANGES: {scenario['exchanges']}
"""
            else:
                user_prompt += f"""INTERACTION TYPE: {scenario['conversation_type']} ({'single question and answer' if scenario['conversation_type'] == 'question_answer' else 'multiple message exchanges'})
SUPPORT SCENARIO: {scenario['support_scenario']}
QUESTION STRUCTURE: {scenario['question_structure']}
CUSTOMER KNOWLEDGE LEVEL: {scenario['tech_knowledge']}
CHANNEL: {scenario['channel']}
NUMBER OF EXCHANGES: {scenario['exchanges']}
"""
        elif domain_key == "medical":
            if language == "en":
                user_prompt += f"""INTERACTION TYPE: {scenario['conversation_type']}
MEDICAL SCENARIO: {scenario['medical_scenario']}
QUESTION STRUCTURE: {scenario['question_structure']}
MEDICAL SPECIALTY: {scenario['medical_specialty']}
PATIENT DEMOGRAPHIC: {scenario['patient_demographic']}
NUMBER OF EXCHANGES: {scenario['exchanges']}
"""
            else:
                user_prompt += f"""INTERACTION TYPE: {scenario['conversation_type']}
MEDICAL SCENARIO: {scenario['medical_scenario']}
QUESTION STRUCTURE: {scenario['question_structure']}
MEDICAL SPECIALTY: {scenario['medical_specialty']}
PATIENT DEMOGRAPHIC: {scenario['patient_demographic']}
NUMBER OF EXCHANGES: {scenario['exchanges']}
"""
        
        # Add domain-specific parameters for other domains
        # (Similar blocks for other domains)
        
        # Add format instructions
        if language == "en":
            user_prompt += f"""
Format as JSON:
{{
    "messages": [
        {{"role": "user", "content": "user message"}},
        {{"role": "assistant", "content": "assistant response"}},
        ...additional messages if needed...
    ],
    "metadata": {{
        "domain": "{scenario['domain_name']}",
        "subdomain": "{scenario['subdomain_name']}",
"""
        else:
            user_prompt += f"""
Format as JSON:
{{
    "messages": [
        {{"role": "user", "content": "user message"}},
        {{"role": "assistant", "content": "assistant response"}},
        ...additional messages if needed...
    ],
    "metadata": {{
        "domain": "{scenario['domain_name']}",
        "subdomain": "{scenario['subdomain_name']}",
"""

        # Add common metadata fields and closing instructions
        if language == "en":
            user_prompt += f"""        "complexity_level": "{scenario['complexity_level']}",
        "communication_style": "{scenario['communication_style']}",
        "emotional_tone": "{scenario['emotional_tone']}",
        "exchanges": {scenario.get('exchanges', 1)}
    }}
}}

Create a realistic and specific content for the given domain and subdomain.
Return ONLY valid JSON without explanations or markdown formatting.
"""
        else:
            user_prompt += f"""        "complexity_level": "{scenario['complexity_level']}",
        "communication_style": "{scenario['communication_style']}",
        "emotional_tone": "{scenario['emotional_tone']}",
        "exchanges": {scenario.get('exchanges', 1)}
    }}
}}

Create a realistic and specific content for the given domain and subdomain.
Return ONLY valid JSON without explanations or markdown formatting.
"""
        
        return user_prompt
    
    def generate_example(self, scenario: Dict[str, Any], format_type: str, language: str = "en") -> Optional[Dict[str, Any]]:
        """Generate a single synthetic example using LLM (with optional agent system)"""

        # Use agent system if enabled
        if self.use_agents and self.agent_coordinator:
            try:
                # Extract parameters from scenario
                domain = scenario['domain']
                subdomain = scenario['subdomain']

                # Remove domain/subdomain from parameters to avoid duplication
                parameters = {k: v for k, v in scenario.items()
                             if k not in ['domain', 'subdomain', 'domain_name', 'subdomain_name']}

                # Generate with agent system
                final_example, agent_responses = self.agent_coordinator.generate_example(
                    domain=domain,
                    subdomain=subdomain,
                    parameters=parameters,
                    language=language,
                    format_type=format_type
                )

                if final_example:
                    # Log agent activity
                    print(f"Agent system: Generated example with {len(agent_responses)} agent interactions")
                    return final_example
                else:
                    print("Agent system failed, falling back to standard generation")
                    # Fall through to standard generation

            except Exception as e:
                print(f"Error in agent generation: {e}")
                # Fall through to standard generation

        # Standard generation (original implementation)
        user_prompt = self.get_prompt_in_language(scenario, format_type, language)

        try:
            # Generate text from LLM
            result = self.llm_provider.generate_text(user_prompt, temperature=0.7)

            # Extract JSON from the result
            if isinstance(self.llm_provider, OllamaProvider):
                data = self.llm_provider.extract_json_from_text(result)
            else:
                # Try to extract JSON directly
                start_idx = result.find('{')
                end_idx = result.rfind('}') + 1

                if start_idx != -1 and end_idx > start_idx:
                    clean_json = result[start_idx:end_idx]
                    data = json.loads(clean_json)
                else:
                    print("Could not find valid JSON in response")
                    return None

            # Convert to instruction format if needed
            if format_type == 'instruction' and 'messages' in data:
                messages = data['messages']
                metadata = data.get('metadata', {})

                if len(messages) >= 2 and messages[0]['role'] == 'user' and messages[1]['role'] == 'assistant':
                    return {
                        "instruction": messages[0]['content'],
                        "output": messages[1]['content'],
                        "metadata": metadata
                    }

            return data

        except Exception as e:
            print(f"Error generating example: {e}")
            return None
    
    async def generate_example_async(self, scenario: Dict[str, Any], format_type: str, language: str = "en") -> Optional[Dict[str, Any]]:
        """Generate a single example asynchronously using an executor"""
        with concurrent.futures.ThreadPoolExecutor() as executor:
            return await asyncio.get_event_loop().run_in_executor(
                executor, self.generate_example, scenario, format_type, language
            )
    
    async def generate_examples_batch(
        self,
        job_id: int,
        domain: str,
        format_type: str,
        count: int,
        subdomain: Optional[str] = None,
        language: str = "en"
    ) -> List[Dict[str, Any]]:
        """Generate a batch of examples asynchronously"""
        examples = []
        tasks = []
        errors = []
        
        # Generate scenario parameters
        scenarios = []
        for _ in range(count):
            scenarios.append(self.generate_scenario_parameters(domain, subdomain))
        
        # Create tasks for async generation
        for scenario in scenarios:
            tasks.append(self.generate_example_async(scenario, format_type, language))
        
        # Execute tasks with progress tracking
        examples_generated = 0
        for task in asyncio.as_completed(tasks):
            try:
                example = await task
                if example:
                    examples.append(example)
                    examples_generated += 1
                    # Update the job status in the database
                    db.update_generation_job(job_id, examples_generated=examples_generated)
                else:
                    errors.append("Failed to generate example")
            except Exception as e:
                errors.append(str(e))
        
        return examples, errors
    
    def save_examples(self, examples: List[Dict[str, Any]], file_path: str) -> bool:
        """Save generated examples to a JSONL file"""
        try:
            # Ensure directory exists
            os.makedirs(os.path.dirname(file_path), exist_ok=True)

            # Write examples to JSONL file
            with open(file_path, 'w', encoding='utf-8') as f:
                for example in examples:
                    f.write(json.dumps(example, ensure_ascii=False) + '\n')

            return True
        except Exception as e:
            print(f"Error saving examples: {e}")
            return False

    def get_agent_stats(self) -> Optional[Dict[str, Any]]:
        """Get agent coordinator statistics"""
        if self.agent_coordinator:
            return self.agent_coordinator.get_stats()
        return None

# Functions for the API
async def start_generation_job(job_id: int, llm_provider: LLMProvider, params: Dict[str, Any] = None):
    """Start a generation job in the background"""
    try:
        # Get job information
        job = db.get_generation_job(job_id)
        if not job:
            print(f"Generation job {job_id} not found")
            return

        # Extract parameters
        params = job['parameters']
        domain = params.get('domain')
        subdomain = params.get('subdomain')
        format_type = params.get('format', 'chat')
        language = params.get('language', 'en')
        count = job['examples_requested']

        # Extract agent configuration
        use_agents = params.get('use_agents', False)
        agent_config = None

        if use_agents:
            # Build agent config from parameters
            agent_config = AgentConfig(
                enable_critic=params.get('enable_critic', True),
                enable_refiner=params.get('enable_refiner', True),
                enable_diversity=params.get('enable_diversity', True),
                enable_domain_expert=params.get('enable_domain_expert', True),
                min_quality_score=params.get('min_quality_score', 7.0),
                max_refinement_iterations=params.get('max_refinement_iterations', 2),
                diversity_threshold=params.get('diversity_threshold', 0.7),
                temperature_generation=params.get('temperature_generation', 0.8),
                temperature_critique=params.get('temperature_critique', 0.3),
                temperature_refinement=params.get('temperature_refinement', 0.6)
            )
            print(f"Agent system enabled with quality threshold: {agent_config.min_quality_score}")

        # Create generator with agent support
        generator = DatasetGenerator(llm_provider, use_agents=use_agents, agent_config=agent_config)

        # Update job status
        db.update_generation_job(job_id, status="running")
        
        # Generate examples
        examples, errors = await generator.generate_examples_batch(
            job_id, domain, format_type, count, subdomain, language
        )
        
        # Create timestamp and dataset name
        timestamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
        dataset_name = f"{domain}_{timestamp}"
        
        # Define file path
        file_path = f"data/datasets/{dataset_name}.jsonl"
        
        # Save examples
        success = generator.save_examples(examples, file_path)
        
        if success:
            # Create dataset record
            dataset_id = db.create_dataset(
                name=dataset_name,
                domain=domain,
                subdomain=subdomain,
                format=format_type,
                file_path=file_path,
                example_count=len(examples),
                metadata=params
            )
            
            # Add examples to database
            db.add_examples(dataset_id, examples)
            
            # Update job status
            db.update_generation_job(
                job_id, 
                status="completed", 
                examples_generated=len(examples),
                completed_at=datetime.now().isoformat(),
                dataset_id=dataset_id,
                errors=errors if errors else None
            )
            
            # Update dataset record with job information
            db.update_dataset(dataset_id, metadata={"job_id": job_id})
        else:
            # Update job status to failed
            db.update_generation_job(
                job_id, 
                status="failed", 
                completed_at=datetime.now().isoformat(),
                errors=["Failed to save examples"] + errors if errors else ["Failed to save examples"]
            )
    except Exception as e:
        print(f"Error in generation job {job_id}: {e}")
        # Update job status to failed
        db.update_generation_job(
            job_id, 
            status="failed", 
            completed_at=datetime.now().isoformat(),
            errors=[str(e)]
        )

def create_generation_job(params: Dict[str, Any], examples_requested: int) -> int:
    """Create a new generation job"""
    job_id = db.create_generation_job(examples_requested, params)
    return job_id

def get_job_status(job_id: int) -> Dict[str, Any]:
    """Get the status of a generation job"""
    job = db.get_generation_job(job_id)
    if not job:
        return {"error": "Job not found"}
    
    return {
        "id": job["id"],
        "status": job["status"],
        "examples_generated": job["examples_generated"],
        "examples_requested": job["examples_requested"],
        "started_at": job["started_at"],
        "completed_at": job["completed_at"],
        "dataset_id": job["dataset_id"],
        "errors": job["errors"] if "errors" in job else None
    }

def cancel_job(job_id: int) -> bool:
    """Cancel a running generation job"""
    job = db.get_generation_job(job_id)
    if not job or job["status"] not in ["pending", "running"]:
        return False
    
    # Update job status to cancelled
    db.update_generation_job(
        job_id, 
        status="cancelled", 
        completed_at=datetime.now().isoformat()
    )
    
    return True