"""
Quality control service for LLM Dataset Creator
Evaluates and improves the quality of generated datasets
"""

import os
import json
import time
import asyncio
import concurrent.futures
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime
from pathlib import Path

from llm_providers import LLMProvider
import database as db

class QualityController:
    """Quality controller for LLM datasets"""
    
    def __init__(self, llm_provider: LLMProvider):
        self.llm_provider = llm_provider
    
    def check_example_structure(self, example: Dict[str, Any], format_type: str) -> List[str]:
        """Check if example has the required structure"""
        issues = []
        
        if format_type == 'chat':
            if 'messages' not in example:
                issues.append("Missing 'messages' field")
            elif not isinstance(example['messages'], list):
                issues.append("'messages' is not a list")
            elif len(example['messages']) < 2:
                issues.append("Too few messages (need at least 2)")
            else:
                # Check message structure
                for i, msg in enumerate(example['messages']):
                    if not isinstance(msg, dict):
                        issues.append(f"Message {i} is not an object")
                    elif 'role' not in msg:
                        issues.append(f"Message {i} is missing 'role'")
                    elif 'content' not in msg:
                        issues.append(f"Message {i} is missing 'content'")
                    elif msg['role'] not in ['user', 'assistant', 'system']:
                        issues.append(f"Message {i} has invalid role: {msg['role']}")
            
            # Check metadata
            if 'metadata' not in example:
                issues.append("Missing 'metadata' field")
            elif not isinstance(example['metadata'], dict):
                issues.append("'metadata' is not an object")
        
        elif format_type == 'instruction':
            if 'instruction' not in example:
                issues.append("Missing 'instruction' field")
            elif not isinstance(example['instruction'], str):
                issues.append("'instruction' is not a string")
            
            if 'output' not in example:
                issues.append("Missing 'output' field")
            elif not isinstance(example['output'], str):
                issues.append("'output' is not a string")
            
            # Check metadata
            if 'metadata' not in example:
                issues.append("Missing 'metadata' field")
            elif not isinstance(example['metadata'], dict):
                issues.append("'metadata' is not an object")
        
        return issues
    
    def evaluate_example_quality(self, example: Dict[str, Any], format_type: str, index: int) -> Optional[Dict[str, Any]]:
        """Use LLM to evaluate the quality of an example"""
        
        # Create prompt for quality evaluation
        if format_type == 'chat':
            messages = json.dumps(example.get('messages', []), indent=2)
            metadata = json.dumps(example.get('metadata', {}), indent=2)
            
            eval_prompt = f"""
Evaluate the quality of this fine-tuning example for an LLM. Score the example from 0-10 and identify any issues or ways to improve it.

EXAMPLE ID: {index}
FORMAT: Chat

MESSAGES:
{messages}

METADATA:
{metadata}

Please evaluate this example on the following criteria:
1. Content quality (natural, coherent, informative)
2. Instruction-response alignment (response addresses the input appropriately)
3. Factual correctness (no obvious factual errors)
4. Formatting and structure (properly formatted, complete)
5. Helpfulness and utility for training

Return your assessment as a valid JSON object with these fields:
- score: numeric score from 0-10
- issues: array of specific issues identified
- fix_suggestions: specific suggestions to improve the example
- keep_or_fix: recommendation ("keep", "fix", or "remove")

JSON FORMAT ONLY, NO ADDITIONAL TEXT.
"""
        else:  # instruction format
            instruction = example.get('instruction', '')
            output = example.get('output', '')
            metadata = json.dumps(example.get('metadata', {}), indent=2)
            
            eval_prompt = f"""
Evaluate the quality of this fine-tuning example for an LLM. Score the example from 0-10 and identify any issues or ways to improve it.

EXAMPLE ID: {index}
FORMAT: Instruction

INSTRUCTION:
{instruction}

OUTPUT:
{output}

METADATA:
{metadata}

Please evaluate this example on the following criteria:
1. Content quality (natural, coherent, informative)
2. Instruction-response alignment (response addresses the instruction appropriately)
3. Factual correctness (no obvious factual errors)
4. Formatting and structure (properly formatted, complete)
5. Helpfulness and utility for training

Return your assessment as a valid JSON object with these fields:
- score: numeric score from 0-10
- issues: array of specific issues identified
- fix_suggestions: specific suggestions to improve the example
- keep_or_fix: recommendation ("keep", "fix", or "remove")

JSON FORMAT ONLY, NO ADDITIONAL TEXT.
"""
        
        try:
            # Generate text from LLM
            result = self.llm_provider.generate_text(eval_prompt, temperature=0.2)
            
            # Extract JSON from the result
            if hasattr(self.llm_provider, 'extract_json_from_text'):
                return self.llm_provider.extract_json_from_text(result)
            else:
                # Try to extract JSON manually
                try:
                    # Clean up the response to extract JSON
                    if "```json" in result:
                        result = result.split("```json")[1].split("```")[0].strip()
                    elif "```" in result:
                        result = result.split("```")[1].split("```")[0].strip()
                    
                    # Find JSON object bounds
                    start_idx = result.find('{')
                    end_idx = result.rfind('}') + 1
                    
                    if start_idx != -1 and end_idx > start_idx:
                        clean_json = result[start_idx:end_idx]
                        return json.loads(clean_json)
                    else:
                        print(f"Could not find valid JSON in evaluation response for example {index}")
                        return None
                except Exception as e:
                    print(f"Error parsing evaluation JSON: {e}")
                    return None
            
        except Exception as e:
            print(f"Error evaluating example {index}: {e}")
            return None
    
    def fix_example(self, example: Dict[str, Any], format_type: str, evaluation: Dict[str, Any], index: int) -> Optional[Dict[str, Any]]:
        """Use LLM to fix issues with an example based on evaluation"""
        
        if format_type == 'chat':
            messages = json.dumps(example.get('messages', []), indent=2)
            metadata = json.dumps(example.get('metadata', {}), indent=2)
            issues = json.dumps(evaluation.get('issues', []), indent=2)
            suggestions = json.dumps(evaluation.get('fix_suggestions', []), indent=2)
            
            fix_prompt = f"""
Fix the following chat example based on the identified issues and suggestions.

EXAMPLE ID: {index}
FORMAT: Chat

ORIGINAL MESSAGES:
{messages}

METADATA:
{metadata}

ISSUES IDENTIFIED:
{issues}

IMPROVEMENT SUGGESTIONS:
{suggestions}

Create an improved version of this example. Keep the same domain and intent, but fix the identified issues.
Return the complete fixed example as a valid JSON object with 'messages' and 'metadata' fields.
The 'messages' array should follow the correct format with 'role' and 'content' for each message.

JSON FORMAT ONLY, NO ADDITIONAL TEXT.
"""
        else:  # instruction format
            instruction = example.get('instruction', '')
            output = example.get('output', '')
            metadata = json.dumps(example.get('metadata', {}), indent=2)
            issues = json.dumps(evaluation.get('issues', []), indent=2)
            suggestions = json.dumps(evaluation.get('fix_suggestions', []), indent=2)
            
            fix_prompt = f"""
Fix the following instruction example based on the identified issues and suggestions.

EXAMPLE ID: {index}
FORMAT: Instruction

ORIGINAL INSTRUCTION:
{instruction}

ORIGINAL OUTPUT:
{output}

METADATA:
{metadata}

ISSUES IDENTIFIED:
{issues}

IMPROVEMENT SUGGESTIONS:
{suggestions}

Create an improved version of this example. Keep the same domain and intent, but fix the identified issues.
Return the complete fixed example as a valid JSON object with 'instruction', 'output', and 'metadata' fields.

JSON FORMAT ONLY, NO ADDITIONAL TEXT.
"""
        
        try:
            # Generate text from LLM
            result = self.llm_provider.generate_text(fix_prompt, temperature=0.7)
            
            # Extract JSON from the result
            if hasattr(self.llm_provider, 'extract_json_from_text'):
                fixed_example = self.llm_provider.extract_json_from_text(result)
            else:
                # Try to extract JSON manually
                try:
                    # Clean up the response to extract JSON
                    if "```json" in result:
                        result = result.split("```json")[1].split("```")[0].strip()
                    elif "```" in result:
                        result = result.split("```")[1].split("```")[0].strip()
                    
                    # Find JSON object bounds
                    start_idx = result.find('{')
                    end_idx = result.rfind('}') + 1
                    
                    if start_idx != -1 and end_idx > start_idx:
                        clean_json = result[start_idx:end_idx]
                        fixed_example = json.loads(clean_json)
                    else:
                        print(f"Could not find valid JSON in fix response for example {index}")
                        return None
                except Exception as e:
                    print(f"Error parsing fix JSON: {e}")
                    return None
            
            # Verify the fixed example has the required structure
            structure_issues = self.check_example_structure(fixed_example, format_type)
            if structure_issues:
                print(f"Warning: Fixed example still has structure issues: {structure_issues}")
            
            return fixed_example
            
        except Exception as e:
            print(f"Error fixing example {index}: {e}")
            return None
    
    async def evaluate_example_async(self, example: Dict[str, Any], format_type: str, index: int) -> Tuple[Dict[str, Any], Optional[Dict[str, Any]]]:
        """Evaluate an example asynchronously using an executor"""
        with concurrent.futures.ThreadPoolExecutor() as executor:
            evaluation = await asyncio.get_event_loop().run_in_executor(
                executor, self.evaluate_example_quality, example, format_type, index
            )
            return example, evaluation
    
    async def fix_example_async(self, example: Dict[str, Any], format_type: str, evaluation: Dict[str, Any], index: int) -> Tuple[Dict[str, Any], Optional[Dict[str, Any]]]:
        """Fix an example asynchronously using an executor"""
        with concurrent.futures.ThreadPoolExecutor() as executor:
            fixed_example = await asyncio.get_event_loop().run_in_executor(
                executor, self.fix_example, example, format_type, evaluation, index
            )
            return example, fixed_example
    
    def save_examples(self, examples: List[Dict[str, Any]], file_path: str) -> bool:
        """Save examples to a JSONL file"""
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
    
    def load_examples(self, file_path: str) -> List[Dict[str, Any]]:
        """Load examples from a JSONL file"""
        examples = []
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if line:
                        example = json.loads(line)
                        examples.append(example)
        except Exception as e:
            print(f"Error loading examples: {e}")
        
        return examples
    
    def determine_format_type(self, examples: List[Dict[str, Any]]) -> str:
        """Determine the format type of the examples (chat or instruction)"""
        if not examples:
            return "unknown"
        
        sample = examples[0]
        if "messages" in sample:
            return "chat"
        elif "instruction" in sample and "output" in sample:
            return "instruction"
        else:
            return "unknown"

# Functions for the API
async def process_quality_job(job_id: int, llm_provider: LLMProvider, batch_size: int = 10, quality_threshold: float = 7.0, auto_fix: bool = False, auto_remove: bool = False):
    """Process a quality control job"""
    try:
        # Get job information
        job = db.get_quality_job(job_id)
        if not job:
            print(f"Quality job {job_id} not found")
            return
        
        # Get dataset information
        dataset_id = job['dataset_id']
        dataset = db.get_dataset(dataset_id)
        if not dataset:
            print(f"Dataset {dataset_id} not found")
            db.update_quality_job(job_id, status="failed", errors=["Dataset not found"])
            return
        
        # Create quality controller
        controller = QualityController(llm_provider)
        
        # Update job status
        db.update_quality_job(job_id, status="running")
        
        # Load examples
        file_path = dataset['file_path']
        examples = controller.load_examples(file_path)
        
        if not examples:
            db.update_quality_job(job_id, status="failed", errors=["No examples found in dataset"])
            return
        
        # Determine format type
        format_type = controller.determine_format_type(examples)
        if format_type == "unknown":
            db.update_quality_job(job_id, status="failed", errors=["Unknown dataset format"])
            return
        
        # Process examples in batches
        processed_examples = []
        quality_scores = []
        stats = {
            'total': len(examples),
            'kept': 0,
            'fixed': 0,
            'removed': 0,
            'error': 0
        }
        
        # Process examples in batches
        for start_idx in range(0, len(examples), batch_size):
            batch = examples[start_idx:start_idx + batch_size]
            
            # Create evaluation tasks
            eval_tasks = []
            for i, example in enumerate(batch):
                global_idx = start_idx + i
                eval_tasks.append(controller.evaluate_example_async(example, format_type, global_idx))
            
            # Execute evaluation tasks
            evaluations = {}
            for task in asyncio.as_completed(eval_tasks):
                example, evaluation = await task
                if evaluation:
                    evaluations[id(example)] = evaluation
            
            # Process evaluations
            for example in batch:
                evaluation = evaluations.get(id(example))
                
                if not evaluation:
                    # Could not evaluate, keep original
                    processed_examples.append(example)
                    stats['error'] += 1
                    continue
                
                # Track quality score
                score = float(evaluation.get('score', 0))
                quality_scores.append(score)
                
                # Decision based on score and recommendation
                if score >= quality_threshold:
                    # Quality is good, keep example
                    processed_examples.append(example)
                    stats['kept'] += 1
                else:
                    recommendation = evaluation.get('keep_or_fix', 'fix')
                    
                    if recommendation == 'remove' and auto_remove:
                        # Remove example
                        stats['removed'] += 1
                    elif recommendation == 'fix' or (recommendation == 'remove' and not auto_remove):
                        if auto_fix:
                            # Automatically fix the example
                            fixed_example = await controller.fix_example_async(example, format_type, evaluation, start_idx + batch.index(example))
                            if fixed_example and fixed_example[1]:
                                processed_examples.append(fixed_example[1])
                                stats['fixed'] += 1
                            else:
                                # Fix failed, keep original
                                processed_examples.append(example)
                                stats['error'] += 1
                        else:
                            # Keep original example without fixing
                            processed_examples.append(example)
                            stats['kept'] += 1
                    else:
                        # Keep example despite low score
                        processed_examples.append(example)
                        stats['kept'] += 1
            
            # Update job progress
            db.update_quality_job(
                job_id,
                examples_processed=start_idx + len(batch),
                examples_kept=stats['kept'],
                examples_fixed=stats['fixed'],
                examples_removed=stats['removed']
            )
        
        # Create output file path
        output_file_path = f"{os.path.splitext(file_path)[0]}_improved.jsonl"
        
        # Save processed examples
        success = controller.save_examples(processed_examples, output_file_path)
        
        if success:
            # Calculate average quality score
            avg_score = sum(quality_scores) / len(quality_scores) if quality_scores else 0
            
            # Create new dataset record for improved dataset
            improved_dataset_id = db.create_dataset(
                name=f"{dataset['name']}_improved",
                domain=dataset['domain'],
                subdomain=dataset.get('subdomain'),
                format=dataset['format'],
                file_path=output_file_path,
                example_count=len(processed_examples),
                description=f"Improved version of {dataset['name']}",
                metadata={
                    "original_dataset_id": dataset_id,
                    "quality_job_id": job_id,
                    "avg_quality_score": avg_score,
                    "quality_threshold": quality_threshold,
                    "auto_fix": auto_fix,
                    "auto_remove": auto_remove
                }
            )
            
            # Add examples to database
            db.add_examples(improved_dataset_id, processed_examples, quality_scores)
            
            # Update job status
            db.update_quality_job(
                job_id,
                status="completed",
                examples_processed=len(examples),
                examples_total=len(examples),
                examples_kept=stats['kept'],
                examples_fixed=stats['fixed'],
                examples_removed=stats['removed'],
                avg_quality_score=avg_score,
                result_file_path=output_file_path,
                completed_at=datetime.now().isoformat()
            )
            
            # Update dataset metadata
            db.update_dataset(dataset_id, metadata={"has_improved_version": True, "improved_dataset_id": improved_dataset_id})
        else:
            # Update job status to failed
            db.update_quality_job(
                job_id,
                status="failed",
                completed_at=datetime.now().isoformat(),
                errors=["Failed to save processed examples"]
            )
    except Exception as e:
        print(f"Error in quality job {job_id}: {e}")
        # Update job status to failed
        db.update_quality_job(
            job_id,
            status="failed",
            completed_at=datetime.now().isoformat(),
            errors=[str(e)]
        )

def create_quality_job(dataset_id: int, parameters: Dict[str, Any]) -> int:
    """Create a new quality control job"""
    # Get dataset information
    dataset = db.get_dataset(dataset_id)
    if not dataset:
        raise ValueError(f"Dataset {dataset_id} not found")
    
    # Create job record
    job_id = db.create_quality_job(
        dataset_id=dataset_id,
        examples_total=dataset['example_count'],
        parameters=parameters
    )
    
    return job_id

def get_job_status(job_id: int) -> Dict[str, Any]:
    """Get the status of a quality control job"""
    job = db.get_quality_job(job_id)
    if not job:
        return {"error": "Job not found"}
    
    return {
        "id": job["id"],
        "dataset_id": job["dataset_id"],
        "status": job["status"],
        "examples_processed": job["examples_processed"],
        "examples_total": job["examples_total"],
        "examples_kept": job["examples_kept"],
        "examples_fixed": job["examples_fixed"],
        "examples_removed": job["examples_removed"],
        "avg_quality_score": job["avg_quality_score"],
        "started_at": job["started_at"],
        "completed_at": job["completed_at"],
        "result_file_path": job["result_file_path"],
        "errors": job["errors"] if "errors" in job else None
    }

def cancel_job(job_id: int) -> bool:
    """Cancel a running quality control job"""
    job = db.get_quality_job(job_id)
    if not job or job["status"] not in ["pending", "running"]:
        return False
    
    # Update job status to cancelled
    db.update_quality_job(
        job_id,
        status="cancelled",
        completed_at=datetime.now().isoformat()
    )
    
    return True