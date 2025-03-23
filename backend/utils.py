"""
Utility functions for LLM Dataset Creator
"""

import os
import json
import random
import glob
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime

def scan_for_datasets() -> List[str]:
    """Scan data directory for JSONL files that might be datasets"""
    current_dir = Path.cwd()
    
    # Try the data/datasets directory first
    datasets_dir = current_dir / "data" / "datasets"
    if datasets_dir.exists():
        # Find all JSONL files in the datasets directory
        jsonl_files = []
        for path in datasets_dir.glob('**/*.jsonl'):
            if path.is_file():
                jsonl_files.append(str(path))
        
        return jsonl_files
    
    # Fallback to scanning the current directory and subdirectories
    jsonl_files = []
    for path in current_dir.glob('**/*.jsonl'):
        if path.is_file():
            jsonl_files.append(str(path))
    
    return jsonl_files

def load_jsonl(file_path: str) -> List[Dict[str, Any]]:
    """Load data from a JSONL file"""
    data = []
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    item = json.loads(line)
                    data.append(item)
    except Exception as e:
        print(f"Error loading JSONL file {file_path}: {e}")
    
    return data

def save_jsonl(data: List[Dict[str, Any]], file_path: str) -> bool:
    """Save data to a JSONL file"""
    try:
        # Ensure directory exists
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        
        with open(file_path, 'w', encoding='utf-8') as f:
            for item in data:
                f.write(json.dumps(item, ensure_ascii=False) + '\n')
        
        return True
    except Exception as e:
        print(f"Error saving JSONL file {file_path}: {e}")
        return False

def detect_format_type(file_path: str) -> str:
    """Detect the format type of a dataset (chat or instruction)"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            # Read the first line
            first_line = f.readline().strip()
            if not first_line:
                return "unknown"
            
            # Parse JSON
            example = json.loads(first_line)
            
            # Determine format type
            if "messages" in example:
                return "chat"
            elif "instruction" in example and "output" in example:
                return "instruction"
            else:
                return "unknown"
    except Exception as e:
        print(f"Error detecting format type: {e}")
        return "unknown"

def count_examples(file_path: str) -> int:
    """Count the number of examples in a JSONL file"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            count = sum(1 for line in f if line.strip())
        return count
    except Exception as e:
        print(f"Error counting examples: {e}")
        return 0

def get_example_preview(file_path: str, count: int = 5) -> List[Dict[str, Any]]:
    """Get a preview of examples from a dataset"""
    try:
        examples = []
        with open(file_path, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                if i >= count:
                    break
                
                line = line.strip()
                if line:
                    example = json.loads(line)
                    examples.append(example)
        
        return examples
    except Exception as e:
        print(f"Error getting example preview: {e}")
        return []

def generate_dataset_name(domain: str, subdomain: Optional[str] = None) -> str:
    """Generate a unique name for a dataset"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    if subdomain:
        return f"{domain}_{subdomain}_{timestamp}"
    else:
        return f"{domain}_{timestamp}"

def format_example_for_display(example: Dict[str, Any], format_type: str) -> Dict[str, Any]:
    """Format an example for display in the UI"""
    if format_type == 'chat':
        # For chat format, extract messages and metadata
        messages = example.get('messages', [])
        metadata = example.get('metadata', {})
        
        # Format each message
        formatted_messages = []
        for msg in messages:
            formatted_messages.append({
                'role': msg.get('role', ''),
                'content': msg.get('content', '')
            })
        
        return {
            'type': 'chat',
            'messages': formatted_messages,
            'metadata': metadata
        }
    
    elif format_type == 'instruction':
        # For instruction format, extract instruction, output, and metadata
        instruction = example.get('instruction', '')
        output = example.get('output', '')
        metadata = example.get('metadata', {})
        
        return {
            'type': 'instruction',
            'instruction': instruction,
            'output': output,
            'metadata': metadata
        }
    
    else:
        # Unknown format, return as is
        return {
            'type': 'unknown',
            'content': example
        }

def merge_datasets(file_paths: List[str], output_path: str) -> Tuple[bool, int]:
    """Merge multiple datasets into one"""
    try:
        all_examples = []
        
        # Load examples from each dataset
        for file_path in file_paths:
            examples = load_jsonl(file_path)
            all_examples.extend(examples)
        
        # Save merged dataset
        success = save_jsonl(all_examples, output_path)
        
        return success, len(all_examples)
    except Exception as e:
        print(f"Error merging datasets: {e}")
        return False, 0

def filter_dataset(
    file_path: str,
    output_path: str,
    domain: Optional[str] = None,
    subdomain: Optional[str] = None,
    min_quality: Optional[float] = None,
    max_examples: Optional[int] = None
) -> Tuple[bool, int]:
    """Filter a dataset based on criteria"""
    try:
        examples = load_jsonl(file_path)
        filtered_examples = []
        
        for example in examples:
            # Get metadata
            metadata = example.get('metadata', {})
            
            # Apply domain filter
            if domain and metadata.get('domain') != domain:
                continue
            
            # Apply subdomain filter
            if subdomain and metadata.get('subdomain') != subdomain:
                continue
            
            # Apply quality filter
            if min_quality is not None and metadata.get('quality_score', 0) < min_quality:
                continue
            
            # Add example to filtered list
            filtered_examples.append(example)
            
            # Check max examples limit
            if max_examples and len(filtered_examples) >= max_examples:
                break
        
        # Save filtered dataset
        success = save_jsonl(filtered_examples, output_path)
        
        return success, len(filtered_examples)
    except Exception as e:
        print(f"Error filtering dataset: {e}")
        return False, 0

def sample_dataset(file_path: str, output_path: str, count: int) -> Tuple[bool, int]:
    """Create a random sample of a dataset"""
    try:
        examples = load_jsonl(file_path)
        
        # If requesting more examples than available, use all examples
        if count >= len(examples):
            sampled_examples = examples
        else:
            sampled_examples = random.sample(examples, count)
        
        # Save sampled dataset
        success = save_jsonl(sampled_examples, output_path)
        
        return success, len(sampled_examples)
    except Exception as e:
        print(f"Error sampling dataset: {e}")
        return False, 0

def convert_format(file_path: str, output_path: str, from_format: str, to_format: str) -> Tuple[bool, int]:
    """Convert dataset between chat and instruction formats"""
    try:
        examples = load_jsonl(file_path)
        converted_examples = []
        
        for example in examples:
            # Convert from chat to instruction
            if from_format == 'chat' and to_format == 'instruction':
                messages = example.get('messages', [])
                metadata = example.get('metadata', {})
                
                # Need at least a user message and assistant response
                if len(messages) < 2:
                    continue
                
                # Find first user and assistant messages
                user_message = None
                assistant_message = None
                
                for msg in messages:
                    if msg.get('role') == 'user' and user_message is None:
                        user_message = msg.get('content', '')
                    elif msg.get('role') == 'assistant' and assistant_message is None:
                        assistant_message = msg.get('content', '')
                
                if user_message and assistant_message:
                    converted_examples.append({
                        'instruction': user_message,
                        'output': assistant_message,
                        'metadata': metadata
                    })
            
            # Convert from instruction to chat
            elif from_format == 'instruction' and to_format == 'chat':
                instruction = example.get('instruction', '')
                output = example.get('output', '')
                metadata = example.get('metadata', {})
                
                if instruction and output:
                    converted_examples.append({
                        'messages': [
                            {'role': 'user', 'content': instruction},
                            {'role': 'assistant', 'content': output}
                        ],
                        'metadata': metadata
                    })
        
        # Save converted dataset
        success = save_jsonl(converted_examples, output_path)
        
        return success, len(converted_examples)
    except Exception as e:
        print(f"Error converting dataset format: {e}")
        return False, 0

def export_to_csv(file_path: str, output_path: str, format_type: str) -> bool:
    """Export dataset to CSV format for easier viewing"""
    try:
        import pandas as pd
        
        examples = load_jsonl(file_path)
        csv_data = []
        
        for example in examples:
            metadata = example.get('metadata', {})
            
            if format_type == 'instruction':
                instruction = example.get('instruction', '')
                output = example.get('output', '')
                
                csv_row = {
                    'domain': metadata.get('domain', ''),
                    'subdomain': metadata.get('subdomain', ''),
                    'complexity_level': metadata.get('complexity_level', ''),
                    'communication_style': metadata.get('communication_style', ''),
                    'emotional_tone': metadata.get('emotional_tone', ''),
                }
                
                # Add all available metadata
                for key, value in metadata.items():
                    if key not in csv_row:
                        csv_row[key] = value
                        
                # Add content
                csv_row.update({
                    'instruction': instruction[:100] + '...' if len(instruction) > 100 else instruction,
                    'output': output[:100] + '...' if len(output) > 100 else output
                })
                
                csv_data.append(csv_row)
            else:  # chat format
                messages = example.get('messages', [])
                
                # Extract first user message and assistant response
                first_message = ""
                first_response = ""
                
                for msg in messages:
                    if msg.get('role') == 'user' and not first_message:
                        content = msg.get('content', '')
                        first_message = content[:100] + '...' if len(content) > 100 else content
                    elif msg.get('role') == 'assistant' and not first_response:
                        content = msg.get('content', '')
                        first_response = content[:100] + '...' if len(content) > 100 else content
                
                csv_row = {
                    'domain': metadata.get('domain', ''),
                    'subdomain': metadata.get('subdomain', ''),
                    'complexity_level': metadata.get('complexity_level', ''),
                    'communication_style': metadata.get('communication_style', ''),
                    'emotional_tone': metadata.get('emotional_tone', ''),
                }
                
                # Add all available metadata
                for key, value in metadata.items():
                    if key not in csv_row:
                        csv_row[key] = value
                        
                # Add content
                csv_row.update({
                    'first_message': first_message,
                    'first_response': first_response,
                    'messages_count': len(messages)
                })
                
                csv_data.append(csv_row)
        
        # Create DataFrame and save to CSV
        df = pd.DataFrame(csv_data)
        df.to_csv(output_path, index=False, encoding='utf-8')
        
        return True
    except Exception as e:
        print(f"Error exporting to CSV: {e}")
        return False

def get_disk_usage(path):
    """Get disk usage statistics for a given path"""
    try:
        import shutil
        total, used, free = shutil.disk_usage(path)
        return total, used, free
    except Exception:
        # Fallback if shutil doesn't work
        return 100000000000, 50000000000, 50000000000  # 100GB total, 50GB used, 50GB free

def get_cpu_usage():
    """Get CPU usage percentage"""
    try:
        import psutil
        return psutil.cpu_percent(interval=0.1)
    except ImportError:
        # Fallback if psutil is not available
        return 25.0  # Return a default value

def get_memory_usage():
    """Get memory usage statistics"""
    try:
        import psutil
        memory = psutil.virtual_memory()
        return {
            "total": memory.total,
            "available": memory.available,
            "used": memory.used,
            "percent": memory.percent
        }
    except ImportError:
        # Fallback if psutil is not available
        total = 8000000000  # 8GB
        used = 4000000000   # 4GB
        return {
            "total": total,
            "available": total - used,
            "used": used,
            "percent": 50.0
        }

def get_cache_size():
    """Estimate cache size by scanning temporary files"""
    try:
        import os
        from pathlib import Path
        
        # Assume cache is in a cache directory
        cache_dir = Path("/tmp/llm_dataset_creator_cache")
        
        if not cache_dir.exists():
            return 0
            
        total_size = 0
        for path in cache_dir.glob('**/*'):
            if path.is_file():
                total_size += path.stat().st_size
                
        return total_size
    except Exception:
        # Fallback if error occurs
        return 1000000  # 1MB