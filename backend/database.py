"""
Database models and functions for LLM Dataset Creator
Uses SQLite for simplicity, can be replaced with PostgreSQL for production
"""

import os
import json
import sqlite3
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
from pathlib import Path

# Database setup
DATABASE_FILE = "llm_dataset_creator.db"

def get_db_connection():
    """Get a database connection"""
    conn = sqlite3.connect(DATABASE_FILE)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    """Initialize the database schema"""
    conn = get_db_connection()
    cursor = conn.cursor()
    
    # Create datasets table
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS datasets (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT NOT NULL,
        description TEXT,
        domain TEXT NOT NULL,
        subdomain TEXT,
        format TEXT NOT NULL,
        example_count INTEGER NOT NULL,
        created_at TEXT NOT NULL,
        updated_at TEXT NOT NULL,
        file_path TEXT NOT NULL,
        metadata TEXT
    )
    ''')
    
    # Create generation_jobs table
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS generation_jobs (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        dataset_id INTEGER,
        status TEXT NOT NULL,
        parameters TEXT NOT NULL,
        started_at TEXT NOT NULL,
        completed_at TEXT,
        examples_generated INTEGER DEFAULT 0,
        examples_requested INTEGER NOT NULL,
        errors TEXT,
        FOREIGN KEY (dataset_id) REFERENCES datasets (id)
    )
    ''')
    
    # Create quality_jobs table
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS quality_jobs (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        dataset_id INTEGER NOT NULL,
        status TEXT NOT NULL,
        parameters TEXT NOT NULL,
        started_at TEXT NOT NULL,
        completed_at TEXT,
        examples_processed INTEGER DEFAULT 0,
        examples_total INTEGER NOT NULL,
        examples_kept INTEGER DEFAULT 0,
        examples_fixed INTEGER DEFAULT 0,
        examples_removed INTEGER DEFAULT 0,
        avg_quality_score REAL,
        result_file_path TEXT,
        errors TEXT,
        FOREIGN KEY (dataset_id) REFERENCES datasets (id)
    )
    ''')
    
    # Create examples table (optional for detailed tracking)
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS examples (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        dataset_id INTEGER NOT NULL,
        example_index INTEGER NOT NULL,
        content TEXT NOT NULL,
        quality_score REAL,
        status TEXT,
        metadata TEXT,
        FOREIGN KEY (dataset_id) REFERENCES datasets (id)
    )
    ''')
    
    conn.commit()
    conn.close()

# Dataset Operations
def create_dataset(
    name: str,
    domain: str,
    format: str,
    file_path: str,
    description: str = "",
    subdomain: Optional[str] = None,
    example_count: int = 0,
    metadata: Optional[Dict[str, Any]] = None
) -> int:
    """Create a new dataset record"""
    conn = get_db_connection()
    cursor = conn.cursor()
    
    now = datetime.now().isoformat()
    
    cursor.execute('''
    INSERT INTO datasets (name, description, domain, subdomain, format, example_count, 
                         created_at, updated_at, file_path, metadata)
    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    ''', (
        name, description, domain, subdomain, format, example_count,
        now, now, file_path, json.dumps(metadata or {})
    ))
    
    dataset_id = cursor.lastrowid
    conn.commit()
    conn.close()
    
    return dataset_id

def update_dataset(
    dataset_id: int,
    name: Optional[str] = None,
    description: Optional[str] = None,
    example_count: Optional[int] = None,
    file_path: Optional[str] = None,
    metadata: Optional[Dict[str, Any]] = None
) -> bool:
    """Update a dataset record"""
    conn = get_db_connection()
    cursor = conn.cursor()
    
    # Get current dataset data
    cursor.execute('SELECT * FROM datasets WHERE id = ?', (dataset_id,))
    dataset = cursor.fetchone()
    
    if not dataset:
        conn.close()
        return False
    
    # Prepare update fields
    updated_at = datetime.now().isoformat()
    update_fields = []
    params = []
    
    if name is not None:
        update_fields.append('name = ?')
        params.append(name)
    
    if description is not None:
        update_fields.append('description = ?')
        params.append(description)
    
    if example_count is not None:
        update_fields.append('example_count = ?')
        params.append(example_count)
    
    if file_path is not None:
        update_fields.append('file_path = ?')
        params.append(file_path)
    
    if metadata is not None:
        current_metadata = json.loads(dataset['metadata'])
        current_metadata.update(metadata)
        update_fields.append('metadata = ?')
        params.append(json.dumps(current_metadata))
    
    update_fields.append('updated_at = ?')
    params.append(updated_at)
    
    # Build and execute the update query
    query = f'UPDATE datasets SET {", ".join(update_fields)} WHERE id = ?'
    params.append(dataset_id)
    
    cursor.execute(query, params)
    conn.commit()
    conn.close()
    
    return True

def get_dataset(dataset_id: int) -> Optional[Dict[str, Any]]:
    """Get a dataset by ID"""
    conn = get_db_connection()
    cursor = conn.cursor()
    
    cursor.execute('SELECT * FROM datasets WHERE id = ?', (dataset_id,))
    dataset = cursor.fetchone()
    
    conn.close()
    
    if dataset:
        result = dict(dataset)
        result['metadata'] = json.loads(result['metadata'])
        return result
    
    return None

def get_datasets(
    domain: Optional[str] = None,
    format: Optional[str] = None,
    sort_by: str = "created_at",
    sort_order: str = "DESC",
    limit: int = 100,
    file_path: Optional[str] = None
) -> List[Dict[str, Any]]:
    """Get datasets with optional filtering"""
    conn = get_db_connection()
    cursor = conn.cursor()
    
    query = 'SELECT * FROM datasets'
    params = []
    
    # Add filters
    filters = []
    if domain:
        filters.append('domain = ?')
        params.append(domain)
    
    if format:
        filters.append('format = ?')
        params.append(format)
        
    if file_path:
        filters.append('file_path = ?')
        params.append(file_path)
    
    if filters:
        query += ' WHERE ' + ' AND '.join(filters)
    
    # Add sorting
    valid_sort_fields = ['created_at', 'updated_at', 'name', 'example_count']
    if sort_by not in valid_sort_fields:
        sort_by = 'created_at'
    
    valid_sort_orders = ['ASC', 'DESC']
    if sort_order not in valid_sort_orders:
        sort_order = 'DESC'
    
    query += f' ORDER BY {sort_by} {sort_order}'
    
    # Add limit
    query += ' LIMIT ?'
    params.append(limit)
    
    cursor.execute(query, params)
    datasets = cursor.fetchall()
    
    conn.close()
    
    # Convert rows to dictionaries and parse metadata
    result = []
    for dataset in datasets:
        item = dict(dataset)
        item['metadata'] = json.loads(item['metadata'])
        result.append(item)
    
    return result

def delete_dataset(dataset_id: int) -> bool:
    """Delete a dataset and associated jobs"""
    conn = get_db_connection()
    cursor = conn.cursor()
    
    try:
        # Get dataset file path to delete the file
        cursor.execute('SELECT file_path FROM datasets WHERE id = ?', (dataset_id,))
        dataset = cursor.fetchone()
        
        if dataset and dataset['file_path']:
            file_path = dataset['file_path']
            if os.path.exists(file_path):
                os.remove(file_path)
        
        # Delete associated data
        cursor.execute('DELETE FROM examples WHERE dataset_id = ?', (dataset_id,))
        cursor.execute('DELETE FROM quality_jobs WHERE dataset_id = ?', (dataset_id,))
        cursor.execute('DELETE FROM generation_jobs WHERE dataset_id = ?', (dataset_id,))
        
        # Delete the dataset record
        cursor.execute('DELETE FROM datasets WHERE id = ?', (dataset_id,))
        
        conn.commit()
        conn.close()
        return True
    except Exception as e:
        print(f"Error deleting dataset: {e}")
        conn.rollback()
        conn.close()
        return False

# Generation Job Operations
def create_generation_job(
    examples_requested: int,
    parameters: Dict[str, Any],
    dataset_id: Optional[int] = None
) -> int:
    """Create a new generation job"""
    conn = get_db_connection()
    cursor = conn.cursor()
    
    now = datetime.now().isoformat()
    
    cursor.execute('''
    INSERT INTO generation_jobs (
        dataset_id, status, parameters, started_at, examples_requested
    ) VALUES (?, ?, ?, ?, ?)
    ''', (
        dataset_id, 'pending', json.dumps(parameters), now, examples_requested
    ))
    
    job_id = cursor.lastrowid
    conn.commit()
    conn.close()
    
    return job_id

def update_generation_job(
    job_id: int,
    status: Optional[str] = None,
    examples_generated: Optional[int] = None,
    completed_at: Optional[str] = None,
    dataset_id: Optional[int] = None,
    errors: Optional[List[str]] = None
) -> bool:
    """Update a generation job"""
    conn = get_db_connection()
    cursor = conn.cursor()
    
    update_fields = []
    params = []
    
    if status is not None:
        update_fields.append('status = ?')
        params.append(status)
    
    if examples_generated is not None:
        update_fields.append('examples_generated = ?')
        params.append(examples_generated)
    
    if completed_at is not None:
        update_fields.append('completed_at = ?')
        params.append(completed_at)
    
    if dataset_id is not None:
        update_fields.append('dataset_id = ?')
        params.append(dataset_id)
    
    if errors is not None:
        update_fields.append('errors = ?')
        params.append(json.dumps(errors))
    
    if not update_fields:
        conn.close()
        return False
    
    query = f'UPDATE generation_jobs SET {", ".join(update_fields)} WHERE id = ?'
    params.append(job_id)
    
    cursor.execute(query, params)
    conn.commit()
    conn.close()
    
    return True

def get_generation_job(job_id: int) -> Optional[Dict[str, Any]]:
    """Get a generation job by ID"""
    conn = get_db_connection()
    cursor = conn.cursor()
    
    cursor.execute('SELECT * FROM generation_jobs WHERE id = ?', (job_id,))
    job = cursor.fetchone()
    
    conn.close()
    
    if job:
        result = dict(job)
        result['parameters'] = json.loads(result['parameters'])
        if result['errors']:
            result['errors'] = json.loads(result['errors'])
        return result
    
    return None

def get_generation_jobs(
    dataset_id: Optional[int] = None,
    status: Optional[str] = None,
    limit: int = 20
) -> List[Dict[str, Any]]:
    """Get generation jobs with optional filtering"""
    conn = get_db_connection()
    cursor = conn.cursor()
    
    query = 'SELECT * FROM generation_jobs'
    params = []
    
    # Add filters
    filters = ['status != "deleted"']  # Всегда исключаем удаленные задачи
    
    if dataset_id is not None:
        filters.append('dataset_id = ?')
        params.append(dataset_id)
    
    if status:
        filters.append('status = ?')
        params.append(status)
    
    query += ' WHERE ' + ' AND '.join(filters)
    
    # Add sorting and limit
    query += ' ORDER BY started_at DESC LIMIT ?'
    params.append(limit)
    
    cursor.execute(query, params)
    jobs = cursor.fetchall()
    
    conn.close()
    
    # Convert rows to dictionaries and parse JSON
    result = []
    for job in jobs:
        item = dict(job)
        item['parameters'] = json.loads(item['parameters'])
        if item['errors']:
            item['errors'] = json.loads(item['errors'])
        result.append(item)
    
    return result

# Quality Control Job Operations
def create_quality_job(
    dataset_id: int,
    examples_total: int,
    parameters: Dict[str, Any]
) -> int:
    """Create a new quality control job"""
    conn = get_db_connection()
    cursor = conn.cursor()
    
    now = datetime.now().isoformat()
    
    cursor.execute('''
    INSERT INTO quality_jobs (
        dataset_id, status, parameters, started_at, examples_total
    ) VALUES (?, ?, ?, ?, ?)
    ''', (
        dataset_id, 'pending', json.dumps(parameters), now, examples_total
    ))
    
    job_id = cursor.lastrowid
    conn.commit()
    conn.close()
    
    return job_id

def update_quality_job(
    job_id: int,
    status: Optional[str] = None,
    examples_processed: Optional[int] = None,
    examples_kept: Optional[int] = None,
    examples_fixed: Optional[int] = None,
    examples_removed: Optional[int] = None,
    examples_total: Optional[int] = None,
    avg_quality_score: Optional[float] = None,
    result_file_path: Optional[str] = None,
    completed_at: Optional[str] = None,
    errors: Optional[List[str]] = None
) -> bool:
    """Update a quality control job"""
    conn = get_db_connection()
    cursor = conn.cursor()
    
    update_fields = []
    params = []
    
    if status is not None:
        update_fields.append('status = ?')
        params.append(status)
    
    if examples_processed is not None:
        update_fields.append('examples_processed = ?')
        params.append(examples_processed)
    
    if examples_kept is not None:
        update_fields.append('examples_kept = ?')
        params.append(examples_kept)
    
    if examples_fixed is not None:
        update_fields.append('examples_fixed = ?')
        params.append(examples_fixed)
    
    if examples_removed is not None:
        update_fields.append('examples_removed = ?')
        params.append(examples_removed)
        
    if examples_total is not None:
        update_fields.append('examples_total = ?')
        params.append(examples_total)
    
    if avg_quality_score is not None:
        update_fields.append('avg_quality_score = ?')
        params.append(avg_quality_score)
    
    if result_file_path is not None:
        update_fields.append('result_file_path = ?')
        params.append(result_file_path)
    
    if completed_at is not None:
        update_fields.append('completed_at = ?')
        params.append(completed_at)
    
    if errors is not None:
        update_fields.append('errors = ?')
        params.append(json.dumps(errors))
    
    if not update_fields:
        conn.close()
        return False
    
    query = f'UPDATE quality_jobs SET {", ".join(update_fields)} WHERE id = ?'
    params.append(job_id)
    
    cursor.execute(query, params)
    conn.commit()
    conn.close()
    
    return True

def get_quality_job(job_id: int) -> Optional[Dict[str, Any]]:
    """Get a quality control job by ID"""
    conn = get_db_connection()
    cursor = conn.cursor()
    
    cursor.execute('SELECT * FROM quality_jobs WHERE id = ?', (job_id,))
    job = cursor.fetchone()
    
    conn.close()
    
    if job:
        result = dict(job)
        result['parameters'] = json.loads(result['parameters'])
        if result['errors']:
            result['errors'] = json.loads(result['errors'])
        return result
    
    return None

def get_quality_jobs(
    dataset_id: Optional[int] = None,
    status: Optional[str] = None,
    limit: int = 20
) -> List[Dict[str, Any]]:
    """Get quality control jobs with optional filtering"""
    conn = get_db_connection()
    cursor = conn.cursor()
    
    query = 'SELECT * FROM quality_jobs'
    params = []
    
    # Add filters
    filters = ['status != "deleted"']  # Всегда исключаем удаленные задачи
    
    if dataset_id is not None:
        filters.append('dataset_id = ?')
        params.append(dataset_id)
    
    if status:
        filters.append('status = ?')
        params.append(status)
    
    query += ' WHERE ' + ' AND '.join(filters)
    
    # Add sorting and limit
    query += ' ORDER BY started_at DESC LIMIT ?'
    params.append(limit)
    
    cursor.execute(query, params)
    jobs = cursor.fetchall()
    
    conn.close()
    
    # Convert rows to dictionaries and parse JSON
    result = []
    for job in jobs:
        item = dict(job)
        item['parameters'] = json.loads(item['parameters'])
        if item['errors']:
            item['errors'] = json.loads(item['errors'])
        result.append(item)
    
    return result

# Example Operations
def add_examples(
    dataset_id: int,
    examples: List[Dict[str, Any]],
    quality_scores: Optional[List[float]] = None
) -> Tuple[int, List[int]]:
    """Add multiple examples to the database"""
    conn = get_db_connection()
    cursor = conn.cursor()
    
    example_ids = []
    
    try:
        for i, example in enumerate(examples):
            quality_score = None
            if quality_scores and i < len(quality_scores):
                quality_score = quality_scores[i]
            
            cursor.execute('''
            INSERT INTO examples (
                dataset_id, example_index, content, quality_score, status, metadata
            ) VALUES (?, ?, ?, ?, ?, ?)
            ''', (
                dataset_id,
                i,
                json.dumps(example),
                quality_score,
                'original',
                json.dumps(example.get('metadata', {}))
            ))
            
            example_ids.append(cursor.lastrowid)
        
        conn.commit()
    except Exception as e:
        print(f"Error adding examples: {e}")
        conn.rollback()
        example_ids = []
    finally:
        conn.close()
    
    return len(example_ids), example_ids

def update_example(
    example_id: int,
    content: Optional[Dict[str, Any]] = None,
    quality_score: Optional[float] = None,
    status: Optional[str] = None
) -> bool:
    """Update an example"""
    conn = None
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        
        # Get current example data
        print(f"Getting example data for ID {example_id}")
        cursor.execute('SELECT * FROM examples WHERE id = ?', (example_id,))
        example = cursor.fetchone()
        
        if not example:
            print(f"Example {example_id} not found")
            raise Exception(f"Example {example_id} not found")
        
        # Get dataset file path
        print(f"Getting dataset info for ID {example['dataset_id']}")
        cursor.execute('SELECT file_path FROM datasets WHERE id = ?', (example['dataset_id'],))
        dataset = cursor.fetchone()
        
        if not dataset:
            print(f"Dataset {example['dataset_id']} not found")
            raise Exception(f"Dataset {example['dataset_id']} not found")
        
        update_fields = []
        params = []
        
        if content is not None:
            print(f"Updating content for example {example_id}")
            update_fields.append('content = ?')
            params.append(json.dumps(content))
        
        if quality_score is not None:
            update_fields.append('quality_score = ?')
            params.append(quality_score)
        
        if status is not None:
            update_fields.append('status = ?')
            params.append(status)
        
        if not update_fields:
            print("No fields to update")
            raise Exception("No fields to update")
        
        query = f'UPDATE examples SET {", ".join(update_fields)} WHERE id = ?'
        params.append(example_id)
        
        print(f"Executing update query: {query}")
        cursor.execute(query, params)
        
        # Update the JSONL file
        if content is not None and dataset['file_path']:
            # Use relative path from backend directory
            file_path = dataset['file_path']
            print(f"Updating JSONL file at: {file_path}")
            
            if not os.path.exists(file_path):
                print(f"File not found: {file_path}")
                raise Exception(f"JSONL file not found at {file_path}")
            
            # Read all examples
            print("Reading all examples from file")
            examples = []
            with open(file_path, 'r') as f:
                for line in f:
                    examples.append(json.loads(line))
            
            # Update the example at the correct index
            if 0 <= example['example_index'] < len(examples):
                print(f"Updating example at index {example['example_index']}")
                examples[example['example_index']] = content['content']
                
                # Write back all examples
                print("Writing updated examples back to file")
                with open(file_path, 'w') as f:
                    for ex in examples:
                        f.write(json.dumps(ex) + '\n')
                print("File update completed successfully")
            else:
                print(f"Invalid example index: {example['example_index']}")
                raise Exception(f"Invalid example index: {example['example_index']}")
        
        print("Committing database changes")
        conn.commit()
        print(f"Example {example_id} updated successfully")
        return True
        
    except Exception as e:
        print(f"Error in update_example: {e}")
        if conn:
            conn.rollback()
        raise Exception(str(e))
    finally:
        if conn:
            conn.close()

def get_example(example_id: int) -> Optional[Dict[str, Any]]:
    """Get an example by ID"""
    conn = get_db_connection()
    cursor = conn.cursor()
    
    cursor.execute('SELECT * FROM examples WHERE id = ?', (example_id,))
    example = cursor.fetchone()
    
    conn.close()
    
    if example:
        result = dict(example)
        result['content'] = json.loads(result['content'])
        result['metadata'] = json.loads(result['metadata'])
        return result
    
    return None

def get_examples(
    dataset_id: int,
    status: Optional[str] = None,
    min_quality: Optional[float] = None,
    max_quality: Optional[float] = None,
    offset: int = 0,
    limit: int = 100
) -> List[Dict[str, Any]]:
    """Get examples with optional filtering"""
    conn = get_db_connection()
    cursor = conn.cursor()
    
    query = 'SELECT * FROM examples WHERE dataset_id = ?'
    params = [dataset_id]
    
    # Add filters
    if status:
        query += ' AND status = ?'
        params.append(status)
    
    if min_quality is not None:
        query += ' AND quality_score >= ?'
        params.append(min_quality)
    
    if max_quality is not None:
        query += ' AND quality_score <= ?'
        params.append(max_quality)
    
    # Add sorting, offset, and limit
    query += ' ORDER BY example_index ASC LIMIT ? OFFSET ?'
    params.extend([limit, offset])
    
    cursor.execute(query, params)
    examples = cursor.fetchall()
    
    conn.close()
    
    # Convert rows to dictionaries and parse JSON
    result = []
    for example in examples:
        item = dict(example)
        item['content'] = json.loads(item['content'])
        item['metadata'] = json.loads(item['metadata'])
        result.append(item)
    
    return result

def count_examples(
    dataset_id: int,
    status: Optional[str] = None,
    min_quality: Optional[float] = None,
    max_quality: Optional[float] = None
) -> int:
    """Count examples with optional filtering"""
    conn = get_db_connection()
    cursor = conn.cursor()
    
    query = 'SELECT COUNT(*) as count FROM examples WHERE dataset_id = ?'
    params = [dataset_id]
    
    # Add filters
    if status:
        query += ' AND status = ?'
        params.append(status)
    
    if min_quality is not None:
        query += ' AND quality_score >= ?'
        params.append(min_quality)
    
    if max_quality is not None:
        query += ' AND quality_score <= ?'
        params.append(max_quality)
    
    cursor.execute(query, params)
    result = cursor.fetchone()
    
    conn.close()
    
    return result['count'] if result else 0

def get_examples_stats(dataset_id: int) -> Dict[str, Any]:
    """Get statistics about examples in a dataset"""
    conn = get_db_connection()
    cursor = conn.cursor()
    
    # Get total count
    cursor.execute('SELECT COUNT(*) as count FROM examples WHERE dataset_id = ?', (dataset_id,))
    total = cursor.fetchone()['count']
    
    # Get counts by status
    cursor.execute('''
    SELECT status, COUNT(*) as count 
    FROM examples 
    WHERE dataset_id = ? 
    GROUP BY status
    ''', (dataset_id,))
    status_counts = {row['status']: row['count'] for row in cursor.fetchall()}
    
    # Get quality score statistics
    cursor.execute('''
    SELECT 
        AVG(quality_score) as avg_score,
        MIN(quality_score) as min_score,
        MAX(quality_score) as max_score
    FROM examples 
    WHERE dataset_id = ? AND quality_score IS NOT NULL
    ''', (dataset_id,))
    quality_stats = cursor.fetchone()
    
    conn.close()
    
    return {
        'total': total,
        'status_counts': status_counts,
        'quality_stats': dict(quality_stats) if quality_stats else None
    }

# Ensure data directory exists
def ensure_data_dir():
    """Ensure the data directory exists"""
    os.makedirs('data', exist_ok=True)
    os.makedirs('data/datasets', exist_ok=True)

# Initialize database
def initialize():
    """Initialize the database and ensure directories exist"""
    ensure_data_dir()
    init_db()

if __name__ == "__main__":
    initialize()