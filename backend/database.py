"""
Database models and functions for LLM Dataset Creator
Uses PostgreSQL for production
"""

import os
import json
import psycopg2
from psycopg2.extras import RealDictCursor
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
from pathlib import Path

# Database setup
DB_HOST = os.getenv("POSTGRES_HOST", "localhost")
DB_NAME = os.getenv("POSTGRES_DB", "llm_dataset_creator")
DB_USER = os.getenv("POSTGRES_USER", "postgres")
DB_PASS = os.getenv("POSTGRES_PASSWORD", "postgres")
DB_PORT = os.getenv("POSTGRES_PORT", "5432")

def ensure_data_dir():
    """Ensure data directory exists"""
    data_dir = Path("data")
    data_dir.mkdir(exist_ok=True)
    return data_dir

def get_db_connection():
    """Get a database connection"""
    conn = psycopg2.connect(
        host=DB_HOST,
        database=DB_NAME,
        user=DB_USER,
        password=DB_PASS,
        port=DB_PORT
    )
    return conn

def init_db():
    """Initialize the database schema"""
    conn = get_db_connection()
    cursor = conn.cursor()
    
    # Create datasets table
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS datasets (
        id SERIAL PRIMARY KEY,
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
        id SERIAL PRIMARY KEY,
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
        id SERIAL PRIMARY KEY,
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
        id SERIAL PRIMARY KEY,
        dataset_id INTEGER NOT NULL,
        example_index INTEGER NOT NULL,
        content TEXT NOT NULL,
        quality_score REAL,
        status TEXT,
        metadata TEXT,
        FOREIGN KEY (dataset_id) REFERENCES datasets (id)
    )
    ''')

    # Create templates table
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS templates (
        id SERIAL PRIMARY KEY,
        name TEXT NOT NULL,
        domain TEXT NOT NULL,
        subdomain TEXT,
        content TEXT NOT NULL,
        variables TEXT,
        description TEXT,
        created_at TEXT NOT NULL,
        updated_at TEXT NOT NULL,
        version INTEGER DEFAULT 1
    )
    ''')

    # Create template_versions table
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS template_versions (
        id SERIAL PRIMARY KEY,
        template_id INTEGER NOT NULL,
        version INTEGER NOT NULL,
        name TEXT NOT NULL,
        content TEXT NOT NULL,
        variables TEXT,
        description TEXT,
        created_at TEXT NOT NULL,
        FOREIGN KEY (template_id) REFERENCES templates (id)
    )
    ''')

    # Create webhooks table
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS webhooks (
        id SERIAL PRIMARY KEY,
        url TEXT NOT NULL,
        events TEXT NOT NULL,
        secret TEXT,
        active INTEGER DEFAULT 1,
        created_at TEXT NOT NULL,
        metadata TEXT
    )
    ''')

    # Create webhook_logs table
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS webhook_logs (
        id SERIAL PRIMARY KEY,
        webhook_id INTEGER NOT NULL,
        event TEXT NOT NULL,
        payload TEXT,
        status_code INTEGER,
        response_body TEXT,
        created_at TEXT NOT NULL,
        FOREIGN KEY (webhook_id) REFERENCES webhooks (id)
    )
    ''')

    # Create scheduled_jobs table
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS scheduled_jobs (
        id SERIAL PRIMARY KEY,
        name TEXT NOT NULL,
        cron_expression TEXT NOT NULL,
        task_type TEXT NOT NULL,
        parameters TEXT,
        active INTEGER DEFAULT 1,
        last_run_at TEXT,
        next_run_at TEXT,
        created_at TEXT NOT NULL
    )
    ''')

    # Create prompts table
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS prompts (
        id SERIAL PRIMARY KEY,
        name TEXT NOT NULL,
        description TEXT,
        content TEXT NOT NULL,
        variables TEXT,
        created_at TEXT NOT NULL,
        updated_at TEXT NOT NULL,
        metadata TEXT
    )
    ''')

    # Create reviews table
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS reviews (
        id SERIAL PRIMARY KEY,
        dataset_id INTEGER NOT NULL,
        reviewer_id TEXT NOT NULL,
        status TEXT NOT NULL,
        feedback TEXT,
        created_at TEXT NOT NULL,
        updated_at TEXT NOT NULL,
        FOREIGN KEY (dataset_id) REFERENCES datasets (id)
    )
    ''')

    # Create comments table
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS comments (
        id SERIAL PRIMARY KEY,
        dataset_id INTEGER NOT NULL,
        example_id INTEGER,
        user_id TEXT NOT NULL,
        content TEXT NOT NULL,
        parent_id INTEGER,
        created_at TEXT NOT NULL,
        FOREIGN KEY (dataset_id) REFERENCES datasets (id)
    )
    ''')

    # Create dataset_versions table
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS dataset_versions (
        id SERIAL PRIMARY KEY,
        dataset_id INTEGER NOT NULL,
        version_number INTEGER NOT NULL,
        commit_message TEXT,
        author TEXT,
        file_path TEXT NOT NULL,
        example_count INTEGER NOT NULL,
        created_at TEXT NOT NULL,
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
    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
    RETURNING id
    ''', (
        name, description, domain, subdomain, format, example_count,
        now, now, file_path, json.dumps(metadata or {})
    ))
    
    dataset_id = cursor.fetchone()[0]
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
    cursor = conn.cursor(cursor_factory=RealDictCursor)
    
    # Get current dataset data
    cursor.execute('SELECT * FROM datasets WHERE id = %s', (dataset_id,))
    dataset = cursor.fetchone()
    
    if not dataset:
        conn.close()
        return False
    
    # Prepare update fields
    updated_at = datetime.now().isoformat()
    update_fields = []
    params = []
    
    if name is not None:
        update_fields.append('name = %s')
        params.append(name)
    
    if description is not None:
        update_fields.append('description = %s')
        params.append(description)
    
    if example_count is not None:
        update_fields.append('example_count = %s')
        params.append(example_count)
    
    if file_path is not None:
        update_fields.append('file_path = %s')
        params.append(file_path)
    
    if metadata is not None:
        current_metadata = json.loads(dataset['metadata'])
        current_metadata.update(metadata)
        update_fields.append('metadata = %s')
        params.append(json.dumps(current_metadata))
    
    update_fields.append('updated_at = %s')
    params.append(updated_at)
    
    # Build and execute the update query
    query = f'UPDATE datasets SET {", ".join(update_fields)} WHERE id = %s'
    params.append(dataset_id)
    
    cursor.execute(query, params)
    conn.commit()
    conn.close()
    
    return True

def get_dataset(dataset_id: int) -> Optional[Dict[str, Any]]:
    """Get a dataset by ID"""
    conn = get_db_connection()
    cursor = conn.cursor(cursor_factory=RealDictCursor)
    
    cursor.execute('SELECT * FROM datasets WHERE id = %s', (dataset_id,))
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
    cursor = conn.cursor(cursor_factory=RealDictCursor)
    
    query = 'SELECT * FROM datasets'
    params = []
    
    # Add filters
    filters = []
    if domain:
        filters.append('domain = %s')
        params.append(domain)
    
    if format:
        filters.append('format = %s')
        params.append(format)
        
    if file_path:
        filters.append('file_path = %s')
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
    query += ' LIMIT %s'
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

def get_examples(
    dataset_id: int,
    status: Optional[str] = None,
    min_quality: Optional[float] = None,
    max_quality: Optional[float] = None,
    offset: int = 0,
    limit: int = 100
) -> List[Dict[str, Any]]:
    """
    Get examples for a dataset with optional filtering and pagination.
    Returns a list of examples (dicts).
    """
    conn = get_db_connection()
    cursor = conn.cursor(cursor_factory=RealDictCursor)
    try:
        # Build query with filters
        query = 'SELECT * FROM examples WHERE dataset_id = %s'
        params = [dataset_id]
        
        if status:
            query += ' AND status = %s'
            params.append(status)
        
        if min_quality is not None:
            query += ' AND quality_score >= %s'
            params.append(min_quality)
        
        if max_quality is not None:
            query += ' AND quality_score <= %s'
            params.append(max_quality)
        
        query += ' ORDER BY example_index ASC LIMIT %s OFFSET %s'
        params.extend([limit, offset])
        
        cursor.execute(query, params)
        rows = cursor.fetchall()
        
        examples = []
        for row in rows:
            # content stored as JSON string
            try:
                example = json.loads(row['content'])
            except Exception:
                example = {'content': row['content']}
            
            # Add metadata
            example['id'] = row['id']
            if row.get('quality_score') is not None:
                example['quality_score'] = row['quality_score']
            if row.get('status'):
                example['status'] = row['status']
            if row.get('metadata'):
                try:
                    example['metadata'] = json.loads(row['metadata'])
                except Exception:
                    pass
            
            examples.append(example)
        
        conn.close()
        return examples
    except Exception as e:
        print(f"Error loading examples from DB: {e}")
        conn.close()
        return []

def count_examples(
    dataset_id: int,
    status: Optional[str] = None,
    min_quality: Optional[float] = None,
    max_quality: Optional[float] = None
) -> int:
    """Count examples with optional filtering"""
    conn = get_db_connection()
    cursor = conn.cursor()
    try:
        query = 'SELECT COUNT(*) FROM examples WHERE dataset_id = %s'
        params = [dataset_id]
        
        if status:
            query += ' AND status = %s'
            params.append(status)
        
        if min_quality is not None:
            query += ' AND quality_score >= %s'
            params.append(min_quality)
        
        if max_quality is not None:
            query += ' AND quality_score <= %s'
            params.append(max_quality)
        
        cursor.execute(query, params)
        count = cursor.fetchone()[0]
        conn.close()
        return count
    except Exception as e:
        print(f"Error counting examples: {e}")
        conn.close()
        return 0

def delete_dataset(dataset_id: int) -> bool:
    """Delete a dataset and associated jobs"""
    conn = get_db_connection()
    cursor = conn.cursor(cursor_factory=RealDictCursor)
    
    try:
        # Get dataset file path to delete the file
        cursor.execute('SELECT file_path FROM datasets WHERE id = %s', (dataset_id,))
        dataset = cursor.fetchone()
        
        if dataset and dataset['file_path']:
            file_path = dataset['file_path']
            if os.path.exists(file_path):
                os.remove(file_path)
        
        # Delete associated data
        cursor.execute('DELETE FROM examples WHERE dataset_id = %s', (dataset_id,))
        cursor.execute('DELETE FROM quality_jobs WHERE dataset_id = %s', (dataset_id,))
        cursor.execute('DELETE FROM generation_jobs WHERE dataset_id = %s', (dataset_id,))
        
        # Delete the dataset record
        cursor.execute('DELETE FROM datasets WHERE id = %s', (dataset_id,))
        
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
    ) VALUES (%s, %s, %s, %s, %s)
    RETURNING id
    ''', (
        dataset_id, 'pending', json.dumps(parameters), now, examples_requested
    ))
    
    job_id = cursor.fetchone()[0]
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
        update_fields.append('status = %s')
        params.append(status)
    
    if examples_generated is not None:
        update_fields.append('examples_generated = %s')
        params.append(examples_generated)
    
    if completed_at is not None:
        update_fields.append('completed_at = %s')
        params.append(completed_at)
    
    if dataset_id is not None:
        update_fields.append('dataset_id = %s')
        params.append(dataset_id)
    
    if errors is not None:
        update_fields.append('errors = %s')
        params.append(json.dumps(errors))
    
    if not update_fields:
        conn.close()
        return False
    
    query = f'UPDATE generation_jobs SET {", ".join(update_fields)} WHERE id = %s'
    params.append(job_id)
    
    cursor.execute(query, params)
    conn.commit()
    conn.close()
    
    return True

def get_generation_job(job_id: int) -> Optional[Dict[str, Any]]:
    """Get a generation job by ID"""
    conn = get_db_connection()
    cursor = conn.cursor(cursor_factory=RealDictCursor)
    
    cursor.execute('SELECT * FROM generation_jobs WHERE id = %s', (job_id,))
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
    cursor = conn.cursor(cursor_factory=RealDictCursor)
    
    query = 'SELECT * FROM generation_jobs'
    params = []
    
    # Add filters
    filters = ["status != 'deleted'"]  # Всегда исключаем удаленные задачи
    
    if dataset_id is not None:
        filters.append('dataset_id = %s')
        params.append(dataset_id)
    
    if status:
        filters.append('status = %s')
        params.append(status)
    
    query += ' WHERE ' + ' AND '.join(filters)
    
    # Add sorting and limit
    query += ' ORDER BY started_at DESC LIMIT %s'
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
    ) VALUES (%s, %s, %s, %s, %s)
    RETURNING id
    ''', (
        dataset_id, 'pending', json.dumps(parameters), now, examples_total
    ))
    
    job_id = cursor.fetchone()[0]
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
        update_fields.append('status = %s')
        params.append(status)
    
    if examples_processed is not None:
        update_fields.append('examples_processed = %s')
        params.append(examples_processed)
    
    if examples_kept is not None:
        update_fields.append('examples_kept = %s')
        params.append(examples_kept)
    
    if examples_fixed is not None:
        update_fields.append('examples_fixed = %s')
        params.append(examples_fixed)
    
    if examples_removed is not None:
        update_fields.append('examples_removed = %s')
        params.append(examples_removed)
        
    if examples_total is not None:
        update_fields.append('examples_total = %s')
        params.append(examples_total)
    
    if avg_quality_score is not None:
        update_fields.append('avg_quality_score = %s')
        params.append(avg_quality_score)
    
    if result_file_path is not None:
        update_fields.append('result_file_path = %s')
        params.append(result_file_path)
    
    if completed_at is not None:
        update_fields.append('completed_at = %s')
        params.append(completed_at)
    
    if errors is not None:
        update_fields.append('errors = %s')
        params.append(json.dumps(errors))
    
    if not update_fields:
        conn.close()
        return False
    
    query = f'UPDATE quality_jobs SET {", ".join(update_fields)} WHERE id = %s'
    params.append(job_id)
    
    cursor.execute(query, params)
    conn.commit()
    conn.close()
    
    return True

def get_quality_job(job_id: int) -> Optional[Dict[str, Any]]:
    """Get a quality control job by ID"""
    conn = get_db_connection()
    cursor = conn.cursor(cursor_factory=RealDictCursor)
    
    cursor.execute('SELECT * FROM quality_jobs WHERE id = %s', (job_id,))
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
    cursor = conn.cursor(cursor_factory=RealDictCursor)
    
    query = 'SELECT * FROM quality_jobs'
    params = []
    
    # Add filters
    filters = ["status != 'deleted'"]  # Всегда исключаем удаленные задачи
    
    if dataset_id is not None:
        filters.append('dataset_id = %s')
        params.append(dataset_id)
    
    if status:
        filters.append('status = %s')
        params.append(status)
    
    query += ' WHERE ' + ' AND '.join(filters)
    
    # Add sorting and limit
    query += ' ORDER BY started_at DESC LIMIT %s'
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
            ) VALUES (%s, %s, %s, %s, %s, %s)
            RETURNING id
            ''', (
                dataset_id,
                i,
                json.dumps(example),
                quality_score,
                'original',
                json.dumps(example.get('metadata', {}))
            ))
            
            example_ids.append(cursor.fetchone()[0])
        
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
        cursor = conn.cursor(cursor_factory=RealDictCursor)
        
        # Get current example data
        print(f"Getting example data for ID {example_id}")
        cursor.execute('SELECT * FROM examples WHERE id = %s', (example_id,))
        example = cursor.fetchone()
        
        if not example:
            print(f"Example {example_id} not found")
            raise Exception(f"Example {example_id} not found")
        
        # Get dataset file path
        print(f"Getting dataset info for ID {example['dataset_id']}")
        cursor.execute('SELECT file_path FROM datasets WHERE id = %s', (example['dataset_id'],))
        dataset = cursor.fetchone()
        
        if not dataset:
            print(f"Dataset {example['dataset_id']} not found")
            raise Exception(f"Dataset {example['dataset_id']} not found")
            
        file_path = dataset['file_path']
        print(f"Dataset file path: {file_path}")
        
        # Update database record
        update_fields = []
        params = []
        
        if content is not None:
            update_fields.append('content = %s')
            params.append(json.dumps(content))
            
        if quality_score is not None:
            update_fields.append('quality_score = %s')
            params.append(quality_score)
            
        if status is not None:
            update_fields.append('status = %s')
            params.append(status)
            
        if update_fields:
            query = f'UPDATE examples SET {", ".join(update_fields)} WHERE id = %s'
            params.append(example_id)
            cursor.execute(query, params)
            
        # Update file content if content changed
        if content is not None and os.path.exists(file_path):
            print(f"Updating file content at {file_path}")
            with open(file_path, 'r') as f:
                data = json.load(f)
                
            # Find and update the example in the file
            # Note: This assumes example_index matches list index, which might not be true if examples were deleted
            # A better approach would be to use a unique ID in the file or search by content
            # For now, we'll try to match by content of the old example
            
            old_content = json.loads(example['content'])
            updated = False
            
            for i, item in enumerate(data):
                # Simple content matching (might need improvement)
                if json.dumps(item, sort_keys=True) == json.dumps(old_content, sort_keys=True):
                    data[i] = content
                    updated = True
                    break
            
            if updated:
                with open(file_path, 'w') as f:
                    json.dump(data, f, indent=2)
                print("File updated successfully")
            else:
                print("Warning: Could not find example in file to update")
        
        conn.commit()
        return True
        
    except Exception as e:
        print(f"Error updating example: {e}")
        if conn:
            conn.rollback()
        return False
    finally:
        if conn:
            conn.close()

# Template Operations
def create_template(
    name: str,
    domain: str,
    content: str,
    subdomain: Optional[str] = None,
    variables: Optional[Dict[str, Any]] = None,
    description: Optional[str] = None
) -> int:
    """Create a new template"""
    conn = get_db_connection()
    cursor = conn.cursor()
    
    now = datetime.now().isoformat()
    
    cursor.execute('''
    INSERT INTO templates (name, domain, subdomain, content, variables, description, created_at, updated_at, version)
    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
    RETURNING id
    ''', (
        name, domain, subdomain, content, 
        json.dumps(variables) if variables else None, 
        description, now, now, 1
    ))
    
    template_id = cursor.fetchone()[0]
    conn.commit()
    conn.close()
    
    return template_id

def get_templates(domain: Optional[str] = None) -> List[Dict[str, Any]]:
    """Get templates with optional domain filtering"""
    conn = get_db_connection()
    cursor = conn.cursor(cursor_factory=RealDictCursor)
    
    query = 'SELECT * FROM templates'
    params = []
    
    if domain:
        query += ' WHERE domain = %s'
        params.append(domain)
        
    query += ' ORDER BY updated_at DESC'
    
    cursor.execute(query, params)
    templates = cursor.fetchall()
    conn.close()
    
    result = []
    for t in templates:
        t_dict = dict(t)
        if t_dict['variables'] and isinstance(t_dict['variables'], str):
            t_dict['variables'] = json.loads(t_dict['variables'])
        result.append(t_dict)
        
    return result

def get_template(template_id: int) -> Optional[Dict[str, Any]]:
    """Get a template by ID"""
    conn = get_db_connection()
    cursor = conn.cursor(cursor_factory=RealDictCursor)
    
    cursor.execute('SELECT * FROM templates WHERE id = %s', (template_id,))
    template = cursor.fetchone()
    conn.close()
    
    if template:
        result = dict(template)
        if result['variables'] and isinstance(result['variables'], str):
            result['variables'] = json.loads(result['variables'])
        return result
        
    return None

def update_template(
    template_id: int,
    name: Optional[str] = None,
    content: Optional[str] = None,
    variables: Optional[Dict[str, Any]] = None,
    description: Optional[str] = None
) -> bool:
    """Update a template and save version history"""
    conn = get_db_connection()
    cursor = conn.cursor(cursor_factory=RealDictCursor)
    
    # Get current template
    cursor.execute('SELECT * FROM templates WHERE id = %s', (template_id,))
    current = cursor.fetchone()
    
    if not current:
        conn.close()
        return False
        
    # Save current version to history
    current_version = current['version'] if 'version' in current.keys() else 1
    
    cursor.execute(
        '''
        INSERT INTO template_versions (template_id, version, name, content, variables, description, created_at)
        VALUES (%s, %s, %s, %s, %s, %s, %s)
        ''',
        (
            template_id,
            current_version,
            current['name'],
            current['content'],
            current['variables'],
            current['description'],
            datetime.now().isoformat()
        )
    )
    
    # Update template
    updated_at = datetime.now().isoformat()
    new_version = current_version + 1
    
    update_fields = []
    params = []
    
    if name is not None:
        update_fields.append('name = %s')
        params.append(name)
        
    if content is not None:
        update_fields.append('content = %s')
        params.append(content)
        
    if variables is not None:
        update_fields.append('variables = %s')
        params.append(json.dumps(variables))
        
    if description is not None:
        update_fields.append('description = %s')
        params.append(description)
        
    update_fields.append('updated_at = %s')
    params.append(updated_at)
    
    update_fields.append('version = %s')
    params.append(new_version)
    
    params.append(template_id)
    
    query = f'UPDATE templates SET {", ".join(update_fields)} WHERE id = %s'
    cursor.execute(query, params)
    conn.commit()
    conn.close()
    
    return True

def get_template_versions(template_id: int) -> List[Dict[str, Any]]:
    """Get version history for a template"""
    conn = get_db_connection()
    cursor = conn.cursor(cursor_factory=RealDictCursor)
    
    cursor.execute(
        'SELECT * FROM template_versions WHERE template_id = %s ORDER BY version DESC',
        (template_id,)
    )
    versions = cursor.fetchall()
    conn.close()
    
    return [dict(v) for v in versions]

def restore_template_version(template_id: int, version_id: int) -> bool:
    """Restore a specific version"""
    conn = get_db_connection()
    cursor = conn.cursor(cursor_factory=RealDictCursor)
    
    # Get the version to restore
    cursor.execute('SELECT * FROM template_versions WHERE id = %s AND template_id = %s', (version_id, template_id))
    version_data = cursor.fetchone()
    
    if not version_data:
        conn.close()
        return False
        
    # Get current template to save as history before restoring
    cursor.execute('SELECT * FROM templates WHERE id = %s', (template_id,))
    current = cursor.fetchone()
    current_version = current['version'] if 'version' in current.keys() else 1
    
    # Save current state
    cursor.execute(
        '''
        INSERT INTO template_versions (template_id, version, name, content, variables, description, created_at)
        VALUES (%s, %s, %s, %s, %s, %s, %s)
        ''',
        (
            template_id,
            current_version,
            current['name'],
            current['content'],
            current['variables'],
            current['description'],
            datetime.now().isoformat()
        )
    )
    
    # Restore old data as new version
    new_version = current_version + 1
    updated_at = datetime.now().isoformat()
    
    cursor.execute(
        '''
        UPDATE templates 
        SET name = %s, content = %s, variables = %s, description = %s, updated_at = %s, version = %s
        WHERE id = %s
        ''',
        (
            version_data['name'],
            version_data['content'],
            version_data['variables'],
            version_data['description'],
            updated_at,
            new_version,
            template_id
        )
    )
    
    conn.commit()
    conn.close()
    return True

def delete_template(template_id: int) -> bool:
    """Delete a template"""
    conn = get_db_connection()
    cursor = conn.cursor()
    
    cursor.execute('DELETE FROM templates WHERE id = %s', (template_id,))
    rows_affected = cursor.rowcount
    
    # Also delete versions
    cursor.execute('DELETE FROM template_versions WHERE template_id = %s', (template_id,))
    
    conn.commit()
    conn.close()
    
    return rows_affected > 0
