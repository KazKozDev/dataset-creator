import os
import sys
import json
from datetime import datetime

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import database as db
from default_templates import PREDEFINED_TEMPLATES

def init_templates():
    print("Initializing templates...")
    
    # Get existing templates to avoid duplicates
    existing = db.get_templates()
    existing_names = {t['name'] for t in existing}
    
    count = 0
    for template_data in PREDEFINED_TEMPLATES:
        if template_data['name'] in existing_names:
            print(f"Skipping {template_data['name']} (already exists)")
            continue
            
        try:
            # Combine parts into content field
            content = f"System: {template_data['system_prompt']}\n\nUser: {template_data['user_template']}\n\nAssistant: {template_data['assistant_template']}"
            
            db.create_template(
                name=template_data['name'],
                domain=template_data['domain'],
                content=content,
                subdomain=template_data.get('subdomain'),
                variables=template_data.get('variables'),
                description=template_data.get('description')
            )
            print(f"Created template: {template_data['name']}")
            count += 1
        except Exception as e:
            print(f"Failed to create {template_data['name']}: {e}")
            
    print(f"Done! Created {count} new templates.")

if __name__ == "__main__":
    init_templates()
