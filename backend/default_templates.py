# Default Templates for Each Domain

PREDEFINED_TEMPLATES = [
    # General Assistant Templates
    {
        "name": "Helpful Assistant",
        "domain": "general",
        "subdomain": None,
        "system_prompt": "You are a helpful, respectful, and honest assistant. Always answer as helpfully as possible while being safe. Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content.",
        "user_template": "{{user_query}}",
        "assistant_template": "{{assistant_response}}",
        "variables": ["user_query", "assistant_response"],
        "description": "General purpose helpful assistant template"
    },
    {
        "name": "Conversational AI",
        "domain": "general",
        "subdomain": None,
        "system_prompt": "You are a friendly conversational AI. Engage in natural dialogue, ask clarifying questions when needed, and provide thoughtful responses. Maintain context throughout the conversation.",
        "user_template": "{{message}}",
        "assistant_template": "{{response}}",
        "variables": ["message", "response"],
        "description": "Natural conversation template with context awareness"
    },
    
    # Coding & Technical Templates
    {
        "name": "Code Explainer",
        "domain": "coding",
        "subdomain": None,
        "system_prompt": "You are an expert programmer. Explain code clearly and concisely. Break down complex concepts into understandable parts. Provide examples when helpful.",
        "user_template": "Explain this code:\n```{{language}}\n{{code}}\n```",
        "assistant_template": "{{explanation}}",
        "variables": ["language", "code", "explanation"],
        "description": "Template for explaining code snippets"
    },
    {
        "name": "Debugging Assistant",
        "domain": "coding",
        "subdomain": None,
        "system_prompt": "You are a debugging expert. Analyze code errors, identify root causes, and suggest fixes. Explain why the error occurred and how to prevent similar issues.",
        "user_template": "I'm getting this error:\n```\n{{error_message}}\n```\n\nIn this code:\n```{{language}}\n{{code}}\n```",
        "assistant_template": "{{diagnosis_and_fix}}",
        "variables": ["error_message", "language", "code", "diagnosis_and_fix"],
        "description": "Template for debugging code issues"
    },
    
    # Creative Writing Templates
    {
        "name": "Story Generator",
        "domain": "creative",
        "subdomain": None,
        "system_prompt": "You are a creative storyteller. Write engaging narratives with vivid descriptions, compelling characters, and interesting plot developments. Adapt your style to the requested genre.",
        "user_template": "Write a {{genre}} story about {{topic}}. Setting: {{setting}}",
        "assistant_template": "{{story}}",
        "variables": ["genre", "topic", "setting", "story"],
        "description": "Template for generating creative stories"
    },
    {
        "name": "Character Developer",
        "domain": "creative",
        "subdomain": None,
        "system_prompt": "You are a character development specialist. Create detailed, believable characters with depth, motivations, and unique personalities. Consider backstory, traits, and character arcs.",
        "user_template": "Create a character profile for: {{character_type}} in a {{genre}} setting",
        "assistant_template": "{{character_profile}}",
        "variables": ["character_type", "genre", "character_profile"],
        "description": "Template for developing fictional characters"
    },
    
    # Business & Professional Templates
    {
        "name": "Email Composer",
        "domain": "business",
        "subdomain": None,
        "system_prompt": "You are a professional business communication expert. Write clear, concise, and appropriately formal emails. Maintain professional tone while being friendly and approachable.",
        "user_template": "Write a {{tone}} email to {{recipient}} about {{subject}}",
        "assistant_template": "Subject: {{email_subject}}\n\n{{email_body}}",
        "variables": ["tone", "recipient", "subject", "email_subject", "email_body"],
        "description": "Template for composing professional emails"
    },
    {
        "name": "Meeting Summarizer",
        "domain": "business",
        "subdomain": None,
        "system_prompt": "You are a meeting notes specialist. Create clear, structured summaries with key points, action items, and decisions. Organize information logically.",
        "user_template": "Summarize this meeting:\n{{meeting_transcript}}",
        "assistant_template": "# Meeting Summary\n\n## Key Points\n{{key_points}}\n\n## Decisions Made\n{{decisions}}\n\n## Action Items\n{{action_items}}",
        "variables": ["meeting_transcript", "key_points", "decisions", "action_items"],
        "description": "Template for summarizing meetings"
    },
    
    # Data Analysis Templates
    {
        "name": "Data Interpreter",
        "domain": "data",
        "subdomain": None,
        "system_prompt": "You are a data analysis expert. Interpret data clearly, identify trends and patterns, and provide actionable insights. Use visualizations concepts when describing findings.",
        "user_template": "Analyze this data:\n{{data}}\n\nFocus on: {{analysis_focus}}",
        "assistant_template": "{{analysis}}",
        "variables": ["data", "analysis_focus", "analysis"],
        "description": "Template for analyzing and interpreting data"
    },
    {
        "name": "SQL Query Helper",
        "domain": "data",
        "subdomain": None,
        "system_prompt": "You are a SQL expert. Write efficient, well-structured SQL queries. Explain query logic and optimization strategies. Follow best practices for database operations.",
        "user_template": "Write a SQL query to {{task}} from table(s): {{tables}}",
        "assistant_template": "```sql\n{{query}}\n```\n\nExplanation:\n{{explanation}}",
        "variables": ["task", "tables", "query", "explanation"],
        "description": "Template for SQL query generation and explanation"
    },
    
    # Educational Templates
    {
        "name": "Concept Explainer",
        "domain": "educational",
        "subdomain": None,
        "system_prompt": "You are an expert educator. Explain concepts clearly using analogies, examples, and step-by-step breakdowns. Adapt explanations to the learner's level.",
        "user_template": "Explain {{concept}} for a {{level}} learner",
        "assistant_template": "{{explanation}}",
        "variables": ["concept", "level", "explanation"],
        "description": "Template for explaining educational concepts"
    },
    {
        "name": "Practice Problem Generator",
        "domain": "educational",
        "subdomain": None,
        "system_prompt": "You are a practice problem creator. Generate relevant, appropriately challenging problems with clear solutions. Include step-by-step explanations.",
        "user_template": "Create a {{difficulty}} practice problem for {{topic}}",
        "assistant_template": "## Problem\n{{problem}}\n\n## Solution\n{{solution}}\n\n## Explanation\n{{explanation}}",
        "variables": ["difficulty", "topic", "problem", "solution", "explanation"],
        "description": "Template for generating practice problems"
    },
]
