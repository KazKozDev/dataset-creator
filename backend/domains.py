"""
Domain configurations for LLM Dataset Creator
Contains comprehensive domain definitions with subdomains and parameters
"""

# Comprehensive domain configuration with specialized subdomains
DOMAINS = {
    # 1. Customer Support
    "support": {
        "name": "Customer Support",
        "description": "Support dialogues for various products and services",
        "examples": ["troubleshooting", "product information", "customer complaints"],
        "subdomains": {
            "tech_support": {
                "name": "Technical Support",
                "description": "Troubleshooting hardware, software, and service issues",
                "scenarios": ["device troubleshooting", "software installation", "connectivity issues"]
            },
            "account_support": {
                "name": "Account Support",
                "description": "Handling account-related inquiries and issues",
                "scenarios": ["account access", "billing questions", "subscription management"]
            },
            "product_inquiry": {
                "name": "Product Inquiry",
                "description": "Answering questions about products and features",
                "scenarios": ["feature information", "product comparison", "compatibility inquiries"]
            },
            "service_issue": {
                "name": "Service Issue",
                "description": "Resolving problems with services",
                "scenarios": ["service outage", "quality problems", "setup assistance"]
            },
            "returns_refunds": {
                "name": "Returns and Refunds",
                "description": "Processing returns, refunds, and exchanges",
                "scenarios": ["return policy", "refund status", "exchange process"]
            }
        },
        "specific_params": {
            "conversation_types": {
                "question_answer": 60,    # Single-turn interactions (60%)
                "multi_turn": 40          # Complex issue resolution requiring multiple exchanges (40%)
            },
            "scenarios": {
                "technical_troubleshooting": 30,  # Step-by-step problem resolution (30%)
                "product_information": 25,        # Explanations of features, capabilities (25%)
                "account_management": 15,         # Account-related requests (15%)
                "complaint_handling": 15,         # Addressing dissatisfaction professionally (15%)
                "policy_explanation": 10,         # Explaining company policies (10%)
                "edge_case": 5                    # Unusual situations requiring special handling (5%)
            },
            "customer_knowledge": {
                "beginner": 40,      # Beginner (40%)
                "intermediate": 40,  # Intermediate (40%)
                "advanced": 20       # Advanced (20%)
            },
            "channels": {
                "chat": 60,            # Chat (60%)
                "email": 30,           # Email (30%)
                "phone_transcript": 10 # Phone transcript (10%)
            }
        }
    },
    
    # 2. Medical Information
    "medical": {
        "name": "Medical Information",
        "description": "Medical consultations, documentation, and diagnostic dialogues",
        "examples": ["doctor consultation", "medical history", "diagnosis explanation"],
        "subdomains": {
            "patient_consultation": {
                "name": "Patient Consultation",
                "description": "Direct dialogues between healthcare providers and patients",
                "scenarios": ["initial visit", "follow-up consultation", "specialist referral"]
            },
            "clinical_documentation": {
                "name": "Clinical Documentation",
                "description": "Medical reports, histories, and records",
                "scenarios": ["medical history", "progress notes", "discharge summary"]
            },
            "diagnostic_reasoning": {
                "name": "Diagnostic Reasoning",
                "description": "Analysis and interpretation of symptoms and test results",
                "scenarios": ["differential diagnosis", "test interpretation", "treatment planning"]
            },
            "patient_education": {
                "name": "Patient Education",
                "description": "Materials explaining conditions and treatments to patients",
                "scenarios": ["condition explanation", "treatment instructions", "preventive care"]
            },
            "medical_emergency": {
                "name": "Medical Emergency",
                "description": "Urgent medical situations requiring immediate response",
                "scenarios": ["triage assessment", "emergency instructions", "critical care"]
            }
        },
        "specific_params": {
            "conversation_types": {
                "consultation": 40,    # Provider-patient consultation (40%)
                "documentation": 30,   # Medical documentation (30%)
                "explanation": 30      # Medical information explanation (30%)
            },
            "scenarios": {
                "initial_consultation": 25,   # Initial consultation (25%)
                "follow_up": 20,              # Follow-up visit (20%)
                "chronic_management": 15,     # Chronic condition management (15%)
                "test_results": 15,           # Test results explanation (15%)
                "treatment_plan": 15,         # Treatment planning (15%)
                "emergency_guidance": 10      # Emergency guidance (10%)
            },
            "medical_specialties": {
                "general_practice": 25,  # General practice (25%)
                "cardiology": 15,        # Cardiology (15%)
                "neurology": 10,         # Neurology (10%)
                "oncology": 10,          # Oncology (10%)
                "pediatrics": 10,        # Pediatrics (10%)
                "psychiatry": 10,        # Psychiatry (10%)
                "dermatology": 10,       # Dermatology (10%)
                "other": 10              # Other specialties (10%)
            },
            "patient_demographics": {
                "adult": 60,         # Adult patients (60%)
                "elderly": 20,       # Elderly patients (20%)
                "pediatric": 15,     # Children (15%)
                "pregnant": 5        # Pregnant patients (5%)
            }
        }
    },
    
    # 3. Legal Documentation
    "legal": {
        "name": "Legal Documentation",
        "description": "Legal documents, consultations, and contract work",
        "examples": ["contract analysis", "legal consultation", "court documents"],
        "subdomains": {
            "contract_law": {
                "name": "Contract Law",
                "description": "Creation and analysis of contracts and agreements",
                "scenarios": ["contract drafting", "contract review", "breach analysis"]
            },
            "litigation": {
                "name": "Litigation",
                "description": "Court-related documentation and processes",
                "scenarios": ["pleadings", "motions", "case analysis"]
            },
            "regulatory_compliance": {
                "name": "Regulatory Compliance",
                "description": "Ensuring adherence to laws and regulations",
                "scenarios": ["compliance assessment", "regulatory filings", "audit response"]
            },
            "client_consultation": {
                "name": "Client Consultation",
                "description": "Legal advice and counsel to clients",
                "scenarios": ["initial consultation", "case strategy", "rights explanation"]
            },
            "intellectual_property": {
                "name": "Intellectual Property",
                "description": "Patents, trademarks, copyrights, and IP protection",
                "scenarios": ["patent application", "IP infringement", "licensing agreements"]
            }
        },
        "specific_params": {
            "document_types": {
                "contract": 30,         # Contracts (30%)
                "consultation": 25,     # Legal consultations (25%)
                "case_analysis": 15,    # Case analysis (15%)
                "legal_memo": 15,       # Legal memos (15%)
                "brief": 10,            # Legal briefs (10%)
                "legal_letter": 5       # Legal letters (5%)
            },
            "practice_areas": {
                "corporate": 20,        # Corporate law (20%)
                "civil": 20,            # Civil law (20%)
                "family": 15,           # Family law (15%)
                "criminal": 15,         # Criminal law (15%)
                "intellectual_property": 10,  # Intellectual property (10%)
                "real_estate": 10,      # Real estate (10%)
                "tax": 5,               # Tax law (5%)
                "immigration": 5        # Immigration law (5%)
            },
            "complexity_levels": {
                "basic": 30,            # Basic level (30%)
                "intermediate": 50,     # Intermediate level (50%)
                "complex": 20           # Complex level (20%)
            },
            "client_knowledge": {
                "minimal": 50,          # Minimal legal knowledge (50%)
                "moderate": 40,         # Moderate knowledge (40%)
                "sophisticated": 10     # Advanced knowledge (10%)
            }
        }
    },
    
    # 4. Educational Content
    "education": {
        "name": "Educational Content",
        "description": "Educational materials, tutoring dialogues, and instructional content",
        "examples": ["concept explanation", "student questions", "lesson plans"],
        "subdomains": {
            "instructional_content": {
                "name": "Instructional Content",
                "description": "Direct teaching of concepts and skills",
                "scenarios": ["lesson delivery", "concept explanation", "skill instruction"]
            },
            "tutoring_session": {
                "name": "Tutoring Session",
                "description": "One-on-one educational support",
                "scenarios": ["homework help", "exam preparation", "concept clarification"]
            },
            "assessment_material": {
                "name": "Assessment Material",
                "description": "Questions, tests, and evaluation content",
                "scenarios": ["quiz questions", "essay prompts", "problem sets"]
            },
            "student_feedback": {
                "name": "Student Feedback",
                "description": "Evaluative comments on student work",
                "scenarios": ["assignment feedback", "improvement suggestions", "progress evaluation"]
            },
            "educational_planning": {
                "name": "Educational Planning",
                "description": "Curriculum and learning path development",
                "scenarios": ["lesson planning", "curriculum design", "learning objectives"]
            }
        },
        "specific_params": {
            "content_types": {
                "explanation": 35,       # Material explanation (35%)
                "tutoring": 25,          # Tutoring (25%)
                "assessment": 20,        # Knowledge assessment (20%)
                "feedback": 10,          # Feedback (10%)
                "study_plan": 10         # Study plan (10%)
            },
            "subject_areas": {
                "math": 20,              # Mathematics (20%)
                "science": 20,           # Science (20%)
                "humanities": 15,        # Humanities (15%)
                "language": 15,          # Languages (15%)
                "programming": 15,       # Programming (15%)
                "arts": 10,              # Arts (10%)
                "professional": 5        # Professional skills (5%)
            },
            "education_levels": {
                "elementary": 15,        # Elementary school (15%)
                "middle_school": 20,     # Middle school (20%)
                "high_school": 25,       # High school (25%)
                "undergraduate": 25,     # Undergraduate (25%)
                "graduate": 10,          # Graduate (10%)
                "professional": 5        # Professional education (5%)
            },
            "instruction_approaches": {
                "direct": 30,            # Direct instruction (30%)
                "socratic": 25,          # Socratic method (25%)
                "problem_based": 20,     # Problem-based learning (20%)
                "project_based": 15,     # Project-based learning (15%)
                "inquiry_based": 10      # Inquiry-based learning (10%)
            }
        }
    },
    
    # 5. Business Communication
    "business": {
        "name": "Business Communication",
        "description": "Business correspondence, reports, and presentations",
        "examples": ["business emails", "meeting notes", "business proposals"],
        "subdomains": {
            "internal_comms": {
                "name": "Internal Communications",
                "description": "Communication within an organization",
                "scenarios": ["team updates", "policy announcements", "procedure documentation"]
            },
            "external_comms": {
                "name": "External Communications",
                "description": "Communication with clients, vendors, and stakeholders",
                "scenarios": ["client emails", "vendor negotiations", "stakeholder updates"]
            },
            "reporting": {
                "name": "Business Reporting",
                "description": "Performance reports and business metrics",
                "scenarios": ["quarterly reports", "KPI dashboards", "progress updates"]
            },
            "proposals": {
                "name": "Business Proposals",
                "description": "Formal business proposals and plans",
                "scenarios": ["project proposals", "business plans", "investment pitches"]
            },
            "executive_comms": {
                "name": "Executive Communications",
                "description": "High-level strategic communication",
                "scenarios": ["executive summaries", "board presentations", "strategic plans"]
            }
        },
        "specific_params": {
            "document_types": {
                "email": 30,             # Email (30%)
                "report": 20,            # Reports (20%)
                "proposal": 15,          # Proposals (15%)
                "meeting_notes": 15,     # Meeting notes (15%)
                "memo": 10,              # Memos (10%)
                "presentation": 10       # Presentations (10%)
            },
            "business_contexts": {
                "internal_communication": 35,  # Internal communication (35%)
                "client_facing": 25,          # Client communication (25%)
                "vendor_management": 15,      # Vendor management (15%)
                "executive_level": 15,        # Executive communication (15%)
                "cross_department": 10        # Cross-departmental communication (10%)
            },
            "business_sectors": {
                "technology": 25,         # Technology (25%)
                "finance": 20,            # Finance (20%)
                "healthcare": 15,         # Healthcare (15%)
                "manufacturing": 15,      # Manufacturing (15%)
                "retail": 10,             # Retail (10%)
                "consulting": 10,         # Consulting (10%)
                "non_profit": 5           # Non-profit (5%)
            },
            "formality_levels": {
                "formal": 50,             # Formal style (50%)
                "semi_formal": 40,        # Semi-formal style (40%)
                "casual_professional": 10 # Casual professional (10%)
            }
        }
    },
    
    # 6. Technical Documentation
    "technical": {
        "name": "Technical Documentation",
        "description": "Technical manuals, guides, API docs, and specifications",
        "examples": ["API documentation", "user manuals", "technical specifications"],
        "subdomains": {
            "api_documentation": {
                "name": "API Documentation",
                "description": "Documentation for application programming interfaces",
                "scenarios": ["endpoint descriptions", "authentication guides", "integration examples"]
            },
            "user_manuals": {
                "name": "User Manuals",
                "description": "End-user focused product documentation",
                "scenarios": ["installation guides", "feature instructions", "troubleshooting guides"]
            },
            "technical_specifications": {
                "name": "Technical Specifications",
                "description": "Detailed technical descriptions of systems",
                "scenarios": ["system requirements", "architecture documents", "data models"]
            },
            "developer_guides": {
                "name": "Developer Guides",
                "description": "Documentation for developers and engineers",
                "scenarios": ["implementation guides", "best practices", "code examples"]
            },
            "reference_documentation": {
                "name": "Reference Documentation",
                "description": "Comprehensive technical reference materials",
                "scenarios": ["function references", "parameter descriptions", "error code listings"]
            }
        },
        "specific_params": {
            "document_types": {
                "reference": 30,           # Reference documentation (30%)
                "guide": 25,               # Guides and tutorials (25%)
                "specification": 20,       # Technical specifications (20%)
                "api_doc": 15,             # API documentation (15%)
                "troubleshooting": 10      # Troubleshooting docs (10%)
            },
            "technical_domains": {
                "software": 35,            # Software (35%)
                "web_services": 20,        # Web services (20%)
                "hardware": 15,            # Hardware (15%)
                "network": 15,             # Networking (15%)
                "database": 10,            # Databases (10%)
                "embedded": 5              # Embedded systems (5%)
            },
            "audience_expertise": {
                "beginner": 20,            # Beginner (20%)
                "intermediate": 50,        # Intermediate (50%)
                "expert": 30               # Expert (30%)
            },
            "documentation_styles": {
                "procedural": 40,          # Step-by-step procedures (40%)
                "conceptual": 30,          # Conceptual explanations (30%)
                "reference": 20,           # Reference information (20%)
                "tutorial": 10             # Tutorial style (10%)
            }
        }
    },
    
    # 7. Sales and Negotiation
    "sales": {
        "name": "Sales and Negotiation",
        "description": "Sales pitches, negotiation dialogues, and customer interactions",
        "examples": ["sales pitch", "price negotiation", "objection handling"],
        "subdomains": {
            "sales_pitch": {
                "name": "Sales Pitch",
                "description": "Product and service presentations to prospects",
                "scenarios": ["product presentation", "solution selling", "feature demonstration"]
            },
            "negotiation": {
                "name": "Negotiation",
                "description": "Price and terms negotiations",
                "scenarios": ["price negotiation", "contract terms", "service level agreements"]
            },
            "objection_handling": {
                "name": "Objection Handling",
                "description": "Addressing customer concerns and objections",
                "scenarios": ["price objections", "competitor comparisons", "implementation concerns"]
            },
            "discovery": {
                "name": "Discovery",
                "description": "Needs assessment and prospect qualification",
                "scenarios": ["needs analysis", "qualification questions", "problem identification"]
            },
            "closing": {
                "name": "Closing",
                "description": "Techniques for finalizing sales",
                "scenarios": ["closing techniques", "commitment questions", "next steps planning"]
            }
        },
        "specific_params": {
            "interaction_types": {
                "initial_pitch": 30,        # Initial pitch (30%)
                "discovery_call": 25,       # Discovery call (25%)
                "demo_presentation": 20,    # Demo/presentation (20%)
                "negotiation": 15,          # Negotiation (15%)
                "closing_conversation": 10  # Closing conversation (10%)
            },
            "customer_types": {
                "new_prospect": 40,         # New prospect (40%)
                "existing_customer": 30,    # Existing customer (30%)
                "returning_customer": 20,   # Returning customer (20%)
                "referral": 10              # Referral (10%)
            },
            "sale_complexity": {
                "simple_transaction": 30,   # Simple transaction (30%)
                "solution_sale": 40,        # Solution sale (40%)
                "complex_enterprise": 30    # Complex enterprise sale (30%)
            },
            "objection_types": {
                "price": 30,                # Price objections (30%)
                "competition": 20,          # Competition comparisons (20%)
                "implementation": 20,       # Implementation concerns (20%)
                "timing": 15,               # Timing objections (15%)
                "authority": 15             # Decision authority (15%)
            }
        }
    },
    
    # 8. Financial Analysis
    "financial": {
        "name": "Financial Analysis",
        "description": "Financial reports, investment analysis, and market evaluations",
        "examples": ["earnings report", "investment recommendation", "market analysis"],
        "subdomains": {
            "investment_analysis": {
                "name": "Investment Analysis",
                "description": "Analysis of investment opportunities",
                "scenarios": ["stock analysis", "investment recommendations", "risk assessment"]
            },
            "financial_reporting": {
                "name": "Financial Reporting",
                "description": "Financial statements and performance reports",
                "scenarios": ["earnings reports", "financial statements", "performance metrics"]
            },
            "market_analysis": {
                "name": "Market Analysis",
                "description": "Analysis of market trends and conditions",
                "scenarios": ["market trends", "competitive analysis", "sector outlook"]
            },
            "valuation": {
                "name": "Valuation",
                "description": "Determining the value of assets and companies",
                "scenarios": ["company valuation", "asset pricing", "DCF analysis"]
            },
            "risk_assessment": {
                "name": "Risk Assessment",
                "description": "Evaluation of financial risks",
                "scenarios": ["risk factors", "mitigation strategies", "exposure analysis"]
            }
        },
        "specific_params": {
            "report_types": {
                "earnings_report": 25,      # Earnings report (25%)
                "analysis_report": 25,      # Analysis report (25%)
                "recommendation": 20,       # Investment recommendation (20%)
                "forecast": 15,             # Financial forecast (15%)
                "risk_assessment": 15       # Risk assessment (15%)
            },
            "financial_sectors": {
                "equities": 30,             # Equities (30%)
                "fixed_income": 20,         # Fixed income (20%)
                "banking": 15,              # Banking (15%)
                "real_estate": 15,          # Real estate (15%)
                "commodities": 10,          # Commodities (10%)
                "cryptocurrency": 10        # Cryptocurrency (10%)
            },
            "time_horizons": {
                "short_term": 30,           # Short-term (30%)
                "medium_term": 40,          # Medium-term (40%)
                "long_term": 30             # Long-term (30%)
            },
            "audience_types": {
                "retail_investor": 35,      # Retail investor (35%)
                "institutional": 30,        # Institutional client (30%)
                "internal": 20,             # Internal stakeholders (20%)
                "regulatory": 15            # Regulatory bodies (15%)
            }
        }
    },
    
    # 9. Research Summaries
    "research": {
        "name": "Research Summaries",
        "description": "Academic research summaries, literature reviews, and scientific reports",
        "examples": ["literature review", "research abstract", "findings summary"],
        "subdomains": {
            "literature_review": {
                "name": "Literature Review",
                "description": "Synthesis of existing research",
                "scenarios": ["systematic review", "meta-analysis", "research synthesis"]
            },
            "methodology": {
                "name": "Methodology",
                "description": "Research methodology descriptions",
                "scenarios": ["experimental design", "data collection", "analytical approach"]
            },
            "results_analysis": {
                "name": "Results Analysis",
                "description": "Analysis and interpretation of research results",
                "scenarios": ["statistical analysis", "findings interpretation", "data visualization"]
            },
            "research_abstract": {
                "name": "Research Abstract",
                "description": "Concise summaries of research papers",
                "scenarios": ["journal abstract", "conference abstract", "thesis summary"]
            },
            "discussion_section": {
                "name": "Discussion Section",
                "description": "Interpretation and implications of research",
                "scenarios": ["findings discussion", "limitations analysis", "future research"]
            }
        },
        "specific_params": {
            "research_types": {
                "empirical": 40,            # Empirical research (40%)
                "theoretical": 25,          # Theoretical research (25%)
                "review": 20,               # Review paper (20%)
                "case_study": 15            # Case study (15%)
            },
            "academic_fields": {
                "natural_sciences": 25,     # Natural sciences (25%)
                "social_sciences": 25,      # Social sciences (25%)
                "medical": 20,              # Medical research (20%)
                "computer_science": 15,     # Computer science (15%)
                "humanities": 15            # Humanities (15%)
            },
            "complexity_levels": {
                "undergraduate": 25,        # Undergraduate level (25%)
                "graduate": 45,             # Graduate level (45%)
                "expert": 30                # Expert level (30%)
            },
            "publication_types": {
                "journal_article": 40,      # Journal article (40%)
                "conference_paper": 25,     # Conference paper (25%)
                "thesis": 20,               # Thesis/dissertation (20%)
                "grant_proposal": 15        # Grant proposal (15%)
            }
        }
    },
    
    # 10. Coaching and Mentoring
    "coaching": {
        "name": "Coaching and Mentoring",
        "description": "Professional development conversations and coaching sessions",
        "examples": ["career coaching", "skills development", "performance feedback"],
        "subdomains": {
            "career_coaching": {
                "name": "Career Coaching",
                "description": "Career development and progression guidance",
                "scenarios": ["career planning", "job transition", "promotion preparation"]
            },
            "skills_development": {
                "name": "Skills Development",
                "description": "Developing specific professional skills",
                "scenarios": ["skill assessment", "development plan", "progress review"]
            },
            "performance_coaching": {
                "name": "Performance Coaching",
                "description": "Improving job performance",
                "scenarios": ["performance feedback", "goal setting", "improvement strategies"]
            },
            "leadership_development": {
                "name": "Leadership Development",
                "description": "Developing leadership capabilities",
                "scenarios": ["leadership skills", "team management", "executive presence"]
            },
            "life_coaching": {
                "name": "Life Coaching",
                "description": "Personal growth and life balance",
                "scenarios": ["work-life balance", "personal goals", "habit formation"]
            }
        },
        "specific_params": {
            "coaching_types": {
                "career": 30,               # Career coaching (30%)
                "skills": 25,               # Skills coaching (25%)
                "performance": 20,          # Performance coaching (20%)
                "leadership": 15,           # Leadership coaching (15%)
                "life": 10                  # Life coaching (10%)
            },
            "client_stages": {
                "early_career": 30,         # Early career (30%)
                "mid_career": 40,           # Mid-career (40%)
                "senior_level": 20,         # Senior level (20%)
                "transition": 10            # Career transition (10%)
            },
            "coaching_approaches": {
                "directive": 25,            # Directive coaching (25%)
                "non_directive": 30,        # Non-directive coaching (30%)
                "solution_focused": 25,     # Solution-focused (25%)
                "behavioral": 20            # Behavioral coaching (20%)
            },
            "session_types": {
                "initial_assessment": 20,   # Initial assessment (20%)
                "goal_setting": 25,         # Goal setting (25%)
                "progress_review": 30,      # Progress review (30%)
                "challenge_session": 25     # Challenge session (25%)
            }
        }
    },
    
    # 11. Creative Writing
    "creative": {
        "name": "Creative Writing",
        "description": "Narrative content, storytelling, and creative scenarios",
        "examples": ["short stories", "dialogue writing", "character development"],
        "subdomains": {
            "narrative": {
                "name": "Narrative Writing",
                "description": "Story-driven creative content",
                "scenarios": ["short stories", "narrative scenes", "plot development"]
            },
            "dialogue": {
                "name": "Dialogue Writing",
                "description": "Character conversations and dialogue",
                "scenarios": ["character interactions", "dialogue scenes", "conversation flow"]
            },
            "descriptive": {
                "name": "Descriptive Writing",
                "description": "Rich descriptive content",
                "scenarios": ["setting descriptions", "character descriptions", "sensory writing"]
            },
            "character_development": {
                "name": "Character Development",
                "description": "Creating and developing characters",
                "scenarios": ["character profiles", "backstories", "character arcs"]
            },
            "creative_nonfiction": {
                "name": "Creative Nonfiction",
                "description": "Creative approaches to factual content",
                "scenarios": ["personal essays", "memoirs", "travel writing"]
            }
        },
        "specific_params": {
            "writing_forms": {
                "short_story": 30,          # Short story (30%)
                "scene": 25,                # Scene (25%)
                "dialogue": 20,             # Dialogue (20%)
                "description": 15,          # Description (15%)
                "character_study": 10       # Character study (10%)
            },
            "genres": {
                "literary": 20,             # Literary (20%)
                "science_fiction": 15,      # Science fiction (15%)
                "fantasy": 15,              # Fantasy (15%)
                "thriller": 15,             # Thriller (15%)
                "romance": 15,              # Romance (15%)
                "historical": 10,           # Historical (10%)
                "humor": 10                 # Humor (10%)
            },
            "tone_styles": {
                "dramatic": 25,             # Dramatic (25%)
                "contemplative": 20,        # Contemplative (20%)
                "humorous": 20,             # Humorous (20%)
                "suspenseful": 15,          # Suspenseful (15%)
                "whimsical": 10,            # Whimsical (10%)
                "dark": 10                  # Dark (10%)
            },
            "perspective": {
                "first_person": 40,         # First person (40%)
                "third_person_limited": 35, # Third person limited (35%)
                "third_person_omniscient": 20, # Third person omniscient (20%)
                "second_person": 5          # Second person (5%)
            }
        }
    },
    
    # 12. Meeting Summaries
    "meetings": {
        "name": "Meeting Summaries",
        "description": "Meeting notes, transcripts, and summarization",
        "examples": ["meeting minutes", "action items", "decision summary"],
        "subdomains": {
            "executive_meetings": {
                "name": "Executive Meetings",
                "description": "High-level strategic meeting documentation",
                "scenarios": ["board meetings", "executive team", "strategic planning"]
            },
            "team_meetings": {
                "name": "Team Meetings",
                "description": "Operational and tactical team meeting records",
                "scenarios": ["status updates", "team planning", "project reviews"]
            },
            "client_meetings": {
                "name": "Client Meetings",
                "description": "Documentation of client interactions",
                "scenarios": ["client presentations", "requirement gathering", "feedback sessions"]
            },
            "project_meetings": {
                "name": "Project Meetings",
                "description": "Project-specific meeting documentation",
                "scenarios": ["kickoff meetings", "milestone reviews", "retrospectives"]
            },
            "decision_records": {
                "name": "Decision Records",
                "description": "Documentation of key decisions",
                "scenarios": ["decision logs", "rationale documentation", "approval records"]
            }
        },
        "specific_params": {
            "meeting_types": {
                "status_update": 25,        # Status update (25%)
                "decision_making": 25,      # Decision making (25%)
                "planning": 20,             # Planning (20%)
                "brainstorming": 15,        # Brainstorming (15%)
                "review": 15                # Review (15%)
            },
            "meeting_contexts": {
                "internal_team": 40,        # Internal team (40%)
                "cross_functional": 25,     # Cross-functional (25%)
                "client_facing": 20,        # Client facing (20%)
                "executive": 15             # Executive (15%)
            },
            "summary_styles": {
                "detailed_minutes": 30,     # Detailed minutes (30%)
                "action_oriented": 30,      # Action-oriented (30%)
                "decision_focused": 25,     # Decision-focused (25%)
                "brief_summary": 15         # Brief summary (15%)
            },
            "participation_levels": {
                "small_group": 40,          # Small group (3-5 people) (40%)
                "medium_group": 35,         # Medium group (6-10 people) (35%)
                "large_group": 25           # Large group (11+ people) (25%)
            }
        }
    }
}

# Common parameters for all domains
COMMON_PARAMS = {
    "question_structures": {
        "diagnostic": 25,             # Identifying root causes (25%)
        "information_providing": 25,  # Clear explanations and solutions (25%)
        "clarification": 20,          # Asking for more details (20%)
        "verification": 15,           # Confirming understanding (15%)
        "open_closed_mix": 15         # Mix of open-ended and closed questions (15%)
    },
    "communication_styles": {
        "standard_professional": 70,  # Clear, business-appropriate (70%)
        "empathetic": 15,             # For emotional situations (15%)
        "technical": 10,              # Detailed technical information (10%)
        "deescalation": 5             # Handling frustrated individuals (5%)
    },
    "emotional_tones": {
        "neutral": 40,      # Neutral (40%)
        "frustrated": 15,   # Frustrated (15%)
        "concerned": 10,    # Concerned (10%)
        "confused": 10,     # Confused (10%)
        "angry": 5,         # Angry (5%)
        "urgent": 5,        # Urgent (5%) 
        "polite": 10,       # Polite (10%)
        "appreciative": 5   # Appreciative (5%)
    },
    "complexity_levels": {
        "simple": 30,       # Simple (30%) 
        "medium": 50,       # Medium (50%)
        "complex": 20       # Complex (20%)
    }
}

# Translation dictionaries for multi-language support
TRANSLATIONS = {
    "domains": {
        "support": {"en": "Customer Support", "ru": "Клиентская поддержка"},
        "medical": {"en": "Medical Information", "ru": "Медицинская информация"},
        "legal": {"en": "Legal Documentation", "ru": "Юридическая документация"},
        "education": {"en": "Educational Content", "ru": "Образовательные материалы"},
        "business": {"en": "Business Communication", "ru": "Деловая коммуникация"},
        "technical": {"en": "Technical Documentation", "ru": "Техническая документация"},
        "sales": {"en": "Sales and Negotiation", "ru": "Продажи и переговоры"},
        "financial": {"en": "Financial Analysis", "ru": "Финансовый анализ"},
        "research": {"en": "Research Summaries", "ru": "Научные обзоры"},
        "coaching": {"en": "Coaching and Mentoring", "ru": "Коучинг и наставничество"},
        "creative": {"en": "Creative Writing", "ru": "Креативное письмо"},
        "meetings": {"en": "Meeting Summaries", "ru": "Протоколы встреч"}
    },
    "common": {
        "emotional_tone": {"en": "Emotional Tone", "ru": "Эмоциональный тон"},
        "complexity": {"en": "Complexity", "ru": "Сложность"},
        "exchanges": {"en": "Exchanges", "ru": "Обмены сообщениями"},
        "domain": {"en": "Domain", "ru": "Домен"},
        "subdomain": {"en": "Subdomain", "ru": "Поддомен"}
    }
}

def get_domains():
    """Return the list of all domains"""
    return DOMAINS

def get_domain(domain_key):
    """Get a specific domain by key"""
    return DOMAINS.get(domain_key)

def get_subdomain(domain_key, subdomain_key):
    """Get a specific subdomain by domain and subdomain keys"""
    domain = DOMAINS.get(domain_key)
    if domain and subdomain_key in domain.get("subdomains", {}):
        return domain["subdomains"][subdomain_key]
    return None

def get_common_params():
    """Return common parameters for all domains"""
    return COMMON_PARAMS

def get_translations():
    """Return translations for UI elements"""
    return TRANSLATIONS