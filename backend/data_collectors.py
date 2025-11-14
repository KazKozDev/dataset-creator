"""
Data Collection Modules for Dataset Creator
Supports web scraping, API integrations, and file parsing
"""

import os
import json
import re
import time
from typing import Dict, Any, List, Optional, Tuple
from pathlib import Path
from datetime import datetime
from urllib.parse import urljoin, urlparse
import asyncio
import aiohttp
from bs4 import BeautifulSoup
import feedparser

# Import LLM provider for synthetic data generation
from llm_providers import LLMProvider


class WebScraperCollector:
    """Collect data by scraping websites"""

    def __init__(self, llm_provider: Optional[LLMProvider] = None):
        self.llm_provider = llm_provider
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }

    async def fetch_url(self, session: aiohttp.ClientSession, url: str, timeout: int = 30) -> Optional[str]:
        """Fetch content from a URL"""
        try:
            async with session.get(url, headers=self.headers, timeout=timeout) as response:
                if response.status == 200:
                    return await response.text()
                else:
                    print(f"Failed to fetch {url}: status {response.status}")
                    return None
        except asyncio.TimeoutError:
            print(f"Timeout fetching {url}")
            return None
        except Exception as e:
            print(f"Error fetching {url}: {e}")
            return None

    def extract_text_from_html(self, html: str, selector: Optional[str] = None) -> str:
        """Extract text content from HTML"""
        soup = BeautifulSoup(html, 'html.parser')

        # Remove script and style elements
        for script in soup(["script", "style", "nav", "footer", "header"]):
            script.decompose()

        # Extract text based on selector or get all text
        if selector:
            elements = soup.select(selector)
            text = '\n'.join([elem.get_text(strip=True) for elem in elements])
        else:
            text = soup.get_text(separator='\n', strip=True)

        # Clean up whitespace
        text = re.sub(r'\n\s*\n', '\n\n', text)
        return text.strip()

    def extract_links(self, html: str, base_url: str, pattern: Optional[str] = None) -> List[str]:
        """Extract links from HTML"""
        soup = BeautifulSoup(html, 'html.parser')
        links = []

        for link in soup.find_all('a', href=True):
            url = urljoin(base_url, link['href'])

            # Filter by pattern if provided
            if pattern and not re.search(pattern, url):
                continue

            # Only include http/https links
            if url.startswith(('http://', 'https://')):
                links.append(url)

        return list(set(links))  # Remove duplicates

    async def scrape_urls(
        self,
        urls: List[str],
        selector: Optional[str] = None,
        max_concurrent: int = 5
    ) -> List[Dict[str, Any]]:
        """Scrape multiple URLs concurrently"""
        results = []
        semaphore = asyncio.Semaphore(max_concurrent)

        async def scrape_with_semaphore(session, url):
            async with semaphore:
                html = await self.fetch_url(session, url)
                if html:
                    text = self.extract_text_from_html(html, selector)
                    return {
                        'url': url,
                        'content': text,
                        'scraped_at': datetime.now().isoformat()
                    }
                return None

        async with aiohttp.ClientSession() as session:
            tasks = [scrape_with_semaphore(session, url) for url in urls]
            results = await asyncio.gather(*tasks)

        return [r for r in results if r]

    async def crawl_website(
        self,
        start_url: str,
        max_pages: int = 100,
        link_pattern: Optional[str] = None,
        content_selector: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Crawl a website starting from a URL"""
        visited = set()
        to_visit = {start_url}
        results = []

        async with aiohttp.ClientSession() as session:
            while to_visit and len(visited) < max_pages:
                url = to_visit.pop()

                if url in visited:
                    continue

                visited.add(url)
                print(f"Crawling: {url} ({len(visited)}/{max_pages})")

                html = await self.fetch_url(session, url)
                if not html:
                    continue

                # Extract content
                text = self.extract_text_from_html(html, content_selector)
                results.append({
                    'url': url,
                    'content': text,
                    'scraped_at': datetime.now().isoformat()
                })

                # Extract links for further crawling
                if len(visited) < max_pages:
                    links = self.extract_links(html, url, link_pattern)
                    # Only add links from same domain
                    base_domain = urlparse(start_url).netloc
                    for link in links:
                        if urlparse(link).netloc == base_domain and link not in visited:
                            to_visit.add(link)

                # Respect rate limiting
                await asyncio.sleep(1)

        return results

    def convert_to_training_examples(
        self,
        scraped_data: List[Dict[str, Any]],
        format_type: str = 'chat'
    ) -> List[Dict[str, Any]]:
        """Convert scraped data into training examples using LLM"""
        if not self.llm_provider:
            raise ValueError("LLM provider required for conversion")

        examples = []

        for data in scraped_data:
            content = data['content']

            # Create prompt for LLM to generate training examples
            prompt = f"""Convert the following web content into a training example for fine-tuning.
Create a realistic question-answer or instruction-response pair based on the content.

WEB CONTENT:
{content[:2000]}  # Limit to first 2000 chars

Format as JSON:
{{
    "messages": [
        {{"role": "user", "content": "question based on content"}},
        {{"role": "assistant", "content": "answer based on content"}}
    ],
    "metadata": {{
        "source": "web_scraping",
        "source_url": "{data['url']}"
    }}
}}

Return ONLY valid JSON without explanations or markdown formatting.
"""

            try:
                result = self.llm_provider.generate_text(prompt, temperature=0.7)

                # Extract JSON
                start_idx = result.find('{')
                end_idx = result.rfind('}') + 1

                if start_idx != -1 and end_idx > start_idx:
                    clean_json = result[start_idx:end_idx]
                    example = json.loads(clean_json)
                    examples.append(example)
            except Exception as e:
                print(f"Error converting content from {data['url']}: {e}")

        return examples


class APICollector:
    """Collect data from various APIs"""

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key

    async def fetch_github_issues(
        self,
        repo: str,
        state: str = 'all',
        max_issues: int = 100
    ) -> List[Dict[str, Any]]:
        """Fetch GitHub issues from a repository"""
        headers = {'Accept': 'application/vnd.github.v3+json'}
        if self.api_key:
            headers['Authorization'] = f'token {self.api_key}'

        url = f'https://api.github.com/repos/{repo}/issues'
        params = {'state': state, 'per_page': min(max_issues, 100)}

        issues = []

        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(url, headers=headers, params=params) as response:
                    if response.status == 200:
                        data = await response.json()

                        for issue in data:
                            issues.append({
                                'title': issue['title'],
                                'body': issue['body'] or '',
                                'state': issue['state'],
                                'comments_count': issue['comments'],
                                'created_at': issue['created_at'],
                                'url': issue['html_url'],
                                'labels': [label['name'] for label in issue.get('labels', [])]
                            })
                    else:
                        print(f"Failed to fetch GitHub issues: {response.status}")
        except Exception as e:
            print(f"Error fetching GitHub issues: {e}")

        return issues

    async def fetch_github_discussions(
        self,
        repo: str,
        max_discussions: int = 50
    ) -> List[Dict[str, Any]]:
        """Fetch GitHub discussions (requires GraphQL)"""
        if not self.api_key:
            print("GitHub API key required for discussions")
            return []

        # GitHub GraphQL endpoint
        url = 'https://api.github.com/graphql'
        headers = {'Authorization': f'bearer {self.api_key}'}

        owner, name = repo.split('/')

        query = """
        query($owner: String!, $name: String!, $first: Int!) {
            repository(owner: $owner, name: $name) {
                discussions(first: $first) {
                    nodes {
                        title
                        body
                        answer {
                            body
                        }
                        comments(first: 5) {
                            nodes {
                                body
                            }
                        }
                        createdAt
                        url
                    }
                }
            }
        }
        """

        variables = {
            'owner': owner,
            'name': name,
            'first': max_discussions
        }

        discussions = []

        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    url,
                    headers=headers,
                    json={'query': query, 'variables': variables}
                ) as response:
                    if response.status == 200:
                        data = await response.json()

                        for disc in data.get('data', {}).get('repository', {}).get('discussions', {}).get('nodes', []):
                            discussions.append({
                                'title': disc['title'],
                                'body': disc['body'],
                                'answer': disc.get('answer', {}).get('body', ''),
                                'comments': [c['body'] for c in disc.get('comments', {}).get('nodes', [])],
                                'created_at': disc['createdAt'],
                                'url': disc['url']
                            })
                    else:
                        print(f"Failed to fetch GitHub discussions: {response.status}")
        except Exception as e:
            print(f"Error fetching GitHub discussions: {e}")

        return discussions

    async def fetch_stackoverflow_questions(
        self,
        tag: str,
        max_questions: int = 100
    ) -> List[Dict[str, Any]]:
        """Fetch StackOverflow questions by tag"""
        url = 'https://api.stackexchange.com/2.3/questions'
        params = {
            'order': 'desc',
            'sort': 'votes',
            'tagged': tag,
            'site': 'stackoverflow',
            'pagesize': min(max_questions, 100),
            'filter': 'withbody'  # Include question body
        }

        questions = []

        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(url, params=params) as response:
                    if response.status == 200:
                        data = await response.json()

                        for q in data.get('items', []):
                            questions.append({
                                'title': q['title'],
                                'body': q.get('body', ''),
                                'score': q['score'],
                                'answer_count': q['answer_count'],
                                'tags': q['tags'],
                                'created_at': datetime.fromtimestamp(q['creation_date']).isoformat(),
                                'url': q['link'],
                                'is_answered': q.get('is_answered', False)
                            })
                    else:
                        print(f"Failed to fetch StackOverflow questions: {response.status}")
        except Exception as e:
            print(f"Error fetching StackOverflow questions: {e}")

        return questions

    async def fetch_rss_feed(self, feed_url: str, max_entries: int = 50) -> List[Dict[str, Any]]:
        """Fetch and parse RSS/Atom feed"""
        entries = []

        try:
            # Fetch feed using feedparser
            feed = feedparser.parse(feed_url)

            for entry in feed.entries[:max_entries]:
                entries.append({
                    'title': entry.get('title', ''),
                    'content': entry.get('summary', entry.get('description', '')),
                    'link': entry.get('link', ''),
                    'published': entry.get('published', ''),
                    'author': entry.get('author', '')
                })
        except Exception as e:
            print(f"Error parsing RSS feed: {e}")

        return entries

    def convert_to_training_examples(
        self,
        api_data: List[Dict[str, Any]],
        data_type: str,
        format_type: str = 'chat'
    ) -> List[Dict[str, Any]]:
        """Convert API data into training examples"""
        examples = []

        for data in api_data:
            if data_type == 'github_issues':
                example = {
                    'messages': [
                        {'role': 'user', 'content': f"Issue: {data['title']}\n\n{data['body']}"},
                        {'role': 'assistant', 'content': f"This is a GitHub issue in '{data['state']}' state with {data['comments_count']} comments."}
                    ],
                    'metadata': {
                        'source': 'github_issues',
                        'url': data['url'],
                        'labels': data['labels']
                    }
                }

            elif data_type == 'github_discussions':
                if data['answer']:
                    example = {
                        'messages': [
                            {'role': 'user', 'content': f"{data['title']}\n\n{data['body']}"},
                            {'role': 'assistant', 'content': data['answer']}
                        ],
                        'metadata': {
                            'source': 'github_discussions',
                            'url': data['url']
                        }
                    }
                else:
                    continue  # Skip unanswered discussions

            elif data_type == 'stackoverflow':
                if data['is_answered']:
                    example = {
                        'messages': [
                            {'role': 'user', 'content': f"{data['title']}\n\n{data['body']}"},
                            {'role': 'assistant', 'content': f"This question has {data['answer_count']} answers. Tags: {', '.join(data['tags'])}"}
                        ],
                        'metadata': {
                            'source': 'stackoverflow',
                            'url': data['url'],
                            'tags': data['tags'],
                            'score': data['score']
                        }
                    }
                else:
                    continue  # Skip unanswered questions

            elif data_type == 'rss':
                example = {
                    'messages': [
                        {'role': 'user', 'content': f"Tell me about: {data['title']}"},
                        {'role': 'assistant', 'content': data['content']}
                    ],
                    'metadata': {
                        'source': 'rss_feed',
                        'url': data['link'],
                        'published': data['published']
                    }
                }

            else:
                continue

            # Convert to instruction format if needed
            if format_type == 'instruction' and 'messages' in example:
                messages = example['messages']
                if len(messages) >= 2 and messages[0]['role'] == 'user' and messages[1]['role'] == 'assistant':
                    example = {
                        'instruction': messages[0]['content'],
                        'output': messages[1]['content'],
                        'metadata': example.get('metadata', {})
                    }

            examples.append(example)

        return examples


class FileParserCollector:
    """Parse various file formats to extract training data"""

    def __init__(self, llm_provider: Optional[LLMProvider] = None):
        self.llm_provider = llm_provider

    def parse_text_file(self, file_path: str) -> str:
        """Parse plain text file"""
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                return f.read()
        except Exception as e:
            print(f"Error reading text file: {e}")
            return ""

    def parse_json_file(self, file_path: str) -> Any:
        """Parse JSON file"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            print(f"Error reading JSON file: {e}")
            return None

    def parse_jsonl_file(self, file_path: str) -> List[Dict[str, Any]]:
        """Parse JSONL file"""
        data = []
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if line:
                        data.append(json.loads(line))
        except Exception as e:
            print(f"Error reading JSONL file: {e}")

        return data

    def parse_markdown_file(self, file_path: str) -> Dict[str, Any]:
        """Parse Markdown file into structured data"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()

            # Extract sections
            sections = []
            current_section = {'title': '', 'content': '', 'level': 0}

            for line in content.split('\n'):
                # Check for headings
                if line.startswith('#'):
                    if current_section['content']:
                        sections.append(current_section)

                    level = len(re.match(r'^#+', line).group())
                    title = line.lstrip('#').strip()
                    current_section = {'title': title, 'content': '', 'level': level}
                else:
                    current_section['content'] += line + '\n'

            if current_section['content']:
                sections.append(current_section)

            return {
                'file_path': file_path,
                'sections': sections
            }
        except Exception as e:
            print(f"Error parsing Markdown file: {e}")
            return {'file_path': file_path, 'sections': []}

    def chunk_text(self, text: str, chunk_size: int = 1000, overlap: int = 100) -> List[str]:
        """Split text into overlapping chunks"""
        chunks = []
        start = 0

        while start < len(text):
            end = start + chunk_size
            chunk = text[start:end]
            chunks.append(chunk)
            start = end - overlap

        return chunks

    def convert_documents_to_qa_pairs(
        self,
        documents: List[str],
        examples_per_doc: int = 3
    ) -> List[Dict[str, Any]]:
        """Convert documents into Q&A pairs using LLM"""
        if not self.llm_provider:
            raise ValueError("LLM provider required for conversion")

        examples = []

        for doc in documents:
            # Limit document size
            doc_text = doc[:3000]

            prompt = f"""Based on the following document, generate {examples_per_doc} diverse question-answer pairs that could be used for training.

DOCUMENT:
{doc_text}

Generate questions that:
1. Cover different aspects of the document
2. Vary in complexity
3. Have clear, accurate answers based on the document

Format as JSON array:
[
    {{
        "messages": [
            {{"role": "user", "content": "question 1"}},
            {{"role": "assistant", "content": "answer 1"}}
        ],
        "metadata": {{"source": "document_parsing"}}
    }},
    {{
        "messages": [
            {{"role": "user", "content": "question 2"}},
            {{"role": "assistant", "content": "answer 2"}}
        ],
        "metadata": {{"source": "document_parsing"}}
    }}
]

Return ONLY valid JSON array without explanations or markdown formatting.
"""

            try:
                result = self.llm_provider.generate_text(prompt, temperature=0.7)

                # Extract JSON array
                start_idx = result.find('[')
                end_idx = result.rfind(']') + 1

                if start_idx != -1 and end_idx > start_idx:
                    clean_json = result[start_idx:end_idx]
                    doc_examples = json.loads(clean_json)
                    examples.extend(doc_examples)
            except Exception as e:
                print(f"Error converting document to Q&A pairs: {e}")

        return examples


# Factory function to create collectors
def create_collector(
    collector_type: str,
    llm_provider: Optional[LLMProvider] = None,
    api_key: Optional[str] = None
):
    """Factory function to create data collectors"""
    if collector_type == 'web':
        return WebScraperCollector(llm_provider)
    elif collector_type == 'api':
        return APICollector(api_key)
    elif collector_type == 'file':
        return FileParserCollector(llm_provider)
    else:
        raise ValueError(f"Unknown collector type: {collector_type}")
