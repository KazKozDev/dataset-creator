"""
LangChain Exporter

Exports datasets to LangChain-compatible formats
"""

import json
from typing import List, Dict, Any, Optional
from pathlib import Path
from .base import BaseExporter


class LangChainExporter(BaseExporter):
    """Exporter for LangChain document format"""

    def export(
        self,
        examples: List[Dict[str, Any]],
        output_filename: str,
        export_type: str = "documents",
        **kwargs
    ) -> str:
        """
        Export examples to LangChain format

        Args:
            examples: List of conversation examples
            output_filename: Name of the output file
            export_type: "documents", "chat", or "qa_pairs"
            **kwargs: Additional parameters

        Returns:
            Path to the exported file
        """
        # Validate examples
        is_valid, errors = self.validate_examples(examples)
        if not is_valid:
            raise ValueError(f"Invalid examples: {'; '.join(errors)}")

        # Convert based on export type
        if export_type == "documents":
            converted = self._convert_to_documents(examples)
        elif export_type == "chat":
            converted = self._convert_to_chat(examples)
        elif export_type == "qa_pairs":
            converted = self._convert_to_qa_pairs(examples)
        else:
            raise ValueError(f"Unsupported export_type: {export_type}")

        # Save to JSONL (LangChain prefers JSONL for large datasets)
        output_path = self.get_output_path(output_filename)
        if not output_path.suffix:
            output_path = output_path.with_suffix('.jsonl')

        self.save_jsonl(converted, output_path)

        # Create loader script
        self._create_loader_script(output_path, export_type)

        return str(output_path)

    def validate_example(self, example: Dict[str, Any]) -> bool:
        """Validate example format"""
        return any(key in example for key in ["conversation", "messages", "text", "content", "prompt", "input"])

    def _convert_to_documents(self, examples: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Convert to LangChain Document format

        Format:
        {
            "page_content": "...",
            "metadata": {...}
        }
        """
        documents = []

        for idx, example in enumerate(examples):
            # Extract text content
            page_content = ""

            if "text" in example:
                page_content = example["text"]
            elif "content" in example:
                page_content = example["content"]
            elif "conversation" in example or "messages" in example:
                # Convert conversation to text
                messages = example.get("conversation", example.get("messages", []))
                parts = []
                for msg in messages:
                    role = msg.get("role", "unknown")
                    content = msg.get("content", "")
                    parts.append(f"{role.capitalize()}: {content}")
                page_content = "\n\n".join(parts)
            elif "prompt" in example:
                page_content = example["prompt"]
                if "completion" in example or "response" in example:
                    page_content += "\n\n" + example.get("completion", example.get("response", ""))
            elif "input" in example and "output" in example:
                page_content = f"Input: {example['input']}\n\nOutput: {example['output']}"

            # Extract metadata
            metadata = {
                "source": "synthetic",
                "id": example.get("id", idx)
            }

            # Add domain info if available
            if "domain" in example:
                metadata["domain"] = example["domain"]
            if "subdomain" in example:
                metadata["subdomain"] = example["subdomain"]
            if "language" in example:
                metadata["language"] = example["language"]

            # Add any existing metadata
            if "metadata" in example and isinstance(example["metadata"], dict):
                metadata.update(example["metadata"])

            documents.append({
                "page_content": page_content,
                "metadata": metadata
            })

        return documents

    def _convert_to_chat(self, examples: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Convert to LangChain chat message format

        Format:
        {
            "messages": [
                {"role": "system/human/ai", "content": "..."}
            ],
            "metadata": {...}
        }
        """
        chat_examples = []

        for idx, example in enumerate(examples):
            messages = []

            # If already in message format
            if "conversation" in example or "messages" in example:
                original_messages = example.get("conversation", example.get("messages", []))

                for msg in original_messages:
                    role = msg.get("role", "human")

                    # Map roles to LangChain format
                    if role in ["assistant", "bot"]:
                        role = "ai"
                    elif role in ["user", "person"]:
                        role = "human"

                    messages.append({
                        "role": role,
                        "content": msg.get("content", "")
                    })

            # Convert from instruction format
            elif "instruction" in example:
                # Add system message with instruction
                messages.append({
                    "role": "system",
                    "content": example["instruction"]
                })

                # Add user input if present
                if example.get("input"):
                    messages.append({
                        "role": "human",
                        "content": example["input"]
                    })

                # Add assistant output
                if example.get("output"):
                    messages.append({
                        "role": "ai",
                        "content": example["output"]
                    })

            # Convert from prompt-completion
            elif "prompt" in example:
                messages.append({
                    "role": "human",
                    "content": example["prompt"]
                })

                if "completion" in example or "response" in example:
                    messages.append({
                        "role": "ai",
                        "content": example.get("completion", example.get("response", ""))
                    })

            # Extract metadata
            metadata = {"id": example.get("id", idx)}
            if "domain" in example:
                metadata["domain"] = example["domain"]
            if "metadata" in example and isinstance(example["metadata"], dict):
                metadata.update(example["metadata"])

            if messages:
                chat_examples.append({
                    "messages": messages,
                    "metadata": metadata
                })

        return chat_examples

    def _convert_to_qa_pairs(self, examples: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Convert to Q&A pairs format for retrieval

        Format:
        {
            "question": "...",
            "answer": "...",
            "metadata": {...}
        }
        """
        qa_pairs = []

        for idx, example in enumerate(examples):
            question = ""
            answer = ""

            # Extract from conversation
            if "conversation" in example or "messages" in example:
                messages = example.get("conversation", example.get("messages", []))

                # Find first user message as question and first assistant as answer
                for msg in messages:
                    role = msg.get("role", "")
                    content = msg.get("content", "")

                    if not question and role in ["user", "human"]:
                        question = content
                    elif question and not answer and role in ["assistant", "bot", "ai"]:
                        answer = content
                        break

            # From instruction format
            elif "instruction" in example:
                question = example["instruction"]
                if example.get("input"):
                    question += "\n" + example["input"]
                answer = example.get("output", "")

            # From prompt-completion
            elif "prompt" in example:
                question = example["prompt"]
                answer = example.get("completion", example.get("response", ""))

            # From simple input-output
            elif "input" in example and "output" in example:
                question = example["input"]
                answer = example["output"]

            # Extract metadata
            metadata = {"id": example.get("id", idx)}
            if "domain" in example:
                metadata["domain"] = example["domain"]

            if question and answer:
                qa_pairs.append({
                    "question": question,
                    "answer": answer,
                    "metadata": metadata
                })

        return qa_pairs

    def _create_loader_script(self, data_path: Path, export_type: str) -> None:
        """Create a Python script to load the data with LangChain"""

        if export_type == "documents":
            loader_code = f'''"""
Load documents with LangChain
"""

from langchain.document_loaders import JSONLoader
from langchain.schema import Document
import json

# Load documents
documents = []
with open("{data_path.name}", "r", encoding="utf-8") as f:
    for line in f:
        data = json.loads(line)
        doc = Document(
            page_content=data["page_content"],
            metadata=data.get("metadata", {{}})
        )
        documents.append(doc)

print(f"Loaded {{len(documents)}} documents")

# Example: Create a vector store
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma

embeddings = OpenAIEmbeddings()
vectorstore = Chroma.from_documents(documents, embeddings)

# Query the vector store
query = "Your query here"
results = vectorstore.similarity_search(query, k=3)

for doc in results:
    print(doc.page_content)
    print(doc.metadata)
    print()
'''

        elif export_type == "chat":
            loader_code = f'''"""
Load chat messages with LangChain
"""

from langchain.schema import HumanMessage, AIMessage, SystemMessage
import json

# Load chat examples
chat_examples = []
with open("{data_path.name}", "r", encoding="utf-8") as f:
    for line in f:
        data = json.loads(line)
        messages = []

        for msg in data["messages"]:
            role = msg["role"]
            content = msg["content"]

            if role == "human":
                messages.append(HumanMessage(content=content))
            elif role == "ai":
                messages.append(AIMessage(content=content))
            elif role == "system":
                messages.append(SystemMessage(content=content))

        chat_examples.append(messages)

print(f"Loaded {{len(chat_examples)}} chat examples")

# Example: Use with a chat model
from langchain.chat_models import ChatOpenAI

llm = ChatOpenAI()

# Use the first example
response = llm(chat_examples[0])
print(response.content)
'''

        else:  # qa_pairs
            loader_code = f'''"""
Load Q&A pairs with LangChain
"""

import json

# Load Q&A pairs
qa_pairs = []
with open("{data_path.name}", "r", encoding="utf-8") as f:
    for line in f:
        data = json.loads(line)
        qa_pairs.append(data)

print(f"Loaded {{len(qa_pairs)}} Q&A pairs")

# Example: Create a retrieval QA chain
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain.llms import OpenAI
from langchain.schema import Document

# Convert to documents for retrieval
documents = []
for qa in qa_pairs:
    doc_text = f"Q: {{qa['question']}}\\nA: {{qa['answer']}}"
    doc = Document(
        page_content=doc_text,
        metadata=qa.get("metadata", {{}})
    )
    documents.append(doc)

# Create vector store
embeddings = OpenAIEmbeddings()
vectorstore = Chroma.from_documents(documents, embeddings)

# Create QA chain
qa_chain = RetrievalQA.from_chain_type(
    llm=OpenAI(),
    retriever=vectorstore.as_retriever()
)

# Ask a question
result = qa_chain.run("Your question here")
print(result)
'''

        # Save loader script
        script_path = data_path.with_name(f"load_{data_path.stem}.py")
        with open(script_path, 'w', encoding='utf-8') as f:
            f.write(loader_code)

        # Make executable
        script_path.chmod(0o755)

    def export_for_fine_tuning(
        self,
        examples: List[Dict[str, Any]],
        output_filename: str
    ) -> str:
        """
        Export in format suitable for fine-tuning with LangChain

        Args:
            examples: List of examples
            output_filename: Output filename

        Returns:
            Path to exported file
        """
        # Use chat format for fine-tuning
        return self.export(examples, output_filename, export_type="chat")

    def export_for_retrieval(
        self,
        examples: List[Dict[str, Any]],
        output_filename: str,
        chunk_size: int = 1000,
        chunk_overlap: int = 200
    ) -> str:
        """
        Export in format optimized for retrieval

        Args:
            examples: List of examples
            output_filename: Output filename
            chunk_size: Size of text chunks
            chunk_overlap: Overlap between chunks

        Returns:
            Path to exported file
        """
        # Convert to documents
        documents = self._convert_to_documents(examples)

        # Optionally split long documents into chunks
        chunked_documents = []
        for doc in documents:
            content = doc["page_content"]

            # If document is longer than chunk_size, split it
            if len(content) > chunk_size:
                chunks = []
                start = 0

                while start < len(content):
                    end = start + chunk_size
                    chunk = content[start:end]
                    chunks.append(chunk)
                    start = end - chunk_overlap

                # Create separate documents for each chunk
                for i, chunk in enumerate(chunks):
                    chunk_metadata = doc["metadata"].copy()
                    chunk_metadata["chunk"] = i
                    chunk_metadata["total_chunks"] = len(chunks)

                    chunked_documents.append({
                        "page_content": chunk,
                        "metadata": chunk_metadata
                    })
            else:
                chunked_documents.append(doc)

        # Save to JSONL
        output_path = self.get_output_path(output_filename)
        if not output_path.suffix:
            output_path = output_path.with_suffix('.jsonl')

        self.save_jsonl(chunked_documents, output_path)

        return str(output_path)
