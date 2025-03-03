import json
import ollama
import numpy as np
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
import faiss
from pathlib import Path
import pickle
import asyncio
from collections import defaultdict
import json

# Load JSON files instead of using imports from google_drive_embeddings
with open('codebase_structure.json', 'r') as f:
    output_code = json.load(f)

with open('folder_tree.json', 'r') as f:
    output_tree = json.load(f)

with open('model_relationships.json', 'r') as f:
    output_models = json.load(f)


@dataclass
class Document:
    content: str
    metadata: Dict[str, Any]
    embedding: Optional[np.ndarray] = None

class CodebaseRAG:
    EMBEDDING_DIM = 1024  # mxbai-embed-large dimension
    
    def __init__(self, cache_dir: Optional[str] = None):
        self.documents = []
        self.index = None
        self.embeddings = None
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self.query_cache = {}
        self.model_map = {}
        
    def _get_embedding(self, text: str) -> np.ndarray:
        """Get embedding from mxbai-embed-large model"""
        response = ollama.embed(
            model='mxbai-embed-large',
            input=text
        )
        return np.array(response['embeddings'][0])

    def process_code_data(self, code_data: List[Dict]):
        """Process code files and their chunks"""
        for file_info in code_data:
            if not file_info.get('content') and not file_info.get('chunks'):
                continue

            file_path = file_info.get('path', '')
            file_type = file_info.get('type', 'unknown')
            
            # Process file chunks
            if file_info.get('chunks'):
                for chunk in file_info['chunks']:
                    content = f"File: {file_path}\nType: {chunk['type']}\n\n{chunk['content']}"
                    
                    metadata = {
                        'type': 'code_chunk',
                        'file_type': file_type,
                        'path': file_path,
                        'chunk_type': chunk['type'],
                        'relationships': chunk['metadata'].get('relationships', []),
                        'dependencies': chunk['metadata'].get('dependencies', [])
                    }
                    
                    if chunk['type'] == 'method':
                        metadata.update({
                            'parent_class': chunk['metadata'].get('parent_class'),
                            'method_name': chunk['metadata'].get('method_name')
                        })
                    
                    self.documents.append(Document(
                        content=content,
                        metadata=metadata
                    ))

    def process_file_tree(self, tree_data: Dict):
        """Process file tree structure"""
        for app_name, app_info in tree_data.items():
            content = f"App: {app_name}\nFiles:\n"
            for file in app_info.get('files', []):
                content += f"- {file}\n"
            if app_info.get('subdirs'):
                content += "\nSubdirectories:\n"
                for subdir in app_info['subdirs']:
                    content += f"- {subdir}\n"
                    
            self.documents.append(Document(
                content=content,
                metadata={
                    'type': 'file_structure',
                    'app_name': app_name
                }
            ))

    def process_models_structure(self, models_data: List[Dict]):
        """Process Django models structure"""
        for app in models_data:
            app_name = app['app_name']
            
            for model in app.get('models', []):
                content = f"App: {app_name}\nModel: {model['name']}\n"
                if model.get('relationships'):
                    content += "\nRelationships:\n"
                    for rel in model['relationships']:
                        content += f"- {rel['type']} to {rel['target']}\n"
                
                doc = Document(
                    content=content,
                    metadata={
                        'type': 'model',
                        'app_name': app_name,
                        'model_name': model['name'],
                        'relationships': model.get('relationships', [])
                    }
                )
                self.documents.append(doc)
                self.model_map[model['name']] = doc

    def _enhance_with_relationships(self):
        """Enhance documents with relationship information"""
        for doc in self.documents:
            if doc.metadata['type'] == 'code_chunk':
                # Link code chunks to models they reference
                referenced_models = []
                for model_name in self.model_map:
                    if model_name.lower() in doc.content.lower():
                        referenced_models.append(self.model_map[model_name])
                doc.metadata['referenced_models'] = referenced_models
                
            elif doc.metadata['type'] == 'model':
                # Link models to their related models
                if 'relationships' in doc.metadata:
                    related_models = []
                    for rel in doc.metadata['relationships']:
                        target = rel['target']
                        if target in self.model_map:
                            related_models.append({
                                'model': self.model_map[target],
                                'relationship_type': rel['type']
                            })
                    doc.metadata['related_models'] = related_models

    def build_index(self):
        """Build FAISS index from documents"""
        if not self.documents:
            raise ValueError("No documents added yet")
            
        # Get embeddings for all documents
        embeddings = []
        for doc in self.documents:
            if doc.embedding is None:
                doc.embedding = self._get_embedding(doc.content)
            embeddings.append(doc.embedding)
        
        self.embeddings = np.array(embeddings)
        
        # Verify embedding dimensions
        if self.embeddings.shape[1] != self.EMBEDDING_DIM:
            raise ValueError(f"Expected embedding dimension {self.EMBEDDING_DIM}, got {self.embeddings.shape[1]}")
        
        # Build optimized index
        self.index = faiss.IndexIDMap(faiss.IndexFlatIP(self.EMBEDDING_DIM))
        self.index.add_with_ids(
            self.embeddings.astype('float32'),
            np.arange(len(self.documents))
        )
        
        # Enhance relationships
        self._enhance_with_relationships()
        
        # Save if cache directory is set
        if self.cache_dir:
            self._save_index()
    
    def _save_index(self):
        """Save index and data to cache"""
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        faiss.write_index(self.index, str(self.cache_dir / "index.faiss"))
        
        with open(self.cache_dir / "documents.pkl", "wb") as f:
            pickle.dump(self.documents, f)
    
    def _load_index(self) -> bool:
        """Load index and data from cache"""
        if not self.cache_dir or not (self.cache_dir / "index.faiss").exists():
            return False
            
        self.index = faiss.read_index(str(self.cache_dir / "index.faiss"))
        
        with open(self.cache_dir / "documents.pkl", "rb") as f:
            self.documents = pickle.load(f)
            
        return True

    def _detect_model_terms(self, query: str) -> List[str]:
        """Detect model names in query"""
        query_terms = query.lower().split()
        return [
            model_name for model_name in self.model_map.keys()
            if model_name.lower() in query_terms
        ]

    def _rerank_results(self, results: List[Document], query: str) -> List[Document]:
        """Rerank results based on relevance scoring"""
        model_terms = self._detect_model_terms(query)
        query_terms = query.lower().split()
        
        scored_results = []
        for doc in results:
            score = 1.0
            
            # Boost score for model matches
            if doc.metadata['type'] == 'model' and doc.metadata['model_name'] in model_terms:
                score *= 1.5
            
            # Boost score for code that implements referenced models
            if doc.metadata['type'] == 'code_chunk':
                if any(term in doc.content.lower() for term in model_terms):
                    score *= 1.3
                if doc.metadata.get('referenced_models'):
                    score *= 1.2
            
            # Boost score for relevant method names
            if doc.metadata.get('method_name'):
                if any(term in doc.metadata['method_name'].lower() for term in query_terms):
                    score *= 1.4
            
            scored_results.append((doc, score))
        
        return [doc for doc, _ in sorted(scored_results, key=lambda x: x[1], reverse=True)]

    def query(self, query_text: str, k: int = 8, code_k: int = 30) -> List[Document]:
        cache_key = f"{query_text}_{k}_{code_k}"
        if cache_key in self.query_cache:
            return self.query_cache[cache_key]
            
        if not self.index:
            raise ValueError("Index not built yet. Call build_index() first")
        
        model_terms = self._detect_model_terms(query_text)
        enhanced_query = f"{query_text} [models: {' '.join(model_terms)}]"
        
        query_embedding = self._get_embedding(enhanced_query)
        
        # Get maximum possible candidates
        max_k = max(k, code_k)
        distances, indices = self.index.search(
            query_embedding.reshape(1, -1).astype('float32'), 
            max_k * 3
        )
        
        # Get initial results
        results = [self.documents[idx] for idx in indices[0]]
        
        # Split results by type
        code_results = [doc for doc in results if doc.metadata['type'] == 'code_chunk'][:code_k]
        other_results = [doc for doc in results if doc.metadata['type'] != 'code_chunk'][:k]
        
        # Combine and rerank
        combined_results = other_results + code_results
        reranked_results = self._rerank_results(combined_results, query_text)
        
        self.query_cache[cache_key] = reranked_results
        return reranked_results

    def generate_system_prompt(self) -> str:
        """Generate system prompt for LLM"""
        return """You are an expert coding assistant with deep knowledge of this specific Django codebase. 
        Your role is to help developers understand and work with the code effectively. When answering:
        
        0. IF YOU DONT KNOW SOMETHING- TELL THAT YOU CANT REACH THAT INFORMATION
        1. Reference specific code sections and explain their purpose
        2. Highlight relationships between different components (especially Django models)
        3. Explain architectural decisions when relevant
        4. Use code examples from the actual codebase
        
        Base your responses only on the provided context. If something isn't clear from the context, acknowledge this.
        When discussing Django models, mention their relationships and foreign keys if relevant."""

    def _format_context(self, relevant_docs: List[Document]) -> str:
        """Format retrieved documents into context"""
        context = "Relevant code context:\n\n"
        
        for i, doc in enumerate(relevant_docs, 1):
            context += f"[Section {i}]\n"
            context += f"Type: {doc.metadata['type']}\n"
            
            if doc.metadata['type'] == 'model':
                context += f"Model: {doc.metadata['model_name']}\n"
                context += f"App: {doc.metadata['app_name']}\n"
                if doc.metadata.get('relationships'):
                    context += "Relationships:\n"
                    for rel in doc.metadata['relationships']:
                        context += f"- {rel['type']} to {rel['target']}\n"
            
            elif doc.metadata['type'] == 'code_chunk':
                context += f"File: {doc.metadata['path']}\n"
                if doc.metadata.get('chunk_type') == 'method':
                    context += f"Method: {doc.metadata.get('method_name', 'Unknown')}\n"
                    if doc.metadata.get('parent_class'):
                        context += f"Class: {doc.metadata['parent_class']}\n"
            
            context += f"\nContent:\n{doc.content}\n\n"
            
            # Add relationship context
            if doc.metadata.get('referenced_models'):
                context += "References models:\n"
                for ref_model in doc.metadata['referenced_models']:
                    context += f"- {ref_model.metadata['model_name']}\n"
            
        return context

    async def query_with_context(
        self, 
        query_text: str, 
        model_name: str = "deepseek-r1:8b"
    ) -> str:
        """Query the RAG system and generate response using LLM"""
        # Get relevant documents
        relevant_docs = self.query(query_text)
        
        # Format context and create full prompt
        context = self._format_context(relevant_docs)
        system_prompt = self.generate_system_prompt()
        
        user_prompt = f"""Question: {query_text}

        {context}

        Please provide a clear, detailed response based on the code context above."""

        # Query LLM
        response = ollama.chat(
            model=model_name,
            messages=[
                {
                    "role": "system",
                    "content": system_prompt
                },
                {
                    "role": "user",
                    "content": user_prompt
                }
            ]
        )
        
        return response['message']['content']

def create_rag_system(
    code_data: List[Dict],
    models_data: List[Dict],
    file_tree: Dict,
    cache_dir: Optional[str] = None
) -> CodebaseRAG:
    """Create and initialize the RAG system"""
    rag = CodebaseRAG(cache_dir=cache_dir)
    
    # Try loading from cache first
    if not rag._load_index():
        # Process all data sources
        rag.process_code_data(code_data)
        rag.process_models_structure(models_data)
        rag.process_file_tree(file_tree)
        
        # Build index
        rag.build_index()
    
    return rag

def count_tokens(text: str) -> int:
    """Rough estimation of token count"""
    return len(text.split())

async def main():
    rag = create_rag_system(
        code_data=output_code,
        models_data=output_models,
        file_tree=output_tree,
        cache_dir="./rag_cache"
    )
    
    query = "give me views from categorical_data app"
    
    # print("\n=== Original Query ===")
    print(query)
    
    # Get components
    docs = rag.query(query)
    context = rag._format_context(docs)
    system_prompt = rag.generate_system_prompt()
    user_prompt = f"""Question: {query}

    {context}

    Please provide a clear, detailed response based on the code context above."""
    
    # print("\n=== Token Counts by Message ===")
    
    
    print("\n=== LLM Messages ===")
    print("1. System message:")
    print(system_prompt)
    print("\n2. User message:")
    print(user_prompt)

    print("\n=== LLM Response ===")
    response = await rag.query_with_context(query)
    print(response)

    print(f"System message tokens: {count_tokens(system_prompt)}")
    print(f"User message tokens: {count_tokens(user_prompt)}")
    print(f"Total input tokens: {count_tokens(system_prompt) + count_tokens(user_prompt)}")
    print(f"\nResponse tokens: {count_tokens(response)}")

if __name__ == "__main__":
    asyncio.run(main())