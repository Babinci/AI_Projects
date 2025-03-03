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
    EMBEDDING_DIM = 1024

    def __init__(self, cache_dir: Optional[str] = None):
        self.structural_documents = []  # For models and file structure
        self.code_documents = []  # For code chunks
        self.structural_index = None
        self.code_index = None
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
        
    def _detect_model_terms(self, query: str) -> List[str]:
        """Detect model names in query"""
        query_terms = query.lower().split()
        return [
            model_name for model_name in self.model_map.keys()
            if model_name.lower() in query_terms
        ]
        
    def build_index(self):
        """Build separate indices for structural and code content"""
        if not self.structural_documents and not self.code_documents:
            raise ValueError("No documents added yet")
            
        # Build structural index
        structural_embeddings = []
        for doc in self.structural_documents:
            if doc.embedding is None:
                doc.embedding = self._get_embedding(doc.content)
            structural_embeddings.append(doc.embedding)
        
        if structural_embeddings:
            self.structural_index = faiss.IndexIDMap(faiss.IndexFlatIP(self.EMBEDDING_DIM))
            self.structural_index.add_with_ids(
                np.array(structural_embeddings).astype('float32'),
                np.arange(len(self.structural_documents))
            )
            
        # Build code index
        code_embeddings = []
        for doc in self.code_documents:
            if doc.embedding is None:
                doc.embedding = self._get_embedding(doc.content)
            code_embeddings.append(doc.embedding)
            
        if code_embeddings:
            self.code_index = faiss.IndexIDMap(faiss.IndexFlatIP(self.EMBEDDING_DIM))
            self.code_index.add_with_ids(
                np.array(code_embeddings).astype('float32'),
                np.arange(len(self.code_documents))
            )

    def process_code_data(self, code_data: List[Dict]):
        """Process code files with view file prioritization"""
        for file_info in code_data:
            if not file_info.get('content') and not file_info.get('chunks'):
                continue

            file_path = file_info.get('path', '')
            file_type = file_info.get('type', 'unknown').lower()
            
            # Boost view files and their chunks
            view_boost = 1.5 if file_type == 'view' or 'views.py' in file_path else 1.0

            if file_info.get('chunks'):
                for chunk in file_info['chunks']:
                    content = f"File: {file_path}\nType: {chunk['type']}\n\n{chunk['content']}"
                    metadata = {
                        'type': 'code_chunk',
                        'file_type': file_type,
                        'path': file_path,
                        'chunk_type': chunk['type'],
                        'is_view_file': ('views.py' in file_path),
                        'relationships': chunk['metadata'].get('relationships', []),
                        'dependencies': chunk['metadata'].get('dependencies', []),
                        'view_boost': view_boost
                    }
                    
                    # Add view-specific context
                    if metadata['is_view_file']:
                        content = f"[VIEW_FILE] {content}"
                        metadata['view_boost'] = 1.7  # Higher boost for direct matches

                    self.code_documents.append(Document(
                        content=content,
                        metadata=metadata
                    ))

    def process_structure_data(self, tree_data: Dict, models_data: List[Dict]):
        """Process file tree and models into structural_documents"""
        # Process file tree
        for app_name, app_info in tree_data.items():
            content = f"App: {app_name}\nFiles:\n"
            files = app_info.get('files', [])
            for file in files:
                content += f"- {file}\n"
            if app_info.get('subdirs'):
                content += "\nSubdirectories:\n"
                for subdir in app_info['subdirs']:
                    content += f"- {subdir}\n"
                    
            self.structural_documents.append(Document(
                content=content,
                metadata={
                    'type': 'file_structure',
                    'app_name': app_name,
                    'files': files
                }
            ))
            
        # Process models
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
                self.structural_documents.append(doc)
                self.model_map[model['name']] = doc

    def _extract_context_terms(self, docs: List[Document]) -> Dict[str, float]:
        """Extract relevant terms from structural documents with weights"""
        terms = defaultdict(float)
        
        for doc in docs:
            # Extract app names
            app_name = doc.metadata.get('app_name', '').lower()
            if app_name:
                terms[app_name] += 1.2
                
            if doc.metadata['type'] == 'model':
                # Add model name with high weight
                model_name = doc.metadata['model_name'].lower()
                terms[model_name] += 1.5
                
                # Add related model names
                for rel in doc.metadata.get('relationships', []):
                    target = rel['target'].lower()
                    terms[target] += 0.8
                    
            elif doc.metadata['type'] == 'file_structure':
                # Add relevant file names (excluding extensions and common files)
                for file in doc.metadata.get('files', []):
                    name = Path(file).stem.lower()
                    if name not in ['__init__', 'apps']:
                        terms[name] += 1.0
                        
        return terms

    def _enhance_query(self, query: str, context_terms: Dict[str, float]) -> str:
        """Enhance the query with structural context"""
        enhanced_query = query
        
        # Add top weighted terms to query
        top_terms = sorted(context_terms.items(), key=lambda x: x[1], reverse=True)[:3]
        if top_terms:
            term_str = ' '.join(term for term, _ in top_terms)
            enhanced_query += f" context: {term_str}"
            
        return enhanced_query


    def _query_structural(self, query_text: str, k: int) -> List[Document]:
        """Query structural index"""
        if not self.structural_index:
            return []
            
        query_embedding = self._get_embedding(query_text)
        distances, indices = self.structural_index.search(
            query_embedding.reshape(1, -1).astype('float32'),
            k
        )
        
        return [self.structural_documents[idx] for idx in indices[0]]

    def _query_code(self, query_text: str, k: int) -> List[Document]:
        """Query code index"""
        if not self.code_index:
            return []
            
        query_embedding = self._get_embedding(query_text)
        distances, indices = self.code_index.search(
            query_embedding.reshape(1, -1).astype('float32'),
            k
        )
        
        return [self.code_documents[idx] for idx in indices[0]]

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
            
        return context

    async def query_with_context(
        self, 
        query_text: str, 
        model_name: str = "deepseek-r1:7b"
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

    def _save_index(self):
        """Save indices and data to cache"""
        if not self.cache_dir:
            return
            
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Save structural index and documents
        if self.structural_index:
            faiss.write_index(self.structural_index, str(self.cache_dir / "structural_index.faiss"))
        
        # Save code index and documents
        if self.code_index:
            faiss.write_index(self.code_index, str(self.cache_dir / "code_index.faiss"))
        
        # Save documents and model map
        with open(self.cache_dir / "data.pkl", "wb") as f:
            pickle.dump({
                'structural_documents': self.structural_documents,
                'code_documents': self.code_documents,
                'model_map': self.model_map
            }, f)
    
    def _load_index(self) -> bool:
        """Load indices and data from cache"""
        if not self.cache_dir:
            return False
            
        structural_path = self.cache_dir / "structural_index.faiss"
        code_path = self.cache_dir / "code_index.faiss"
        data_path = self.cache_dir / "data.pkl"
        
        if not (structural_path.exists() and code_path.exists() and data_path.exists()):
            return False
            
        try:
            # Load indices
            self.structural_index = faiss.read_index(str(structural_path))
            self.code_index = faiss.read_index(str(code_path))
            
            # Load documents and model map
            with open(data_path, "rb") as f:
                data = pickle.load(f)
                self.structural_documents = data['structural_documents']
                self.code_documents = data['code_documents']
                self.model_map = data['model_map']
                
            return True
        except Exception as e:
            print(f"Error loading cache: {e}")
            return False
    
    def _rerank_results(self, combined_results, query_text, context_terms):
        """Enhanced reranking with view file prioritization"""
        query_terms = query_text.lower().split()
        scored_results = []
        
        for doc in combined_results:
            score = doc.metadata.get('view_boost', 1.0)
            
            # Direct view file match boost
            if doc.metadata.get('is_view_file') and 'view' in query_terms:
                score *= 2.0
                
            # Existing scoring logic
            if doc.metadata['type'] == 'model' and doc.metadata['model_name'] in query_terms:
                score *= 1.5
                
            # Add method name relevance scoring
            if doc.metadata.get('method_name'):
                method_terms = doc.metadata['method_name'].lower().split('_')
                score += 0.2 * sum(1 for term in query_terms if term in method_terms)
            
            scored_results.append((doc, score))
            
        sorted_results = sorted(scored_results, key=lambda x: x[1], reverse=True)[:15]
        return [doc for (doc, _) in sorted_results]

    def query(self, query_text: str, k: int = 8, code_k: int = 30) -> List[Document]:
        """Enhanced two-phase query with view file detection"""
        # Phase 1: Check for view-related terms
        view_terms = {'view', 'views', 'endpoint', 'api'}
        if any(term in query_text.lower() for term in view_terms):
            context_terms = {'view_file': 1.7}
        else:
            context_terms = self._extract_context_terms(self._query_structural(query_text, 3))

        # Phase 2: Query expansion for view files
        if 'view' in query_text.lower():
            expanded_query = f"{query_text} [views.py] [API endpoints] [View classes]"
            code_results = self._query_code(expanded_query, code_k)
        else:
            code_results = self._query_code(query_text, code_k)

        # Combine and return boosted results
        return self._rerank_results(code_results, query_text, context_terms)

def create_rag_system(
    code_data: List[Dict],
    models_data: List[Dict],
    file_tree: Dict,
    cache_dir: Optional[str] = None
) -> CodebaseRAG:
    """Create and initialize the enhanced RAG system"""
    rag = CodebaseRAG(cache_dir=cache_dir)
    
    # Try loading from cache first
    if not rag._load_index():
        # Process structure and code separately
        rag.process_structure_data(file_tree, models_data)
        rag.process_code_data(code_data)
        
        # Build indices
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