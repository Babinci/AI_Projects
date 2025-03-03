"""
Google Docs Embedding Processor with Semantic Chunking
"""
from dotenv import load_dotenv
from google.oauth2 import service_account
from googleapiclient.discovery import build
import os
import json
from datetime import datetime
from dataclasses import dataclass
from typing import Dict, List, Any
import ast
import re
from typing import Dict, List, Optional
from dataclasses import dataclass
import hashlib


# Configuration
load_dotenv("/home/wojtek/AI_Projects/credentials/.env")
CREDENTIALS_PATH = os.getenv("GOOGLE_DOCS_CREDENTIALS_PATH")
DOCUMENT_IDS = {
    "jobassistant models":"1rFpXPHDwuPxFjtVpfTURAzvrY3UvyrZdkza7Jhgp1Zw",
    "jobassistant folder tree":"1N_u96gXBpUrVz1KBZBENQGBU2iBiqoNLzVVs4tunVpc",
    "jobassistant relevant code":"1677H1qNjhfH7I44BBEhVrm00lR499yZDXY4h9MMruvM"
}
def initialize_google_docs_service():
    """Authenticate and create Google Docs API service instance"""
    creds = service_account.Credentials.from_service_account_file(
        CREDENTIALS_PATH,
        scopes=['https://www.googleapis.com/auth/documents.readonly']
    )
    return build('docs', 'v1', credentials=creds)

def fetch_document(service, document_id):
    """Fetch complete document structure from Google Docs API"""
    try:
        return service.documents().get(documentId=document_id).execute()
    except Exception as e:
        print(f"Error fetching document {document_id}: {str(e)}")
        return None

service = initialize_google_docs_service()


def parse_google_doc_structure(document):
    """Process Google Docs content into structured code files"""
    parsed_data = []
    current_script = None
    current_content = []

    for element in document.get('body', {}).get('content', []):
        if 'paragraph' not in element:
            continue

        para = element['paragraph']
        style = para.get('paragraphStyle', {}).get('namedStyleType', 'NORMAL_TEXT')

        # Extract clean text content
        text = ''.join([
            e.get('textRun', {}).get('content', '')
            for e in para.get('elements', [])
        ]).strip()

        # Handle header (script path)
        if style == 'HEADING_2':
            if current_script is not None:
                parsed_data.append({
                    'script_path': current_script,
                    'content': '\n'.join(current_content).strip()
                })
                current_content = []
            current_script = text
        # Handle code content
        elif style == 'NORMAL_TEXT' and current_script is not None:
            code_line = ''.join([
                e.get('textRun', {}).get('content', '').rstrip('\n')
                for e in para.get('elements', [])
            ])
            if code_line:
                current_content.append(code_line)

    # Add final section
    if current_script is not None:
        parsed_data.append({
            'script_path': current_script,
            'content': '\n'.join(current_content).strip()
        })

    return parsed_data
def parse_models_document(document):
    """Process Google Docs model relationships into structured data"""
    structured_models = []
    current_app = None
    current_model = None

    for element in document.get('body', {}).get('content', []):
        if 'paragraph' not in element:
            continue

        text = ''.join([
            e.get('textRun', {}).get('content', '')
            for e in element['paragraph'].get('elements', [])
        ]).strip()

        # Detect app section
        if text.startswith('App: '):
            current_app = text.split('App: ')[1]
            current_model = None
            structured_models.append({
                'app_name': current_app,
                'models': []
            })

        # Detect model section
        elif text.startswith('Model: '):
            model_info = text.split('Model: ')[1]
            model_name = model_info.split(' (')[0]  # Remove "(no relations)" if present
            current_model = {
                'name': model_name,
                'relationships': []
            }
            structured_models[-1]['models'].append(current_model)

        # Process relationships
        elif current_model and text.startswith('• '):
            rel_type, target = text[2:].split(' -> ')
            current_model['relationships'].append({
                'type': rel_type.strip(),
                'target': target.strip()
            })

    return structured_models

def parse_folder_tree(document):
    """Process Google Docs folder tree into structured dictionary"""
    folder_structure = {}
    current_app = None

    for element in document.get('body', {}).get('content', []):
        if 'paragraph' not in element:
            continue

        text = ''.join([
            e.get('textRun', {}).get('content', '')
            for e in element['paragraph'].get('elements', [])
        ]).strip()

        # Process app directories
        if text.startswith('├──') or text.startswith('└──'):
            app_name = text.split('──')[1].strip()
            if not text.endswith('.py'):  # It's a directory
                current_app = app_name
                folder_structure[current_app] = {
                    'files': [],
                    'subdirs': []
                }

        # Process files and subdirectories within apps
        elif text.startswith('│   ├──') or text.startswith('│   └──'):
            if current_app:
                item_name = text.split('──')[1].strip()
                if item_name.endswith('.py'):
                    folder_structure[current_app]['files'].append(item_name)
                else:
                    folder_structure[current_app]['subdirs'].append(item_name)

    return folder_structure

@dataclass
class CodeChunk:
    id: str
    type: str  # 'class', 'function', 'import', 'other'
    content: str
    start_line: int
    end_line: int
    metadata: Dict

class CodeProcessor:
    def __init__(self, max_chunk_size=1000):
        self.max_chunk_size = max_chunk_size

    def process_file(self, file_data: Dict) -> Dict:
        if not file_data['script_path'] or not file_data['content']:
            return self._process_empty_file(file_data)

        try:
            tree = ast.parse(file_data['content'])
            chunks = self._parse_ast(tree, file_data['content'])
        except SyntaxError:
            chunks = self._fallback_chunking(file_data['content'])

        return {
            'file_id': self._generate_file_id(file_data['script_path']),
            'path': file_data['script_path'],
            'type': self._detect_file_type(file_data['script_path']),
            'content': file_data['content'],
            'chunks': chunks,
            'metadata': self._extract_file_metadata(file_data['content'])
        }

    def _process_empty_file(self, file_data: Dict) -> Dict:
        return {
            'file_id': self._generate_file_id(file_data['script_path']),
            'path': file_data['script_path'],
            'type': self._detect_file_type(file_data['script_path']),
            'content': '',
            'chunks': [],
            'metadata': {'is_empty': True}
        }

    def _fallback_chunking(self, content: str) -> List[Dict]:
        """Fallback method for when AST parsing fails"""
        chunks = []
        lines = content.split('\n')
        current_chunk = []
        current_chunk_start = 0

        for i, line in enumerate(lines):
            current_chunk.append(line)

            # Start new chunk at class or function definitions
            if (line.strip().startswith('class ') or line.strip().startswith('def ')) and current_chunk:
                if len(current_chunk) > 1:  # Don't create empty chunks
                    chunk_content = '\n'.join(current_chunk[:-1])
                    chunks.append({
                        'id': self._generate_chunk_id(chunk_content),
                        'type': 'other',
                        'content': chunk_content,
                        'start_line': current_chunk_start,
                        'end_line': i,
                        'metadata': {}
                    })
                current_chunk = [line]
                current_chunk_start = i

        # Add the last chunk
        if current_chunk:
            chunk_content = '\n'.join(current_chunk)
            chunks.append({
                'id': self._generate_chunk_id(chunk_content),
                'type': 'other',
                'content': chunk_content,
                'start_line': current_chunk_start,
                'end_line': len(lines),
                'metadata': {}
            })

        return chunks

    def _parse_ast(self, tree: ast.AST, content: str) -> List[Dict]:
        chunks = []
        for node in ast.walk(tree):
            if isinstance(node, (ast.ClassDef, ast.FunctionDef)):
                chunk = self._create_chunk_from_node(node, content)
                if chunk:
                    chunks.append(chunk)

                    # Handle large classes/functions by creating sub-chunks
                    if len(chunk['content']) > self.max_chunk_size:
                        sub_chunks = self._create_sub_chunks(chunk)
                        chunks.extend(sub_chunks)

        return chunks

    def _create_chunk_from_node(self, node: ast.AST, content: str) -> Optional[Dict]:
        source_lines = content.split('\n')
        chunk_content = '\n'.join(source_lines[node.lineno-1:node.end_lineno])

        metadata = {
            'name': node.name if hasattr(node, 'name') else None,
            'type': type(node).__name__,
            'dependencies': self._extract_dependencies(node),
            'relationships': self._extract_relationships(node)
        }

        return {
            'id': self._generate_chunk_id(chunk_content),
            'type': 'class' if isinstance(node, ast.ClassDef) else 'function',
            'content': chunk_content,
            'start_line': node.lineno,
            'end_line': node.end_lineno,
            'metadata': metadata
        }

    def _create_sub_chunks(self, chunk: Dict) -> List[Dict]:
        """Split large chunks into smaller, logical pieces"""
        sub_chunks = []
        content = chunk['content']

        # Split by methods if it's a class
        if chunk['type'] == 'class':
            method_pattern = r'(\s+def\s+[^\n]+:)'
            methods = re.split(method_pattern, content)
            current_pos = 0

            for i, section in enumerate(methods):
                if section.strip().startswith('def'):
                    sub_chunks.append({
                        'id': f"{chunk['id']}_method_{i}",
                        'type': 'method',
                        'content': section,
                        'parent_chunk_id': chunk['id'],
                        'metadata': {
                            'parent_class': chunk['metadata']['name'],
                            'method_name': re.search(r'def\s+([^\(]+)', section).group(1)
                        }
                    })
                current_pos += len(section)

        return sub_chunks

    def _extract_relationships(self, node: ast.AST) -> List[Dict]:
        relationships = []
        for child in ast.walk(node):
            if isinstance(child, ast.Call) and isinstance(child.func, ast.Name):
                if child.func.id in ['ForeignKey', 'OneToOneField', 'ManyToManyField']:
                    relationships.append({
                        'type': child.func.id,
                        'target': self._get_relationship_target(child)
                    })
        return relationships

    def _extract_dependencies(self, node: ast.AST) -> List[str]:
        deps = []
        for child in ast.walk(node):
            if isinstance(child, ast.Import):
                deps.extend(name.name for name in child.names)
            elif isinstance(child, ast.ImportFrom):
                if child.module:
                    deps.append(child.module)
        return deps

    def _detect_file_type(self, path: str) -> str:
        """
        Detect Django file type based on filename and path.

        Args:
            path (str): File path

        Returns:
            str: Detected file type
        """
        if 'models.py' in path:
            return 'model'
        elif 'views.py' in path:
            return 'view'
        elif 'serializers.py' in path:
            return 'serializer'
        elif 'admin.py' in path:
            return 'admin'
        elif 'tests' in path:  # Handles both tests.py and tests/ directory
            return 'test'
        elif 'signals.py' in path:
            return 'signal'
        elif 'urls.py' in path:
            return 'url'
        elif 'forms.py' in path:
            return 'form'
        elif 'managers.py' in path:
            return 'manager'
        elif 'middleware.py' in path:
            return 'middleware'
        elif 'validators.py' in path:
            return 'validator'
        elif 'apps.py' in path:
            return 'app_config'
        elif 'tasks.py' in path:  # For Celery/async tasks
            return 'task'
        elif 'permissions.py' in path:
            return 'permission'
        elif 'filters.py' in path:  # For DRF or Django filters
            return 'filter'
        elif 'exceptions.py' in path:
            return 'exception'
        elif 'constants.py' in path:
            return 'constant'
        elif 'utils.py' in path or 'helpers.py' in path:
            return 'utility'
        return 'other'

    def _generate_file_id(self, path: str) -> str:
        return hashlib.md5(path.encode()).hexdigest()

    def _generate_chunk_id(self, content: str) -> str:
        return hashlib.md5(content.encode()).hexdigest()

    def _extract_file_metadata(self, content: str) -> Dict:
        return {
            'size': len(content),
            'imports': self._extract_imports(content),
            'has_classes': bool(re.search(r'class\s+\w+', content)),
            'has_functions': bool(re.search(r'def\s+\w+', content))
        }

    def _extract_imports(self, content: str) -> List[str]:
        imports = []
        try:
            tree = ast.parse(content)
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    imports.extend(name.name for name in node.names)
                elif isinstance(node, ast.ImportFrom):
                    if node.module:
                        imports.append(node.module)
        except:
            pass
        return imports

def process_codebase(files: List[Dict]) -> List[Dict]:
    processor = CodeProcessor()
    processed_files = []

    for file_data in files:
        processed = processor.process_file(file_data)
        processed_files.append(processed)

    return processed_files


###fetching codebase
document_code = fetch_document(service, DOCUMENT_IDS["jobassistant relevant code"])
structured_data = parse_google_doc_structure(document_code)
output_code = process_codebase(structured_data)
with open('codebase_structure.json', 'w') as f:
    json.dump(output_code, f, indent=2)
document_tree = fetch_document(service, DOCUMENT_IDS['jobassistant folder tree'])
output_tree = parse_folder_tree(document_tree)
with open('folder_tree.json', 'w') as f:
    json.dump(output_tree, f, indent=2)

document_models = fetch_document(service, DOCUMENT_IDS['jobassistant models'])
output_models = parse_models_document(document_models)
with open('model_relationships.json', 'w') as f:
    json.dump(output_models, f, indent=2)