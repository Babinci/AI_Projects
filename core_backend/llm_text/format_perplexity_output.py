import json
import re
import sys
import logging
import os
from typing import Dict, Union, Any, Tuple, Optional

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def fix_polish_encoding(text: str) -> str:
    """
    Fixes encoding of Polish characters from various Unicode formats.
    
    Handles both \\u and \\U escape sequences, as well as HTML entities.
    
    Args:
        text (str): Text with escaped Polish characters
        
    Returns:
        str: Text with proper Polish characters
    """
    if not isinstance(text, str):
        logger.warning(f"Expected string input, got {type(text)}. Converting to string.")
        text = str(text)
    
    # Dictionary mapping Unicode escape sequences to Polish characters
    polish_chars = {
        # Lowercase letters
        r'\u0105': 'ą', r'\u0107': 'ć', r'\u0119': 'ę', r'\u0142': 'ł',
        r'\u0144': 'ń', r'\u00f3': 'ó', r'\u015b': 'ś', r'\u017a': 'ź',
        r'\u017c': 'ż',
        # Uppercase letters
        r'\u0104': 'Ą', r'\u0106': 'Ć', r'\u0118': 'Ę', r'\u0141': 'Ł',
        r'\u0143': 'Ń', r'\u00d3': 'Ó', r'\u015a': 'Ś', r'\u0179': 'Ź',
        r'\u017b': 'Ż',
        # HTML entities
        '&oacute;': 'ó', '&Oacute;': 'Ó',
        '&aogon;': 'ą', '&Aogon;': 'Ą',
        '&eogon;': 'ę', '&Eogon;': 'Ę',
        '&lstrok;': 'ł', '&Lstrok;': 'Ł',
        '&nacute;': 'ń', '&Nacute;': 'Ń',
        '&sacute;': 'ś', '&Sacute;': 'Ś',
        '&zacute;': 'ź', '&Zacute;': 'Ź',
        '&zdot;': 'ż', '&Zdot;': 'Ż',
        '&cacute;': 'ć', '&Cacute;': 'Ć'
    }
    
    # Replace all Unicode escape sequences
    for code, char in polish_chars.items():
        text = text.replace(code, char)
    
    # Handle \U format as well
    for code, char in polish_chars.items():
        if code.startswith(r'\u'):
            text = text.replace(code.replace(r'\u', r'\U000'), char)
    
    # Handle additional JSON-style escaping
    text = text.replace('\\\\u', '\\u')  # Handle double-escaped sequences
    
    # Use regex to catch any remaining Unicode escapes in the format \uXXXX
    def replace_unicode_escape(match):
        try:
            return chr(int(match.group(1), 16))
        except:
            return match.group(0)
    
    text = re.sub(r'\\u([0-9a-fA-F]{4})', replace_unicode_escape, text)
    
    return text

def extract_json_content(input_data: Union[str, Dict]) -> str:
    """
    Extracts content field from JSON data, handling different structures.
    
    Args:
        input_data: Either a JSON string or already parsed dict
        
    Returns:
        str: The extracted content string
        
    Raises:
        ValueError: If content can't be extracted from the input data
    """
    try:
        # If we have a string, try to parse it as JSON
        if isinstance(input_data, str):
            try:
                data = json.loads(input_data)
            except json.JSONDecodeError as e:
                logger.error(f"Failed to parse JSON: {e}")
                raise ValueError(f"Invalid JSON format: {e}")
        else:
            data = input_data
            
        # Try different ways to access content field
        if isinstance(data, dict):
            # Try direct access to 'content' field
            if 'content' in data:
                return data['content']
            
            # Check for nested 'content' field
            for key, value in data.items():
                if isinstance(value, dict) and 'content' in value:
                    return value['content']
            
            # If no 'content' field found, but there's just one key, try its value
            if len(data) == 1:
                value = next(iter(data.values()))
                if isinstance(value, str):
                    return value
                
            # If the dict contains potentially useful string data, return the whole dict
            # Convert to string to handle further
            return json.dumps(data)
        
        # If data is a list, try to extract useful content
        elif isinstance(data, list):
            # If it's a list of dicts, look for one with 'content'
            for item in data:
                if isinstance(item, dict) and 'content' in item:
                    return item['content']
            
            # Otherwise convert the whole list to string
            return json.dumps(data)
        
        # If data is a scalar value like string, return it directly
        elif isinstance(data, str):
            return data
        else:
            # For other types, convert to string
            return str(data)
            
    except Exception as e:
        logger.error(f"Error extracting content: {e}")
        raise ValueError(f"Could not extract content from input data: {e}")

def parse_sections(content: str) -> list:
    """
    Parses content and splits it into sections based on date headers.
    Handles multiple date formats.
    
    Args:
        content (str): The content to parse
        
    Returns:
        list: List of (date, section_content) tuples
    """
    # Create more flexible pattern for date headers
    # Match both ## DD.MM.YYYY and ## YYYY-MM-DD formats
    date_patterns = [
        r'##\s+(\d{2}\.\d{2}\.\d{4})',  # DD.MM.YYYY
        r'##\s+(\d{4}-\d{2}-\d{2})',    # YYYY-MM-DD
        r'##\s+(\d{2}/\d{2}/\d{4})',    # DD/MM/YYYY
        r'##\s+(\d{4}/\d{2}/\d{2})'     # YYYY/MM/DD
    ]
    
    sections = []
    current_content = content
    
    # First try to split by date headers with strict pattern matching
    for pattern in date_patterns:
        splits = re.split(pattern, current_content)
        if len(splits) > 1:
            # First element is any content before first date header
            if splits[0].strip():
                sections.append(("Introduction", splits[0].strip()))
                
            # Process date-content pairs
            for i in range(1, len(splits), 2):
                if i+1 < len(splits):
                    date = splits[i].strip()
                    section_content = splits[i+1].strip()
                    sections.append((date, section_content))
            
            return sections
    
    # If no date sections found, try to look for any headers as section dividers
    header_pattern = r'(#{1,3})\s+(.+?)\s*(?=\n|$)'
    headers = re.finditer(header_pattern, content)
    last_pos = 0
    
    for match in headers:
        # Get content before current header
        if match.start() > last_pos:
            prev_content = content[last_pos:match.start()].strip()
            if prev_content:
                if not sections:  # If this is the first section, call it Introduction
                    sections.append(("Introduction", prev_content))
                else:
                    # Add content to previous section
                    prev_header = sections[-1][0]
                    prev_content = sections[-1][1] + "\n\n" + prev_content
                    sections[-1] = (prev_header, prev_content)
        
        # Add new section
        header_level = len(match.group(1))  # Number of # characters
        header_text = match.group(2)
        next_content_start = match.end()
        sections.append((header_text, ""))  # Content will be added on next iteration
        last_pos = next_content_start
    
    # Add the last section's content
    if last_pos < len(content):
        if sections:
            last_header = sections[-1][0]
            last_content = content[last_pos:].strip()
            sections[-1] = (last_header, last_content)
        else:
            # If no headers found at all, treat the entire content as one section
            sections.append(("Content", content.strip()))
    
    return sections

def parse_news_items(section_content: str) -> list:
    """
    Parses section content to extract individual news items.
    
    Args:
        section_content (str): Content of a section
        
    Returns:
        list: List of (title, body, tags) tuples
    """
    news_items = []
    
    # First try to split by bold titles using regex pattern **title**
    # This handles cases where titles are clearly marked with double asterisks
    title_pattern = r'\*\*(.+?)\*\*'
    splits = re.split(title_pattern, section_content)
    
    if len(splits) > 1:
        # Process title-content pairs
        for i in range(1, len(splits), 2):
            if i+1 < len(splits):
                title = splits[i].strip()
                body = splits[i+1].strip()
                
                # Extract tags
                tags, body = extract_tags(body)
                
                news_items.append((title, body, tags))
    else:
        # Alternative approach: try to split by newlines and look for patterns
        # This handles cases where the format isn't consistent
        lines = section_content.split('\n')
        current_title = None
        current_body = []
        current_tags = ""
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
                
            # Check if line is a title (heuristic: short line, possibly ending with colon)
            if current_title is None and len(line) < 100 and not line.startswith('#'):
                current_title = line.strip('*: ')
            # Check if line contains tags
            elif line.lower().startswith('tagi:') or line.lower().startswith('tags:'):
                current_tags = line
                # If we have a title and body, add the item
                if current_title and current_body:
                    body_text = '\n'.join(current_body).strip()
                    tags, _ = extract_tags(current_tags)
                    news_items.append((current_title, body_text, tags))
                    # Reset for next item
                    current_title = None
                    current_body = []
                    current_tags = ""
            # Otherwise, it's part of the body
            elif current_title is not None:
                current_body.append(line)
        
        # Don't forget the last item
        if current_title and current_body:
            body_text = '\n'.join(current_body).strip()
            tags, _ = extract_tags(current_tags) if current_tags else ("", "")
            news_items.append((current_title, body_text, tags))
    
    return news_items

def extract_tags(text: str) -> Tuple[str, str]:
    """
    Extracts tags from text and returns both the tags and the text without tags.
    
    Args:
        text (str): Text that may contain tags
        
    Returns:
        tuple: (tags, text_without_tags)
    """
    # Common patterns for tags
    tag_patterns = [
        r'(?:Tagi|Tags|TAGI|TAGS):\s*((?:#\w+(?:[-_]\w+)*(?:\s+|$))+)',  # Tagi: #tag1 #tag2
        r'(?:Tagi|Tags|TAGI|TAGS):\s*((?:\w+(?:,\s*\w+)*)+)',  # Tagi: tag1, tag2
        r'(\[(?:#\w+(?:[-_]\w+)*(?:\s+|$))+\])',  # [#tag1 #tag2]
        r'(#\w+(?:[-_]\w+)*(?:\s+#\w+(?:[-_]\w+)*)*)'  # #tag1 #tag2
    ]
    
    for pattern in tag_patterns:
        match = re.search(pattern, text)
        if match:
            tags = match.group(1).strip()
            # Remove the tags from the text
            cleaned_text = text.replace(match.group(0), "").strip()
            return tags, cleaned_text
    
    # No tags found
    return "", text

def format_news_to_markdown(json_content: Union[str, Dict]) -> str:
    """
    Converts news data from JSON to nicely formatted Markdown.
    
    Args:
        json_content: Either a JSON string or already parsed dict
        
    Returns:
        str: Formatted Markdown content
        
    Raises:
        ValueError: If content can't be processed properly
    """
    try:
        # Extract content from the JSON
        content = extract_json_content(json_content)
        
        # Fix Polish character encoding
        content = fix_polish_encoding(content)
        
        # Begin building the Markdown document
        markdown = "# Przegląd najważniejszych wydarzeń\n\n"
        
        # Parse content into sections
        sections = parse_sections(content)
        
        if not sections:
            logger.warning("No sections found in the content.")
            return markdown + "_Nie znaleziono sekcji w dokumencie._"
        
        # Process each section
        for date, section_content in sections:
            # Skip empty sections
            if not section_content.strip():
                continue
                
            # Add section header
            if date.lower() != "introduction":
                markdown += f"\n## {date}\n\n"
            
            # Parse news items in this section
            news_items = parse_news_items(section_content)
            
            if not news_items:
                # If no structured news items found, just add the raw content
                markdown += f"{section_content}\n\n"
                continue
                
            # Format each news item
            for title, body, tags in news_items:
                # Skip items with empty titles
                if not title.strip():
                    continue
                    
                markdown += f"**{title}**  \n{body}  \n"
                if tags:
                    markdown += f"*Tagi: {tags}*\n\n"
                else:
                    markdown += "\n"
        
        return markdown
        
    except Exception as e:
        logger.error(f"Error formatting news to Markdown: {e}")
        return f"# Błąd przetwarzania\n\nWystąpił błąd podczas konwersji do formatu Markdown: {str(e)}"

def detect_encoding(file_path: str) -> str:
    """
    Attempts to detect the encoding of a file.
    
    Args:
        file_path (str): Path to the file
        
    Returns:
        str: Detected encoding or 'utf-8' as fallback
    """
    encodings = ['utf-8', 'cp1250', 'iso-8859-2', 'windows-1250']
    
    for enc in encodings:
        try:
            with open(file_path, 'r', encoding=enc) as f:
                f.read()
                return enc
        except UnicodeDecodeError:
            continue
    
    logger.warning(f"Could not detect encoding for {file_path}, defaulting to utf-8")
    return 'utf-8'

def validate_output_path(output_file: str) -> str:
    """
    Validates and adjusts output file path if necessary.
    
    Args:
        output_file (str): Proposed output file path
        
    Returns:
        str: Valid output file path
    """
    # Check if directory exists
    output_dir = os.path.dirname(output_file)
    if output_dir and not os.path.exists(output_dir):
        try:
            os.makedirs(output_dir)
            logger.info(f"Created directory: {output_dir}")
        except OSError as e:
            logger.warning(f"Could not create directory {output_dir}: {e}")
            # Fallback to current directory
            output_file = os.path.basename(output_file)
    
    # Check if file already exists
    if os.path.exists(output_file):
        base, ext = os.path.splitext(output_file)
        i = 1
        while os.path.exists(f"{base}_{i}{ext}"):
            i += 1
        output_file = f"{base}_{i}{ext}"
        logger.info(f"Output file already exists, using {output_file} instead")
    
    return output_file

def main():
    """
    Main function to process command line arguments and convert JSON to Markdown.
    """
    if len(sys.argv) > 1:
        # Process file
        input_file = sys.argv[1]
        try:
            # Detect encoding
            encoding = detect_encoding(input_file)
            logger.info(f"Using encoding: {encoding}")
            
            # Read the file
            with open(input_file, 'r', encoding=encoding) as f:
                content = f.read()
            
            # Process the content
            markdown = format_news_to_markdown(content)
            
            # Determine output file name
            output_file = input_file.rsplit('.', 1)[0] + '.md'
            output_file = validate_output_path(output_file)
            
            # Write the output
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(markdown)
                
            logger.info(f"Successfully wrote formatted text to: {output_file}")
            print(f"Pomyślnie zapisano sformatowany tekst do: {output_file}")
        
        except Exception as e:
            logger.error(f"Error processing file: {e}")
            print(f"Błąd: {str(e)}")
    else:
        # Handle stdin/stdout if no file is provided
        try:
            content = sys.stdin.read()
            markdown = format_news_to_markdown(content)
            sys.stdout.write(markdown)
        except Exception as e:
            logger.error(f"Error processing input: {e}")
            print(f"Błąd: {str(e)}")
            print("Użycie: python format_polish_text.py <plik_json>")
            print("   lub: cat plik.json | python format_polish_text.py > wynik.md")
            sys.exit(1)

if __name__ == "__main__":
    main()