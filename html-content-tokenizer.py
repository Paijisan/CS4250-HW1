import os
import sys
import re
import json
from datetime import datetime
from bs4 import BeautifulSoup
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

# Dictionary mapping folder name patterns to language codes
LANGUAGE_MAPPING = {
    'en': 'english',
    'eng': 'english',
    'english': 'english',
    'fr': 'french',
    'fre': 'french',
    'french': 'french',
    'es': 'spanish',
    'spa': 'spanish',
    'spanish': 'spanish',
    'de': 'german',
    'ger': 'german',
    'german': 'german',
    'it': 'italian',
    'ita': 'italian',
    'italian': 'italian',
    'pt': 'portuguese',
    'por': 'portuguese',
    'portuguese': 'portuguese',
    'ru': 'russian',
    'rus': 'russian',
    'russian': 'russian',
    'nl': 'dutch',
    'dut': 'dutch',
    'dutch': 'dutch',
    'ja': 'japanese',
    'jpn': 'japanese',
    'japanese': 'japanese',
    'zh': 'chinese',
    'chi': 'chinese',
    'chinese': 'chinese',
    'ar': 'arabic',
    'ara': 'arabic',
    'arabic': 'arabic',
}

def download_nltk_resources():
    """Download required NLTK resources."""
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        print("Downloading NLTK punkt tokenizer...")
        nltk.download('punkt', quiet=True)
    
    try:
        nltk.data.find('corpora/stopwords')
    except LookupError:
        print("Downloading NLTK stopwords...")
        nltk.download('stopwords', quiet=True)

def extract_content_from_html(html_content):
    """Extract the main text content from HTML, removing code blocks."""
    soup = BeautifulSoup(html_content, 'html.parser')
    
    # Remove all script and style elements
    for script_or_style in soup(['script', 'style']):
        script_or_style.decompose()
    
    # Remove all code blocks (pre, code tags)
    for code_block in soup(['pre', 'code']):
        code_block.decompose()
    
    # Get text
    text = soup.get_text()
    
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text

def tokenize_text(text, language=None):
    """
    Tokenize the extracted text.
    
    Args:
        text (str): The text to tokenize
        language (str, optional): Language name for stopword removal
    
    Returns:
        list: Tokenized words
    """
    # Simple tokenization using NLTK
    tokens = word_tokenize(text)
    
    # Remove stopwords if language is supported
    if language:
        try:
            stop_words = set(stopwords.words(language))
            tokens = [token for token in tokens if token.lower() not in stop_words]
        except Exception as e:
            print(f"Warning: Stopword removal for '{language}' failed - {str(e)}")
    
    return tokens

def process_html_file(file_path, language=None):
    """Process an HTML file: extract content and tokenize."""
    try:
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as file:
            html_content = file.read()
        
        # Extract main content
        text_content = extract_content_from_html(html_content)
        
        # Tokenize the content
        tokens = tokenize_text(text_content, language)
        
        return {
            'text': text_content[:100] + "..." if len(text_content) > 100 else text_content,  # Just a preview
            'tokens': tokens,
            'token_count': len(tokens)
        }
    except Exception as e:
        return {'error': str(e)}

def detect_language_from_folder_name(folder_path):
    """Try to detect language from folder name."""
    folder_name = os.path.basename(folder_path).lower()
    
    # Check for exact matches first
    if folder_name in LANGUAGE_MAPPING:
        return LANGUAGE_MAPPING[folder_name]
    
    # Check for partial matches (if folder contains language name)
    for lang_code, lang_name in LANGUAGE_MAPPING.items():
        if lang_code in folder_name or lang_name in folder_name:
            return lang_name
    
    # Default to English if no match
    return 'english'

def process_folder(folder_path):
    """Process all HTML files in a folder."""
    if not os.path.isdir(folder_path):
        print(f"Error: {folder_path} is not a directory")
        return
    
    # Detect language from folder name
    language = detect_language_from_folder_name(folder_path)
    print(f"Detected language: {language}")
    
    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(os.path.dirname(folder_path), f"tokenized_{os.path.basename(folder_path)}_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)
    
    # Find all HTML files
    html_files = []
    for root, _, files in os.walk(folder_path):
        for file in files:
            if file.lower().endswith(('.html', '.htm')):
                html_files.append(os.path.join(root, file))
    
    if not html_files:
        print(f"No HTML files found in {folder_path}")
        return
    
    print(f"Found {len(html_files)} HTML files to process")
    
    # Process each file
    results = {}
    total_tokens = 0
    processed_count = 0
    error_count = 0
    
    for file_path in html_files:
        rel_path = os.path.relpath(file_path, folder_path)
        print(f"Processing {rel_path}...", end='', flush=True)
        
        result = process_html_file(file_path, language)
        
        if 'error' in result:
            print(f" Error: {result['error']}")
            error_count += 1
        else:
            print(f" {result['token_count']} tokens")
            results[rel_path] = result
            total_tokens += result['token_count']
            
            # Save tokens to individual file
            token_file = os.path.join(output_dir, f"{os.path.splitext(rel_path)[0]}.tokens.txt")
            os.makedirs(os.path.dirname(token_file), exist_ok=True)
            with open(token_file, 'w', encoding='utf-8') as f:
                f.write('\n'.join(result['tokens']))
            
            processed_count += 1
    
    # Save summary
    summary = {
        'processed_files': processed_count,
        'error_files': error_count,
        'total_tokens': total_tokens,
        'language': language,
        'timestamp': timestamp
    }
    
    with open(os.path.join(output_dir, 'summary.json'), 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2)
    
    # Create a combined tokens file
    all_tokens = []
    for result in results.values():
        all_tokens.extend(result['tokens'])
    
    with open(os.path.join(output_dir, 'all_tokens.txt'), 'w', encoding='utf-8') as f:
        f.write('\n'.join(all_tokens))
    
    print(f"\nProcessing complete!")
    print(f"Processed {processed_count} files ({error_count} errors)")
    print(f"Total tokens: {total_tokens}")
    print(f"Results saved to: {output_dir}")
    
    # Create a readme file with instructions
    with open(os.path.join(output_dir, 'README.txt'), 'w', encoding='utf-8') as f:
        f.write("HTML Content Tokenizer Results\n")
        f.write("============================\n\n")
        f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Source folder: {folder_path}\n")
        f.write(f"Language: {language}\n")
        f.write(f"Files processed: {processed_count}\n")
        f.write(f"Total tokens: {total_tokens}\n\n")
        f.write("Files:\n")
        f.write("- summary.json: Contains processing statistics\n")
        f.write("- all_tokens.txt: Combined tokens from all files\n")
        f.write("- *.tokens.txt: Individual tokenized files\n")

def main():
    # Download NLTK resources
    download_nltk_resources()
    
    # Check arguments
    if len(sys.argv) > 1:
        folder_path = sys.argv[1]
        process_folder(folder_path)
    else:
        # No arguments provided, allow for input
        print("HTML Content Tokenizer")
        print("=====================")
        print("Drag a folder onto this executable to process all HTML files in it.")
        print("The folder name will be used to detect the language.")
        print("\nSupported languages:")
        for lang in sorted(set(LANGUAGE_MAPPING.values())):
            print(f"- {lang}")
        
        folder_path = input("\nOr enter the folder path manually: ").strip()
        if folder_path and os.path.isdir(folder_path):
            process_folder(folder_path)
        else:
            print("No valid folder provided. Exiting.")

if __name__ == "__main__":
    try:
        main()
        # Keep console open after completion
        input("\nPress Enter to exit...")
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        input("\nPress Enter to exit...")
