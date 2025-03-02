#!/usr/bin/env python3

import nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet
import jieba
import jieba.posseg as pseg
import tinysegmenter
import argparse
import sys
import json

# Try importing optional libraries, with graceful fallbacks
try:
    from fugashi import Tagger
    has_fugashi = True
except ImportError:
    has_fugashi = False

try:
    from konlpy.tag import Okt
    has_konlpy = True
except ImportError:
    has_konlpy = False

# Download required NLTK resources for English
try:
    nltk.download('punkt', quiet=True)
    nltk.download('wordnet', quiet=True)
    nltk.download('averaged_perceptron_tagger', quiet=True)
except:
    print("Warning: NLTK resources could not be downloaded. English processing may be limited.")

class MultilingualStemmer:
    def __init__(self, verbose=False):
        self.verbose = verbose
        
        # English processing tools
        self.lemmatizer = WordNetLemmatizer()
        
        # Japanese tokenizer (MeCab wrapper)
        if has_fugashi:
            try:
                self.ja_tagger = Tagger('-Owakati')
                if self.verbose:
                    print("Japanese MeCab initialized successfully.")
            except:
                self.ja_tagger = None
                if self.verbose:
                    print("Japanese MeCab not available despite fugashi being installed.")
        else:
            self.ja_tagger = None
            if self.verbose:
                print("Japanese MeCab not available. Install with: pip install fugashi[unidic]")
        
        # Backup Japanese tokenizer
        self.ja_segmenter = tinysegmenter.TinySegmenter()
        
        # Korean processor (KoNLPy with Okt/Twitter)
        if has_konlpy:
            try:
                self.ko_processor = Okt()
                if self.verbose:
                    print("Korean processor initialized successfully.")
            except:
                self.ko_processor = None
                if self.verbose:
                    print("Korean processor failed to initialize despite konlpy being installed.")
        else:
            self.ko_processor = None
            if self.verbose:
                print("Korean processor not available. Install with: pip install konlpy")
        
        # Chinese tokenizer
        jieba.initialize()
        if self.verbose:
            print("Chinese processor initialized successfully.")
    
    def get_wordnet_pos(self, tag):
        """Map NLTK POS tag to WordNet POS tag for English"""
        tag_dict = {
            'J': wordnet.ADJ,
            'N': wordnet.NOUN,
            'V': wordnet.VERB,
            'R': wordnet.ADV
        }
        return tag_dict.get(tag[0].upper(), wordnet.NOUN)
    
    def detect_language(self, text):
        """Simple language detection based on character sets"""
        # Check for Chinese characters
        if any('\u4e00' <= char <= '\u9fff' for char in text):
            return 'zh'
        # Check for Korean characters
        elif any('\uac00' <= char <= '\ud7a3' for char in text):
            return 'ko'
        # Check for Japanese-specific characters (hiragana and katakana)
        elif any('\u3040' <= char <= '\u309f' for char in text) or any('\u30a0' <= char <= '\u30ff' for char in text):
            return 'ja'
        # Default to English
        else:
            return 'en'
    
    def tokenize(self, text, language=None):
        """Tokenize text based on detected or specified language"""
        if language is None:
            language = self.detect_language(text)
            if self.verbose:
                print(f"Detected language: {language}")
        
        if language == 'zh':
            # Chinese tokenization with jieba
            return list(jieba.cut(text))
        elif language == 'ja':
            # Japanese tokenization with MeCab if available
            if self.ja_tagger:
                return [word.surface for word in self.ja_tagger(text)]
            else:
                # Fallback to TinySegmenter
                return self.ja_segmenter.tokenize(text)
        elif language == 'ko':
            # Korean tokenization with KoNLPy
            if self.ko_processor:
                return self.ko_processor.morphs(text)
            else:
                raise ValueError("Korean processor not available. Install with: pip install konlpy")
        else:
            # English or other languages using NLTK
            return nltk.word_tokenize(text)
    
    def stem_chinese(self, tokens):
        """Chinese word normalization using jieba's POS tagging"""
        processed_tokens = []
        
        # Use jieba's part-of-speech tagging
        text = "".join(tokens)
        words_with_pos = pseg.cut(text)
        
        for word, pos in words_with_pos:
            # For verbs, try to get the root form
            if pos.startswith('v'):
                # Remove common aspect markers like 了, 过, 着
                word = word.rstrip('了过着')
            processed_tokens.append(word)
            
        return processed_tokens
    
    def stem_japanese(self, tokens):
        """Japanese lemmatization using MeCab/UniDic"""
        processed_tokens = []
        
        if self.ja_tagger:
            # If MeCab is available, use it for better results
            text = "".join(tokens)
            words = self.ja_tagger(text)
            
            for word in words:
                # Get the lemma (dictionary form) if available
                lemma = word.feature.lemma
                if lemma is not None:
                    processed_tokens.append(lemma)
                else:
                    processed_tokens.append(word.surface)
        else:
            # Simplified approach - remove common conjugation endings
            common_endings = ['ます', 'でした', 'ました', 'ている', 'です']
            
            for token in tokens:
                for ending in common_endings:
                    if token.endswith(ending):
                        token = token[:-len(ending)]
                        break
                processed_tokens.append(token)
            
        return processed_tokens
    
    def stem_korean(self, tokens):
        """Korean stemming using KoNLPy's Okt"""
        if not self.ko_processor:
            raise ValueError("Korean processor not available. Install with: pip install konlpy")
        
        # Join tokens back (if they were already tokenized)
        text = " ".join(tokens)
        
        # Get lemmatized forms using Okt
        lemmatized = self.ko_processor.normalize(text)
        
        # Further process with the lemmatizer to get stems
        stems = []
        for word in self.ko_processor.pos(lemmatized, stem=True):
            # word is a tuple (token, pos)
            stems.append(word[0])
            
        return stems
    
    def lemmatize_english(self, tokens):
        """Lemmatize English tokens with POS tagging"""
        pos_tags = nltk.pos_tag(tokens)
        
        lemmatized_tokens = []
        for word, tag in pos_tags:
            wordnet_pos = self.get_wordnet_pos(tag)
            lemmatized_tokens.append(self.lemmatizer.lemmatize(word, wordnet_pos))
        
        return lemmatized_tokens
    
    def process_document(self, text_or_tokens, language=None):
        """
        Process a document in English, Chinese, Japanese, or Korean
        
        Args:
            text_or_tokens: Either raw text or pre-tokenized list
            language: 'en', 'zh', 'ja', 'ko', or None for auto-detection
            
        Returns:
            List of processed tokens
        """
        # Handle both text and pre-tokenized input
        if isinstance(text_or_tokens, str):
            if language is None:
                language = self.detect_language(text_or_tokens)
            tokens = self.tokenize(text_or_tokens, language)
        else:
            tokens = text_or_tokens
            if language is None:
                # Try to detect language from joined tokens
                sample_text = "".join(tokens[:10]) if tokens else ""
                language = self.detect_language(sample_text)
                if self.verbose:
                    print(f"Detected language from tokens: {language}")
        
        # Process based on language
        if language == 'zh':
            return self.stem_chinese(tokens)
        elif language == 'ja':
            return self.stem_japanese(tokens)
        elif language == 'ko':
            return self.stem_korean(tokens)
        else:  # 'en' or other
            return self.lemmatize_english(tokens)

def main():
    # Set up command line arguments
    parser = argparse.ArgumentParser(description='Multilingual document stemmer for English, Chinese, Japanese, and Korean.')
    parser.add_argument('--input', '-i', type=str, help='Input file containing tokenized document (one token per line or JSON array)')
    parser.add_argument('--output', '-o', type=str, help='Output file for stemmed tokens')
    parser.add_argument('--language', '-l', type=str, choices=['en', 'zh', 'ja', 'ko'], help='Language of document (en, zh, ja, ko)')
    parser.add_argument('--format', '-f', type=str, choices=['json', 'text'], default='text', help='Input/output format (json or text, default: text)')
    parser.add_argument('--verbose', '-v', action='store_true', help='Enable verbose output')
    
    args = parser.parse_args()
    
    # Initialize stemmer
    stemmer = MultilingualStemmer(verbose=args.verbose)
    
    # Read input tokens
    if args.input:
        with open(args.input, 'r', encoding='utf-8') as f:
            if args.format == 'json':
                try:
                    tokens = json.load(f)
                    if not isinstance(tokens, list):
                        print("Error: JSON input must be an array of tokens")
                        sys.exit(1)
                except json.JSONDecodeError:
                    print("Error: Invalid JSON input")
                    sys.exit(1)
            else:  # text format
                tokens = [line.strip() for line in f if line.strip()]
    else:
        # Read from stdin
        if args.verbose:
            print("Reading tokenized input from stdin (one token per line)...")
            print("Press Ctrl+D (Unix) or Ctrl+Z (Windows) when finished.")
        
        if args.format == 'json':
            try:
                tokens = json.load(sys.stdin)
                if not isinstance(tokens, list):
                    print("Error: JSON input must be an array of tokens")
                    sys.exit(1)
            except json.JSONDecodeError:
                print("Error: Invalid JSON input")
                sys.exit(1)
        else:  # text format
            tokens = [line.strip() for line in sys.stdin if line.strip()]
    
    if args.verbose:
        print(f"Read {len(tokens)} tokens.")
    
    # Process the document
    stemmed_tokens = stemmer.process_document(tokens, language=args.language)
    
    # Output the results
    if args.output:
        with open(args.output, 'w', encoding='utf-8') as f:
            if args.format == 'json':
                json.dump(stemmed_tokens, f, ensure_ascii=False, indent=2)
            else:  # text format
                for token in stemmed_tokens:
                    f.write(f"{token}\n")
    else:
        # Write to stdout
        if args.format == 'json':
            print(json.dumps(stemmed_tokens, ensure_ascii=False, indent=2))
        else:  # text format
            for token in stemmed_tokens:
                print(token)

if __name__ == "__main__":
    main()
