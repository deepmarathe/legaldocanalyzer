import re
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import syllables
import spacy
from collections import Counter
import logging
import io
import base64
from wordcloud import WordCloud
import matplotlib.pyplot as plt

# Ensure NLTK resources are available
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)

class LegalLanguageComplexityAnalyzer:
    """
    Analyzes legal document text for complexity using established linguistic metrics
    and provides plain language alternatives.
    """
    
    def __init__(self):
        """Initialize the analyzer with necessary resources and models"""
        self.logger = logging.getLogger(__name__)
        
        # Load legal jargon dictionary - comprehensive list of legal terms
        self.legal_jargon = set([
            "aforementioned", "hereinafter", "whereas", "hereby", "thereto", 
            "herein", "aforesaid", "whereof", "hereof", "therein", "hereunder",
            "thereunder", "hereinabove", "hereinbefore", "heretofore", "hereunto",
            "thereupon", "wherein", "inter alia", "mutatis mutandis", "prima facie",
            "ab initio", "de jure", "de facto", "bona fide", "amicus curiae",
            "ex parte", "in limine", "pro rata", "sua sponte", "res judicata",
            "stare decisis", "voir dire", "force majeure", "pari passu",
            "habeas corpus", "quantum meruit", "quid pro quo", "sine qua non",
            "pro bono", "caveat emptor", "sub judice", "ultra vires", "per se",
            "ipso facto", "gravamen", "locus", "mens rea", "actus reus", "tort",
            "estoppel", "novation", "bailment", "indemnity", "covenant", "easement",
            "encumbrance", "lien", "riparian", "jurisdiction", "adjudication",
            "adjudicate", "arbitration", "appellant", "appellee", "affidavit",
            "jurisprudence", "pursuant", "statute", "statutory", "promulgate",
            "provisional", "remedial", "remand", "deposition", "interrogatory",
            "subpoena", "testimony", "hearsay", "injunction", "adjournment",
            "demurrer", "pleading", "petitioner", "respondent", "stipulation"
        ])
        
        # Common complex legal phrases and simpler alternatives
        self.complex_phrases = {
            r"at the discretion of": "decided by",
            r"in accordance with": "following",
            r"pursuant to": "under",
            r"prior to": "before",
            r"subsequent to": "after",
            r"in the event that": "if",
            r"notwithstanding": "despite",
            r"for the purpose of": "to",
            r"in the amount of": "for",
            r"in connection with": "about",
            r"with reference to": "about",
            r"with regard to": "about",
            r"with respect to": "about",
            r"in lieu of": "instead of",
            r"in excess of": "more than",
            r"in compliance with": "following",
            r"for the duration of": "during",
            r"at such time as": "when",
            r"in the absence of": "without",
            r"on the grounds that": "because",
            r"in the vicinity of": "near",
            r"afford an opportunity": "allow",
            r"at the present time": "now",
            r"at this point in time": "now",
            r"commence": "begin",
            r"constitute": "form",
            r"endeavor": "try",
            r"utilize": "use",
            r"initiate": "start",
            r"terminate": "end",
            r"ascertain": "find out",
            r"demonstrate": "show",
            r"effectuate": "cause",
            r"necessitate": "require",
            r"render": "make",
            r"herein": "in this document",
            r"hereto": "to this",
            r"hereof": "of this",
            r"hereinafter": "after this",
            r"aforementioned": "mentioned above",
            r"foregoing": "previous",
            r"thereafter": "then",
            r"whereby": "by which",
            r"forthwith": "immediately"
        }
        
        try:
            # Load spaCy model for more advanced NLP tasks
            self.nlp = spacy.load("en_core_web_sm")
        except OSError:
            self.logger.warning("SpaCy model not found. Using basic tokenization.")
            self.nlp = None
    
    def analyze_text_complexity(self, text: str) -> Dict[str, Any]:
        """
        Analyze the complexity of legal text using multiple metrics.
        
        Args:
            text: The legal document text to analyze
            
        Returns:
            Dict: Analysis results including readability metrics, jargon usage, 
                 sentence complexity, and simplified text
        """
        try:
            # Guard against invalid input
            if not isinstance(text, str) or not text.strip():
                return self._generate_error_response("Invalid or empty text provided")
            
            # Store original text length for reference
            original_length = len(text)
            
            # Preprocess text for analysis
            preprocessed_text = self._preprocess_text(text)
            
            # Guard against preprocessing failures
            if not preprocessed_text:
                return self._generate_error_response("Text preprocessing failed")
                
            # Extract sentences and words
            sentences = self._extract_sentences(preprocessed_text)
            if not sentences:
                return self._generate_error_response("No sentences could be extracted")
                
            # Calculate all metrics
            readability_metrics = self._calculate_readability_metrics(preprocessed_text, sentences)
            jargon_analysis = self._analyze_legal_jargon(preprocessed_text)
            sentence_analysis = self._analyze_sentence_complexity(sentences)
            simplification = self._generate_simplified_text(text)
            word_frequency = self._analyze_word_frequency(preprocessed_text)
            
            # Generate visualizations
            visualizations = self._generate_visualizations(
                readability_metrics, 
                jargon_analysis,
                sentence_analysis,
                word_frequency
            )
            
            # Compile complete analysis
            analysis = {
                'readability_metrics': readability_metrics,
                'jargon_analysis': jargon_analysis,
                'sentence_analysis': sentence_analysis,
                'simplification': simplification,
                'word_frequency': word_frequency,
                'visualizations': visualizations,
                'meta': {
                    'original_length': original_length,
                    'sentence_count': len(sentences),
                    'overall_complexity_category': self._categorize_overall_complexity(readability_metrics),
                    'analysis_version': '1.0'
                }
            }
            
            return analysis
            
        except Exception as e:
            self.logger.error(f"Error in complexity analysis: {str(e)}")
            return self._generate_error_response(f"Analysis failed: {str(e)}")
    
    def _preprocess_text(self, text: str) -> str:
        """
        Preprocess text to prepare for analysis.
        
        Args:
            text: Raw document text
            
        Returns:
            str: Preprocessed text
        """
        try:
            # Remove excessive whitespace
            cleaned_text = re.sub(r'\s+', ' ', text)
            
            # Remove non-textual elements like page numbers, headers, footers
            # This uses common patterns found in legal documents
            cleaned_text = re.sub(r'\b[Pp]age\s+\d+\s+of\s+\d+\b', '', cleaned_text)
            cleaned_text = re.sub(r'\b\d+\s*\|\s*[Pp]age\b', '', cleaned_text)
            
            # Remove standard document references that don't contribute to complexity analysis
            cleaned_text = re.sub(r'(?i)exhibit\s+[a-z]', '', cleaned_text)
            cleaned_text = re.sub(r'(?i)appendix\s+[a-z]', '', cleaned_text)
            
            return cleaned_text.strip()
            
        except Exception as e:
            self.logger.warning(f"Text preprocessing error: {str(e)}")
            return text  # Return original text as fallback
    
    def _extract_sentences(self, text: str) -> List[str]:
        """
        Extract sentences from text using NLP.
        
        Args:
            text: Preprocessed document text
            
        Returns:
            List[str]: List of sentences
        """
        try:
            if self.nlp:
                # Use spaCy for better sentence boundary detection
                doc = self.nlp(text)
                sentences = [sent.text.strip() for sent in doc.sents if sent.text.strip()]
            else:
                # Fallback to NLTK
                sentences = [s.strip() for s in sent_tokenize(text) if s.strip()]
                
            return sentences
            
        except Exception as e:
            self.logger.warning(f"Sentence extraction error: {str(e)}")
            # Emergency fallback - split by periods with basic cleanup
            return [s.strip() + '.' for s in text.split('.') if s.strip()]
    
    def _calculate_readability_metrics(self, text: str, sentences: List[str]) -> Dict[str, Any]:
        """
        Calculate standard readability metrics.
        
        Args:
            text: Preprocessed document text
            sentences: List of extracted sentences
            
        Returns:
            Dict: Multiple readability scores and statistics
        """
        try:
            # Count words and syllables
            words = word_tokenize(text)
            word_count = len(words)
            
            # Guard against empty input
            if word_count == 0:
                return {
                    'flesch_reading_ease': 0,
                    'flesch_kincaid_grade': 0,
                    'gunning_fog': 0,
                    'smog_index': 0,
                    'automated_readability_index': 0,
                    'coleman_liau_index': 0,
                    'lix_index': 0,
                    'word_count': 0,
                    'sentence_count': 0,
                    'avg_words_per_sentence': 0,
                    'avg_syllables_per_word': 0,
                    'complex_word_count': 0,
                    'complex_word_percentage': 0
                }
            
            sentence_count = len(sentences)
            avg_words_per_sentence = word_count / max(1, sentence_count)
            
            # Calculate syllables
            syllable_counts = [self._count_syllables(word) for word in words]
            total_syllables = sum(syllable_counts)
            avg_syllables_per_word = total_syllables / max(1, word_count)
            
            # Count complex words (3+ syllables)
            complex_words = [word for word, count in zip(words, syllable_counts) if count >= 3]
            complex_word_count = len(complex_words)
            complex_word_percentage = (complex_word_count / max(1, word_count)) * 100
            
            # Calculate character count (for some formulas)
            char_count = sum(len(word) for word in words)
            avg_characters_per_word = char_count / max(1, word_count)
            
            # Calculate various readability scores
            
            # Flesch Reading Ease Score
            # Higher scores indicate text that is easier to read (0-100 scale)
            flesch_reading_ease = 206.835 - (1.015 * avg_words_per_sentence) - (84.6 * avg_syllables_per_word)
            flesch_reading_ease = max(0, min(100, flesch_reading_ease))  # Clamp between 0 and 100
            
            # Flesch-Kincaid Grade Level
            # Indicates the US grade level needed to comprehend the text
            flesch_kincaid_grade = (0.39 * avg_words_per_sentence) + (11.8 * avg_syllables_per_word) - 15.59
            
            # Gunning Fog Index
            # Estimates years of formal education needed to understand the text
            gunning_fog = 0.4 * (avg_words_per_sentence + 100 * (complex_word_count / max(1, word_count)))
            
            # SMOG Index
            # Estimates years of education needed to understand the text, based on complex words
            smog_index = 1.043 * ((complex_word_count * (30 / max(1, sentence_count))) ** 0.5) + 3.1291
            
            # Automated Readability Index
            # Another grade-level estimator using character count instead of syllables
            ari = 4.71 * avg_characters_per_word + 0.5 * avg_words_per_sentence - 21.43
            
            # Coleman-Liau Index
            # Grade-level estimator using character count
            l = (char_count / max(1, word_count)) * 100  # Average characters per 100 words
            s = (sentence_count / max(1, word_count)) * 100  # Average sentences per 100 words
            coleman_liau = 0.0588 * l - 0.296 * s - 15.8
            
            # LIX Index (Swedish readability metric, useful for technical documents)
            # <40 = easy, 40-50 = normal, 50-60 = difficult, >60 = very difficult
            long_words = len([word for word in words if len(word) > 6])
            lix = (word_count / max(1, sentence_count)) + (long_words * 100 / max(1, word_count))
            
            return {
                'flesch_reading_ease': round(flesch_reading_ease, 2),
                'flesch_kincaid_grade': round(flesch_kincaid_grade, 2),
                'gunning_fog': round(gunning_fog, 2),
                'smog_index': round(smog_index, 2),
                'automated_readability_index': round(ari, 2),
                'coleman_liau_index': round(coleman_liau, 2),
                'lix_index': round(lix, 2),
                'word_count': word_count,
                'sentence_count': sentence_count,
                'avg_words_per_sentence': round(avg_words_per_sentence, 2),
                'avg_syllables_per_word': round(avg_syllables_per_word, 2),
                'complex_word_count': complex_word_count,
                'complex_word_percentage': round(complex_word_percentage, 2)
            }
            
        except Exception as e:
            self.logger.error(f"Readability metrics calculation error: {str(e)}")
            return {
                'error': f"Failed to calculate readability metrics: {str(e)}",
                'flesch_reading_ease': 0,
                'flesch_kincaid_grade': 0
            }
    
    def _count_syllables(self, word: str) -> int:
        """
        Count syllables in a word with fallback mechanisms.
        
        Args:
            word: The word to count syllables for
            
        Returns:
            int: Number of syllables
        """
        try:
            # Try using the syllables library first
            count = syllables.estimate(word)
            
            # If syllables library returns 0, use a backup method
            if count == 0:
                # Basic syllable counting heuristic
                word = word.lower()
                if len(word) <= 3:
                    return 1
                    
                # Remove ending e
                if word.endswith('e'):
                    word = word[:-1]
                
                # Count vowel groups
                vowels = "aeiouy"
                prev_is_vowel = False
                count = 0
                
                for char in word:
                    is_vowel = char in vowels
                    if is_vowel and not prev_is_vowel:
                        count += 1
                    prev_is_vowel = is_vowel
                
                if count == 0:
                    count = 1  # Every word has at least one syllable
            
            return count
            
        except Exception as e:
            self.logger.warning(f"Syllable counting error for word '{word}': {str(e)}")
            # Fallback to character-based estimation
            return max(1, len(word) // 3)
    
    def _analyze_legal_jargon(self, text: str) -> Dict[str, Any]:
        """
        Analyze usage of legal jargon and archaic terms.
        
        Args:
            text: Preprocessed document text
            
        Returns:
            Dict: Analysis of legal jargon usage
        """
        try:
            # Tokenize text into words
            words = word_tokenize(text.lower())
            total_words = len(words)
            
            if total_words == 0:
                return {
                    'jargon_count': 0,
                    'jargon_percentage': 0,
                    'jargon_instances': [],
                    'jargon_density': 0
                }
            
            # Count legal jargon occurrences
            jargon_instances = []
            for term in self.legal_jargon:
                pattern = r'\b' + re.escape(term) + r'\b'
                matches = re.finditer(pattern, text.lower())
                for match in matches:
                    # Get some context around the match
                    start = max(0, match.start() - 40)
                    end = min(len(text), match.end() + 40)
                    context = text[start:end]
                    
                    jargon_instances.append({
                        'term': term,
                        'position': match.start(),
                        'context': context
                    })
            
            # Count complex phrases
            phrase_instances = []
            for phrase, simple_alt in self.complex_phrases.items():
                pattern = r'\b' + phrase + r'\b'
                matches = re.finditer(pattern, text.lower())
                for match in matches:
                    start = max(0, match.start() - 40)
                    end = min(len(text), match.end() + 40)
                    context = text[start:end]
                    
                    phrase_instances.append({
                        'phrase': phrase,
                        'alternative': simple_alt,
                        'position': match.start(),
                        'context': context
                    })
            
            # Compile statistics
            jargon_count = len(jargon_instances)
            jargon_percentage = (jargon_count / max(1, total_words)) * 100
            
            # Calculate jargon density (occurrences per 100 words)
            jargon_density = (jargon_count / max(1, total_words)) * 100
            
            # Find most frequent jargon terms
            if jargon_instances:
                term_counts = Counter([instance['term'] for instance in jargon_instances])
                most_frequent = term_counts.most_common(10)
            else:
                most_frequent = []
            
            return {
                'jargon_count': jargon_count,
                'jargon_percentage': round(jargon_percentage, 2),
                'jargon_density': round(jargon_density, 2),
                'most_frequent_terms': most_frequent,
                'jargon_instances': jargon_instances[:50],  # Limit to 50 instances for brevity
                'complex_phrases': phrase_instances[:50]
            }
            
        except Exception as e:
            self.logger.error(f"Legal jargon analysis error: {str(e)}")
            return {
                'error': f"Failed to analyze legal jargon: {str(e)}",
                'jargon_count': 0,
                'jargon_percentage': 0
            }
    
    def _analyze_sentence_complexity(self, sentences: List[str]) -> Dict[str, Any]:
        """
        Analyze the complexity of sentences.
        
        Args:
            sentences: List of sentences from the document
            
        Returns:
            Dict: Sentence complexity analysis
        """
        try:
            sentence_count = len(sentences)
            
            if sentence_count == 0:
                return {
                    'avg_sentence_length': 0,
                    'long_sentence_percentage': 0,
                    'complex_sentence_percentage': 0,
                    'sentence_length_distribution': {},
                    'example_complex_sentences': []
                }
            
            # Analyze each sentence
            sentence_analysis = []
            sentence_lengths = []
            
            for sentence in sentences:
                words = word_tokenize(sentence)
                word_count = len(words)
                sentence_lengths.append(word_count)
                
                # Check for complex sentence indicators
                subordinate_clauses = self._count_subordinate_clauses(sentence)
                has_passive_voice = self._detect_passive_voice(sentence)
                
                complexity_score = self._calculate_sentence_complexity_score(
                    word_count, subordinate_clauses, has_passive_voice
                )
                
                sentence_analysis.append({
                    'text': sentence,
                    'word_count': word_count,
                    'subordinate_clauses': subordinate_clauses,
                    'has_passive_voice': has_passive_voice,
                    'complexity_score': complexity_score
                })
            
            # Calculate statistics
            avg_sentence_length = sum(sentence_lengths) / max(1, sentence_count)
            
            # Categorize sentences by length
            long_sentence_threshold = 30  # Words
            long_sentences = [s for s in sentence_analysis if s['word_count'] > long_sentence_threshold]
            long_sentence_percentage = (len(long_sentences) / max(1, sentence_count)) * 100
            
            # Categorize sentences by complexity
            complexity_threshold = 7  # Complexity score
            complex_sentences = [s for s in sentence_analysis if s['complexity_score'] > complexity_threshold]
            complex_sentence_percentage = (len(complex_sentences) / max(1, sentence_count)) * 100
            
            # Calculate sentence length distribution
            length_ranges = [(0, 10), (11, 20), (21, 30), (31, 40), (41, 50), (51, float('inf'))]
            length_distribution = {}
            
            for start, end in length_ranges:
                category = f"{start}-{end if end != float('inf') else '+'}"
                count = sum(1 for length in sentence_lengths if start <= length <= end)
                percentage = (count / max(1, sentence_count)) * 100
                length_distribution[category] = {
                    'count': count,
                    'percentage': round(percentage, 2)
                }
            
            # Find example complex sentences to highlight
            example_complex_sentences = sorted(
                complex_sentences, 
                key=lambda s: s['complexity_score'], 
                reverse=True
            )[:5]  # Top 5 most complex sentences
            
            return {
                'avg_sentence_length': round(avg_sentence_length, 2),
                'long_sentence_percentage': round(long_sentence_percentage, 2),
                'complex_sentence_percentage': round(complex_sentence_percentage, 2),
                'sentence_length_distribution': length_distribution,
                'example_complex_sentences': example_complex_sentences,
                'sentence_complexity_stats': {
                    'min': min(sentence_lengths),
                    'max': max(sentence_lengths),
                    'median': np.median(sentence_lengths),
                    'std_dev': round(np.std(sentence_lengths), 2)
                }
            }
            
        except Exception as e:
            self.logger.error(f"Sentence complexity analysis error: {str(e)}")
            return {
                'error': f"Failed to analyze sentence complexity: {str(e)}",
                'avg_sentence_length': 0
            }
    
    def _count_subordinate_clauses(self, sentence: str) -> int:
        """
        Count subordinate clauses in a sentence.
        
        Args:
            sentence: The sentence to analyze
            
        Returns:
            int: Number of subordinate clauses detected
        """
        try:
            # Common subordinating conjunctions
            subordinators = [
                'after', 'although', 'as', 'because', 'before', 'if', 'since', 
                'than', 'that', 'though', 'unless', 'until', 'when', 'whenever', 
                'where', 'whereas', 'wherever', 'whether', 'while', 'who', 
                'whoever', 'whom', 'whose', 'why', 'provided that', 'assuming that',
                'even if', 'even though', 'in order that', 'so that', 'such that'
            ]
            
            # Advanced detection with spaCy if available
            if self.nlp:
                doc = self.nlp(sentence)
                # Count subordinate clauses based on dependency parsing
                clause_count = 0
                for token in doc:
                    # Check for subordinate clause markers in dependency tree
                    if (token.dep_ in ['advcl', 'ccomp', 'xcomp', 'acl', 'relcl'] or 
                        (token.lower_ in subordinators and token.pos_ == 'SCONJ')):
                        clause_count += 1
                return clause_count
            else:
                # Fallback: Simple word matching
                words = word_tokenize(sentence.lower())
                return sum(1 for word in words if word in subordinators)
                
        except Exception as e:
            self.logger.warning(f"Subordinate clause detection error: {str(e)}")
            return 0
    
    def _detect_passive_voice(self, sentence: str) -> bool:
        """
        Detect passive voice in a sentence.
        
        Args:
            sentence: The sentence to analyze
            
        Returns:
            bool: True if passive voice is detected
        """
        try:
            if self.nlp:
                doc = self.nlp(sentence)
                
                # Check for passive voice construction
                for token in doc:
                    if (token.dep_ == "nsubjpass" or  # Nominal subject (passive)
                        token.dep_ == "auxpass"):      # Auxiliary verb (passive)
                        return True
                return False
            else:
                # Fallback: Simple pattern matching
                # Look for common passive voice patterns: be + past participle
                be_verbs = ['am', 'is', 'are', 'was', 'were', 'be', 'been', 'being']
                words = word_tokenize(sentence.lower())
                
                for i, word in enumerate(words[:-1]):
                    if word in be_verbs and i < len(words) - 1:
                        next_word = words[i + 1]
                        # Check if next word might be a past participle (ends with -ed or -en)
                        if next_word.endswith('ed') or next_word.endswith('en'):
                            return True
                return False
                
        except Exception as e:
            self.logger.warning(f"Passive voice detection error: {str(e)}")
            return False
    
    def _calculate_sentence_complexity_score(self, 
                                           word_count: int, 
                                           subordinate_clauses: int, 
                                           has_passive_voice: bool) -> float:
        """
        Calculate a complexity score for a sentence based on multiple factors.
        
        Args:
            word_count: Number of words in the sentence
            subordinate_clauses: Number of subordinate clauses
            has_passive_voice: Whether passive voice is used
            
        Returns:
            float: Complexity score (higher = more complex)
        """
        try:
            # Base score is word count divided by 10 (for normalization)
            score = word_count / 10
            
            # Add points for subordinate clauses
            score += subordinate_clauses * 2
            
            # Add points for passive voice
            if has_passive_voice:
                score += 2
                
            return round(score, 2)
            
        except Exception as e:
            self.logger.warning(f"Complexity score calculation error: {str(e)}")
            return 0
    
    def _generate_simplified_text(self, text: str) -> Dict[str, Any]:
        """
        Generate a simplified version of the legal text.
        
        Args:
            text: Original document text
            
        Returns:
            Dict: Simplified text and transformation statistics
        """
        try:
            # Initialize tracking variables for transformations
            simplified_text = text
            transformations = []
            
            # 1. Replace complex phrases with simpler alternatives
            for complex_phrase, simple_alternative in self.complex_phrases.items():
                pattern = r'\b' + re.escape(complex_phrase) + r'\b'
                original = simplified_text
                simplified_text = re.sub(pattern, simple_alternative, simplified_text, flags=re.IGNORECASE)
                
                # Count replacements
                if original != simplified_text:
                    replacement_count = len(re.findall(pattern, original, re.IGNORECASE))
                    transformations.append({
                        'type': 'phrase_replacement',
                        'original': complex_phrase,
                        'replacement': simple_alternative,
                        'count': replacement_count
                    })
            
            # 2. Break down long sentences
            if self.nlp:
                original_doc = self.nlp(simplified_text)
                sentences = list(original_doc.sents)
                
                long_sentence_threshold = 30  # words
                simplified_sentences = []
                
                long_sentences_count = 0
                sentences_split = 0
                
                for sent in sentences:
                    word_count = len([token for token in sent if not token.is_punct])
                    
                    if word_count > long_sentence_threshold:
                        long_sentences_count += 1
                        split_sentences = self._split_long_sentence(sent.text)
                        simplified_sentences.extend(split_sentences)
                        sentences_split += len(split_sentences) - 1
                    else:
                        simplified_sentences.append(sent.text)
                
                if sentences_split > 0:
                    transformations.append({
                        'type': 'sentence_splitting',
                        'long_sentences_count': long_sentences_count,
                        'sentences_split': sentences_split
                    })
                    
                    # Reconstruct the text with split sentences
                    simplified_text = ' '.join(simplified_sentences)
            
            # 3. Remove redundant legal boilerplate when possible
            boilerplate_patterns = [
                (r'this\s+agreement\s+is\s+made\s+and\s+entered\s+into\s+as\s+of\s+the\s+date\s+first\s+written\s+above', 
                 'This agreement starts on the date above'),
                (r'for\s+good\s+and\s+valuable\s+consideration,\s+the\s+receipt\s+and\s+sufficiency\s+of\s+which\s+is\s+hereby\s+acknowledged', 
                 'for payment received'),
                (r'now,\s+therefore,\s+in\s+consideration\s+of\s+the\s+mutual\s+covenants\s+contained\s+herein\s+and\s+other\s+good\s+and\s+valuable\s+consideration', 
                 'Therefore, in exchange for payment and promises in this agreement'),
            ]
            
            for pattern, replacement in boilerplate_patterns:
                original = simplified_text
                simplified_text = re.sub(pattern, replacement, simplified_text, flags=re.IGNORECASE)
                
                if original != simplified_text:
                    transformations.append({
                        'type': 'boilerplate_removal',
                        'original': pattern,
                        'replacement': replacement
                    })
            
            # Calculate simplification metrics
            original_words = len(word_tokenize(text))
            simplified_words = len(word_tokenize(simplified_text))
            word_reduction = original_words - simplified_words
            word_reduction_percentage = (word_reduction / max(1, original_words)) * 100
            
            # Calculate readability improvement
            original_metrics = self._calculate_readability_metrics(
                text, 
                self._extract_sentences(text)
            )
            simplified_metrics = self._calculate_readability_metrics(
                simplified_text,
                self._extract_sentences(simplified_text)
            )
            
            flesch_improvement = simplified_metrics['flesch_reading_ease'] - original_metrics['flesch_reading_ease']
            grade_level_reduction = original_metrics['flesch_kincaid_grade'] - simplified_metrics['flesch_kincaid_grade']
            
            return {
                'original_text': text[:1000] + ('...' if len(text) > 1000 else ''),  # First 1000 chars for reference
                'simplified_text': simplified_text,
                'transformations': transformations,
                'metrics': {
                    'word_count_original': original_words,
                    'word_count_simplified': simplified_words,
                    'word_reduction': word_reduction,
                    'word_reduction_percentage': round(word_reduction_percentage, 2),
                    'flesch_reading_ease_original': original_metrics['flesch_reading_ease'],
                    'flesch_reading_ease_simplified': simplified_metrics['flesch_reading_ease'],
                    'flesch_reading_ease_improvement': round(flesch_improvement, 2),
                    'grade_level_original': original_metrics['flesch_kincaid_grade'],
                    'grade_level_simplified': simplified_metrics['flesch_kincaid_grade'],
                    'grade_level_reduction': round(grade_level_reduction, 2)
                }
            }
            
        except Exception as e:
            self.logger.error(f"Text simplification error: {str(e)}")
            return {
                'error': f"Failed to generate simplified text: {str(e)}",
                'original_text': text[:1000] + ('...' if len(text) > 1000 else ''),
                'simplified_text': text[:1000] + ('...' if len(text) > 1000 else '')
            }
    
    def _split_long_sentence(self, sentence: str) -> List[str]:
        """
        Split a long sentence into multiple shorter sentences.
        
        Args:
            sentence: Long sentence to split
            
        Returns:
            List[str]: List of shorter sentences
        """
        try:
            # If spaCy is available, use more sophisticated approach
            if self.nlp:
                doc = self.nlp(sentence)
                
                # Identify potential split points
                split_candidates = []
                
                for token in doc:
                    # Good split points: coordinating conjunctions, semicolons, or certain subordinating clauses
                    if (token.pos_ == "CCONJ" and token.dep_ == "cc") or \
                       (token.text == ";" or token.text == ":") or \
                       (token.dep_ == "mark" and token.head.dep_ == "advcl"):
                        split_candidates.append(token.i)
                
                # If no natural split points, fall back to comma-based splitting
                if not split_candidates and len(doc) > 30:
                    for token in doc:
                        if token.text == "," and token.i > 10:  # Don't split too early in the sentence
                            split_candidates.append(token.i)
                
                # If we have split candidates, create new sentences
                if split_candidates:
                    splits = sorted(split_candidates)
                    result = []
                    
                    start_idx = 0
                    for split_idx in splits:
                        # Extract chunk and adjust to make grammatical
                        chunk = doc[start_idx:split_idx].text
                        
                        # Only add non-empty chunks
                        if chunk.strip():
                            # Ensure the chunk ends with proper punctuation
                            if not chunk.strip().endswith(('.', '!', '?', ':', ';')):
                                chunk += '.'
                            result.append(chunk)
                        
                        # Move start index
                        start_idx = split_idx + 1
                    
                    # Add the final chunk
                    final_chunk = doc[start_idx:].text
                    if final_chunk.strip():
                        result.append(final_chunk)
                    
                    # If we managed to split the sentence, return the result
                    if len(result) > 1:
                        return result
            
            # Fallback: Simple splitting at commas and conjunctions
            pattern = r'(,\s+(?:and|but|or|however|therefore|thus|nevertheless|nonetheless|consequently|accordingly)\s+|\.\s+|\;\s+)'
            parts = re.split(pattern, sentence)
            
            # Rejoin parts to form complete sentences
            result = []
            current = ""
            
            for i, part in enumerate(parts):
                if re.match(pattern, part):
                    # This is a split point
                    if current:
                        current += part
                        result.append(current)
                        current = ""
                else:
                    # This is content
                    if not current and part:
                        # This is the start of a new sentence
                        current = part
                    elif part:
                        # This continues the current sentence
                        current += part
            
            # Add any remaining content
            if current:
                result.append(current)
            
            # If we couldn't split, return the original sentence
            return result if len(result) > 1 else [sentence]
            
        except Exception as e:
            self.logger.warning(f"Sentence splitting error: {str(e)}")
            return [sentence]  # Return original sentence if splitting fails
    
    def _analyze_word_frequency(self, text: str) -> Dict[str, Any]:
        """
        Analyze word frequency and identify commonly used terms.
        
        Args:
            text: Document text
            
        Returns:
            Dict: Word frequency analysis
        """
        try:
            # Tokenize and normalize words
            words = word_tokenize(text.lower())
            
            # Remove punctuation and stopwords
            stopwords = set([
                'the', 'a', 'an', 'and', 'or', 'but', 'if', 'of', 'at', 'by',
                'for', 'with', 'about', 'against', 'between', 'into', 'through',
                'during', 'before', 'after', 'above', 'below', 'to', 'from',
                'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again',
                'further', 'then', 'once', 'here', 'there', 'when', 'where', 'why',
                'how', 'all', 'any', 'both', 'each', 'few', 'more', 'most', 'other',
                'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so',
                'than', 'too', 'very', 's', 't', 'will', 'just', 'don', 'should'
            ])
            
            filtered_words = [word for word in words if word.isalpha() and word not in stopwords]
            
            # Count word frequency
            word_counts = Counter(filtered_words)
            most_common = word_counts.most_common(50)  # Top 50 most common words
            
            # Calculate total word count for percentage
            total_words = len(filtered_words)
            
            # Calculate weighted frequency (accounting for word length)
            weighted_frequency = {}
            for word, count in word_counts.items():
                # Longer words are given higher weight as they're often more important in legal text
                weight = min(2.0, 0.5 + (len(word) / 10))
                weighted_frequency[word] = count * weight
            
            weighted_common = sorted(
                weighted_frequency.items(), 
                key=lambda x: x[1], 
                reverse=True
            )[:50]
            
            return {
                'most_common': [{'word': word, 'count': count, 'percentage': round((count/max(1, total_words))*100, 2)} 
                               for word, count in most_common],
                'weighted_common': [{'word': word, 'weighted_score': round(score, 2)} 
                                   for word, score in weighted_common],
                'total_unique_words': len(word_counts),
                'total_filtered_words': total_words
            }
            
        except Exception as e:
            self.logger.error(f"Word frequency analysis error: {str(e)}")
            return {
                'error': f"Failed to analyze word frequency: {str(e)}",
                'most_common': [],
                'total_unique_words': 0
            }
    
    def _generate_visualizations(self, 
                               readability_metrics: Dict[str, Any],
                               jargon_analysis: Dict[str, Any],
                               sentence_analysis: Dict[str, Any],
                               word_frequency: Dict[str, Any]) -> Dict[str, str]:
        """
        Generate visualizations for the complexity analysis.
        
        Args:
            readability_metrics: Readability scores and statistics
            jargon_analysis: Legal jargon analysis results
            sentence_analysis: Sentence complexity analysis
            word_frequency: Word frequency analysis
            
        Returns:
            Dict: Base64-encoded visualizations
        """
        try:
            visualizations = {}
            
            # 1. Readability Metrics Gauge Chart
            try:
                fig = make_subplots(
                    rows=1, cols=2,
                    specs=[[{"type": "indicator"}, {"type": "indicator"}]],
                    subplot_titles=("Flesch Reading Ease", "Grade Level")
                )
                
                # Flesch Reading Ease gauge
                fig.add_trace(
                    go.Indicator(
                        mode="gauge+number",
                        value=readability_metrics['flesch_reading_ease'],
                        title={'text': "Flesch Reading Ease"},
                        gauge={
                            'axis': {'range': [0, 100]},
                            'bar': {'color': "darkblue"},
                            'steps': [
                                {'range': [0, 30], 'color': "red"},
                                {'range': [30, 50], 'color': "orange"},
                                {'range': [50, 70], 'color': "yellow"},
                                {'range': [70, 100], 'color': "green"}
                            ],
                            'threshold': {
                                'line': {'color': "black", 'width': 4},
                                'thickness': 0.75,
                                'value': readability_metrics['flesch_reading_ease']
                            }
                        }
                    ),
                    row=1, col=1
                )
                
                # Grade Level gauge
                fig.add_trace(
                    go.Indicator(
                        mode="gauge+number",
                        value=readability_metrics['flesch_kincaid_grade'],
                        title={'text': "Grade Level"},
                        gauge={
                            'axis': {'range': [0, 20]},
                            'bar': {'color': "darkblue"},
                            'steps': [
                                {'range': [0, 8], 'color': "green"},
                                {'range': [8, 12], 'color': "yellow"},
                                {'range': [12, 16], 'color': "orange"},
                                {'range': [16, 20], 'color': "red"}
                            ],
                            'threshold': {
                                'line': {'color': "black", 'width': 4},
                                'thickness': 0.75,
                                'value': readability_metrics['flesch_kincaid_grade']
                            }
                        }
                    ),
                    row=1, col=2
                )
                
                fig.update_layout(
                    height=400,
                    width=700,
                    margin=dict(l=40, r=40, t=60, b=40)
                )
                
                # Convert to base64
                img_bytes = io.BytesIO()
                fig.write_image(img_bytes, format='png')
                img_bytes.seek(0)
                readability_viz = base64.b64encode(img_bytes.read()).decode('utf-8')
                visualizations['readability_gauge'] = readability_viz
                
            except Exception as e:
                self.logger.warning(f"Readability visualization error: {str(e)}")
            
            # 2. Sentence Length Distribution
            try:
                if 'sentence_length_distribution' in sentence_analysis:
                    distribution = sentence_analysis['sentence_length_distribution']
                    
                    categories = list(distribution.keys())
                    counts = [data['count'] for data in distribution.values()]
                    percentages = [data['percentage'] for data in distribution.values()]
                    
                    fig = go.Figure()
                    
                    fig.add_trace(go.Bar(
                        x=categories,
                        y=counts,
                        name='Count',
                        marker_color='blue',
                        opacity=0.7
                    ))
                    
                    fig.add_trace(go.Scatter(
                        x=categories,
                        y=percentages,
                        name='Percentage',
                        yaxis='y2',
                        mode='lines+markers',
                        marker=dict(color='red')
                    ))
                    
                    fig.update_layout(
                        title='Sentence Length Distribution',
                        xaxis_title='Word Count Range',
                        yaxis_title='Number of Sentences',
                        yaxis2=dict(
                            title='Percentage',
                            overlaying='y',
                            side='right',
                            range=[0, 100]
                        ),
                        legend=dict(x=0.01, y=0.99),
                        height=400,
                        width=700
                    )
                    
                    img_bytes = io.BytesIO()
                    fig.write_image(img_bytes, format='png')
                    img_bytes.seek(0)
                    sentence_viz = base64.b64encode(img_bytes.read()).decode('utf-8')
                    visualizations['sentence_distribution'] = sentence_viz
                    
            except Exception as e:
                self.logger.warning(f"Sentence distribution visualization error: {str(e)}")
            
            # 3. Word Cloud for most common terms
            try:
                if 'most_common' in word_frequency and word_frequency['most_common']:
                    # Create word cloud
                    word_dict = {item['word']: item['count'] for item in word_frequency['most_common']}
                    
                    wordcloud = WordCloud(
                        width=700,
                        height=400,
                        background_color='white',
                        colormap='viridis',
                        max_words=100
                    ).generate_from_frequencies(word_dict)
                    
                    plt.figure(figsize=(10, 6))
                    plt.imshow(wordcloud, interpolation='bilinear')
                    plt.axis("off")
                    plt.tight_layout()
                    
                    img_bytes = io.BytesIO()
                    plt.savefig(img_bytes, format='png')
                    img_bytes.seek(0)
                    wordcloud_viz = base64.b64encode(img_bytes.read()).decode('utf-8')
                    plt.close()
                    
                    visualizations['word_cloud'] = wordcloud_viz
                    
            except Exception as e:
                self.logger.warning(f"Word cloud visualization error: {str(e)}")
            
            # 4. Complexity Comparison with Standard Texts
            try:
                # Reference values for different text types
                reference_scores = {
                    'Average Contract': {
                        'flesch_reading_ease': 35.0,
                        'flesch_kincaid_grade': 14.0,
                        'gunning_fog': 17.0
                    },
                    'Supreme Court Opinions': {
                        'flesch_reading_ease': 30.0,
                        'flesch_kincaid_grade': 15.0,
                        'gunning_fog': 18.0
                    },
                    'Newspaper Article': {
                        'flesch_reading_ease': 60.0,
                        'flesch_kincaid_grade': 9.0,
                        'gunning_fog': 12.0
                    },
                    'Plain Language Guidelines': {
                        'flesch_reading_ease': 70.0,
                        'flesch_kincaid_grade': 7.0,
                        'gunning_fog': 9.0
                    }
                }
                
                # Create comparison chart
                current_doc = {
                    'flesch_reading_ease': readability_metrics['flesch_reading_ease'],
                    'flesch_kincaid_grade': readability_metrics['flesch_kincaid_grade'],
                    'gunning_fog': readability_metrics['gunning_fog']
                }
                
                # Prepare data for radar chart
                categories = ['Flesch Reading Ease', 'Flesch-Kincaid Grade', 'Gunning Fog']
                
                fig = go.Figure()
                
                # Add current document
                fig.add_trace(go.Scatterpolar(
                    r=[
                        current_doc['flesch_reading_ease'],
                        current_doc['flesch_kincaid_grade'],
                        current_doc['gunning_fog']
                    ],
                    theta=categories,
                    fill='toself',
                    name='Current Document'
                ))
                
                # Add reference texts
                for text_type, scores in reference_scores.items():
                    fig.add_trace(go.Scatterpolar(
                        r=[
                            scores['flesch_reading_ease'],
                            scores['flesch_kincaid_grade'],
                            scores['gunning_fog']
                        ],
                        theta=categories,
                        fill='toself',
                        name=text_type
                    ))
                
                fig.update_layout(
                    title='Complexity Comparison with Standard Text Types',
                    polar=dict(
                        radialaxis=dict(
                            visible=True,
                            range=[0, 100]
                        )
                    ),
                    showlegend=True,
                    height=500,
                    width=700
                )
                
                img_bytes = io.BytesIO()
                fig.write_image(img_bytes, format='png')
                img_bytes.seek(0)
                comparison_viz = base64.b64encode(img_bytes.read()).decode('utf-8')
                visualizations['complexity_comparison'] = comparison_viz
                
            except Exception as e:
                self.logger.warning(f"Complexity comparison visualization error: {str(e)}")
            
            return visualizations
            
        except Exception as e:
            self.logger.error(f"Visualization generation error: {str(e)}")
            return {}
    
    def _categorize_overall_complexity(self, metrics: Dict[str, Any]) -> Dict[str, Any]:
        """
        Categorize the overall complexity of the document.
        
        Args:
            metrics: Readability metrics
            
        Returns:
            Dict: Complexity categorization and explanation
        """
        try:
            # Extract key metrics
            flesch = metrics.get('flesch_reading_ease', 0)
            grade = metrics.get('flesch_kincaid_grade', 0)
            fog = metrics.get('gunning_fog', 0)
            
            # Define category thresholds
            categories = [
                {
                    'name': 'Very Complex',
                    'description': 'Extremely complex legal language requiring expert interpretation',
                    'flesch_range': (0, 30),
                    'grade_range': (15, float('inf')),
                    'fog_range': (18, float('inf'))
                },
                {
                    'name': 'Complex',
                    'description': 'Difficult legal language typically found in specialized documents',
                    'flesch_range': (30, 50),
                    'grade_range': (12, 15),
                    'fog_range': (15, 18)
                },
                {
                    'name': 'Moderately Complex',
                    'description': 'Moderate complexity with some legal terminology but generally understandable',
                    'flesch_range': (50, 60),
                    'grade_range': (10, 12),
                    'fog_range': (12, 15)
                },
                {
                    'name': 'Moderately Simple',
                    'description': 'Relatively clear language with minimal complex legal terminology',
                    'flesch_range': (60, 70),
                    'grade_range': (8, 10),
                    'fog_range': (10, 12)
                },
                {
                    'name': 'Plain Language',
                    'description': 'Clear, plain language accessible to general audiences',
                    'flesch_range': (70, 100),
                    'grade_range': (0, 8),
                    'fog_range': (0, 10)
                }
            ]
            
            # Score each category based on how many metrics fit within its ranges
            category_scores = []
            
            for category in categories:
                score = 0
                if category['flesch_range'][0] <= flesch <= category['flesch_range'][1]:
                    score += 1
                if category['grade_range'][0] <= grade <= category['grade_range'][1]:
                    score += 1
                if category['fog_range'][0] <= fog <= category['fog_range'][1]:
                    score += 1
                
                category_scores.append((category, score))
            
            # Select the category with the highest score
            selected_category = max(category_scores, key=lambda x: x[1])[0]
            
            # Generate explanation
            explanation = f"This document is categorized as '{selected_category['name']}'. {selected_category['description']}."
            
            if flesch < 50:
                explanation += " The readability score indicates language that may be challenging for non-legal readers."
            
            if grade > 12:
                explanation += f" The grade level of {grade:.1f} suggests college-level education is needed to fully comprehend this text."
            
            return {
                'category': selected_category['name'],
                'description': selected_category['description'],
                'explanation': explanation
            }
            
        except Exception as e:
            self.logger.error(f"Complexity categorization error: {str(e)}")
            return {
                'category': 'Unknown',
                'description': 'Unable to categorize complexity',
                'explanation': 'Error occurred during complexity analysis'
            }
    
    def _generate_error_response(self, error_message: str) -> Dict[str, Any]:
        """
        Generate a standardized error response.
        
        Args:
            error_message: Error message to include
            
        Returns:
            Dict: Error response structure
        """
        return {
            'error': error_message,
            'readability_metrics': {
                'flesch_reading_ease': 0,
                'flesch_kincaid_grade': 0,
                'gunning_fog': 0
            },
            'jargon_analysis': {
                'jargon_count': 0,
                'jargon_percentage': 0
            },
            'sentence_analysis': {
                'avg_sentence_length': 0,
                'complex_sentence_percentage': 0
            },
            'simplification': {
                'simplified_text': '',
                'metrics': {}
            }
        }