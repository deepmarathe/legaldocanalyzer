#doc1.py
import os
import re
import io
import logging
from typing import Dict, List
from datetime import datetime
from pathlib import Path
import streamlit as st


# Import libraries for document processing
import spacy
import pytesseract
from PIL import Image, ImageEnhance
import pdf2image
import docx2txt
from googletrans import Translator
import pdfplumber
from legal_language_analyzer import LegalLanguageComplexityAnalyzer
# PDF generation imports
from reportlab.lib.pagesizes import letter, A4
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.pdfgen import canvas
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont

# Initialize logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class LegalDocumentProcessor:
    """Base Legal Document Processing System"""

    def __init__(self):
        self.logger = logger  # Use the module-level logger
        try:
            self.nlp = spacy.load('en_core_web_sm')
        except OSError:
            st.error("Spacy model not found. Installing now...")
            os.system("python -m spacy download en_core_web_sm")
            self.nlp = spacy.load('en_core_web_sm')

        # Configure OCR settings
        self.ocr_config = r'--oem 3 --psm 6 -l eng'

    def extract_text(self, file) -> str:
        """Extract text from multiple document formats"""
        try:
            if file is None:
                raise ValueError("No file provided")

            if file.size > 50 * 1024 * 1024:  # 50MB limit
                raise ValueError("File size exceeds 50MB limit")

            if file.type == "application/pdf":
                return self._extract_from_pdf(file)
            elif file.type.startswith('image/'):
                return self._extract_from_image(file)
            elif file.type == "application/msword" or \
                 file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
                return self._extract_from_docx(file)
            elif file.type == "text/plain":
                return file.getvalue().decode('utf-8')
            else:
                raise ValueError(f"Unsupported file type: {file.type}")
        except Exception as e:
            self.logger.error(f"Text extraction error: {str(e)}")
            st.error(f"Failed to extract text: {str(e)}")
            return ""

    def _extract_from_pdf(self, file) -> str:
        """Extract text from PDF files"""
        try:
            st.info("Processing PDF. This may take a few moments...")

            # Convert PDF to images
            file_bytes = file.getvalue()
            images = pdf2image.convert_from_bytes(
                file_bytes,
                dpi=300,
                fmt='PNG',
                grayscale=True,
            )

            text = ""
            total_pages = len(images)

            progress_bar = st.progress(0)
            status_text = st.empty()

            for i, img in enumerate(images, 1):
                status_text.text(f"Processing page {i}/{total_pages}")

                # Preprocess image
                img = self._preprocess_image(img)

                # Extract text
                page_text = pytesseract.image_to_string(
                    img,
                    config=self.ocr_config
                )

                text += f"\n--- Page {i} ---\n" + page_text
                progress_bar.progress(i/total_pages)

            status_text.empty()
            progress_bar.empty()

            cleaned_text = self._clean_extracted_text(text)
            st.success(f"Successfully processed {total_pages} pages")
            return cleaned_text

        except Exception as e:
            st.error(f"PDF processing error: {str(e)}")
            return ""

    def _extract_from_image(self, file) -> str:
        """Extract text from image files"""
        try:
            image = Image.open(file)
            image = self._preprocess_image(image)
            text = pytesseract.image_to_string(image, config=self.ocr_config)
            return self._clean_extracted_text(text)
        except Exception as e:
            st.error(f"Image processing error: {str(e)}")
            return ""

    def _extract_from_docx(self, file) -> str:
        """Extract text from DOCX files"""
        try:
            text = docx2txt.process(io.BytesIO(file.read()))
            return self._clean_extracted_text(text)
        except Exception as e:
            st.error(f"DOCX processing error: {str(e)}")
            return ""

    def _preprocess_image(self, image: Image.Image) -> Image.Image:
        """Preprocess image for better OCR results"""
        try:
            img = image.copy()

            # Convert to RGB if necessary
            if img.mode not in ['L', 'RGB']:
                img = img.convert('RGB')

            # Enhance contrast
            enhancer = ImageEnhance.Contrast(img)
            img = enhancer.enhance(1.5)

            # Enhance sharpness
            enhancer = ImageEnhance.Sharpness(img)
            img = enhancer.enhance(1.5)

            return img
        except Exception as e:
            self.logger.warning(f"Image preprocessing warning: {str(e)}")
            return image

    def _clean_extracted_text(self, text: str) -> str:
        """Clean and normalize extracted text"""
        try:
            # Remove excessive whitespace
            text = re.sub(r'\s+', ' ', text)

            # Remove non-printable characters
            text = ''.join(char for char in text if char.isprintable())

            # Normalize line endings
            text = text.replace('\r', '\n')
            text = re.sub(r'\n\s*\n', '\n\n', text)

            return text.strip()
        except Exception as e:
            self.logger.warning(f"Text cleaning warning: {str(e)}")
            return text.strip()

    def analyze_document(self, text: str) -> Dict:
        """Analyze document content"""
        try:
            if not isinstance(text, str) or not text.strip():
                raise ValueError("Invalid or empty text provided")

            # Process with spaCy
            doc = self.nlp(text[:1000000])  # Limit text length for processing

            # Basic analysis
            analysis = {
                'entities': self._extract_entities(doc),
                'summary': self._generate_summary(doc),
                'statistics': self._generate_statistics(doc)
            }

            return analysis

        except Exception as e:
            self.logger.error(f"Document analysis error: {str(e)}")
            return {
                'entities': [],
                'summary': f"Analysis failed: {str(e)}",
                'statistics': {}
            }

    def _extract_entities(self, doc) -> List[Dict]:
        """Extract named entities"""
        try:
            return [{'text': ent.text, 'label': ent.label_}
                   for ent in doc.ents]
        except Exception as e:
            self.logger.error(f"Entity extraction error: {str(e)}")
            return []

    def _generate_summary(self, doc) -> str:
        """Generate document summary"""
        try:
            sentences = list(doc.sents)
            summary_sents = sentences[:3]
            return ' '.join([sent.text for sent in summary_sents])
        except Exception as e:
            self.logger.error(f"Summary generation error: {str(e)}")
            return "Summary generation failed"

    def _generate_statistics(self, doc) -> Dict:
        """Generate document statistics"""
        try:
            return {
                'word_count': len([token for token in doc if not token.is_space]),
                'sentence_count': len(list(doc.sents)),
                'entity_count': len(doc.ents)
            }
        except Exception as e:
            self.logger.error(f"Statistics generation error: {str(e)}")
            return {}

    # def analyze_document_complete(self, file, target_language: str = 'en') -> Dict:
    #     """Complete enhanced document analysis with legal expertise features"""
    #     try:
    #         # Basic analysis from parent class
    #         text = self.extract_text(file)
    #         if not text:
    #             return {"error": "No text could be extracted from the document"}
    
    #         # Enhanced analysis with legal expertise
    #         analysis = {
    #             'document_type': self._identify_document_type(text),
    #             'simplified_explanation': self._generate_simplified_explanation(text),
    #             'key_terms_explained': self._explain_legal_terms(text),
    #             'red_flags': self._identify_red_flags(text),
    #             'action_items': self._generate_action_items(text),
    #             'property_details': self._extract_property_details(text),
    #             'property_valuation': self._estimate_property_value(text),
    #             'verification_checklist': self._generate_verification_checklist(text),
    #             'translated_content': self._translate_content(text, target_language),
    #             'recommendations': self._generate_recommendations(text),
                
    #             # New legal expertise features
    #             'legal_precedents': self.extract_legal_precedents(text),
    #             'enforceaAbility_analysis': self.analyze_document_enforceability(text),
    #             'terminology_standardization': self.legal_terminology_standardization(text),
    #             'potential_objections': self.generate_legal_objections(text, "buyer"),
    #             'stamp_duty_analysis': self.stamp_duty_calculator(text),
    #             'force_majeure_analysis': self.analyze_force_majeure_clauses(text),
    #             'timeline_requirements': self.interpret_legal_timeline_requirements(text)
    #         }
    
    #         return analysis
    
        except Exception as e:
            self.logger.error(f"Enhanced analysis error: {str(e)}")
            return {"error": f"Analysis failed: {str(e)}"}


class EnhancedLegalDocumentProcessor(LegalDocumentProcessor):
    """Enhanced Legal Document Processing System with Advanced Features"""

    def __init__(self):
        super().__init__()
        self.translator = Translator()
        self.load_legal_terms_database()
        self.load_document_templates()
        self.load_property_valuation_data()

        self.supported_languages = {
            'en': 'English',
            'hi': 'Hindi',
            'ta': 'Tamil',
            'te': 'Telugu',
            'mr': 'Marathi',
            'bn': 'Bengali',
            'gu': 'Gujarati',
            'kn': 'Kannada',
            'ml': 'Malayalam'
        }

    def load_legal_terms_database(self):
        """Load database of legal terms with simple explanations"""
        self.legal_terms = {
            "easement": {
                "simple": "Permission to use someone else's property",
                "example": "Right to walk through neighbor's land to reach your property",
                "importance": "High"
            },
            "encumbrance": {
                "simple": "Any burden on the property",
                "example": "A loan or unpaid tax on the property",
                "importance": "High"
            },
            "covenant": {
                "simple": "A promise about the property",
                "example": "Promise not to build more than two stories",
                "importance": "Medium"
            },
            "consideration": {
                "simple": "The price or payment",
                "example": "The amount you pay to buy the property",
                "importance": "High"
            },
            "deed": {
                "simple": "Legal ownership document",
                "example": "The paper that proves you own the property",
                "importance": "High"
            },
            "lien": {
                "simple": "Legal right over property until debt is paid",
                "example": "Bank's right over property until loan is repaid",
                "importance": "High"
            },
            "mortgage": {
                "simple": "Property loan",
                "example": "Bank loan to buy property",
                "importance": "High"
            }
        }

    def load_document_templates(self):
        """Load document templates for different property documents"""
        self.document_templates = {
            "sale_deed": {
                "required_sections": ["seller details", "buyer details", "property description", "consideration"],
                "essential_clauses": ["payment terms", "possession date", "title clearance"],
                "verification_checklist": ["seller identity", "property title", "encumbrance check"]
            },
            "rental_agreement": {
                "required_sections": ["landlord details", "tenant details", "rent amount", "duration"],
                "essential_clauses": ["payment schedule", "maintenance", "security deposit"],
                "verification_checklist": ["identity proof", "property ownership", "rental terms"]
            },
            "mortgage_deed": {
                "required_sections": ["borrower details", "lender details", "loan amount", "property details"],
                "essential_clauses": ["interest rate", "repayment schedule", "default conditions"],
                "verification_checklist": ["property valuation", "title check", "borrower credentials"]
            }
        }

    def load_property_valuation_data(self):
        """Load property valuation reference data"""
        # This would ideally come from a database or API
        self.valuation_data = {
            "location_multipliers": {
                "urban": 1.5,
                "suburban": 1.2,
                "rural": 1.0
            },
            "property_type_factors": {
                "residential": 1.0,
                "commercial": 1.3,
                "industrial": 1.2
            }
        }

    def analyze_document_complete(self, file, target_language: str = 'en') -> Dict:
        """
        Complete enhanced document analysis with comprehensive legal expertise features.
        
        This method performs a thorough analysis of legal documents with robust error handling
        to ensure that failures in one analysis component don't affect others.
        
        Args:
            file: The uploaded document file object
            target_language: Target language code for translation (default: 'en')
            
        Returns:
            Dict: Comprehensive analysis results dictionary
        """
        # Start with an empty analysis structure with default values
        analysis = {
            'document_type': {'type': 'Unknown', 'confidence': 0.0, 'possible_matches': []},
            'simplified_explanation': {'what_is_it': 'Unable to determine document type', 'key_points': [], 'watch_out_for': []},
            'key_terms_explained': [],
            'red_flags': [],
            'action_items': [],
            'property_details': {},
            'property_valuation': {},
            'verification_checklist': [],
            'translated_content': {'original_text': '', 'translated_text': '', 'language': target_language},
            'recommendations': [],
            'enforceability_analysis': {'overall_score': 0.0, 'risk_assessment': {'level': 'unknown', 'description': 'Analysis not available'}},
            'timeline_requirements': {'critical_deadlines': [], 'all_timelines': []},
            'legal_precedents': {'precedents': [], 'legal_strength': {'strength': 'unknown', 'reasoning': 'Analysis not available'}},
            'stamp_duty_analysis': {'stamp_duty_amount': 0.0, 'registration_fee': 0.0, 'total_charges': 0.0},
            'potential_objections': {'potential_objections': [], 'highest_risk_areas': []},
            'force_majeure_analysis': {'present': False, 'recommendation': 'Analysis not available'},
            'language_complexity': {'readability_metrics': {}, 'jargon_analysis': {}, 'simplification': {}}  # Add this line
        }
        
        try:
            # Step 1: Extract text - this is critical; if it fails, return early with error
            text = self.extract_text(file)
            if not text or len(text.strip()) < 50:  # Ensure we have meaningful text to analyze
                return {"error": "Insufficient text could be extracted from the document. Please check the document quality."}
            
            # Store the extracted text for potential troubleshooting
            analysis['extracted_text_length'] = len(text)
            
            # Step 2: Perform basic analysis functions with individual try-except blocks
            analysis_functions = [
                ('document_type', self._identify_document_type),
                ('simplified_explanation', self._generate_simplified_explanation),
                ('key_terms_explained', self._explain_legal_terms),
                ('red_flags', self._identify_red_flags),
                ('action_items', self._generate_action_items),
                ('property_details', self._extract_property_details),
                ('property_valuation', self._estimate_property_value),
                ('verification_checklist', self._generate_verification_checklist),
                ('recommendations', self._generate_recommendations)
            ]
            
            for key, func in analysis_functions:
                try:
                    analysis[key] = func(text)
                except Exception as e:
                    self.logger.warning(f"Error in {key} analysis: {str(e)}")
                    # Keep the default value already in the analysis dict
                    
            # Step 3: Handle translation separately since it needs target_language parameter
            try:
                analysis['translated_content'] = self._translate_content(text, target_language)
            except Exception as e:
                self.logger.warning(f"Translation error: {str(e)}")
                analysis['translated_content']['error'] = str(e)
            
            # Step 4: Perform advanced legal analysis with individual try-except blocks
            advanced_analysis_functions = [
                ('enforceability_analysis', self.analyze_document_enforceability),
                ('timeline_requirements', self.interpret_legal_timeline_requirements),
                ('legal_precedents', self.extract_legal_precedents),
                ('potential_objections', lambda t: self.generate_legal_objections(t, "buyer")),
                ('language_complexity', self.analyze_language_complexity)  # Add this line
            ]
            
            for key, func in advanced_analysis_functions:
                try:
                    analysis[key] = func(text)
                except Exception as e:
                    self.logger.warning(f"Error in {key} analysis: {str(e)}")
                    # Keep the default value already in the analysis dict
                    
            # Step 5: Handle stamp duty calculation which needs additional potential parameters
            try:
                # Try to get property value from previous analysis
                property_value = None
                if 'property_valuation' in analysis and 'estimated_value' in analysis['property_valuation']:
                    property_value = analysis['property_valuation'].get('estimated_value')
                    
                analysis['stamp_duty_analysis'] = self.stamp_duty_calculator(text, property_value)
            except Exception as e:
                self.logger.warning(f"Stamp duty calculation error: {str(e)}")
                # Keep the default value
                
            # Step 6: Force majeure analysis
            try:
                analysis['force_majeure_analysis'] = self.analyze_force_majeure_clauses(text)
            except Exception as e:
                self.logger.warning(f"Force majeure analysis error: {str(e)}")
                # Keep the default value
            
            # Step 7: Calculate analysis completeness score
            successful_analyses = sum(1 for key, value in analysis.items() 
                                    if key not in ['extracted_text_length'] and 
                                    (isinstance(value, dict) and not value.get('error')))
            
            total_analyses = len(analysis) - 1  # Subtract 1 for extracted_text_length
            analysis['meta'] = {
                'completeness': successful_analyses / total_analyses if total_analyses > 0 else 0,
                'timestamp': datetime.now().isoformat(),
                'document_size_bytes': file.size if hasattr(file, 'size') else 0,
                'document_name': file.name if hasattr(file, 'name') else 'unknown'
            }
            
            return analysis

        except Exception as e:
            self.logger.error(f"Critical error in document analysis: {str(e)}")
            return {
                "error": f"Analysis failed: {str(e)}",
                "partial_results": analysis if 'analysis' in locals() else {}
            }

    def _identify_document_type(self, text: str) -> Dict:
        """Identify document type with detailed analysis"""
        doc_types = {
            r'(?i)sale.*deed|conveyance.*deed': 'Sale Deed',
            r'(?i)lease.*agreement|rental.*agreement': 'Rental Agreement',
            r'(?i)mortgage.*deed|loan.*agreement': 'Mortgage Deed',
            r'(?i)gift.*deed': 'Gift Deed',
            r'(?i)power.*attorney': 'Power of Attorney',
            r'(?i)will|testament': 'Will',
            r'(?i)title.*deed': 'Title Deed'
        }

        identified_type = 'Other'
        confidence = 0.0
        matches = []

        for pattern, doc_type in doc_types.items():
            if re.search(pattern, text):
                matches.append(doc_type)
                identified_type = doc_type
                confidence = 0.9 if len(matches) == 1 else 0.7

        return {
            'type': identified_type,
            'confidence': confidence,
            'possible_matches': matches
        }

    def _generate_simplified_explanation(self, text: str) -> Dict:
        """Generate simple, easy-to-understand explanation"""
        doc_type = self._identify_document_type(text)['type']
        explanations = {
            'Sale Deed': {
                'what_is_it': "This is a document that transfers property ownership from seller to buyer",
                'key_points': [
                    "Shows who is selling and who is buying the property",
                    "States the price being paid",
                    "Describes the property being sold",
                    "Transfers ownership rights"
                ],
                'watch_out_for': [
                    "Check if all sellers have signed",
                    "Verify property description matches reality",
                    "Ensure price and payment terms are clear",
                    "Look for any conditions or restrictions"
                ]
            },
            'Rental Agreement': {
                'what_is_it': "This is a contract between property owner and tenant for temporary use of property",
                'key_points': [
                    "States monthly rent amount and due date",
                    "Shows how long you can stay",
                    "Lists rules and responsibilities",
                    "Mentions security deposit details"
                ],
                'watch_out_for': [
                    "Check maintenance responsibilities",
                    "Verify notice period for leaving",
                    "Understand what's included in rent",
                    "Look for rules about modifications"
                ]
            }
            # Add more document types as needed
        }

        return explanations.get(doc_type, {
            'what_is_it': "This appears to be a property-related legal document",
            'key_points': self._extract_key_points(text),
            'watch_out_for': self._identify_important_clauses(text)
        })

    def _explain_legal_terms(self, text: str) -> List[Dict]:
        """Find and explain legal terms in simple language"""
        explained_terms = []

        for term, details in self.legal_terms.items():
            if re.search(rf'\b{term}\b', text, re.IGNORECASE):
                # Find the context where the term appears
                matches = re.finditer(rf'\b{term}\b', text, re.IGNORECASE)
                for match in matches:
                    start = max(0, match.start() - 100)
                    end = min(len(text), match.end() + 100)
                    context = text[start:end]

                    explained_terms.append({
                        'term': term,
                        'simple_explanation': details['simple'],
                        'example': details['example'],
                        'importance': details['importance'],
                        'context': context,
                        'where_found': f"Page {self._estimate_page_number(start, text)}"
                    })

        return explained_terms

    def _identify_red_flags(self, text: str) -> List[Dict]:
        """Identify potential issues or red flags"""
        red_flags = []

        red_flag_patterns = {
            r'(?i)pending.*litigation': {
                'warning': "There might be ongoing legal disputes",
                'severity': "High",
                'action': "Consult a lawyer immediately"
            },
            r'(?i)encroachment|encumbrance': {
                'warning': "Property might have ownership/boundary issues",
                'severity': "High",
                'action': "Get property survey and encumbrance certificate"
            },
            r'(?i)unpaid.*tax|outstanding.*dues': {
                'warning': "There might be pending payments",
                'severity': "Medium",
                'action': "Request tax payment history and clearance certificate"
            },
            r'(?i)without.*permission|unauthorized': {
                'warning': "There might be unauthorized constructions",
                'severity': "High",
                'action': "Verify building approvals and permits"
            },
            r'(?i)subject.*approval|conditional': {
                'warning': "Transaction needs additional approvals",
                'severity': "Medium",
                'action': "Check which approvals are pending"
            }
        }

        for pattern, details in red_flag_patterns.items():
            matches = re.finditer(pattern, text)
            for match in matches:
                start = max(0, match.start() - 100)
                end = min(len(text), match.end() + 100)
                context = text[start:end]

                red_flags.append({
                    'warning': details['warning'],
                    'severity': details['severity'],
                    'context': context,
                    'suggested_action': details['action'],
                    'location': f"Page {self._estimate_page_number(start, text)}"
                })

        return red_flags

    def _generate_action_items(self, text: str) -> List[Dict]:
        """Generate list of required actions"""
        doc_type = self._identify_document_type(text)['type']
        actions = []

        # Common actions for all documents
        common_actions = [
            {
                'action': "Verify identities of all parties",
                'deadline': "Before signing",
                'documents_needed': ["ID proof", "Address proof"],
                'priority': "High"
            },
            {
                'action': "Check property ownership documents",
                'deadline': "Before signing",
                'documents_needed': ["Title deed", "Tax receipts"],
                'priority': "High"
            }
        ]

        # Document-specific actions
        specific_actions = {
            'Sale Deed': [
                {
                    'action': "Pay stamp duty",
                    'deadline': "Before registration",
                    'documents_needed': ["Stamp duty calculation"],
                    'priority': "High"
                },
                {
                    'action': "Register the deed",
                    'deadline': "Within 4 months of execution",
                    'documents_needed': ["Original deed", "Payment receipts"],
                    'priority': "High"
                }
            ],
            'Rental Agreement': [
                {
                    'action': "Pay security deposit",
                    'deadline': "Before possession",
                    'documents_needed': ["Payment receipt"],
                    'priority': "High"
                },
                {
                    'action': "Document property condition",
                    'deadline': "Before possession",
                    'documents_needed': ["Photographs", "Condition report"],
                    'priority': "Medium"
                }
            ]
        }

        actions.extend(common_actions)
        actions.extend(specific_actions.get(doc_type, []))

        # Add any timeline-specific actions found in the document
        timeline_actions = self._extract_timeline_actions(text)
        actions.extend(timeline_actions)

        return actions

    def _extract_property_details(self, text: str) -> Dict:
        """Extract comprehensive property details"""
        details = {
            'location': self._extract_location(text),
            'area': self._extract_area_details(text),
            'boundaries': self._extract_boundaries(text),
            'features': self._extract_features(text),
            'legal_status': self._extract_legal_status(text)
        }
        return details

    def _estimate_property_value(self, text: str) -> Dict:
        """Estimate property value based on extracted details"""
        try:
            # Extract basic details
            area_details = self._extract_area_details(text)
            location = self._extract_location(text)
            property_type = self._determine_property_type(text)

            # Get valuation factors
            location_multiplier = self.valuation_data['location_multipliers'].get(
                self._determine_location_type(location), 1.0
            )
            type_factor = self.valuation_data['property_type_factors'].get(
                property_type, 1.0
            )

            # Calculate estimated value
            # This is a simplified calculation - in reality, would use more sophisticated methods
            base_value = float(area_details.get('total_area', 0)) * 1000  # Assumed base rate
            estimated_value = base_value * location_multiplier * type_factor

            return {
                'estimated_value': estimated_value,
                'confidence_level': 'Medium',
                'factors_considered': {
                    'location_type': self._determine_location_type(location),
                    'property_type': property_type,
                    'total_area': area_details.get('total_area', 0),
                    'location_multiplier': location_multiplier,
                    'type_factor': type_factor
                },
                'market_trends': self._get_market_trends(location)
            }

        except Exception as e:
            self.logger.error(f"Valuation error: {str(e)}")
            return {}

    def _translate_content(self, text: str, target_language: str) -> Dict:
        """Translate document content to target language with improved handling"""
        try:
            # If target language is English, return original text
            if target_language == 'en':
                return {'original_text': text}

            # Validate target language
            if target_language not in self.supported_languages:
                return {
                    'error': f"Unsupported language: {target_language}",
                    'original_text': text
                }

            # Preprocess text to remove excessive whitespace and newlines
            cleaned_text = re.sub(r'\s+', ' ', text).strip()

            # Break large documents into more manageable chunks
            max_chunk_size = 3000  # Reduced chunk size for better translation quality
            chunks = [cleaned_text[i:i+max_chunk_size] for i in range(0, len(cleaned_text), max_chunk_size)]

            translated_chunks = []
            for chunk in chunks:
                try:
                    # Use try-except for each chunk to handle potential translation failures
                    translation = self.translator.translate(chunk, dest=target_language)
                    translated_chunks.append(translation.text)
                except Exception as chunk_error:
                    self.logger.warning(f"Translation chunk error: {chunk_error}")
                    # Fallback to original chunk if translation fails
                    translated_chunks.append(chunk)

            # Join translated chunks, preserving some original formatting
            translated_text = ' '.join(translated_chunks)

            return {
                'original_text': text,
                'translated_text': translated_text,
                'language': self.supported_languages.get(target_language, target_language)
            }

        except Exception as e:
            self.logger.error(f"Translation error: {str(e)}")
            return {
                'error': f"Translation failed: {str(e)}",
                'original_text': text
            }

    def _generate_verification_checklist(self, text: str) -> List[Dict]:
        """Generate comprehensive verification checklist"""
        doc_type = self._identify_document_type(text)['type']
        checklist = []

        # Basic verification for all documents
        common_checks = [
            {
                'item': "Verify party identities",
                'description': "Check ID proofs of all parties involved",
                'documents_needed': ["Aadhaar Card", "PAN Card", "Passport"],
                'status': False
            },
            {
                'item': "Property ownership verification",
                'description': "Verify current ownership documents",
                'documents_needed': ["Title Deed", "Property Tax Receipts"],
                'status': False
            },
            {
                'item': "Encumbrance check",
                'description': "Check for any existing loans or claims",
                'documents_needed': ["Encumbrance Certificate"],
                'status': False
            }
        ]

        checklist.extend(common_checks)

        # Document-specific checks
        if doc_type == "Sale Deed":
            checklist.extend([
                {
                    'item': "Payment verification",
                    'description': "Verify payment terms and receipts",
                    'documents_needed': ["Payment Receipts", "Bank Statements"],
                    'status': False
                },
                {
                    'item': "Stamp duty payment",
                    'description': "Verify correct stamp duty payment",
                    'documents_needed': ["Stamp Duty Receipt"],
                    'status': False
                }
            ])
        elif doc_type == "Rental Agreement":
            checklist.extend([
                {
                    'item': "Security deposit terms",
                    'description': "Verify security deposit amount and terms",
                    'documents_needed': ["Deposit Receipt"],
                    'status': False
                },
                {
                    'item': "Maintenance terms",
                    'description': "Check maintenance responsibilities",
                    'documents_needed': ["Agreement Clauses"],
                    'status': False
                }
            ])

        return checklist

    def generate_pdf_report(self, analysis: Dict, target_language: str = 'en') -> bytes:
        """Generate comprehensive PDF report with translations"""
        try:
            buffer = io.BytesIO()
            doc = SimpleDocTemplate(
                buffer,
                pagesize=letter,
                rightMargin=72,
                leftMargin=72,
                topMargin=72,
                bottomMargin=72
            )

            styles = getSampleStyleSheet()
            story = []

            # Custom styles
            title_style = ParagraphStyle(
                'CustomTitle',
                parent=styles['Heading1'],
                fontSize=24,
                spaceAfter=30
            )

            heading_style = ParagraphStyle(
                'CustomHeading',
                parent=styles['Heading2'],
                fontSize=16,
                spaceAfter=12
            )

            # Add title
            story.append(Paragraph("Property Document Analysis Report", title_style))
            story.append(Spacer(1, 12))

            # Add simple explanation
            story.append(Paragraph("Simple Explanation", heading_style))
            story.append(Paragraph(analysis['simplified_explanation']['what_is_it'], styles['Normal']))
            story.append(Spacer(1, 12))

            # Add key points
            story.append(Paragraph("Key Points to Understand", heading_style))
            for point in analysis['simplified_explanation']['key_points']:
                story.append(Paragraph(f"• {point}", styles['Normal']))
            story.append(Spacer(1, 12))

            # Add red flags section
            if analysis['red_flags']:
                story.append(Paragraph("⚠️ Important Warnings", heading_style))
                for flag in analysis['red_flags']:
                    story.append(Paragraph(f"Warning: {flag['warning']}", styles['Normal']))
                    story.append(Paragraph(f"Severity: {flag['severity']}", styles['Normal']))
                    story.append(Paragraph(f"What to do: {flag['suggested_action']}", styles['Normal']))
                story.append(Spacer(1, 12))

            # Add action items
            story.append(Paragraph("Required Actions", heading_style))
            for action in analysis['action_items']:
                story.append(Paragraph(f"• {action['action']}", styles['Normal']))
                if action.get('deadline'):
                    story.append(Paragraph(f"  Deadline: {action['deadline']}", styles['Normal']))
            story.append(Spacer(1, 12))

            # Add property details
            if 'property_details' in analysis:
                story.append(Paragraph("Property Details", heading_style))
                details = analysis['property_details']
                if details.get('location'):
                    story.append(Paragraph(f"Location: {details['location']}", styles['Normal']))
                if details.get('area'):
                    story.append(Paragraph(f"Area: {details['area']}", styles['Normal']))
            story.append(Spacer(1, 12))

            # Add verification checklist
            story.append(Paragraph("Document Verification Checklist", heading_style))
            for item in analysis['verification_checklist']:
                story.append(Paragraph(f"□ {item['item']}", styles['Normal']))
                story.append(Paragraph(f"   {item['description']}", styles['Normal']))
            story.append(Spacer(1, 12))

            # Add translated content if available
            if target_language != 'en' and 'translated_content' in analysis:
                story.append(Paragraph(f"Translated Content ({analysis['translated_content']['language']})",
                                    heading_style))
                story.append(Paragraph(analysis['translated_content']['translated_text'], styles['Normal']))

            # Build PDF
            doc.build(story)

            pdf_content = buffer.getvalue()
            buffer.close()

            return pdf_content

        except Exception as e:
            self.logger.error(f"PDF generation error: {str(e)}")
            return b""


    def _extract_key_points(self, text: str) -> List[str]:
        """Extract key points from the document text"""
        try:
            # Basic implementation to extract key points
            # This is a simplified approach and may need more sophisticated NLP techniques
            sentences = self.nlp(text[:5000]).sents

            # Look for sentences that might contain key information
            key_points = []
            for sent in sentences:
                # Filter sentences based on certain criteria
                if (len(sent) > 5 and len(sent) < 30 and  # Reasonable sentence length
                    any(token.pos_ in ['NOUN', 'PROPN', 'NUM'] for token in sent)):  # Contains substantive information
                    key_points.append(sent.text)

                # Limit to first 3-5 key points
                if len(key_points) >= 5:
                    break

            return key_points
        except Exception as e:
            self.logger.warning(f"Key points extraction error: {str(e)}")
            return ["Unable to extract key points"]

    def _identify_important_clauses(self, text: str) -> List[str]:
        """Identify important clauses or sections to watch out for"""
        try:
            # Basic implementation to identify important clauses
            important_patterns = [
                r'(?i)payment\s+terms',
                r'(?i)ownership\s+rights',
                r'(?i)legal\s+obligations',
                r'(?i)conditions\s+and\s+restrictions',
                r'(?i)termination\s+clause'
            ]

            important_clauses = []
            for pattern in important_patterns:
                matches = re.findall(pattern, text)
                if matches:
                    important_clauses.append(f"Found clause matching: {matches[0]}")

            return important_clauses or ["No specific important clauses identified"]
        except Exception as e:
            self.logger.warning(f"Important clauses identification error: {str(e)}")
            return ["Unable to identify important clauses"]

    def _estimate_page_number(self, position: int, text: str) -> int:
        """Estimate page number based on text position"""
        try:
            # Rough estimation based on text position
            # Assumes an average page length of 2000 characters
            page_number = position // 2000 + 1
            return page_number
        except Exception as e:
            self.logger.warning(f"Page number estimation error: {str(e)}")
            return 1

    def _extract_location(self, text: str) -> str:
        """Extract location from the document text"""
        try:
            # Use spaCy to find location entities
            doc = self.nlp(text[:5000])
            locations = [ent.text for ent in doc.ents if ent.label_ in ['GPE', 'LOC']]

            # If no specific location found, return a generic message
            return locations[0] if locations else "Location not specified"
        except Exception as e:
            self.logger.warning(f"Location extraction error: {str(e)}")
            return "Location extraction failed"

    def _extract_area_details(self, text: str) -> Dict:
        """Extract area details from the document text"""
        try:
            # Look for area-related patterns
            area_patterns = [
                r'(\d+(?:\.\d+)?)\s*(?:sq\.?\s*(?:ft|meter|m2|sq\.?\s*m))',
                r'(\d+(?:\.\d+)?)\s*(?:square\s*(?:feet|meters|foot|meter))',
                r'area\s*(?:of)?\s*(\d+(?:\.\d+)?)\s*(?:sq\.?\s*(?:ft|meter|m2|sq\.?\s*m))'
            ]

            for pattern in area_patterns:
                match = re.search(pattern, text, re.IGNORECASE)
                if match:
                    total_area = float(match.group(1))
                    unit = re.search(r'(sq\.?\s*(?:ft|meter|m2|sq\.?\s*m)|square\s*(?:feet|meters|foot|meter))', match.group(0), re.IGNORECASE).group(1)

                    return {
                        'total_area': total_area,
                        'unit': unit,
                        'area_display': f"{total_area} {unit}"  # Add a display-friendly string
                    }

            return {
                'total_area': 0,
                'unit': 'Not specified',
                'area_display': 'Area not specified'
            }
        except Exception as e:
            self.logger.warning(f"Area details extraction error: {str(e)}")
            return {
                'total_area': 0,
                'unit': 'Not specified',
                'area_display': 'Area extraction failed'
            }

    def _extract_boundaries(self, text: str) -> Dict:
        """Extract property boundaries from the document text"""
        try:
            # Look for boundary-related keywords
            boundary_keywords = [
                'north', 'south', 'east', 'west',
                'adjacent', 'borders', 'neighboring', 'boundary'
            ]

            boundaries = {}
            for direction in ['north', 'south', 'east', 'west']:
                match = re.search(rf'{direction}\s*(?:side|boundary)?\s*(?:is)?\s*(.+?)(?:\n|$)', text, re.IGNORECASE)
                if match:
                    boundaries[direction] = match.group(1).strip()

            return boundaries or {'description': 'Boundaries not clearly specified'}
        except Exception as e:
            self.logger.warning(f"Boundaries extraction error: {str(e)}")
            return {'description': 'Boundaries extraction failed'}

    def _extract_features(self, text: str) -> List[str]:
        """Extract property features from the document text"""
        try:
            # Look for common property features
            feature_patterns = [
                r'(?i)(parking)',
                r'(?i)(garden)',
                r'(?i)(balcony)',
                r'(?i)(terrace)',
                r'(?i)(lift)',
                r'(?i)(swimming\s*pool)',
                r'(?i)(security\s*system)'
            ]

            features = []
            for pattern in feature_patterns:
                matches = re.findall(pattern, text)
                features.extend(matches)

            return features or ['No specific features identified']
        except Exception as e:
            self.logger.warning(f"Features extraction error: {str(e)}")
            return ['Features extraction failed']

    def _extract_legal_status(self, text: str) -> Dict:
        """Extract legal status of the property"""
        try:
            # Check for common legal status indicators
            status_patterns = {
                'clear_title': r'(?i)(clear\s*title|free\s*from\s*encumbrances)',
                'disputed': r'(?i)(disputed\s*property|pending\s*litigation)',
                'mortgage': r'(?i)(mortgage|loan\s*attached)',
                'lease': r'(?i)(leased\s*property)'
            }

            legal_status = {}
            for key, pattern in status_patterns.items():
                match = re.search(pattern, text)
                legal_status[key] = bool(match)

            return legal_status
        except Exception as e:
            self.logger.warning(f"Legal status extraction error: {str(e)}")
            return {'error': 'Legal status extraction failed'}

    def _determine_property_type(self, text: str) -> str:
        """Determine the type of property"""
        try:
            # Look for property type keywords
            type_patterns = {
                'residential': r'(?i)(residential|apartment|flat|house)',
                'commercial': r'(?i)(commercial|office|shop|showroom)',
                'industrial': r'(?i)(industrial|warehouse|factory)',
                'agricultural': r'(?i)(agricultural|farm|land)'
            }

            for property_type, pattern in type_patterns.items():
                if re.search(pattern, text):
                    return property_type

            return 'residential'  # Default to residential if no type found
        except Exception as e:
            self.logger.warning(f"Property type determination error: {str(e)}")
            return 'residential'

    def _determine_location_type(self, location: str) -> str:
        """Determine the type of location (urban, suburban, rural)"""
        try:
            # Simple location type determination
            urban_keywords = ['city', 'metro', 'urban', 'downtown', 'center']
            suburban_keywords = ['suburb', 'outskirts', 'peripheral']

            location_lower = location.lower()

            if any(keyword in location_lower for keyword in urban_keywords):
                return 'urban'
            elif any(keyword in location_lower for keyword in suburban_keywords):
                return 'suburban'
            else:
                return 'rural'
        except Exception as e:
            self.logger.warning(f"Location type determination error: {str(e)}")
            return 'rural'

    def _get_market_trends(self, location: str) -> Dict:
        """Get market trends for the location"""
        try:
            # This would typically come from a real-time market data source
            # For now, we'll provide a placeholder
            return {
                'trend': 'Stable',
                'average_price_per_sqft': 5000,  # Example value
                'price_change_percentage': 2.5,
                'data_confidence': 'Low'
            }
        except Exception as e:
            self.logger.warning(f"Market trends retrieval error: {str(e)}")
            return {}

    def _extract_timeline_actions(self, text: str) -> List[Dict]:
        """Extract timeline-specific actions from the document"""
        try:
            # Look for time-related action keywords
            timeline_patterns = [
                r'(?i)(complete\s*by)\s*(\d{1,2}\s*(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)(?:\s*\d{4})?)',
                r'(?i)(deadline)\s*(\d{1,2}\s*(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)(?:\s*\d{4})?)',
                r'(?i)(to\s*be\s*done)\s*(?:before|by)\s*(\d{1,2}\s*(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)(?:\s*\d{4})?)'
            ]

            timeline_actions = []
            for pattern in timeline_patterns:
                matches = re.finditer(pattern, text, re.IGNORECASE)
                for match in matches:
                    timeline_actions.append({
                        'action': 'Timeline-specific action',
                        'deadline': match.group(2),
                        'documents_needed': [],
                        'priority': 'Medium'
                    })

            return timeline_actions
        except Exception as e:
            self.logger.warning(f"Timeline actions extraction error: {str(e)}")
            return []

    def _generate_recommendations(self, text: str) -> List[str]:
        """Generate recommendations based on document analysis"""
        try:
            # Basic recommendation generation
            recommendations = []

            # Check for potential issues or improvements
            if 'mortgage' in text.lower():
                recommendations.append("Consider consulting a financial advisor about loan terms")

            if re.search(r'(?i)unpaid|outstanding|dues', text):
                recommendations.append("Verify and clear any outstanding payments")

            if 'lease' in text.lower():
                recommendations.append("Review lease terms carefully, especially renewal and termination clauses")

            return recommendations or ["No specific recommendations at this time"]
        except Exception as e:
            self.logger.warning(f"Recommendations generation error: {str(e)}")
            return ["Unable to generate recommendations"]


    def extract_legal_precedents(self, document_text: str) -> Dict:
        """
        Extract and analyze legal precedents or case law references in documents
        to determine how they affect document validity
        """
        precedent_patterns = {
            "Supreme Court": r"(?i)((\d{4})\s+SCC\s+(\d+))|(\d+\s+Supreme\s+Court\s+Cases\s+\d+)",
            "High Court": r"(?i)((\d{4})\s+(\w+)\s+HC\s+(\d+))",
            "Tribunal": r"(?i)((\d{4})\s+(\w+)\s+Tribunal\s+(\d+))"
        }
        
        precedents_found = []
        for court_type, pattern in precedent_patterns.items():
            matches = re.finditer(pattern, document_text)
            for match in matches:
                context = self._get_citation_context(document_text, match.span(), 200)
                precedents_found.append({
                    'court': court_type,
                    'citation': match.group(0),
                    'context': context,
                    'impact_analysis': self._analyze_precedent_impact(context)
                })
        
        return {
            'precedents': precedents_found,
            'legal_strength': self._evaluate_precedent_strength(precedents_found),
            'recommendations': self._generate_precedent_recommendations(precedents_found)
        }

    def analyze_document_enforceability(self, text: str) -> Dict:
        """
        Analyze the enforceability of a legal document based on structure,
        content, and completeness against legal requirements
        """
        doc_type = self._identify_document_type(text)['type']
        enforceability = {
            'execution_requirements': self._check_execution_requirements(text, doc_type),
            'essential_clauses': self._check_essential_clauses(text, doc_type),
            'consideration_analysis': self._analyze_consideration(text),
            'party_capacity': self._check_party_capacity(text),
            'vague_language': self._identify_vague_terms(text),
            'contradictory_clauses': self._find_contradictions(text),
            'witness_requirements': self._check_witness_details(text, doc_type)
        }
        
        # Calculate overall enforceability score
        score_factors = [
            enforceability['execution_requirements']['complete'],
            enforceability['essential_clauses']['completeness_score'],
            not enforceability['vague_language']['significant_issues'],
            not enforceability['contradictory_clauses']['found']
        ]
        
        enforceability['overall_score'] = sum(1 for factor in score_factors if factor) / len(score_factors)
        enforceability['risk_assessment'] = self._classify_enforceability_risk(enforceability)
        
        return enforceability
    
    def legal_terminology_standardization(self, text: str) -> Dict:
        """
        Standardize and validate legal terminology according to Indian legal conventions
        and suggest corrections for non-standard or ambiguous terms
        """
        legal_term_patterns = self._load_legal_term_patterns()
        non_standard_terms = []
        
        for category, patterns in legal_term_patterns.items():
            for non_standard, standard in patterns.items():
                if re.search(r'\b' + re.escape(non_standard) + r'\b', text, re.IGNORECASE):
                    occurrences = [m.start() for m in re.finditer(r'\b' + re.escape(non_standard) + r'\b', text, re.IGNORECASE)]
                    non_standard_terms.append({
                        'term': non_standard,
                        'standard_form': standard,
                        'category': category,
                        'occurrences': len(occurrences),
                        'positions': occurrences[:5],  # First 5 positions
                        'legal_impact': self._assess_terminology_impact(non_standard, standard, category)
                    })
        
        return {
            'non_standard_terms': non_standard_terms,
            'standardization_impact': self._assess_overall_standardization_impact(non_standard_terms),
            'suggested_corrections': self._generate_terminology_corrections(text, non_standard_terms)
        }

    def generate_legal_objections(self, text: str, party_perspective: str = "buyer") -> Dict:
        """
        Generate potential legal objections that could be raised by different parties
        to the agreement, allowing users to prepare counterarguments or amendments
        """
        objection_patterns = self._load_objection_patterns_for_party(party_perspective)
        objections = []
        
        for objection_type, patterns in objection_patterns.items():
            for pattern, details in patterns.items():
                matches = re.finditer(pattern, text)
                for match in matches:
                    context = text[max(0, match.start() - 100):min(len(text), match.end() + 100)]
                    objections.append({
                        'type': objection_type,
                        'clause': context,
                        'legal_basis': details['legal_basis'],
                        'potential_argument': details['argument'],
                        'risk_level': details['risk_level'],
                        'suggested_amendment': self._generate_clause_amendment(context, details)
                    })
        
        return {
            'potential_objections': objections,
            'highest_risk_areas': [obj for obj in objections if obj['risk_level'] == 'High'],
            'amendment_recommendations': self._prioritize_amendments(objections)
        }

    def stamp_duty_calculator(self, text: str, property_value: float = None) -> Dict:
        """
        Calculate accurate stamp duty and registration fees based on document type,
        property details, and jurisdiction-specific rates
        """
        doc_type = self._identify_document_type(text)['type']
        jurisdiction = self._extract_jurisdiction(text)
        
        if not property_value:
            # Try to extract from document
            property_value = self._extract_property_value(text)
        
        # Stamp duty rates for different document types and jurisdictions
        stamp_duty_structure = self._get_stamp_duty_structure(jurisdiction, doc_type)
        
        # Calculate applicable stamp duty
        if stamp_duty_structure['calculation_method'] == 'percentage':
            stamp_duty = property_value * (stamp_duty_structure['rate'] / 100)
        else:
            stamp_duty = stamp_duty_structure['fixed_amount']
        
        # Apply any exemptions or concessions
        applicable_exemptions = self._find_applicable_exemptions(text, jurisdiction)
        if applicable_exemptions:
            stamp_duty = self._apply_exemptions(stamp_duty, applicable_exemptions)
        
        # Calculate registration fees
        registration_fee = self._calculate_registration_fee(property_value, jurisdiction)
        
        return {
            'stamp_duty_amount': stamp_duty,
            'registration_fee': registration_fee,
            'total_charges': stamp_duty + registration_fee,
            'applicable_exemptions': applicable_exemptions,
            'payment_procedure': stamp_duty_structure['payment_procedure'],
            'legal_references': stamp_duty_structure['legal_references']
        }

    def analyze_force_majeure_clauses(self, text: str) -> Dict:
        """
        Analyze force majeure clauses in the document for comprehensiveness and protection
        against different scenarios with jurisdiction-specific interpretation
        """
        force_majeure_patterns = [
            r'(?i)force\s+majeure',
            r'(?i)act\s+of\s+god',
            r'(?i)unavoidable\s+circumstances',
            r'(?i)beyond\s+(reasonable\s+)?control'
        ]
        
        # Find force majeure clauses
        force_majeure_text = ""
        for pattern in force_majeure_patterns:
            matches = re.finditer(pattern, text)
            for match in matches:
                # Extract the clause and surrounding context
                start = max(0, self._find_clause_start(text, match.start()))
                end = min(len(text), self._find_clause_end(text, match.end()))
                force_majeure_text += text[start:end] + "\n\n"
        
        # If no force majeure clause found
        if not force_majeure_text:
            return {
                'present': False,
                'recommendation': "No force majeure clause found. Consider adding comprehensive protection."
            }
        
        # Analyze the scope and protections
        covered_events = self._extract_covered_events(force_majeure_text)
        notification_requirements = self._extract_notification_requirements(force_majeure_text)
        duration_provisions = self._extract_duration_provisions(force_majeure_text)
        termination_rights = self._extract_termination_rights(force_majeure_text)
        
        # Assess comprehensiveness
        standard_events = self._get_standard_force_majeure_events()
        missing_events = [event for event in standard_events if event not in covered_events]
        
        return {
            'present': True,
            'clause_text': force_majeure_text,
            'covered_events': covered_events,
            'missing_events': missing_events,
            'notification_requirements': notification_requirements,
            'duration_provisions': duration_provisions,
            'termination_rights': termination_rights,
            'comprehensiveness_score': len(covered_events) / len(standard_events),
            'improvement_suggestions': self._generate_force_majeure_improvements(missing_events)
        }

    def interpret_legal_timeline_requirements(self, text: str) -> Dict:
        """
        Extract and interpret all timeline-related requirements and deadlines
        in the document with legal implications of missing each deadline
        """
        timeline_patterns = [
            r'(?i)within\s+(\d+)\s+(day|week|month|year)s?',
            r'(?i)not\s+later\s+than\s+(\d+)\s+(day|week|month|year)s?',
            r'(?i)before\s+the\s+(\d+)(st|nd|rd|th)\s+(?:day\s+)?of\s+(\w+)',
            r'(?i)on\s+or\s+before\s+(\d{1,2})[\/\.\-](\d{1,2})[\/\.\-](\d{2,4})',
        ]
        
        timelines = []
        for pattern in timeline_patterns:
            matches = re.finditer(pattern, text)
            for match in matches:
                context = text[max(0, match.start() - 100):min(len(text), match.end() + 100)]
                deadline_type = self._identify_deadline_type(context)
                
                timelines.append({
                    'deadline_text': match.group(0),
                    'context': context,
                    'deadline_type': deadline_type,
                    'parties_responsible': self._extract_responsible_parties(context),
                    'legal_consequence': self._identify_legal_consequences(context, deadline_type),
                    'criticality': self._assess_deadline_criticality(deadline_type)
                })
        
        # Sort timelines by criticality and date
        sorted_timelines = sorted(
            timelines, 
            key=lambda x: (x['criticality'] == 'High', x['criticality'] == 'Medium', x['criticality'] == 'Low')
        )
        
        return {
            'critical_deadlines': [t for t in timelines if t['criticality'] == 'High'],
            'all_timelines': sorted_timelines,
            'timeline_visualization': self._generate_timeline_visualization(timelines),
            'calendar_entries': self._generate_calendar_entries(timelines)
        }

    def _get_citation_context(self, text: str, match_span: tuple, context_chars: int = 200) -> str:
        """Extract context around a legal citation"""
        start = max(0, match_span[0] - context_chars)
        end = min(len(text), match_span[1] + context_chars)
        return text[start:end]

    def _analyze_precedent_impact(self, context: str) -> Dict:
        """Analyze the impact of a cited precedent on the document"""
        # Simple impact analysis based on context keywords
        impact_keywords = {
            'positive': ['supports', 'upholds', 'affirms', 'confirms', 'validates'],
            'negative': ['overrules', 'contradicts', 'invalidates', 'denies', 'rejects'],
            'neutral': ['discusses', 'considers', 'examines', 'refers to', 'mentions']
        }
        
        impact = 'neutral'
        for impact_type, keywords in impact_keywords.items():
            if any(keyword in context.lower() for keyword in keywords):
                impact = impact_type
                break
        
        return {
            'impact_type': impact,
            'relevance': 'high' if impact != 'neutral' else 'medium',
            'analysis': f"This precedent appears to {impact} the document's position."
        }

    def _evaluate_precedent_strength(self, precedents: List[Dict]) -> Dict:
        """Evaluate the overall strength of precedents cited"""
        if not precedents:
            return {
                'strength': 'weak',
                'reasoning': 'No legal precedents cited to support document provisions.'
            }
        
        # Count precedents by court and impact
        court_counts = {}
        impact_counts = {'positive': 0, 'negative': 0, 'neutral': 0}
        
        for prec in precedents:
            court = prec['court']
            impact = prec['impact_analysis']['impact_type']
            
            court_counts[court] = court_counts.get(court, 0) + 1
            impact_counts[impact] += 1
        
        # Evaluate strength based on court hierarchy and impact
        supreme_court_count = court_counts.get('Supreme Court', 0)
        high_court_count = court_counts.get('High Court', 0)
        
        if supreme_court_count > 0 and impact_counts['positive'] > impact_counts['negative']:
            strength = 'strong'
            reasoning = f"Document cites {supreme_court_count} Supreme Court precedents with mostly positive impact."
        elif high_court_count > 0 and impact_counts['positive'] > impact_counts['negative']:
            strength = 'moderate'
            reasoning = f"Document cites {high_court_count} High Court precedents with mostly positive impact."
        elif impact_counts['negative'] > impact_counts['positive']:
            strength = 'weak'
            reasoning = "Document cites precedents that mostly have negative impact on its provisions."
        else:
            strength = 'unclear'
            reasoning = "Document cites precedents with mixed or neutral impact."
        
        return {
            'strength': strength,
            'reasoning': reasoning,
            'court_distribution': court_counts,
            'impact_distribution': impact_counts
        }

    def _generate_precedent_recommendations(self, precedents: List[Dict]) -> List[str]:
        """Generate recommendations based on precedent analysis"""
        recommendations = []
        
        if not precedents:
            recommendations.append("Consider adding relevant legal precedents to strengthen document validity.")
            return recommendations
        
        # Check for negative precedents
        negative_precedents = [p for p in precedents if p['impact_analysis']['impact_type'] == 'negative']
        if negative_precedents:
            recommendations.append(f"Review {len(negative_precedents)} precedents that may negatively impact document validity.")
        
        # Check court hierarchy
        supreme_court_precedents = [p for p in precedents if p['court'] == 'Supreme Court']
        if not supreme_court_precedents:
            recommendations.append("Consider adding Supreme Court precedents to strengthen legal position.")
        
        # Check recency
        recent_precedents = [p for p in precedents if int(re.search(r'\d{4}', p['citation']).group(0)) >= 2015]
        if len(recent_precedents) < len(precedents) / 2:
            recommendations.append("Update cited precedents with more recent case law for stronger legal basis.")
        
        return recommendations

    def _check_execution_requirements(self, text: str, doc_type: str) -> Dict:
        """Check if document meets execution requirements"""
        execution_requirements = {
            'Sale Deed': {
                'signature_requirements': ['seller', 'buyer'],
                'witness_count': 2,
                'registration_required': True,
                'stamp_duty_required': True
            },
            'Rental Agreement': {
                'signature_requirements': ['landlord', 'tenant'],
                'witness_count': 2,
                'registration_required': False,
                'stamp_duty_required': True
            },
            'Mortgage Deed': {
                'signature_requirements': ['mortgagor', 'mortgagee'],
                'witness_count': 2,
                'registration_required': True,
                'stamp_duty_required': True
            }
        }
        
        requirements = execution_requirements.get(doc_type, {
            'signature_requirements': ['party1', 'party2'],
            'witness_count': 2,
            'registration_required': True,
            'stamp_duty_required': True
        })
        
        # Check for signatures
        signatures_found = self._find_signatures(text)
        signature_requirements_met = all(party in signatures_found for party in requirements['signature_requirements'])
        
        # Check for witnesses
        witnesses_count = self._count_witnesses(text)
        witness_requirements_met = witnesses_count >= requirements['witness_count']
        
        # Check for registration mentions
        registration_mentioned = bool(re.search(r'(?i)registration|registrar|sub-registrar', text))
        
        # Check for stamp duty mentions
        stamp_duty_mentioned = bool(re.search(r'(?i)stamp\s+duty|stamped|e-stamp', text))
        
        return {
            'complete': signature_requirements_met and witness_requirements_met,
            'signature_requirements_met': signature_requirements_met,
            'signatures_found': signatures_found,
            'witness_requirements_met': witness_requirements_met,
            'witnesses_found': witnesses_count,
            'registration_mentioned': registration_mentioned,
            'registration_required': requirements['registration_required'],
            'stamp_duty_mentioned': stamp_duty_mentioned,
            'stamp_duty_required': requirements['stamp_duty_required']
    }

    def _find_signatures(self, text: str) -> List[str]:
        """Find signature mentions in document"""
        signature_patterns = [
            r'(?i)signed\s+by\s+([^,\.]+)',
            r'(?i)signature\s+of\s+([^,\.]+)',
            r'(?i)(seller|buyer|landlord|tenant|mortgagor|mortgagee|party|lessor|lessee)[\s\']*s\s+signature'
        ]
        
        signatures = []
        for pattern in signature_patterns:
            matches = re.finditer(pattern, text)
            for match in matches:
                if len(match.groups()) > 0:
                    signatures.append(match.group(1).strip())
                else:
                    signatures.append(match.group(0).strip())
        
        return signatures

    def _count_witnesses(self, text: str) -> int:
        """Count witness mentions in document"""
        witness_patterns = [
            r'(?i)witness\s+\d',
            r'(?i)in\s+witness\s+whereof',
            r'(?i)witnessed\s+by',
            r'(?i)in\s+the\s+presence\s+of\s+witness'
        ]
        
        count = 0
        for pattern in witness_patterns:
            matches = re.finditer(pattern, text)
            count += sum(1 for _ in matches)
        
        return max(count, 0)  # Ensure non-negative

    def _check_essential_clauses(self, text: str, doc_type: str) -> Dict:
        """Check if document contains all essential clauses"""
        essential_clauses = {
            'Sale Deed': [
                {'name': 'parties_clause', 'pattern': r'(?i)this\s+deed\s+of\s+sale|between|party\s+of\s+the\s+first\s+part'},
                {'name': 'property_description', 'pattern': r'(?i)schedule\s+of\s+property|property\s+described|description\s+of\s+property'},
                {'name': 'consideration', 'pattern': r'(?i)consideration\s+of\s+(?:rs|inr)|price\s+of\s+(?:rs|inr)|sum\s+of\s+(?:rs|inr)'},
                {'name': 'conveyance', 'pattern': r'(?i)transfer|convey|grant|absolutely\s+sell'},
                {'name': 'possession', 'pattern': r'(?i)possession|handover|delivered\s+possession'}
            ],
            'Rental Agreement': [
                {'name': 'parties_clause', 'pattern': r'(?i)this\s+(?:lease|rental)\s+agreement|between|party\s+of\s+the\s+first\s+part'},
                {'name': 'property_description', 'pattern': r'(?i)schedule\s+of\s+property|premises\s+described|description\s+of\s+property'},
                {'name': 'rent_amount', 'pattern': r'(?i)rent\s+of\s+(?:rs|inr)|monthly\s+rent|agreed\s+rent'},
                {'name': 'duration', 'pattern': r'(?i)period\s+of\s+\d+|term\s+of\s+\d+|duration\s+of\s+\d+'},
                {'name': 'security_deposit', 'pattern': r'(?i)security\s+deposit|caution\s+deposit|advance\s+deposit'}
            ]
        }
        
        clauses = essential_clauses.get(doc_type, [
            {'name': 'parties_clause', 'pattern': r'(?i)this\s+(?:deed|agreement)|between|party\s+of\s+the\s+first\s+part'},
            {'name': 'property_description', 'pattern': r'(?i)schedule\s+of\s+property|property\s+described|description\s+of\s+property'},
            {'name': 'consideration', 'pattern': r'(?i)consideration\s+of\s+(?:rs|inr)|price\s+of\s+(?:rs|inr)|sum\s+of\s+(?:rs|inr)'}
        ])
        
        results = {}
        for clause in clauses:
            results[clause['name']] = bool(re.search(clause['pattern'], text))
        
        completeness_score = sum(1 for present in results.values() if present) / len(results)
        
        return {
            'clause_presence': results,
            'completeness_score': completeness_score,
            'missing_clauses': [name for name, present in results.items() if not present]
        }

    def _analyze_consideration(self, text: str) -> Dict:
        """Analyze consideration clause for adequacy and clarity"""
        # Find consideration mentions
        consideration_patterns = [
            r'(?i)consideration\s+of\s+(?:rs|inr)[\.|\s]*([\d,]+)',
            r'(?i)price\s+of\s+(?:rs|inr)[\.|\s]*([\d,]+)',
            r'(?i)sum\s+of\s+(?:rs|inr)[\.|\s]*([\d,]+)'
        ]
        
        consideration_amounts = []
        for pattern in consideration_patterns:
            matches = re.finditer(pattern, text)
            for match in matches:
                if match.group(1):
                    amount_str = match.group(1).replace(',', '')
                    try:
                        amount = float(amount_str)
                        consideration_amounts.append(amount)
                    except ValueError:
                        pass
        
        # Check for payment details
        payment_methods_mentioned = bool(re.search(r'(?i)bank\s+transfer|cheque|cash|rtgs|neft|dd|demand\s+draft', text))
        payment_proof_mentioned = bool(re.search(r'(?i)receipt|acknowledged|transaction\s+id|reference\s+number', text))
        
        if not consideration_amounts:
            return {
                'consideration_found': False,
                'adequacy': 'unknown',
                'clarity': 'low',
                'issues': ['No clear consideration amount found']
            }
        
        # Simple adequacy check (can be enhanced with property valuation)
        max_amount = max(consideration_amounts)
        adequacy = 'adequate' if max_amount > 10000 else 'questionable'
        
        # Clarity assessment
        clarity_factors = [
            payment_methods_mentioned,
            payment_proof_mentioned,
            len(consideration_amounts) == 1  # Only one amount mentioned (consistency)
        ]
        clarity = 'high' if all(clarity_factors) else 'medium' if sum(clarity_factors) >= 2 else 'low'
        
        issues = []
        if not payment_methods_mentioned:
            issues.append('Payment method not clearly specified')
        if not payment_proof_mentioned:
            issues.append('Payment acknowledgment or proof not mentioned')
        if len(consideration_amounts) > 1:
            issues.append('Multiple different consideration amounts found, may cause ambiguity')
        
        return {
            'consideration_found': True,
            'amount': max_amount,
            'adequacy': adequacy,
            'clarity': clarity,
            'payment_method_specified': payment_methods_mentioned,
            'payment_proof_mentioned': payment_proof_mentioned,
            'issues': issues
        }

    def _check_party_capacity(self, text: str) -> Dict:
        """Check for party capacity mentions and issues"""
        # Check for capacity indicators
        capacity_indicators = {
            'age_mentioned': r'(?i)aged|age\s+\d+|major',
            'sound_mind_mentioned': r'(?i)sound\s+mind|mentally\s+fit|competent',
            'authority_mentioned': r'(?i)authorized|power\s+of\s+attorney|behalf\s+of',
            'company_authorization': r'(?i)board\s+resolution|authorized\s+signatory|company\s+seal'
        }
        
        capacity_results = {}
        for indicator, pattern in capacity_indicators.items():
            capacity_results[indicator] = bool(re.search(pattern, text))
        
        # Check for capacity concerns
        capacity_concerns = {
            'minor_mentioned': r'(?i)minor|under\s+18|not\s+of\s+legal\s+age',
            'guardian_required': r'(?i)guardian|legal\s+representative|natural\s+guardian',
            'mental_capacity_concerns': r'(?i)unsound\s+mind|mentally\s+(?:unfit|unstable)',
            'authority_concerns': r'(?i)without\s+authority|unauthorized|acting\s+beyond'
        }
        
        concern_results = {}
        for concern, pattern in capacity_concerns.items():
            concern_results[concern] = bool(re.search(pattern, text))
        
        # Overall assessment
        has_concerns = any(concern_results.values())
        capacity_indicators_present = sum(1 for present in capacity_results.values() if present)
        
        return {
            'capacity_indicators': capacity_results,
            'concerns': concern_results,
            'has_capacity_concerns': has_concerns,
            'capacity_clarity': 'high' if capacity_indicators_present >= 3 else 'medium' if capacity_indicators_present >= 1 else 'low',
            'recommendation': "Document should explicitly mention parties' age and capacity" if capacity_indicators_present < 2 else ""
        }

    def _identify_vague_terms(self, text: str) -> Dict:
        """Identify vague or ambiguous terms in the document"""
        vague_terms = [
            r'(?i)reasonable',
            r'(?i)substantial',
            r'(?i)appropriate',
            r'(?i)satisfactory',
            r'(?i)adequate',
            r'(?i)as\s+soon\s+as\s+possible',
            r'(?i)from\s+time\s+to\s+time',
            r'(?i)etc\.?',
            r'(?i)and/or',
            r'(?i)necessary\s+or\s+desirable'
        ]
        
        found_terms = {}
        for term in vague_terms:
            matches = list(re.finditer(term, text))
            if matches:
                term_name = matches[0].group(0)
                found_terms[term_name] = {
                    'count': len(matches),
                    'examples': [text[max(0, m.start() - 50):min(len(text), m.end() + 50)] for m in matches[:3]]
                }
        
        significant_issues = len(found_terms) > 3 or any(details['count'] > 5 for details in found_terms.values())
        
        return {
            'vague_terms_found': found_terms,
            'significant_issues': significant_issues,
            'suggestions': self._generate_clarity_suggestions(found_terms)
        }

    def _generate_clarity_suggestions(self, vague_terms: Dict) -> List[str]:
        """Generate suggestions to improve clarity"""
        suggestions = []
        
        clarity_improvements = {
            'reasonable': 'Specify concrete criteria, e.g., "within 30 days" instead of "reasonable time"',
            'substantial': 'Define with specific percentage or amount, e.g., "at least 75% complete"',
            'appropriate': 'Specify what constitutes appropriate behavior or standards',
            'satisfactory': 'Define specific acceptance criteria or standards',
            'adequate': 'Specify minimum requirements explicitly',
            'as soon as possible': 'Set specific deadlines, e.g., "within 14 days"',
            'from time to time': 'Specify frequency, e.g., "quarterly" or "every six months"',
            'etc': 'List all items explicitly rather than using "etc."',
            'and/or': 'Use either "and" or "or" to avoid ambiguity',
            'necessary or desirable': 'Define which party determines necessity and based on what criteria'
        }
        
        for term, details in vague_terms.items():
            for vague_term, improvement in clarity_improvements.items():
                if vague_term in term.lower():
                    suggestions.append(f'Replace "{term}" ({details["count"]} occurrences): {improvement}')
                    break
        
        return suggestions


    
    def _find_contradictions(self, text: str) -> Dict:
        """Find potentially contradictory clauses in the document"""
        # Look for common contradiction indicators
        contradiction_indicators = [
            r'(?i)notwithstanding\s+the\s+foregoing',
            r'(?i)notwithstanding\s+anything\s+(?:to\s+the\s+contrary|contained\s+herein)',
            r'(?i)in\s+spite\s+of\s+clause',
            r'(?i)contrary\s+to\s+clause',
            r'(?i)supersedes',
            r'(?i)overrides',
            r'(?i)take\s+precedence\s+over'
        ]
        
        contradictions = []
        for indicator in contradiction_indicators:
            matches = re.finditer(indicator, text)
            for match in matches:
                start = max(0, match.start() - 150)
                end = min(len(text), match.end() + 150)
                contradictions.append({
                    'indicator': match.group(0),
                    'context': text[start:end],
                    'position': match.start()
                })
        
        return {
            'found': len(contradictions) > 0,
            'count': len(contradictions),
            'contradictions': contradictions,
            'risk': 'high' if len(contradictions) > 2 else 'medium' if len(contradictions) > 0 else 'low'
        }

    def _check_witness_details(self, text: str, doc_type: str) -> Dict:
        """Check for witness details in the document"""
        witness_patterns = [
            r'(?i)witness\s*:',
            r'(?i)in\s+witness\s+whereof',
            r'(?i)witness\s+\d',
            r'(?i)witnessed\s+by'
        ]
        
        witnesses_found = []
        for pattern in witness_patterns:
            matches = re.finditer(pattern, text)
            for match in matches:
                start = max(0, match.start() - 50)
                end = min(len(text), match.end() + 200)
                witness_context = text[start:end]
                
                # Check for witness details within context
                has_name = bool(re.search(r'(?i)name\s*:|\b[A-Z][a-z]+\s+[A-Z][a-z]+\b', witness_context))
                has_address = bool(re.search(r'(?i)address\s*:|resident\s+of|residing\s+at', witness_context))
                
                witnesses_found.append({
                    'context': witness_context,
                    'has_name': has_name,
                    'has_address': has_address,
                    'complete': has_name and has_address
                })
        
        # Requirements by document type
        required_witnesses = 2 if doc_type in ['Sale Deed', 'Mortgage Deed'] else 1
        
        return {
            'witnesses_found': len(witnesses_found),
            'witnesses_required': required_witnesses,
            'witnesses_complete': sum(1 for w in witnesses_found if w['complete']),
            'requirement_met': len(witnesses_found) >= required_witnesses,
            'details_complete': sum(1 for w in witnesses_found if w['complete']) >= required_witnesses,
            'witness_details': witnesses_found
        }
    
    def _classify_enforceability_risk(self, enforceability: Dict) -> Dict:
        """Classify the enforceability risk based on analysis results"""
        # Define risk factors and weights
        risk_factors = {
            'execution_incomplete': not enforceability['execution_requirements']['complete'],
            'missing_essential_clauses': enforceability['essential_clauses']['completeness_score'] < 0.8,
            'consideration_issues': 'consideration_analysis' in enforceability and enforceability['consideration_analysis'].get('clarity', 'low') == 'low',
            'capacity_concerns': enforceability['party_capacity']['has_capacity_concerns'],
            'significant_vague_terms': enforceability['vague_language']['significant_issues'],
            'contradictions_found': enforceability['contradictory_clauses']['found'],
            'witness_issues': 'witness_requirements' in enforceability and not enforceability['witness_requirements']['requirement_met']
        }
        
        # Count risk factors
        risk_count = sum(1 for factor, present in risk_factors.items() if present)
        
        # Classify risk
        if risk_count >= 3 or (risk_factors['execution_incomplete'] and risk_factors['missing_essential_clauses']):
            risk_level = 'high'
        elif risk_count >= 1:
            risk_level = 'medium'
        else:
            risk_level = 'low'
        
        # Generate risk description
        risk_description = f"Document has {risk_count} enforceability risk factors."
        if risk_factors['execution_incomplete']:
            risk_description += " Execution requirements incomplete."
        if risk_factors['missing_essential_clauses']:
            risk_description += f" Missing essential clauses: {', '.join(enforceability['essential_clauses'].get('missing_clauses', []))}."
        
        return {
            'level': risk_level,
            'description': risk_description,
            'risk_factors': {factor: present for factor, present in risk_factors.items() if present}
        }
    
    def _load_legal_term_patterns(self) -> Dict:
        """Load patterns for legal terminology standardization"""
        return {
            'conveyance_terms': {
                'give': 'convey',
                'handover': 'transfer',
                'pass on': 'assign',
                'sell and convey': 'convey',
                'grant, bargain and sell': 'convey',
                'forever quit claim': 'convey'
            },
            'payment_terms': {
                'money': 'consideration',
                'fees': 'consideration',
                'price': 'consideration',
                'amount': 'sum',
                'paid amount': 'consideration amount'
            },
            'party_terms': {
                'first party': 'seller/lessor/transferor',
                'second party': 'buyer/lessee/transferee',
                'vendor': 'seller',
                'purchaser': 'buyer',
                'home owner': 'seller',
                'new owner': 'buyer'
            },
            'condition_terms': {
                'can cancel': 'may terminate',
                'needs to': 'shall',
                'must compulsorily': 'shall',
                'can optionally': 'may',
                'has option to': 'may',
                'breaking the contract': 'breach of agreement'
            }
        }
    
    def _assess_terminology_impact(self, non_standard: str, standard: str, category: str) -> Dict:
        """Assess the legal impact of non-standard terminology"""
        impact_levels = {
            'conveyance_terms': 'high',
            'payment_terms': 'high',
            'party_terms': 'medium',
            'condition_terms': 'high'
        }
        
        impact_level = impact_levels.get(category, 'medium')
        
        impact_descriptions = {
            'high': 'This terminology difference could affect legal interpretation and enforceability',
            'medium': 'This terminology difference may cause ambiguity but likely won\'t affect validity',
            'low': 'This terminology difference is primarily stylistic with minimal legal impact'
        }
        
        return {
            'level': impact_level,
            'description': impact_descriptions[impact_level],
            'recommendation': f'Replace "{non_standard}" with the standard term "{standard}"'
        }
    
    def _assess_overall_standardization_impact(self, terms: List[Dict]) -> Dict:
        """Assess the overall impact of terminology standardization issues"""
        if not terms:
            return {
                'level': 'low',
                'description': 'No terminology standardization issues found',
                'recommendation': 'Document uses standard legal terminology'
            }
        
        high_impact_terms = [term for term in terms if term['legal_impact']['level'] == 'high']
        medium_impact_terms = [term for term in terms if term['legal_impact']['level'] == 'medium']
        
        if high_impact_terms:
            level = 'high'
            description = f'Document contains {len(high_impact_terms)} high-impact non-standard terms that could affect enforceability'
        elif len(medium_impact_terms) > 3:
            level = 'medium'
            description = f'Document contains {len(medium_impact_terms)} medium-impact non-standard terms that may cause ambiguity'
        else:
            level = 'low'
            description = f'Document contains {len(terms)} minor terminology inconsistencies with minimal legal impact'
        
        return {
            'level': level,
            'description': description,
            'high_impact_count': len(high_impact_terms),
            'medium_impact_count': len(medium_impact_terms),
            'low_impact_count': len(terms) - len(high_impact_terms) - len(medium_impact_terms)
        }
    
    def _generate_terminology_corrections(self, text: str, non_standard_terms: List[Dict]) -> Dict:
        """Generate corrected text with standardized terminology"""
        if not non_standard_terms:
            return {
                'corrections_needed': False,
                'corrected_text': text
            }
        
        # Sort terms by position (to replace from end to beginning to maintain positions)
        terms_by_position = []
        for term in non_standard_terms:
            for position in term.get('positions', []):
                terms_by_position.append({
                    'term': term['term'],
                    'standard_form': term['standard_form'],
                    'position': position
                })
        
        # Sort in reverse position order
        terms_by_position.sort(key=lambda x: x['position'], reverse=True)
        
        # Apply corrections
        corrected_text = text
        for term_pos in terms_by_position:
            term = term_pos['term']
            std_form = term_pos['standard_form']
            pos = term_pos['position']
            
            # Replace at specific position
            corrected_text = corrected_text[:pos] + std_form + corrected_text[pos + len(term):]
        
        return {
            'corrections_needed': True,
            'corrected_text': corrected_text,
            'correction_count': len(terms_by_position)
        }
    
    def _load_objection_patterns_for_party(self, party_perspective: str) -> Dict:
        """Load objection patterns from the perspective of a specific party"""
        if party_perspective.lower() in ['buyer', 'purchaser', 'lessee', 'tenant']:
            return {
                'title_objections': {
                    r'(?i)free\s+from\s+all\s+encumbrances': {
                        'legal_basis': 'TPA Section 55(1)(a)',
                        'argument': 'Title should be specifically verified with encumbrance certificate',
                        'risk_level': 'High'
                    },
                    r'(?i)seller\s+warrants?\s+title': {
                        'legal_basis': 'TPA Section 55(2)',
                        'argument': 'Warranty should specify indemnification for title defects',
                        'risk_level': 'Medium'
                    }
                },
                'payment_objections': {
                    r'(?i)consideration\s+paid\s+in\s+full': {
                        'legal_basis': 'Evidence Act Section 91-92',
                        'argument': 'Payment terms should specify mode, receipt confirmation and tax implications',
                        'risk_level': 'Medium'
                    }
                },
                'liability_objections': {
                    r'(?i)seller\s+not\s+liable': {
                        'legal_basis': 'Indian Contract Act Section 23',
                        'argument': 'Blanket liability exclusions may be void as against public policy',
                        'risk_level': 'High'
                    }
                }
            }
        else:  # Seller perspective
            return {
                'payment_objections': {
                    r'(?i)instalments|installments': {
                        'legal_basis': 'Specific Relief Act',
                        'argument': 'Installment terms should include specific default and remedy provisions',
                        'risk_level': 'High'
                    }
                },
                'possession_objections': {
                    r'(?i)possession\s+delivered': {
                        'legal_basis': 'TPA Section 55(1)(f)',
                        'argument': 'Possession clause should specify condition, inventory and utilities status',
                        'risk_level': 'Medium'
                    }
                },
                'representation_objections': {
                    r'(?i)buyer\s+has\s+inspected': {
                        'legal_basis': 'Indian Contract Act Section 17',
                        'argument': 'Buyer inspection clause should not absolve seller from disclosure obligations',
                        'risk_level': 'Medium'
                    }
                }
            }
    
    def _generate_clause_amendment(self, context: str, objection_details: Dict) -> str:
        """Generate suggested clause amendment to address potential objection"""
        amendment_templates = {
            'High': f"SUGGESTED REPLACEMENT: This clause requires significant revision. Based on {objection_details['legal_basis']}, consider: ",
            'Medium': f"SUGGESTED IMPROVEMENT: To strengthen this clause under {objection_details['legal_basis']}, consider adding: ",
            'Low': f"OPTIONAL ENHANCEMENT: For better clarity, consider adding: "
        }
        
        template = amendment_templates.get(objection_details['risk_level'], "SUGGESTED CHANGE: ")
        return template + objection_details['argument']
    
    def _prioritize_amendments(self, objections: List[Dict]) -> List[Dict]:
        """Prioritize amendments based on risk level and legal impact"""
        if not objections:
            return []
        
        # Group by risk level
        high_risk = [obj for obj in objections if obj['risk_level'] == 'High']
        medium_risk = [obj for obj in objections if obj['risk_level'] == 'Medium']
        low_risk = [obj for obj in objections if obj['risk_level'] == 'Low']
        
        # Prioritize amendments
        prioritized = []
        
        if high_risk:
            prioritized.append({
                'priority': 'Critical',
                'description': 'These amendments address critical legal issues that could affect enforceability',
                'amendments': [obj['suggested_amendment'] for obj in high_risk]
            })
        
        if medium_risk:
            prioritized.append({
                'priority': 'Important',
                'description': 'These amendments address important legal issues that could cause disputes',
                'amendments': [obj['suggested_amendment'] for obj in medium_risk]
            })
        
        if low_risk:
            prioritized.append({
                'priority': 'Recommended',
                'description': 'These amendments improve clarity and reduce potential for misinterpretation',
                'amendments': [obj['suggested_amendment'] for obj in low_risk]
            })
        
        return prioritized
    
    def _extract_jurisdiction(self, text: str) -> str:
        """Extract jurisdiction from document text"""
        # Look for explicit jurisdiction mention
        jurisdiction_patterns = [
            r'(?i)jurisdiction\s+of\s+([A-Za-z\s]+)',
            r'(?i)([A-Za-z]+)\s+jurisdiction',
            r'(?i)state\s+of\s+([A-Za-z\s]+)',
            r'(?i)laws?\s+of\s+([A-Za-z\s]+)'
        ]
        
        for pattern in jurisdiction_patterns:
            match = re.search(pattern, text)
            if match and match.group(1):
                jurisdiction = match.group(1).strip()
                # Check if it's a known state
                for state in ["Karnataka", "Maharashtra", "Tamil Nadu", "Delhi", "Uttar Pradesh"]:
                    if state.lower() in jurisdiction.lower():
                        return state
        
        # Look for cities and map to states
        city_state_mapping = {
            'bangalore': 'Karnataka',
            'bengaluru': 'Karnataka',
            'mysore': 'Karnataka',
            'mumbai': 'Maharashtra',
            'pune': 'Maharashtra',
            'nagpur': 'Maharashtra',
            'chennai': 'Tamil Nadu',
            'coimbatore': 'Tamil Nadu',
            'delhi': 'Delhi',
            'new delhi': 'Delhi',
            'lucknow': 'Uttar Pradesh',
            'noida': 'Uttar Pradesh'
        }
        
        for city, state in city_state_mapping.items():
            if re.search(r'\b' + city + r'\b', text, re.IGNORECASE):
                return state
        
        # Default jurisdiction
        return "Karnataka"
    
    def _extract_property_value(self, text: str) -> float:
        """Extract property value from document"""
        value_patterns = [
            r'(?i)consideration\s+of\s+(?:Rs\.?|INR)[\.|\s]*([\d,]+)',
            r'(?i)value\s+of\s+(?:Rs\.?|INR)[\.|\s]*([\d,]+)',
            r'(?i)property\s+valued\s+at\s+(?:Rs\.?|INR)[\.|\s]*([\d,]+)',
            r'(?i)market\s+value\s+of\s+(?:Rs\.?|INR)[\.|\s]*([\d,]+)'
        ]
        
        for pattern in value_patterns:
            match = re.search(pattern, text)
            if match and match.group(1):
                try:
                    return float(match.group(1).replace(',', ''))
                except ValueError:
                    continue
        
        # If no match found, check for numbers near property-related words
        property_contexts = re.finditer(r'(?i)property|land|house|flat|apartment', text)
        
        for match in property_contexts:
            # Look for amounts within 100 characters of property mention
            start = max(0, match.start() - 100)
            end = min(len(text), match.end() + 100)
            context = text[start:end]
            
            # Find any amount in the context
            amount_match = re.search(r'(?:Rs\.?|INR)[\.|\s]*([\d,]+)', context)
            if amount_match and amount_match.group(1):
                try:
                    return float(amount_match.group(1).replace(',', ''))
                except ValueError:
                    continue
        
        # Default value if nothing found
        return 1000000.0  # 10 lakhs default
    
    def _get_stamp_duty_structure(self, jurisdiction: str, doc_type: str) -> Dict:
        """Get stamp duty structure for jurisdiction and document type"""
        # Basic stamp duty rates by jurisdiction and document type
        stamp_duty_rates = {
            'Karnataka': {
                'Sale Deed': {
                    'rate': 5.6,
                    'calculation_method': 'percentage',
                    'fixed_amount': 0,
                    'payment_procedure': 'Pay online through Karnataka IGRS e-Stamps portal',
                    'legal_references': 'Karnataka Stamp Act, Article 20'
                },
                'Rental Agreement': {
                    'rate': 0.5,
                    'calculation_method': 'percentage',
                    'fixed_amount': 0,
                    'payment_procedure': 'Pay online through Karnataka IGRS e-Stamps portal',
                    'legal_references': 'Karnataka Stamp Act, Article 33'
                }
            },
            'Maharashtra': {
                'Sale Deed': {
                    'rate': 6.0,
                    'calculation_method': 'percentage',
                    'fixed_amount': 0,
                    'payment_procedure': 'Pay online through GRAS or e-SBTR',
                    'legal_references': 'Maharashtra Stamp Act, Schedule I, Article 25'
                },
                'Rental Agreement': {
                    'rate': 0.25,
                    'calculation_method': 'percentage',
                    'fixed_amount': 0,
                    'payment_procedure': 'Pay online through GRAS or e-SBTR',
                    'legal_references': 'Maharashtra Stamp Act, Schedule I, Article 36'
                }
            }
        }
        
        # Get structure for specified jurisdiction and document type
        jurisdiction_rates = stamp_duty_rates.get(jurisdiction, stamp_duty_rates['Karnataka'])
        return jurisdiction_rates.get(doc_type, jurisdiction_rates.get('Sale Deed'))
    
    def _calculate_registration_fee(self, property_value: float, jurisdiction: str) -> float:
        """Calculate registration fee based on property value and jurisdiction"""
        registration_fees = {
            'Karnataka': {
                'rate': 1.0,
                'cap': 15000.0
            },
            'Maharashtra': {
                'rate': 1.0,
                'cap': 30000.0
            },
            'Tamil Nadu': {
                'rate': 1.0,
                'cap': 20000.0
            }
        }
        
        fee_structure = registration_fees.get(jurisdiction, registration_fees['Karnataka'])
        fee = property_value * (fee_structure['rate'] / 100)
        
        # Apply cap if applicable
        if fee_structure['cap'] > 0:
            fee = min(fee, fee_structure['cap'])
        
        return fee
    
    def _find_applicable_exemptions(self, text: str, jurisdiction: str) -> List[Dict]:
        """Find applicable stamp duty exemptions"""
        exemption_patterns = {
            'Karnataka': [
                {'pattern': r'(?i)first\s+time\s+buyer', 'type': 'First-time buyer', 'reduction': 0.5},
                {'pattern': r'(?i)scheduled\s+caste|scheduled\s+tribe|sc/st', 'type': 'SC/ST reservation', 'reduction': 0.5},
                {'pattern': r'(?i)woman\s+owner|female\s+buyer', 'type': 'Woman buyer', 'reduction': 1.0}
            ],
            'Maharashtra': [
                {'pattern': r'(?i)woman\s+owner|female\s+buyer', 'type': 'Woman buyer', 'reduction': 1.0},
                {'pattern': r'(?i)slum\s+rehabilitation', 'type': 'Slum rehabilitation', 'reduction': 1.0},
                {'pattern': r'(?i)agriculturist', 'type': 'Agriculturist', 'reduction': 0.5}
            ]
        }
        
        applicable_exemptions = []
        jurisdiction_exemptions = exemption_patterns.get(jurisdiction, [])
        
        for exemption in jurisdiction_exemptions:
            if re.search(exemption['pattern'], text):
                applicable_exemptions.append({
                    'type': exemption['type'],
                    'reduction_percentage': exemption['reduction'],
                    'legal_basis': f"{jurisdiction} Stamp Duty Exemption Notification"
                })
        
        return applicable_exemptions
    
    def _apply_exemptions(self, stamp_duty: float, exemptions: List[Dict]) -> float:
        """Apply exemptions to reduce stamp duty"""
        reduced_duty = stamp_duty
        
        for exemption in exemptions:
            reduction = stamp_duty * (exemption['reduction_percentage'] / 100)
            reduced_duty -= reduction
        
        # Ensure duty doesn't go below zero
        return max(0, reduced_duty)
    
    def _find_clause_start(self, text: str, position: int) -> int:
        """Find the start of a clause containing the given position"""
        # Look backward for paragraph start or numbered clause
        clause_start_pattern = r'[\n\r][\.|\d+\.\s+|\([a-z]\)\s+|[A-Z][A-Z\s]+:|\d+\s*\.\s*]'
        
        # Get text prior to position
        prior_text = text[:position]
        matches = list(re.finditer(clause_start_pattern, prior_text))
        
        if matches:
            # Return position after the last clause start found
            last_match = matches[-1]
            return last_match.end()
        
        # If no clear clause start found, look for a sentence start
        sentence_matches = list(re.finditer(r'[\.\?\!]\s+[A-Z]', prior_text))
        if sentence_matches:
            last_sentence = sentence_matches[-1]
            return last_sentence.end() - 1
        
        # Default to a reasonable amount of prior context
        return max(0, position - 200)
    
    def _find_clause_end(self, text: str, position: int) -> int:
        """Find the end of a clause containing the given position"""
        # Look forward for paragraph end or next clause start
        clause_end_pattern = r'[\n\r][\n\r]|\n\d+\.\s+|\n\([a-z]\)\s+|\n[A-Z][A-Z\s]+:'
        
        # Get text after position
        after_text = text[position:]
        match = re.search(clause_end_pattern, after_text)
        
        if match:
            # Return position of the clause end
            return position + match.start()
        
        # If no clear clause end found, look for multiple sentence endings
        sentence_endings = list(re.finditer(r'[\.\?\!]\s+', after_text))
        if len(sentence_endings) >= 2:
            # Return after second sentence
            return position + sentence_endings[1].end()
        
        # Default to a reasonable amount of following context
        return min(len(text), position + 400)
    
    def _extract_covered_events(self, force_majeure_text: str) -> List[str]:
        """Extract events covered by force majeure clause"""
        # Common force majeure events
        event_patterns = {
            'natural_disaster': r'(?i)earthquake|flood|tsunami|hurricane|typhoon|cyclone|storm|natural\s+disaster',
            'fire': r'(?i)fire|conflagration',
            'war': r'(?i)war|hostility|military\s+operation|invasion',
            'civil_unrest': r'(?i)riot|civil\s+commotion|civil\s+disturbance|protest|strike|lockout',
            'government_action': r'(?i)government\s+action|change\s+in\s+law|regulatory\s+change|expropriation',
            'epidemic': r'(?i)epidemic|pandemic|disease|outbreak|quarantine',
            'utility_failure': r'(?i)utility\s+failure|power\s+outage|water\s+shortage',
            'transportation_disruption': r'(?i)transportation\s+disruption|shipping\s+delay'
        }
        
        covered_events = []
        for event_type, pattern in event_patterns.items():
            if re.search(pattern, force_majeure_text, re.IGNORECASE):
                covered_events.append(event_type)
        
        return covered_events
    
    def _extract_notification_requirements(self, force_majeure_text: str) -> Dict:
        """Extract notification requirements from force majeure clause"""
        # Look for notification patterns
        notification_patterns = {
            'requirement': r'(?i)notify|notice|inform|communication',
            'timeframe': r'(?i)within\s+(\d+)\s+(day|week|month)s?|immediately|promptly'
        }
        
        notification_req = {'required': False, 'timeframe': 'Not specified'}
        
        # Check if notification is required
        req_match = re.search(notification_patterns['requirement'], force_majeure_text)
        notification_req['required'] = bool(req_match)
        
        # Extract timeframe if specified
        time_match = re.search(notification_patterns['timeframe'], force_majeure_text)
        if time_match:
            notification_req['timeframe'] = time_match.group(0)
        
        return notification_req
    
    def _extract_duration_provisions(self, force_majeure_text: str) -> Dict:
        """Extract duration provisions from force majeure clause"""
        # Look for duration-related patterns
        duration_patterns = {
            'time_limit': r'(?i)for\s+a\s+period\s+of\s+(\d+)\s+(day|week|month)s?|not\s+to\s+exceed\s+(\d+)\s+(day|week|month)s?',
            'continuation': r'(?i)continue\s+in\s+effect|remains?\s+in\s+force|persist',
            'termination_right': r'(?i)right\s+to\s+terminate|may\s+cancel|option\s+to\s+(?:cancel|terminate)|if\s+continues?\s+(?:for|beyond)\s+(\d+)\s+(day|week|month)s?'
        }
        
        duration_provisions = {
            'has_time_limit': False,
            'time_limit': 'Not specified',
            'has_termination_right': False,
            'termination_trigger': 'Not specified'
        }
        
        # Check for time limit
        time_match = re.search(duration_patterns['time_limit'], force_majeure_text)
        if time_match:
            duration_provisions['has_time_limit'] = True
            duration_provisions['time_limit'] = time_match.group(0)
        
        # Check for termination right
        term_match = re.search(duration_patterns['termination_right'], force_majeure_text)
        if term_match:
            duration_provisions['has_termination_right'] = True
            duration_provisions['termination_trigger'] = term_match.group(0)
        
        return duration_provisions
    
    def _extract_termination_rights(self, force_majeure_text: str) -> Dict:
        """Extract termination rights from force majeure clause"""
        # Look for termination rights patterns
        termination_patterns = {
            'unilateral': r'(?i)(either|any)\s+party\s+may\s+terminate',
            'affected_party': r'(?i)affected\s+party\s+may\s+terminate',
            'notice_required': r'(?i)notice\s+of\s+termination|written\s+notice',
            'consequences': r'(?i)consequences\s+of\s+termination|effect\s+of\s+termination|upon\s+termination'
        }
        
        termination_rights = {
            'available': False,
            'unilateral': False,
            'notice_required': False,
            'consequences_specified': False
        }
        
        # Check if termination is available
        if re.search(r'(?i)terminate|cancel|rescind|end', force_majeure_text):
            termination_rights['available'] = True
            
            # Check for unilateral right
            if re.search(termination_patterns['unilateral'], force_majeure_text):
                termination_rights['unilateral'] = True
            
            # Check if notice is required
            if re.search(termination_patterns['notice_required'], force_majeure_text):
                termination_rights['notice_required'] = True
            
            # Check if consequences are specified
            if re.search(termination_patterns['consequences'], force_majeure_text):
                termination_rights['consequences_specified'] = True
        
        return termination_rights
    
    def _get_standard_force_majeure_events(self) -> List[str]:
        """Get standard events that should be covered in a comprehensive force majeure clause"""
        return [
            'natural_disaster',
            'fire',
            'war',
            'civil_unrest',
            'government_action',
            'epidemic',
            'utility_failure',
            'transportation_disruption',
            'terrorism',
            'nuclear_incident',
            'labor_dispute'
        ]

    def _generate_force_majeure_improvements(self, missing_events: List[str]) -> List[str]:
        """Generate improvement suggestions for force majeure clause"""
        improvements = []
        
        # Suggest adding missing events
        if missing_events:
            events_desc = {
                    'natural_disaster': 'earthquakes, floods, tsunamis, hurricanes, and other natural disasters',
                    'fire': 'fires and conflagrations',
                    'war': 'war, hostilities, invasion, or military operations',
                    'civil_unrest': 'riots, civil commotion, civil disturbance, protests, strikes, or lockouts',
                    'government_action': 'government actions, changes in laws, regulatory changes, or expropriation',
                    'epidemic': 'epidemics, pandemics, disease outbreaks, or quarantine restrictions',
                    'utility_failure': 'utility failures, power outages, or water shortages',
                    'transportation_disruption': 'transportation disruptions or shipping delays',
                    'terrorism': 'acts of terrorism, sabotage, or threats thereof',
                    'nuclear_incident': 'nuclear incidents, radiation, or radioactive contamination',
                    'labor_dispute': 'labor disputes, strikes, or industrial action'
                }
                
            event_suggestions = [events_desc.get(event, event) for event in missing_events]
            improvements.append(f"Add missing events: {', '.join(event_suggestions)}")
        
        # Standard structure improvements
        standard_improvements = [
            "Include specific notification requirements with clear timeframes",
            "Specify the duration for which force majeure can be claimed before termination rights trigger",
            "Detail the obligations of parties during the force majeure period",
            "Specify consequences of termination due to prolonged force majeure"
        ]
        
        improvements.extend(standard_improvements)
        
        return improvements

    def _identify_deadline_type(self, context: str) -> str:
        """Identify the type of deadline from context"""
        deadline_type_patterns = {
            'payment': r'(?i)pay|payment|amount|consideration|money|fee|charge',
            'possession': r'(?i)possession|handover|deliver|vacate|occupy',
            'registration': r'(?i)registration|register|registrar|sub-registrar',
            'approval': r'(?i)approval|permission|consent|authorize',
            'notice': r'(?i)notice|notify|inform|communication',
            'document': r'(?i)document|submit|furnish|provide'
        }
        
        for deadline_type, pattern in deadline_type_patterns.items():
            if re.search(pattern, context):
                return deadline_type
        
        return 'other'
    
    def _extract_responsible_parties(self, context: str) -> List[str]:
        """Extract parties responsible for meeting the deadline"""
        party_patterns = {
            'seller': r'(?i)seller|vendor|transferor|first\s+party',
            'buyer': r'(?i)buyer|purchaser|transferee|second\s+party',
            'landlord': r'(?i)landlord|lessor|owner',
            'tenant': r'(?i)tenant|lessee|occupant',
            'mortgagor': r'(?i)mortgagor|borrower',
            'mortgagee': r'(?i)mortgagee|lender'
        }
        
        responsible_parties = []
        for party, pattern in party_patterns.items():
            if re.search(pattern, context):
                responsible_parties.append(party)
        
        if not responsible_parties:
            # Check for pronouns indicating parties
            if re.search(r'\b(he|she|they|party)\b', context, re.IGNORECASE):
                # Look at preceding context to determine party
                if re.search(r'(?i)seller|vendor|first\s+party', context):
                    responsible_parties.append('seller')
                elif re.search(r'(?i)buyer|purchaser|second\s+party', context):
                    responsible_parties.append('buyer')
                else:
                    responsible_parties.append('unspecified party')
        
        return responsible_parties or ['both parties']
    
    def _identify_legal_consequences(self, context: str, deadline_type: str) -> Dict:
        """Identify legal consequences of missing the deadline"""
        # Default consequences by deadline type
        default_consequences = {
            'payment': {
                'consequence': 'Interest payment and potential default',
                'severity': 'High',
                'legal_basis': 'Indian Contract Act, Section 55'
            },
            'possession': {
                'consequence': 'Breach of contract, potential for damages',
                'severity': 'High',
                'legal_basis': 'Transfer of Property Act, Section 55'
            },
            'registration': {
                'consequence': 'Document may become inadmissible as evidence',
                'severity': 'High',
                'legal_basis': 'Registration Act, Section 49'
            },
            'approval': {
                'consequence': 'Transaction may not be valid or enforceable',
                'severity': 'Medium',
                'legal_basis': 'Indian Contract Act, Section 10'
            },
            'notice': {
                'consequence': 'Rights dependent on notice may not be enforceable',
                'severity': 'Medium',
                'legal_basis': 'Indian Contract Act, Section 4'
            },
            'document': {
                'consequence': 'Procedural delay or breach of contract',
                'severity': 'Medium',
                'legal_basis': 'Indian Evidence Act, Section 91'
            },
            'other': {
                'consequence': 'Potential breach of contract',
                'severity': 'Medium',
                'legal_basis': 'Indian Contract Act, Section 37'
            }
        }
        
        # Check for specific consequences mentioned in context
        specific_consequence = None
        
        consequence_patterns = {
            r'(?i)interest\s+at\s+(\d+)%': {
                'template': 'Interest payment at {0}%',
                'severity': 'Medium',
                'legal_basis': 'Indian Contract Act, Section 74'
            },
            r'(?i)terminate|cancel|rescind': {
                'template': 'Agreement termination or cancellation',
                'severity': 'High',
                'legal_basis': 'Indian Contract Act, Section 39'
            },
            r'(?i)forfeit|forfeiture': {
                'template': 'Forfeiture of deposit or advance',
                'severity': 'High',
                'legal_basis': 'Indian Contract Act, Section 74'
            },
            r'(?i)penalt(?:y|ies)|fine': {
                'template': 'Financial penalty or fine',
                'severity': 'Medium',
                'legal_basis': 'Indian Contract Act, Section 74'
            },
            r'(?i)damages': {
                'template': 'Liability for damages',
                'severity': 'Medium',
                'legal_basis': 'Indian Contract Act, Section 73'
            }
        }
        
        for pattern, details in consequence_patterns.items():
            match = re.search(pattern, context)
            if match:
                if len(match.groups()) > 0:
                    specific_consequence = {
                        'consequence': details['template'].format(match.group(1)),
                        'severity': details['severity'],
                        'legal_basis': details['legal_basis']
                    }
                else:
                    specific_consequence = {
                        'consequence': details['template'],
                        'severity': details['severity'],
                        'legal_basis': details['legal_basis']
                    }
                break
        
        # Return specific consequence if found, otherwise default
        return specific_consequence or default_consequences[deadline_type]
    
    def _assess_deadline_criticality(self, deadline_type: str) -> str:
        """Assess criticality of deadline based on type"""
        critical_deadlines = ['payment', 'possession', 'registration']
        important_deadlines = ['approval', 'notice']
        
        if deadline_type in critical_deadlines:
            return 'High'
        elif deadline_type in important_deadlines:
            return 'Medium'
        else:
            return 'Low'
    
    def _generate_timeline_visualization(self, timelines: List[Dict]) -> str:
        """Generate a textual timeline visualization"""
        if not timelines:
            return "No timeline requirements found"
        
        # Sort by criticality
        sorted_timelines = sorted(
            timelines, 
            key=lambda x: (x['criticality'] == 'High', x['criticality'] == 'Medium', x['criticality'] == 'Low'),
            reverse=True
        )
        
        visualization = "TIMELINE OF LEGAL DEADLINES:\n\n"
        
        for i, timeline in enumerate(sorted_timelines, 1):
            criticality_marker = "🔴" if timeline['criticality'] == 'High' else "🟠" if timeline['criticality'] == 'Medium' else "🟢"
            visualization += f"{criticality_marker} {i}. {timeline['deadline_text']}\n"
            visualization += f"   Type: {timeline['deadline_type'].capitalize()}\n"
            visualization += f"   Responsible: {', '.join(timeline['parties_responsible'])}\n"
            visualization += f"   Consequence: {timeline['legal_consequence']['consequence']}\n\n"
        
        return visualization
    
    def _generate_calendar_entries(self, timelines: List[Dict]) -> List[Dict]:
        """Generate structured calendar entries for deadlines"""
        calendar_entries = []
        
        for timeline in timelines:
            entry = {
                'title': f"{timeline['deadline_type'].capitalize()} Deadline",
                'description': timeline['deadline_text'],
                'responsible': timeline['parties_responsible'],
                'criticality': timeline['criticality'],
                'consequences': timeline['legal_consequence']['consequence']
            }
            calendar_entries.append(entry)
        
        return calendar_entries

    def analyze_language_complexity(self, text: str) -> Dict:
        """
        Analyze document text using established readability metrics and 
        legal-specific complexity measures.
        
        Args:
            text: The document text to analyze
            
        Returns:
            Dict: Comprehensive language complexity analysis
        """
        try:
            # Create analyzer instance
            analyzer = LegalLanguageComplexityAnalyzer()
            
            # Perform analysis
            analysis = analyzer.analyze_text_complexity(text)
            
            return analysis
        except Exception as e:
            self.logger.error(f"Language complexity analysis error: {str(e)}")
            return {
                "error": f"Language complexity analysis failed: {str(e)}",
                "readability_metrics": {},
                "jargon_analysis": {},
                "simplification": {}
            }
