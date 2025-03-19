import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from typing import Dict, List, Any, Tuple
import re
import json
import base64
import io
from datetime import datetime
import logging

class WillScenarioSimulator:
    """
    Will Scenario Simulator for analyzing inheritance scenarios under Indian Law.
    Simulates asset distribution, identifies potential challenges, and provides 
    recommendations for will creation and execution.
    """
    
    def __init__(self):
        """Initialize the Will Scenario Simulator with legal data and references"""
        self.logger = logging.getLogger(__name__)
        
        # Load Indian succession laws and rules
        self.succession_laws = {
            'Hindu': {
                'name': 'Hindu Succession Act, 1956 (amended 2005)',
                'intestate_rules': {
                    'male': "Equal division among spouse, sons, and daughters. Mother and father receive share if no children.",
                    'female': "Equal division among spouse and children. Parents receive if no children and spouse."
                },
                'eligible_challengers': ['spouse', 'children', 'parents', 'siblings'],
                'mandatory_share': False,
                'notes': "The 2005 amendment gave daughters equal rights in ancestral property."
            },
            'Muslim': {
                'name': 'Muslim Personal Law (Shariat) Application Act, 1937',
                'intestate_rules': {
                    'male': "Distribution as per Sharia Law. Son gets twice the share of daughter. Spouse gets 1/4 if children, 1/2 if no children.",
                    'female': "Distribution as per Sharia Law. Different shares for different relations based on Sunni/Shia denomination."
                },
                'eligible_challengers': ['spouse', 'children', 'parents', 'siblings'],
                'mandatory_share': True,
                'notes': "Muslim law distinguishes between Sunni (Hanafi) and Shia laws of inheritance."
            },
            'Christian': {
                'name': 'Indian Succession Act, 1925',
                'intestate_rules': {
                    'male': "One-third to widow, two-thirds to children equally. If no children, half to widow, half to kin.",
                    'female': "One-third to husband, two-thirds to children equally. If no children, half to husband, half to kin."
                },
                'eligible_challengers': ['spouse', 'children', 'parents', 'siblings'],
                'mandatory_share': False,
                'notes': "Christians in Kerala may be governed by the Travancore and Cochin Succession Acts."
            },
            'Parsi': {
                'name': 'Indian Succession Act, 1925 (Parsi Section)',
                'intestate_rules': {
                    'male': "Equal division among spouse and children. Parents receive if no children.",
                    'female': "Equal division among spouse and children. Parents receive if no children."
                },
                'eligible_challengers': ['spouse', 'children', 'parents'],
                'mandatory_share': False,
                'notes': "Parsis follow more equal gender distribution compared to some other succession laws."
            },
            'JFSA': {
                'name': 'Jewish Law (follower of Judaism)',
                'intestate_rules': {
                    'male': "Eldest son receives double share, other sons divide remainder equally. Daughters inherit if no sons.",
                    'female': "Follows traditional Jewish inheritance principles."
                },
                'eligible_challengers': ['children', 'other heirs'],
                'mandatory_share': True,
                'notes': "Traditional Jewish law gives preference to male heirs, but modern interpretations may vary."
            },
            'Civil': {
                'name': 'Indian Succession Act, 1925 (Part V)',
                'intestate_rules': {
                    'male': "One-third to spouse, rest equally among children. If no children, half to spouse, half to parents.",
                    'female': "One-third to spouse, rest equally among children. If no children, half to spouse, half to parents."
                },
                'eligible_challengers': ['spouse', 'children', 'parents', 'siblings'],
                'mandatory_share': False,
                'notes': "Applies to those married under Special Marriage Act or not covered by religious personal laws."
            }
        }
        
        # Common will challenge reasons
        self.challenge_reasons = {
            'testamentary_capacity': {
                'name': 'Lack of Testamentary Capacity',
                'description': 'The testator lacked mental capacity or understanding when creating the will.',
                'evidence': ['Medical records', 'Witness testimony', 'Abnormal behavior evidence'],
                'legal_refs': ['Section 59 of Indian Succession Act'],
                'risk_factors': ['Age above 75', 'History of mental illness', 'Medication affecting cognition', 'Recent major surgery']
            },
            'undue_influence': {
                'name': 'Undue Influence or Coercion',
                'description': 'The testator was pressured or manipulated by someone to make the will.',
                'evidence': ['Power dynamics', 'Isolation from family', 'Sudden changes to will'],
                'legal_refs': ['Section 61 of Indian Succession Act', 'Sections 15-17 of Contract Act'],
                'risk_factors': ['Caregiver as major beneficiary', 'Recent estrangement from family', 'Deathbed will']
            },
            'fraud': {
                'name': 'Fraud or Forgery',
                'description': 'The will or signature was forged or obtained through deception.',
                'evidence': ['Handwriting analysis', 'Documentary inconsistencies', 'Witness testimony'],
                'legal_refs': ['Section 63 of Indian Succession Act'],
                'risk_factors': ['Lack of witnesses', 'Unusual signatures', 'Informal document preparation']
            },
            'improper_execution': {
                'name': 'Improper Execution',
                'description': 'The will was not properly signed, witnessed, or executed as per law.',
                'evidence': ['Will document analysis', 'Witness testimony'],
                'legal_refs': ['Section 63 of Indian Succession Act'],
                'risk_factors': ['Self-made will', 'Lack of attestation', 'Alterations without initials']
            },
            'newer_will': {
                'name': 'Existence of a Newer Will',
                'description': 'A more recent valid will exists, revoking the earlier one.',
                'evidence': ['Later dated will', 'Revocation documents'],
                'legal_refs': ['Section 70 of Indian Succession Act'],
                'risk_factors': ['Multiple wills created', 'Oral mentions of newer documents', 'Multiple legal advisors used']
            },
            'exclusion': {
                'name': 'Exclusion of Legal Heirs',
                'description': 'Legal heirs were excluded without justification.',
                'evidence': ['Family records', 'Previous statements'],
                'legal_refs': ['Hindu Succession Act', 'Personal law provisions'],
                'risk_factors': ['Disinheriting children', 'Highly unequal distributions', 'Recent family disputes']
            },
            'maintenance': {
                'name': 'Maintenance Rights Violation',
                'description': 'Will doesnt provide for legal maintenance obligations.',
                'evidence': ['Dependency proof', 'Financial records'],
                'legal_refs': ['Section 22 of Hindu Adoptions and Maintenance Act', 'Maintenance and Welfare of Parents and Senior Citizens Act'],
                'risk_factors': ['Dependent spouse excluded', 'Minor children with inadequate provision', 'Dependent parents excluded']
            }
        }
        
        # Asset types and their characteristics
        self.asset_types = {
            'immovable_property': {
                'name': 'Immovable Property',
                'legal_considerations': 'Requires proper registration of transfer. Consider location-specific tax implications.',
                'dispute_likelihood': 'High',
                'transfer_complexity': 'High',
                'documentation_needed': ['Property deed', 'Tax records', 'Clear title documents']
            },
            'financial_investments': {
                'name': 'Financial Investments',
                'legal_considerations': 'Nominee arrangements may conflict with will provisions. Some assets may have separate succession rules.',
                'dispute_likelihood': 'Medium',
                'transfer_complexity': 'Medium',
                'documentation_needed': ['Investment certificates', 'Account details', 'Nomination forms']
            },
            'business_assets': {
                'name': 'Business Assets/Interests',
                'legal_considerations': 'Consider partnership agreements, shareholding patterns, and business succession planning.',
                'dispute_likelihood': 'Very High',
                'transfer_complexity': 'Very High',
                'documentation_needed': ['Business registration', 'Partnership deed', 'Share certificates', 'Business valuation']
            },
            'personal_possessions': {
                'name': 'Personal Possessions',
                'legal_considerations': 'Specific bequests should be clearly identified to avoid disputes.',
                'dispute_likelihood': 'Medium',
                'transfer_complexity': 'Low',
                'documentation_needed': ['Inventory list', 'Valuation certificates for valuables']
            },
            'intellectual_property': {
                'name': 'Intellectual Property',
                'legal_considerations': 'Consider the term of IP rights and specific transfer requirements.',
                'dispute_likelihood': 'Medium',
                'transfer_complexity': 'High',
                'documentation_needed': ['IP registration certificates', 'Licensing agreements']
            },
            'digital_assets': {
                'name': 'Digital Assets',
                'legal_considerations': 'Emerging area with unclear legal framework. Consider access and transfer mechanisms.',
                'dispute_likelihood': 'Low',
                'transfer_complexity': 'Medium',
                'documentation_needed': ['Inventory of digital assets', 'Access information', 'Platform policies']
            }
        }
    
    def simulate_scenario(self, personal_info: Dict, assets: List[Dict], 
                         beneficiaries: List[Dict], distribution: List[Dict]) -> Dict:
        """
        Simulate will scenario based on user inputs.
        
        Args:
            personal_info: Personal details of the testator
            assets: List of assets owned by the testator
            beneficiaries: List of family members and other beneficiaries
            distribution: How assets are to be distributed among beneficiaries
            
        Returns:
            Dict: Simulation results with analysis and visualizations
        """
        try:
            # Validate inputs
            validation_result = self._validate_inputs(personal_info, assets, beneficiaries, distribution)
            if validation_result.get('error'):
                return validation_result
            
            # Process inputs
            applicable_law = self._determine_applicable_law(personal_info)
            total_assets_value = sum(asset.get('value', 0) for asset in assets)
            
            # Analyze the distribution
            distribution_analysis = self._analyze_distribution(assets, beneficiaries, distribution, applicable_law)
            
            # Identify potential challenges
            potential_challenges = self._identify_potential_challenges(
                personal_info, assets, beneficiaries, distribution, applicable_law
            )
            
            # Generate distribution visualization data
            visualizations = self._generate_visualizations(assets, beneficiaries, distribution, distribution_analysis)
            
            # Generate documentation recommendations
            documentation_recs = self._generate_documentation_recommendations(
                personal_info, assets, distribution, potential_challenges
            )
            
            # Calculate intestate scenario for comparison
            intestate_scenario = self._calculate_intestate_scenario(
                personal_info, assets, beneficiaries, applicable_law
            )
            
            # Compile results
            results = {
                'applicable_law': applicable_law,
                'total_assets_value': total_assets_value,
                'distribution_analysis': distribution_analysis,
                'potential_challenges': potential_challenges,
                'visualizations': visualizations,
                'documentation_recommendations': documentation_recs,
                'intestate_scenario': intestate_scenario,
                'timestamp': datetime.now().isoformat()
            }
            
            return results
            
        except Exception as e:
            self.logger.error(f"Error in will scenario simulation: {str(e)}")
            return {
                'error': f"Simulation failed: {str(e)}",
                'status': 'error'
            }
    
    def _validate_inputs(self, personal_info: Dict, assets: List[Dict], 
                       beneficiaries: List[Dict], distribution: List[Dict]) -> Dict:
        """
        Validate input data for the simulation.
        
        Args:
            personal_info: Personal details of the testator
            assets: List of assets owned by the testator
            beneficiaries: List of family members and other beneficiaries
            distribution: How assets are to be distributed among beneficiaries
            
        Returns:
            Dict: Validation result with errors if any
        """
        try:
            # Check for required personal information
            required_personal_fields = ['name', 'age', 'religion', 'gender', 'marital_status']
            for field in required_personal_fields:
                if field not in personal_info or not personal_info[field]:
                    return {
                        'error': f"Missing required personal information: {field}",
                        'status': 'error'
                    }
            
            # Validate age
            try:
                age = int(personal_info['age'])
                if age < 18:
                    return {
                        'error': "Testator must be at least 18 years old",
                        'status': 'error'
                    }
            except ValueError:
                return {
                    'error': "Age must be a valid number",
                    'status': 'error'
                }
            
            # Check if assets list is provided
            if not assets:
                return {
                    'error': "No assets provided for distribution",
                    'status': 'error'
                }
            
            # Validate assets
            asset_ids = set()
            for asset in assets:
                if 'id' not in asset or 'type' not in asset or 'value' not in asset or 'description' not in asset:
                    return {
                        'error': "Each asset must have id, type, value, and description",
                        'status': 'error'
                    }
                
                # Check for duplicate asset IDs
                if asset['id'] in asset_ids:
                    return {
                        'error': f"Duplicate asset ID: {asset['id']}",
                        'status': 'error'
                    }
                asset_ids.add(asset['id'])
                
                # Validate asset value
                try:
                    value = float(asset['value'])
                    if value <= 0:
                        return {
                            'error': f"Asset value must be positive: {asset['description']}",
                            'status': 'error'
                        }
                except ValueError:
                    return {
                        'error': f"Invalid asset value for: {asset['description']}",
                        'status': 'error'
                    }
            
            # Check if beneficiaries list is provided
            if not beneficiaries:
                return {
                    'error': "No beneficiaries provided",
                    'status': 'error'
                }
            
            # Validate beneficiaries
            beneficiary_ids = set()
            for beneficiary in beneficiaries:
                if 'id' not in beneficiary or 'name' not in beneficiary or 'relation' not in beneficiary:
                    return {
                        'error': "Each beneficiary must have id, name, and relation",
                        'status': 'error'
                    }
                
                # Check for duplicate beneficiary IDs
                if beneficiary['id'] in beneficiary_ids:
                    return {
                        'error': f"Duplicate beneficiary ID: {beneficiary['id']}",
                        'status': 'error'
                    }
                beneficiary_ids.add(beneficiary['id'])
            
            # Check if distribution list is provided
            if not distribution:
                return {
                    'error': "No distribution plan provided",
                    'status': 'error'
                }
            
            # Validate distribution
            for item in distribution:
                if 'asset_id' not in item or 'beneficiary_id' not in item or 'percentage' not in item:
                    return {
                        'error': "Each distribution item must have asset_id, beneficiary_id, and percentage",
                        'status': 'error'
                    }
                
                # Check if asset exists
                if item['asset_id'] not in asset_ids:
                    return {
                        'error': f"Unknown asset ID in distribution: {item['asset_id']}",
                        'status': 'error'
                    }
                
                # Check if beneficiary exists
                if item['beneficiary_id'] not in beneficiary_ids:
                    return {
                        'error': f"Unknown beneficiary ID in distribution: {item['beneficiary_id']}",
                        'status': 'error'
                    }
                
                # Validate percentage
                try:
                    percentage = float(item['percentage'])
                    if percentage <= 0 or percentage > 100:
                        return {
                            'error': f"Percentage must be between 0 and 100: {percentage}",
                            'status': 'error'
                        }
                except ValueError:
                    return {
                        'error': f"Invalid percentage value: {item['percentage']}",
                        'status': 'error'
                    }
            
            # Check for asset distribution completeness
            asset_distribution = {}
            for item in distribution:
                asset_id = item['asset_id']
                percentage = float(item['percentage'])
                
                if asset_id not in asset_distribution:
                    asset_distribution[asset_id] = 0
                
                asset_distribution[asset_id] += percentage
            
            for asset_id, total_percentage in asset_distribution.items():
                # Allow small floating-point errors
                if abs(total_percentage - 100) > 0.01 and abs(total_percentage) > 0.01:
                    return {
                        'error': f"Asset {asset_id} distribution does not total 100% (currently: {total_percentage}%)",
                        'status': 'error'
                    }
            
            # All validations passed
            return {
                'status': 'valid'
            }
            
        except Exception as e:
            self.logger.error(f"Input validation error: {str(e)}")
            return {
                'error': f"Validation failed: {str(e)}",
                'status': 'error'
            }
    
    def _determine_applicable_law(self, personal_info: Dict) -> Dict:
        """
        Determine applicable succession law based on personal information.
        
        Args:
            personal_info: Personal details of the testator
            
        Returns:
            Dict: Information about the applicable law
        """
        try:
            religion = personal_info.get('religion', '').capitalize()
            gender = personal_info.get('gender', '').lower()
            domicile = personal_info.get('domicile', 'India')
            
            # Map religion to applicable law
            if religion in ['Hindu', 'Buddhist', 'Jain', 'Sikh']:
                law_key = 'Hindu'
            elif religion == 'Muslim':
                law_key = 'Muslim'
            elif religion == 'Christian':
                law_key = 'Christian'
            elif religion == 'Parsi':
                law_key = 'Parsi'
            elif religion == 'Jewish':
                law_key = 'JFSA'
            else:
                # Default to Civil law (Indian Succession Act)
                law_key = 'Civil'
            
            # Get law details
            law = self.succession_laws.get(law_key, self.succession_laws['Civil'])
            
            # Determine intestate rule based on gender
            intestate_rule = law['intestate_rules'].get(gender, law['intestate_rules'].get('male', ''))
            
            return {
                'name': law['name'],
                'key': law_key,
                'intestate_rule': intestate_rule,
                'eligible_challengers': law['eligible_challengers'],
                'mandatory_share': law['mandatory_share'],
                'notes': law['notes']
            }
            
        except Exception as e:
            self.logger.error(f"Error determining applicable law: {str(e)}")
            return {
                'name': 'Indian Succession Act (Default)',
                'key': 'Civil',
                'intestate_rule': 'Equal division among legal heirs',
                'eligible_challengers': ['spouse', 'children', 'parents', 'siblings'],
                'mandatory_share': False,
                'notes': 'Default law applied due to error or insufficient information'
            }
    
    def _analyze_distribution(self, assets: List[Dict], beneficiaries: List[Dict], 
                            distribution: List[Dict], applicable_law: Dict) -> Dict:
        """
        Analyze the asset distribution plan.
        
        Args:
            assets: List of assets owned by the testator
            beneficiaries: List of family members and other beneficiaries
            distribution: How assets are to be distributed among beneficiaries
            applicable_law: Applicable succession law
            
        Returns:
            Dict: Distribution analysis
        """
        try:
            # Create lookup maps for assets and beneficiaries
            asset_map = {asset['id']: asset for asset in assets}
            beneficiary_map = {ben['id']: ben for ben in beneficiaries}
            
            # Calculate total asset value
            total_value = sum(asset.get('value', 0) for asset in assets)
            
            # Calculate value and percentage for each beneficiary
            beneficiary_values = {}
            beneficiary_percentages = {}
            
            for item in distribution:
                asset_id = item['asset_id']
                ben_id = item['beneficiary_id']
                percentage = float(item['percentage'])
                
                asset = asset_map.get(asset_id, {})
                asset_value = asset.get('value', 0)
                item_value = asset_value * (percentage / 100)
                
                if ben_id not in beneficiary_values:
                    beneficiary_values[ben_id] = 0
                    beneficiary_percentages[ben_id] = 0
                
                beneficiary_values[ben_id] += item_value
                beneficiary_percentages[ben_id] += (item_value / total_value * 100) if total_value > 0 else 0
            
            # Analyze relationships and shares
            legal_heirs = []
            legal_heirs_shares = 0
            non_family_shares = 0
            immediate_family_shares = 0
            
            for ben_id, ben in beneficiary_map.items():
                relation = ben.get('relation', '').lower()
                
                # Mark legal heirs
                is_legal_heir = relation in ['spouse', 'son', 'daughter', 'father', 'mother']
                if is_legal_heir:
                    legal_heirs.append(ben_id)
                    legal_heirs_shares += beneficiary_percentages.get(ben_id, 0)
                
                # Track immediate family vs non-family shares
                if relation in ['spouse', 'son', 'daughter', 'father', 'mother', 'brother', 'sister']:
                    immediate_family_shares += beneficiary_percentages.get(ben_id, 0)
                elif relation not in ['grandson', 'granddaughter', 'uncle', 'aunt', 'nephew', 'niece', 'cousin']:
                    non_family_shares += beneficiary_percentages.get(ben_id, 0)
            
            # Analyze equitability among children
            children_ids = [ben_id for ben_id, ben in beneficiary_map.items() 
                          if ben.get('relation', '').lower() in ['son', 'daughter']]
            children_shares = [beneficiary_percentages.get(ben_id, 0) for ben_id in children_ids]
            
            child_inequality = 0
            if children_shares:
                mean_share = sum(children_shares) / len(children_shares)
                child_inequality = sum(abs(share - mean_share) for share in children_shares) / (2 * mean_share) if mean_share > 0 else 0
            
            # Build beneficiary details
            beneficiary_details = []
            for ben_id, percentage in beneficiary_percentages.items():
                ben = beneficiary_map.get(ben_id, {})
                beneficiary_details.append({
                    'id': ben_id,
                    'name': ben.get('name', ''),
                    'relation': ben.get('relation', ''),
                    'percentage': round(percentage, 2),
                    'value': round(beneficiary_values.get(ben_id, 0), 2),
                    'is_legal_heir': ben_id in legal_heirs
                })
            
            # Sort by value (descending)
            beneficiary_details.sort(key=lambda x: x['value'], reverse=True)
            
            # Track asset distribution by type
            asset_type_distribution = {}
            for asset in assets:
                asset_type = asset.get('type', '')
                if asset_type not in asset_type_distribution:
                    asset_type_distribution[asset_type] = {
                        'total_value': 0,
                        'count': 0,
                        'beneficiaries': set()
                    }
                
                asset_type_distribution[asset_type]['total_value'] += asset.get('value', 0)
                asset_type_distribution[asset_type]['count'] += 1
            
            # Add beneficiaries for each asset type
            for item in distribution:
                asset_id = item['asset_id']
                ben_id = item['beneficiary_id']
                asset = asset_map.get(asset_id, {})
                asset_type = asset.get('type', '')
                
                if asset_type in asset_type_distribution:
                    asset_type_distribution[asset_type]['beneficiaries'].add(ben_id)
            
            # Convert to list and calculate percentages
            asset_type_summary = []
            for asset_type, data in asset_type_distribution.items():
                asset_type_info = self.asset_types.get(asset_type, {'name': asset_type, 'dispute_likelihood': 'Medium'})
                asset_type_summary.append({
                    'type': asset_type,
                    'name': asset_type_info.get('name', asset_type),
                    'dispute_likelihood': asset_type_info.get('dispute_likelihood', 'Medium'),
                    'value': data['total_value'],
                    'percentage': (data['total_value'] / total_value * 100) if total_value > 0 else 0,
                    'count': data['count'],
                    'beneficiary_count': len(data['beneficiaries'])
                })
            
            # Sort by value (descending)
            asset_type_summary.sort(key=lambda x: x['value'], reverse=True)
            
            return {
                'beneficiary_details': beneficiary_details,
                'asset_types': asset_type_summary,
                'legal_heirs_share': round(legal_heirs_shares, 2),
                'non_family_share': round(non_family_shares, 2),
                'immediate_family_share': round(immediate_family_shares, 2),
                'child_inequality_index': round(child_inequality, 2),
                'unique_asset_types': len(asset_type_summary),
                'total_assets_count': len(assets),
                'total_assets_value': total_value
            }
            
        except Exception as e:
            self.logger.error(f"Distribution analysis error: {str(e)}")
            return {
                'error': f"Distribution analysis failed: {str(e)}",
                'beneficiary_details': []
            }
    
    def _identify_potential_challenges(self, personal_info: Dict, assets: List[Dict], 
                                      beneficiaries: List[Dict], distribution: List[Dict],
                                      applicable_law: Dict) -> Dict:
        """
        Identify potential challenges to the will.
        
        Args:
            personal_info: Personal details of the testator
            assets: List of assets owned by the testator
            beneficiaries: List of family members and other beneficiaries
            distribution: How assets are to be distributed among beneficiaries
            applicable_law: Applicable succession law
            
        Returns:
            Dict: Potential challenges analysis
        """
        try:
            # Create beneficiary and distribution maps
            ben_map = {b['id']: b for b in beneficiaries}
            
            # Calculate total value received by each beneficiary
            ben_values = {}
            asset_map = {a['id']: a for a in assets}
            total_value = sum(a.get('value', 0) for a in assets)
            
            for item in distribution:
                asset_id = item['asset_id']
                ben_id = item['beneficiary_id']
                percentage = float(item['percentage'])
                
                asset = asset_map.get(asset_id, {})
                asset_value = asset.get('value', 0)
                item_value = asset_value * (percentage / 100)
                
                if ben_id not in ben_values:
                    ben_values[ben_id] = 0
                
                ben_values[ben_id] += item_value
            
            # Calculate percentages of total estate
            ben_percentages = {ben_id: (value / total_value * 100) if total_value > 0 else 0 
                             for ben_id, value in ben_values.items()}
            
            # Find legal heirs and their treatment
            legal_heirs = []
            excluded_legal_heirs = []
            
            # Track family members by relation
            relation_map = {
                'spouse': [],
                'children': [],
                'parents': [],
                'siblings': []
            }
            
            for ben_id, ben in ben_map.items():
                relation = ben.get('relation', '').lower()
                
                # Group by relation category
                if relation == 'spouse':
                    relation_map['spouse'].append(ben_id)
                    legal_heirs.append(ben_id)
                elif relation in ['son', 'daughter']:
                    relation_map['children'].append(ben_id)
                    legal_heirs.append(ben_id)
                elif relation in ['father', 'mother']:
                    relation_map['parents'].append(ben_id)
                    legal_heirs.append(ben_id)
                elif relation in ['brother', 'sister']:
                    relation_map['siblings'].append(ben_id)
                
                # Check if this legal heir is excluded from the will
                if ben_id in legal_heirs and ben_id not in ben_values:
                    excluded_legal_heirs.append(ben_id)
            
            # Check for additional legal heirs who might not be listed as beneficiaries
            if personal_info.get('has_spouse', False) and not relation_map['spouse']:
                # Spouse exists but not listed as beneficiary
                excluded_legal_heirs.append('Spouse (not listed)')
            
            if personal_info.get('has_children', False) and not relation_map['children']:
                # Children exist but not listed as beneficiaries
                excluded_legal_heirs.append('Children (not listed)')
            
            if personal_info.get('has_living_parents', False) and not relation_map['parents']:
                # Parents alive but not listed as beneficiaries
                excluded_legal_heirs.append('Parents (not listed)')
            
            # Initialize challenges
            challenges = []
            
            # Challenge

            # Check age related factors
            testator_age = int(personal_info.get('age', 0))
            if testator_age > 75:
                challenges.append({
                    'type': 'testamentary_capacity',
                    'reason': self.challenge_reasons['testamentary_capacity']['name'],
                    'description': f"Advanced age ({testator_age} years) may raise questions about testamentary capacity.",
                    'likelihood': 'Medium',
                    'potential_challengers': legal_heirs if legal_heirs else ['Legal heirs'],
                    'mitigation': "Have a medical certificate of soundness of mind at the time of will execution. Ensure proper documentation of the will-making process."
                })
            
            # Check for excluded legal heirs (potential challengers)
            if excluded_legal_heirs:
                for excluded in excluded_legal_heirs:
                    relation = ""
                    name = ""
                    if isinstance(excluded, str):
                        relation = excluded
                    else:
                        relation = ben_map.get(excluded, {}).get('relation', '')
                        name = ben_map.get(excluded, {}).get('name', '')
                    
                    challenges.append({
                        'type': 'exclusion',
                        'reason': self.challenge_reasons['exclusion']['name'],
                        'description': f"{f'{name} ({relation})' if name else relation} has been excluded from the will.",
                        'likelihood': 'High',
                        'potential_challengers': [excluded],
                        'mitigation': "Clearly document reasons for exclusion in the will. Consider providing a token amount instead of complete exclusion."
                    })
            
            # Check for highly unequal distributions among children
            if len(relation_map['children']) > 1:
                child_values = [ben_values.get(child_id, 0) for child_id in relation_map['children']]
                max_value = max(child_values) if child_values else 0
                min_value = min(child_values) if child_values else 0
                
                if max_value > 0 and min_value/max_value < 0.5:
                    # More than 2x difference between children
                    disadvantaged_children = [
                        child_id for child_id in relation_map['children'] 
                        if ben_values.get(child_id, 0) < max_value/2
                    ]
                    
                    challenges.append({
                        'type': 'exclusion',
                        'reason': "Unequal Treatment of Children",
                        'description': "Significant disparity in distribution among children may lead to disputes.",
                        'likelihood': 'High',
                        'potential_challengers': disadvantaged_children,
                        'mitigation': "Clearly document reasons for unequal distribution. Consider balancing the distribution or providing explanation in the will."
                    })
            
            # Check for maintenance obligations
            dependent_family = []
            if personal_info.get('has_dependent_spouse', False):
                dependent_family.append('Spouse')
            
            if personal_info.get('has_minor_children', False):
                dependent_family.append('Minor children')
            
            if personal_info.get('has_dependent_parents', False):
                dependent_family.append('Dependent parents')
            
            if dependent_family:
                # Check if dependents are adequately provided for
                for dependent_type in dependent_family:
                    if dependent_type == 'Spouse' and relation_map['spouse'] and ben_percentages.get(relation_map['spouse'][0], 0) < 20:
                        challenges.append({
                            'type': 'maintenance',
                            'reason': self.challenge_reasons['maintenance']['name'],
                            'description': "Dependent spouse may not have adequate provision for maintenance.",
                            'likelihood': 'High',
                            'potential_challengers': relation_map['spouse'],
                            'mitigation': "Ensure adequate provision for spouse's maintenance. Consider creating a separate maintenance fund."
                        })
                    
                    elif dependent_type == 'Minor children':
                        minor_children = [child for child in relation_map['children'] 
                                        if ben_map.get(child, {}).get('age', 18) < 18]
                        if minor_children:
                            for child in minor_children:
                                if ben_percentages.get(child, 0) < 15:
                                    challenges.append({
                                        'type': 'maintenance',
                                        'reason': self.challenge_reasons['maintenance']['name'],
                                        'description': f"Minor child ({ben_map.get(child, {}).get('name', '')}) may not have adequate provision for maintenance and education.",
                                        'likelihood': 'High',
                                        'potential_challengers': [child],
                                        'mitigation': "Create a trust for minor children's education and maintenance. Ensure adequate provision until they reach majority."
                                    })
                    
                    elif dependent_type == 'Dependent parents' and relation_map['parents']:
                        for parent in relation_map['parents']:
                            if ben_percentages.get(parent, 0) < 10:
                                challenges.append({
                                    'type': 'maintenance',
                                    'reason': self.challenge_reasons['maintenance']['name'],
                                    'description': f"Dependent parent ({ben_map.get(parent, {}).get('name', '')}) may not have adequate provision for maintenance.",
                                    'likelihood': 'Medium',
                                    'potential_challengers': [parent],
                                    'mitigation': "Ensure adequate provision for dependent parents or create a separate maintenance arrangement."
                                })
            
            # Check for Muslim law mandatory shares (if applicable)
            if applicable_law['key'] == 'Muslim' and applicable_law['mandatory_share']:
                # Simplified check for Muslim law compliance
                if relation_map['spouse'] and ben_percentages.get(relation_map['spouse'][0], 0) < 12.5:
                    challenges.append({
                        'type': 'exclusion',
                        'reason': "Violation of Muslim Personal Law",
                        'description': "Spouse's share is below the minimum required by Muslim Personal Law.",
                        'likelihood': 'Very High',
                        'potential_challengers': relation_map['spouse'],
                        'mitigation': "Ensure compliance with mandatory shares under Muslim Personal Law. Consult with an Islamic law expert."
                    })
                
                # Check sons vs daughters (sons should get twice the share of daughters)
                sons = [child for child in relation_map['children'] 
                      if ben_map.get(child, {}).get('relation', '').lower() == 'son']
                daughters = [child for child in relation_map['children'] 
                           if ben_map.get(child, {}).get('relation', '').lower() == 'daughter']
                
                if sons and daughters:
                    son_shares = [ben_percentages.get(son, 0) for son in sons]
                    daughter_shares = [ben_percentages.get(daughter, 0) for daughter in daughters]
                    
                    avg_son_share = sum(son_shares) / len(son_shares) if son_shares else 0
                    avg_daughter_share = sum(daughter_shares) / len(daughter_shares) if daughter_shares else 0
                    
                    if avg_son_share > 0 and avg_daughter_share > 0 and avg_son_share / avg_daughter_share < 1.5:
                        challenges.append({
                            'type': 'exclusion',
                            'reason': "Variance from Muslim Personal Law",
                            'description': "Distribution does not follow the traditional 2:1 ratio between sons and daughters.",
                            'likelihood': 'High',
                            'potential_challengers': sons,
                            'mitigation': "Adjust distribution to comply with Islamic inheritance rules or document explicit consent to deviation."
                        })
            
            # Check for high-value business assets going to multiple beneficiaries
            business_assets = [a for a in assets if a.get('type') == 'business_assets']
            for asset in business_assets:
                asset_id = asset['id']
                # Check if this business asset is split among multiple beneficiaries
                asset_distribution = [d for d in distribution if d['asset_id'] == asset_id]
                if len(asset_distribution) > 1:
                    challenges.append({
                        'type': 'improper_execution',
                        'reason': "Business Succession Risk",
                        'description': f"Business asset '{asset.get('description', '')}' is distributed among multiple beneficiaries, which may lead to operational conflicts.",
                        'likelihood': 'High',
                        'potential_challengers': [d['beneficiary_id'] for d in asset_distribution],
                        'mitigation': "Consider a business succession plan. Create a shareholder agreement or operating agreement to govern business operations post-succession."
                    })
            
            # Check will execution details if provided
            if personal_info.get('will_executed', False):
                will_execution = personal_info.get('will_execution_details', {})
                
                # Check for proper attestation
                if not will_execution.get('properly_attested', True):
                    challenges.append({
                        'type': 'improper_execution',
                        'reason': self.challenge_reasons['improper_execution']['name'],
                        'description': "Will may not be properly attested by witnesses as required by law.",
                        'likelihood': 'Very High',
                        'potential_challengers': legal_heirs,
                        'mitigation': "Ensure will is attested by at least two witnesses who have seen the testator sign. Complete proper attestation formalities."
                    })
                
                # Check for presence of interested witnesses
                if will_execution.get('interested_witnesses', False):
                    challenges.append({
                        'type': 'improper_execution',
                        'reason': self.challenge_reasons['improper_execution']['name'],
                        'description': "Witnesses may be beneficiaries or have an interest in the will, which can invalidate their attestation.",
                        'likelihood': 'High',
                        'potential_challengers': legal_heirs,
                        'mitigation': "Ensure witnesses are disinterested parties with no benefit under the will."
                    })
                
                # Check for registration
                if not will_execution.get('registered', False) and total_value > 1000000:  # 10 Lakhs
                    challenges.append({
                        'type': 'improper_execution',
                        'reason': "Unregistered High-Value Will",
                        'description': "High-value will is not registered, which may invite additional scrutiny.",
                        'likelihood': 'Medium',
                        'potential_challengers': legal_heirs,
                        'mitigation': "Register the will with the Sub-Registrar's office for additional legal validity."
                    })
            
            # Calculate overall challenge risk
            challenge_risk = self._calculate_challenge_risk(challenges, applicable_law)
            
            return {
                'challenges': challenges,
                'excluded_legal_heirs': [{
                    'id': heir if not isinstance(heir, str) else '',
                    'name': ben_map.get(heir, {}).get('name', heir) if not isinstance(heir, str) else heir,
                    'relation': ben_map.get(heir, {}).get('relation', '') if not isinstance(heir, str) else ''
                } for heir in excluded_legal_heirs],
                'legal_heirs': [{
                    'id': heir,
                    'name': ben_map.get(heir, {}).get('name', ''),
                    'relation': ben_map.get(heir, {}).get('relation', '')
                } for heir in legal_heirs],
                'overall_risk': challenge_risk,
                'challenge_type_counts': self._count_challenge_types(challenges)
            }
            
        except Exception as e:
            self.logger.error(f"Error identifying potential challenges: {str(e)}")
            return {
                'error': f"Challenge identification failed: {str(e)}",
                'challenges': [],
                'overall_risk': 'Unknown'
            }
    
    def _calculate_challenge_risk(self, challenges: List[Dict], applicable_law: Dict) -> str:
        """
        Calculate overall challenge risk based on identified challenges.
        
        Args:
            challenges: List of identified challenges
            applicable_law: Information about applicable law
            
        Returns:
            str: Overall risk assessment (Very High, High, Medium, Low, Very Low)
        """
        try:
            if not challenges:
                return 'Very Low'
            
            # Count challenges by likelihood
            likelihood_counts = {
                'Very High': 0,
                'High': 0,
                'Medium': 0,
                'Low': 0
            }
            
            for challenge in challenges:
                likelihood = challenge.get('likelihood', 'Medium')
                likelihood_counts[likelihood] = likelihood_counts.get(likelihood, 0) + 1
            
            # Calculate weighted score
            weights = {
                'Very High': 4,
                'High': 3,
                'Medium': 2,
                'Low': 1
            }
            
            score = sum(count * weights[likelihood] for likelihood, count in likelihood_counts.items())
            
            # Apply multiplier based on applicable law
            law_risk_multiplier = 1.0
            if applicable_law['key'] == 'Muslim' and applicable_law['mandatory_share']:
                law_risk_multiplier = 1.2  # Higher risk due to mandatory shares
            
            score *= law_risk_multiplier
            
            # Map score to risk level
            if score >= 12 or likelihood_counts['Very High'] >= 1:
                return 'Very High'
            elif score >= 8 or likelihood_counts['High'] >= 2:
                return 'High'
            elif score >= 4 or likelihood_counts['High'] >= 1:
                return 'Medium'
            elif score >= 2:
                return 'Low'
            else:
                return 'Very Low'
                
        except Exception as e:
            self.logger.error(f"Error calculating challenge risk: {str(e)}")
            return 'Medium'  # Default to medium risk on error
    
    def _count_challenge_types(self, challenges: List[Dict]) -> Dict[str, int]:
        """
        Count challenges by type.
        
        Args:
            challenges: List of identified challenges
            
        Returns:
            Dict: Counts by challenge type
        """
        try:
            counts = {}
            for challenge in challenges:
                challenge_type = challenge.get('type', 'other')
                counts[challenge_type] = counts.get(challenge_type, 0) + 1
            
            return counts
            
        except Exception as e:
            self.logger.error(f"Error counting challenge types: {str(e)}")
            return {}
    
    def _generate_visualizations(self, assets: List[Dict], beneficiaries: List[Dict], 
                               distribution: List[Dict], 
                               distribution_analysis: Dict) -> Dict[str, str]:
        """
        Generate visualizations for the will scenario.
        
        Args:
            assets: List of assets owned by the testator
            beneficiaries: List of family members and other beneficiaries
            distribution: How assets are to be distributed among beneficiaries
            distribution_analysis: Results of distribution analysis
            
        Returns:
            Dict: Base64 encoded visualizations
        """
        try:
            visualizations = {}
            
            # 1. Asset Distribution by Beneficiary (Pie Chart)
            if 'beneficiary_details' in distribution_analysis:
                fig = go.Figure()
                
                beneficiary_data = distribution_analysis['beneficiary_details']
                
                fig.add_trace(go.Pie(
                    labels=[f"{ben['name']} ({ben['relation']})" for ben in beneficiary_data],
                    values=[ben['value'] for ben in beneficiary_data],
                    textinfo='label+percent',
                    hoverinfo='label+value+percent',
                    marker=dict(
                        colors=px.colors.qualitative.Pastel,
                        line=dict(color='#000000', width=0.5)
                    )
                ))
                
                fig.update_layout(
                    title_text='Asset Distribution by Beneficiary',
                    height=500,
                    width=700
                )
                
                # Convert to base64
                img_bytes = io.BytesIO()
                fig.write_image(img_bytes, format='png')
                img_bytes.seek(0)
                beneficiary_viz = base64.b64encode(img_bytes.read()).decode('utf-8')
                visualizations['beneficiary_distribution'] = beneficiary_viz
            
            # 2. Asset Types Distribution (Bar Chart)
            if 'asset_types' in distribution_analysis:
                fig = go.Figure()
                
                asset_types = distribution_analysis['asset_types']
                
                fig.add_trace(go.Bar(
                    x=[asset['name'] for asset in asset_types],
                    y=[asset['value'] for asset in asset_types],
                    text=[f"{asset['percentage']:.1f}%" for asset in asset_types],
                    textposition='auto',
                    marker_color=[
                        'red' if asset['dispute_likelihood'] == 'Very High' else
                        'orange' if asset['dispute_likelihood'] == 'High' else
                        'yellow' if asset['dispute_likelihood'] == 'Medium' else
                        'green'
                        for asset in asset_types
                    ]
                ))
                
                fig.update_layout(
                    title_text='Distribution by Asset Type (color indicates dispute risk)',
                    xaxis_title='Asset Type',
                    yaxis_title='Value (₹)',
                    height=500,
                    width=700
                )
                
                # Convert to base64
                img_bytes = io.BytesIO()
                fig.write_image(img_bytes, format='png')
                img_bytes.seek(0)
                asset_type_viz = base64.b64encode(img_bytes.read()).decode('utf-8')
                visualizations['asset_type_distribution'] = asset_type_viz
            
            # 3. Beneficiary Network Diagram (Relationship map)
            # Create beneficiary map with simplified relations
            ben_map = {b['id']: b for b in beneficiaries}
            
            # Calculate beneficiary values
            ben_values = {}
            asset_map = {a['id']: a for a in assets}
            
            for item in distribution:
                asset_id = item['asset_id']
                ben_id = item['beneficiary_id']
                percentage = float(item['percentage'])
                
                asset = asset_map.get(asset_id, {})
                asset_value = asset.get('value', 0)
                item_value = asset_value * (percentage / 100)
                
                if ben_id not in ben_values:
                    ben_values[ben_id] = 0
                
                ben_values[ben_id] += item_value
            
            # Group beneficiaries by relation type
            relation_groups = {
                'Spouse': [],
                'Children': [],
                'Parents': [],
                'Siblings': [],
                'Other Family': [],
                'Non-Family': []
            }
            
            for ben_id, ben in ben_map.items():
                relation = ben.get('relation', '').lower()
                name = ben.get('name', '')
                value = ben_values.get(ben_id, 0)
                
                if relation == 'spouse':
                    relation_groups['Spouse'].append({'id': ben_id, 'name': name, 'value': value})
                elif relation in ['son', 'daughter']:
                    relation_groups['Children'].append({'id': ben_id, 'name': name, 'value': value})
                elif relation in ['father', 'mother']:
                    relation_groups['Parents'].append({'id': ben_id, 'name': name, 'value': value})
                elif relation in ['brother', 'sister']:
                    relation_groups['Siblings'].append({'id': ben_id, 'name': name, 'value': value})
                elif relation in ['grandson', 'granddaughter', 'uncle', 'aunt', 'nephew', 'niece', 'cousin']:
                    relation_groups['Other Family'].append({'id': ben_id, 'name': name, 'value': value})
                else:
                    relation_groups['Non-Family'].append({'id': ben_id, 'name': name, 'value': value})
            
            # Create a treemap visualization
            labels = []
            parents = []
            values = []
            colors = []
            
            # Add root
            labels.append('All Beneficiaries')
            parents.append('')
            values.append(sum(ben_values.values()))
            colors.append('white')
            
            # Add relation groups
            group_colors = {
                'Spouse': '#FF9999',
                'Children': '#66B2FF',
                'Parents': '#99FF99',
                'Siblings': '#FFCC99',
                'Other Family': '#CC99FF',
                'Non-Family': '#CCCCCC'
            }
            
            for group_name, members in relation_groups.items():
                if members:  # Only add non-empty groups
                    group_value = sum(member['value'] for member in members)
                    
                    labels.append(group_name)
                    parents.append('All Beneficiaries')
                    values.append(group_value)
                    colors.append(group_colors[group_name])
                    
                    # Add individual beneficiaries within each group
                    for member in members:
                        labels.append(member['name'])
                        parents.append(group_name)
                        values.append(member['value'])
                        colors.append(group_colors[group_name])
            
            if labels:  # Only create visualization if we have data
                fig = go.Figure(go.Treemap(
                    labels=labels,
                    parents=parents,
                    values=values,
                    marker=dict(
                        colors=colors,
                        line=dict(width=1)
                    ),
                    hovertemplate='<b>%{label}</b><br>Value: ₹%{value:,.2f}<br><extra></extra>',
                    textinfo="label+value"
                ))
                
                fig.update_layout(
                    title_text='Beneficiary Relationship Map',
                    height=600,
                    width=800
                )
                
                # Convert to base64
                img_bytes = io.BytesIO()
                fig.write_image(img_bytes, format='png')
                img_bytes.seek(0)
                relationship_viz = base64.b64encode(img_bytes.read()).decode('utf-8')
                visualizations['relationship_map'] = relationship_viz
            
            return visualizations
            
        except Exception as e:
            self.logger.error(f"Error generating visualizations: {str(e)}")
            return {}
    
    def _generate_documentation_recommendations(self, personal_info: Dict, assets: List[Dict], 
                                              distribution: List[Dict], 
                                              potential_challenges: Dict) -> List[Dict]:
        """
        Generate recommendations for will documentation.
        
        Args:
            personal_info: Personal details of the testator
            assets: List of assets owned by the testator
            distribution: How assets are to be distributed among beneficiaries
            potential_challenges: Identified potential challenges
            
        Returns:
            List: Documentation recommendations
        """
        try:
            recommendations = []
            
            # Basic recommendations
            recommendations.append({
                'type': 'essential',
                'title': 'Proper Attestation',
                'description': 'Ensure will is signed by testator and attested by at least two witnesses who have seen the testator sign.',
                'importance': 'Critical',
                'details': 'The witnesses should sign in the presence of the testator and each other. They should not be beneficiaries under the will.'
            })
            
            recommendations.append({
                'type': 'essential',
                'title': 'Clear Identification',
                'description': 'Include full name, address, and other identifying information of the testator.',
                'importance': 'High',
                'details': 'This helps establish the identity of the testator and prevents confusion with others of similar names.'
            })
            
            recommendations.append({
                'type': 'essential',
                'title': 'Sound Mind Declaration',
                'description': 'Include a declaration that the testator is of sound mind and making the will voluntarily.',
                'importance': 'High',
                'details': 'This helps counter challenges based on lack of testamentary capacity.'
            })
            
            # Age-based recommendations
            testator_age = int(personal_info.get('age', 0))
            if testator_age > 70:
                recommendations.append({
                    'type': 'age_related',
                    'title': 'Medical Certificate',
                    'description': 'Attach a recent medical certificate confirming soundness of mind.',
                    'importance': 'High',
                    'details': 'For testators of advanced age, a medical certificate can help prevent challenges based on mental capacity.'
                })
            
            # Challenge-specific recommendations
            challenges = potential_challenges.get('challenges', [])
            for challenge in challenges:
                # Add specific recommendations based on challenge type
                challenge_type = challenge.get('type', '')
                
                if challenge_type == 'testamentary_capacity':
                    recommendations.append({
                        'type': 'challenge_mitigation',
                        'title': 'Video Recording of Will Execution',
                        'description': 'Consider video recording the will execution process.',
                        'importance': 'Medium',
                        'details': 'A video recording can serve as evidence of the testator\'s mental capacity and voluntary execution of the will.'
                    })
                
                elif challenge_type == 'exclusion':
                    recommendations.append({
                        'type': 'challenge_mitigation',
                        'title': 'Exclusion Explanation',
                        'description': 'Document reasons for excluding legal heirs or unequal distributions.',
                        'importance': 'High',
                        'details': 'Clearly stating reasons for exclusion or unequal treatment can help counter challenges from excluded or disadvantaged heirs.'
                    })
                
                elif challenge_type == 'maintenance':
                    recommendations.append({
                        'type': 'challenge_mitigation',
                        'title': 'Maintenance Provisions',
                        'description': 'Include specific provisions for maintenance of dependents.',
                        'importance': 'High',
                        'details': 'Creating a separate maintenance fund or trust for dependents can help fulfill legal obligations and prevent challenges.'
                    })
                
                elif challenge_type == 'improper_execution':
                    recommendations.append({
                        'type': 'challenge_mitigation',
                        'title': 'Professional Will Drafting',
                        'description': 'Have the will drafted by a legal professional specializing in succession law.',
                        'importance': 'High',
                        'details': 'Professional drafting ensures proper legal language and compliance with execution requirements.'
                    })
            
            # Asset-specific recommendations
            asset_types = set(asset.get('type', '') for asset in assets)
            
            if 'immovable_property' in asset_types:
                recommendations.append({
                    'type': 'asset_specific',
                    'title': 'Property Identification',
                    'description': 'Include exact details of immovable properties with survey numbers, boundaries, etc.',
                    'importance': 'High',
                    'details': 'Precise identification of properties prevents disputes over which properties are included in the will.'
                })
            
            if 'business_assets' in asset_types:
                recommendations.append({
                    'type': 'asset_specific',
                    'title': 'Business Succession Plan',
                    'description': 'Create a separate business succession plan if distributing business assets.',
                    'importance': 'High',
                    'details': 'A detailed plan for business succession can prevent operational disruptions and conflicts among inheritors.'
                })
            
            # Consider will registration
            total_value = sum(asset.get('value', 0) for asset in assets)
            if total_value > 1000000:  # 10 Lakhs
                recommendations.append({
                        'type': 'procedural',
                        'title': 'Will Registration',
                        'description': 'Register the will with the Sub-Registrar\'s office.',
                        'importance': 'Medium',
                        'details': 'While not mandatory, registration adds a layer of authenticity and makes the will harder to challenge.'
                    })
            
            # Executor recommendation
            if not personal_info.get('has_executor', False):
                recommendations.append({
                    'type': 'procedural',
                    'title': 'Appoint Executor',
                    'description': 'Appoint a trusted individual or institution as executor of the will.',
                    'importance': 'High',
                    'details': 'An executor ensures proper implementation of the will and can navigate legal procedures after the testator\'s death.'
                })
            
            # Sort by importance
            importance_order = {
                'Critical': 0,
                'High': 1,
                'Medium': 2,
                'Low': 3
            }
            
            recommendations.sort(key=lambda x: importance_order.get(x.get('importance', 'Low'), 3))
            
            return recommendations
            
        except Exception as e:
            self.logger.error(f"Error generating documentation recommendations: {str(e)}")
            return [{
                'type': 'error',
                'title': 'Error Generating Recommendations',
                'description': 'An error occurred while generating recommendations.',
                'importance': 'High',
                'details': 'Please ensure you have provided valid data and try again.'
            }]
    
    def _calculate_intestate_scenario(self, personal_info: Dict, assets: List[Dict], 
                                    beneficiaries: List[Dict], applicable_law: Dict) -> Dict:
        """
        Calculate inheritance distribution under intestate succession.
        
        Args:
            personal_info: Personal details of the testator
            assets: List of assets owned by the testator
            beneficiaries: List of family members and other beneficiaries
            applicable_law: Applicable succession law
            
        Returns:
            Dict: Intestate distribution scenario
        """
        try:
            # Create map of beneficiaries
            ben_map = {b['id']: b for b in beneficiaries}
            
            # Total asset value
            total_value = sum(asset.get('value', 0) for asset in assets)
            
            # Get class of legal heirs based on family structure
            spouse_ids = [b['id'] for b in beneficiaries if b.get('relation', '').lower() == 'spouse']
            children_ids = [b['id'] for b in beneficiaries if b.get('relation', '').lower() in ['son', 'daughter']]
            parent_ids = [b['id'] for b in beneficiaries if b.get('relation', '').lower() in ['father', 'mother']]
            sibling_ids = [b['id'] for b in beneficiaries if b.get('relation', '').lower() in ['brother', 'sister']]
            
            # Initialize beneficiary shares
            shares = {}
            for b in beneficiaries:
                shares[b['id']] = 0
            
            # Apply intestate succession rules based on applicable law
            law_key = applicable_law.get('key', 'Civil')
            gender = personal_info.get('gender', '').lower()
            
            if law_key == 'Hindu':
                # Hindu Succession Act (simplified)
                if children_ids:
                    # If there are children, spouse gets equal share with children
                    total_shares = len(children_ids) + (1 if spouse_ids else 0)
                    share_value = total_value / total_shares if total_shares > 0 else 0
                    
                    # Assign shares
                    for child_id in children_ids:
                        shares[child_id] = share_value
                    
                    if spouse_ids and len(spouse_ids) > 0:
                        shares[spouse_ids[0]] = share_value
                
                elif spouse_ids:
                    # Spouse and parents if no children
                    if parent_ids:
                        # Half to spouse, half to parents
                        spouse_share = total_value / 2
                        parent_share = (total_value / 2) / len(parent_ids) if parent_ids else 0
                        
                        shares[spouse_ids[0]] = spouse_share
                        for parent_id in parent_ids:
                            shares[parent_id] = parent_share
                    else:
                        # All to spouse if no children and no parents
                        shares[spouse_ids[0]] = total_value
                
                elif parent_ids:
                    # Parents if no spouse and no children
                    parent_share = total_value / len(parent_ids) if parent_ids else 0
                    for parent_id in parent_ids:
                        shares[parent_id] = parent_share
                
                elif sibling_ids:
                    # Siblings if no spouse, children, or parents
                    sibling_share = total_value / len(sibling_ids) if sibling_ids else 0
                    for sibling_id in sibling_ids:
                        shares[sibling_id] = sibling_share
            
            elif law_key == 'Muslim':
                # Muslim Personal Law (very simplified)
                # This is a simplified approximation - actual shares would require detailed Sharia calculations
                if gender == 'male':
                    if spouse_ids and children_ids:
                        # Wife gets 1/8, rest to children with son getting twice daughter's share
                        wife_share = total_value / 8
                        if len(spouse_ids) > 0:
                            shares[spouse_ids[0]] = wife_share
                        
                        remaining = total_value - wife_share
                        
                        # Count shares: son gets 2 shares, daughter gets 1 share
                        sons = [child_id for child_id in children_ids 
                               if ben_map.get(child_id, {}).get('relation', '').lower() == 'son']
                        daughters = [child_id for child_id in children_ids 
                                    if ben_map.get(child_id, {}).get('relation', '').lower() == 'daughter']
                        
                        total_shares = (len(sons) * 2) + len(daughters)
                        share_unit = remaining / total_shares if total_shares > 0 else 0
                        
                        for son_id in sons:
                            shares[son_id] = share_unit * 2
                        
                        for daughter_id in daughters:
                            shares[daughter_id] = share_unit
                    
                    elif spouse_ids:
                        # Wife gets 1/4 if no children, rest to other heirs (simplified)
                        wife_share = total_value / 4
                        if len(spouse_ids) > 0:
                            shares[spouse_ids[0]] = wife_share
                        
                        remaining = total_value - wife_share
                        
                        # Distribute remaining among parents and siblings (simplified)
                        remaining_heirs = parent_ids + sibling_ids
                        if remaining_heirs:
                            share = remaining / len(remaining_heirs)
                            for heir_id in remaining_heirs:
                                shares[heir_id] = share
                
                elif gender == 'female':
                    if spouse_ids and children_ids:
                        # Husband gets 1/4, rest to children with son getting twice daughter's share
                        husband_share = total_value / 4
                        if len(spouse_ids) > 0:
                            shares[spouse_ids[0]] = husband_share
                        
                        remaining = total_value - husband_share
                        
                        # Count shares: son gets 2 shares, daughter gets 1 share
                        sons = [child_id for child_id in children_ids 
                               if ben_map.get(child_id, {}).get('relation', '').lower() == 'son']
                        daughters = [child_id for child_id in children_ids 
                                    if ben_map.get(child_id, {}).get('relation', '').lower() == 'daughter']
                        
                        total_shares = (len(sons) * 2) + len(daughters)
                        share_unit = remaining / total_shares if total_shares > 0 else 0
                        
                        for son_id in sons:
                            shares[son_id] = share_unit * 2
                        
                        for daughter_id in daughters:
                            shares[daughter_id] = share_unit
                    
                    elif spouse_ids:
                        # Husband gets 1/2 if no children, rest to other heirs (simplified)
                        husband_share = total_value / 2
                        if len(spouse_ids) > 0:
                            shares[spouse_ids[0]] = husband_share
                        
                        remaining = total_value - husband_share
                        
                        # Distribute remaining among parents and siblings (simplified)
                        remaining_heirs = parent_ids + sibling_ids
                        if remaining_heirs:
                            share = remaining / len(remaining_heirs)
                            for heir_id in remaining_heirs:
                                shares[heir_id] = share
            
            elif law_key in ['Christian', 'Civil']:
                # Indian Succession Act (simplified)
                if gender == 'male':
                    if spouse_ids and children_ids:
                        # One-third to widow, two-thirds to children
                        widow_share = total_value / 3
                        if len(spouse_ids) > 0:
                            shares[spouse_ids[0]] = widow_share
                        
                        remaining = total_value - widow_share
                        child_share = remaining / len(children_ids) if children_ids else 0
                        
                        for child_id in children_ids:
                            shares[child_id] = child_share
                    
                    elif spouse_ids:
                        # Half to widow, half to kin (parents, siblings) if no children
                        widow_share = total_value / 2
                        if len(spouse_ids) > 0:
                            shares[spouse_ids[0]] = widow_share
                        
                        remaining = total_value - widow_share
                        kin = parent_ids + sibling_ids
                        
                        if kin:
                            kin_share = remaining / len(kin)
                            for kin_id in kin:
                                shares[kin_id] = kin_share
                
                elif gender == 'female':
                    if spouse_ids and children_ids:
                        # One-third to husband, two-thirds to children
                        husband_share = total_value / 3
                        if len(spouse_ids) > 0:
                            shares[spouse_ids[0]] = husband_share
                        
                        remaining = total_value - husband_share
                        child_share = remaining / len(children_ids) if children_ids else 0
                        
                        for child_id in children_ids:
                            shares[child_id] = child_share
                    
                    elif spouse_ids:
                        # Half to husband, half to kin (parents, siblings) if no children
                        husband_share = total_value / 2
                        if len(spouse_ids) > 0:
                            shares[spouse_ids[0]] = husband_share
                        
                        remaining = total_value - husband_share
                        kin = parent_ids + sibling_ids
                        
                        if kin:
                            kin_share = remaining / len(kin)
                            for kin_id in kin:
                                shares[kin_id] = kin_share
            
            elif law_key == 'Parsi':
                # Parsi succession (simplified)
                if spouse_ids and children_ids:
                    # Equal division among spouse and children
                    total_heirs = len(children_ids) + (1 if spouse_ids else 0)
                    share_value = total_value / total_heirs if total_heirs > 0 else 0
                    
                    for child_id in children_ids:
                        shares[child_id] = share_value
                    
                    if spouse_ids and len(spouse_ids) > 0:
                        shares[spouse_ids[0]] = share_value
                
                elif spouse_ids:
                    # Half to spouse, half to parents if no children
                    spouse_share = total_value / 2
                    if len(spouse_ids) > 0:
                        shares[spouse_ids[0]] = spouse_share
                    
                    remaining = total_value - spouse_share
                    
                    if parent_ids:
                        parent_share = remaining / len(parent_ids)
                        for parent_id in parent_ids:
                            shares[parent_id] = parent_share
            
            # Format results
            distribution_details = []
            for ben_id, share in shares.items():
                if share > 0:
                    ben = ben_map.get(ben_id, {})
                    distribution_details.append({
                        'id': ben_id,
                        'name': ben.get('name', ''),
                        'relation': ben.get('relation', ''),
                        'share_value': round(share, 2),
                        'share_percentage': round(share / total_value * 100 if total_value > 0 else 0, 2)
                    })
            
            # Sort by share value (descending)
            distribution_details.sort(key=lambda x: x['share_value'], reverse=True)
            
            return {
                'law_name': applicable_law.get('name', ''),
                'intestate_rule': applicable_law.get('intestate_rule', ''),
                'distribution_details': distribution_details,
                'total_value': total_value
            }
            
        except Exception as e:
            self.logger.error(f"Error calculating intestate scenario: {str(e)}")
            return {
                'error': f"Intestate calculation failed: {str(e)}",
                'distribution_details': []
            }