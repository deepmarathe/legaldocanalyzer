import streamlit as st
import json
from datetime import datetime
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from will_scenario_ui import create_will_simulator_ui
from doc1 import EnhancedLegalDocumentProcessor

st.set_page_config(
    page_title="Smart Legal Document Analyzer",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded"
)



def create_enhanced_ui():
    if 'show_will_simulator' in st.session_state and st.session_state.show_will_simulator:
        # Add a back button
        if st.button("← Back to Document Analyzer"):
            st.session_state.show_will_simulator = False
            st.rerun()
        
        # Display the Will Simulator UI
        create_will_simulator_ui()
        return

    """Create enhanced user interface with all features"""
    st.title("Smart Legal Document Analyzer")
    st.write("Upload any property document for a simple, clear explanation with advanced legal expertise")
    
    # Sidebar settings
    with st.sidebar:
        st.header("Analysis Settings")
        target_language = st.selectbox(
            "Select Language",
            options=EnhancedLegalDocumentProcessor().supported_languages.keys(),
            format_func=lambda x: EnhancedLegalDocumentProcessor().supported_languages[x]
        )

        st.markdown("---")

        # Add Will Scenario Simulator button in sidebar
        st.subheader("Additional Tools")
        if st.button("Will Scenario Simulator", use_container_width=True):
            st.session_state.show_will_simulator = True
            st.rerun()

        st.markdown("---")



        with st.expander("Document Types Supported"):
            st.write("""
            - Sale Deeds
            - Rental Agreements
            - Mortgage Documents
            - Property Tax Documents
            - Title Deeds
            - Power of Attorney
            - Gift Deeds
            """)
            
        with st.expander("Legal Expertise Features"):
            st.write("""
            - Legal Precedent Analysis
            - Document Enforceability Assessment
            - Legal Terminology Standardization
            - Counterparty Objection Analysis
            - Stamp Duty Calculation
            - Force Majeure Clause Analysis
            - Legal Timeline Interpretation
            """)

    uploaded_file = st.file_uploader(
        "Upload your document",
        type=['pdf', 'png', 'jpg', 'jpeg', 'docx', 'txt']
    )

    if uploaded_file:
        processor = EnhancedLegalDocumentProcessor()

        with st.spinner("Analyzing your document with legal expertise..."):
            analysis = processor.analyze_document_complete(uploaded_file, target_language)

        if 'error' not in analysis:
            # Create tabs for organized display
            tabs = st.tabs([
                "📑 Simple Explanation",
                "⚠️ Warnings",
                "📋 Action Items",
                "💰 Property Details",
                "✅ Verification",
                "⚖️ Legal Analysis",
                "📝 Document Improvement",
                "📊 Language Complexity",  # Added new tab
                "🌐 Translation"
            ])

            # Simple Explanation Tab
            with tabs[0]:
                st.subheader("What is this document?")
                st.write(analysis['simplified_explanation']['what_is_it'])

                st.subheader("Key Points")
                for point in analysis['simplified_explanation']['key_points']:
                    st.write(f"• {point}")

                with st.expander("Legal Terms Explained"):
                    for term in analysis['key_terms_explained']:
                        col1, col2 = st.columns([1, 2])
                        with col1:
                            st.write(f"**{term['term']}**")
                        with col2:
                            st.write(term['simple_explanation'])
                            if term['importance'] == "High":
                                st.write("🔴 Important term!")

            # Warnings Tab
            with tabs[1]:
                if analysis['red_flags']:
                    for flag in analysis['red_flags']:
                        with st.error(f"Warning: {flag['warning']}"):
                            st.write(f"**Severity:** {flag['severity']}")
                            st.write(f"**What to do:** {flag['suggested_action']}")
                else:
                    st.success("No major issues found in the document")

            # Action Items Tab
            with tabs[2]:
                for action in analysis['action_items']:
                    with st.info(f"**{action['action']}**"):
                        if action.get('deadline'):
                            st.write(f"Deadline: {action['deadline']}")
                        st.write(f"Priority: {action['priority']}")
                        if action.get('documents_needed'):
                            st.write("Documents needed:")
                            for doc in action['documents_needed']:
                                st.write(f"• {doc}")

            # Property Details Tab
            with tabs[3]:
                if 'property_details' in analysis:
                    details = analysis['property_details']
                    cols = st.columns(2)
                    with cols[0]:
                        st.metric("Location", details.get('location', 'Not specified'))
                        
                        # Extract area display as a string
                        area_display = (
                            f"{details.get('area', {}).get('total_area', 0)} {details.get('area', {}).get('unit', 'sq. ft')}"
                            if isinstance(details.get('area', {}).get('total_area'), (int, float))
                            else str(details.get('area', {}).get('area_display', 'Area not specified'))
                        )
                        st.metric("Area", area_display)

                if 'property_valuation' in analysis:
                    st.subheader("Property Valuation")
                    valuation = analysis['property_valuation']
                    st.metric(
                        "Estimated Value",
                        f"₹{valuation.get('estimated_value', 0):,.2f}",
                        help="Based on current market rates"
                    )

            # Verification Tab
            with tabs[4]:
                st.subheader("Document Verification Checklist")
                for item in analysis['verification_checklist']:
                    checked = st.checkbox(
                        item['item'],
                        help=item['description']
                    )
                    if item.get('documents_needed'):
                        st.write("Required documents:")
                        for doc in item['documents_needed']:
                            st.write(f"• {doc}")
                    st.markdown("---")

            # Legal Analysis Tab - NEW
            with tabs[5]:
                st.subheader("Legal Enforceability Analysis")
                
                if 'enforceability_analysis' in analysis:
                    enforceability = analysis['enforceability_analysis']
                    
                    # Show enforceability score with color
                    if 'overall_score' in enforceability:
                        score = enforceability['overall_score']
                        score_color = "green" if score > 0.8 else "orange" if score > 0.6 else "red"
                        st.markdown(f"<h3 style='color: {score_color}'>Enforceability Score: {score:.0%}</h3>", unsafe_allow_html=True)
                    
                    # Show risk assessment
                    if 'risk_assessment' in enforceability:
                        risk = enforceability['risk_assessment']
                        st.write(f"**Risk Assessment:** {risk['description']}")
                    
                    # Show execution requirements
                    if 'execution_requirements' in enforceability:
                        execution = enforceability['execution_requirements']
                        with st.expander("Execution Requirements"):
                            st.write(f"**Signatures Required:** {'✅' if execution.get('signature_requirements_met', False) else '❌'}")
                            st.write(f"**Witness Requirements:** {'✅' if execution.get('witness_requirements_met', False) else '❌'}")
                            st.write(f"**Registration Required:** {'Yes' if execution.get('registration_required', False) else 'No'}")
                            st.write(f"**Registration Mentioned:** {'✅' if execution.get('registration_mentioned', False) else '❌'}")
                    
                    # Show essential clauses
                    if 'essential_clauses' in enforceability:
                        clauses = enforceability['essential_clauses']
                        with st.expander("Essential Clauses"):
                            if 'completeness_score' in clauses:
                                st.write(f"**Completeness:** {clauses['completeness_score']:.0%}")
                            if 'missing_clauses' in clauses and clauses['missing_clauses']:
                                st.write("**Missing Clauses:**")
                                for clause in clauses['missing_clauses']:
                                    st.write(f"• {clause.replace('_', ' ').title()}")
                else:
                    st.warning("Enforceability analysis not available.")
                
                # Show timeline requirements
                st.subheader("Legal Timeline Requirements")
                if 'timeline_requirements' in analysis:
                    timeline = analysis['timeline_requirements']
                    if 'critical_deadlines' in timeline and timeline['critical_deadlines']:
                        st.write("**Critical Deadlines:**")
                        for deadline in timeline['critical_deadlines']:
                            with st.error(deadline['deadline_text']):
                                st.write(f"Responsible: {', '.join(deadline['parties_responsible'])}")
                                st.write(f"Consequence: {deadline['legal_consequence']['consequence']}")
                    else:
                        st.info("No critical deadlines found.")
                else:
                    st.warning("Timeline analysis not available.")
                
                # Show legal precedents
                st.subheader("Legal Precedents Analysis")
                if 'legal_precedents' in analysis:
                    precedents = analysis['legal_precedents']
                    with st.expander("Legal Precedents"):
                        if 'precedents' in precedents and precedents['precedents']:
                            if 'legal_strength' in precedents:
                                st.write(f"**Legal Strength:** {precedents['legal_strength']['strength'].title()}")
                                st.write(f"**Analysis:** {precedents['legal_strength']['reasoning']}")
                            
                            st.write("**Cited Precedents:**")
                            for precedent in precedents['precedents']:
                                st.markdown(f"• **{precedent['citation']}** ({precedent['court']})")
                                if 'impact_analysis' in precedent:
                                    st.write(f"  Impact: {precedent['impact_analysis']['impact_type'].title()}")
                        else:
                            st.write("No legal precedents found in document")
                else:
                    st.warning("Legal precedents analysis not available.")
                
                # Show stamp duty analysis
                st.subheader("Stamp Duty Analysis")
                if 'stamp_duty_analysis' in analysis:
                    stamp_duty = analysis['stamp_duty_analysis']
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("Stamp Duty", f"₹{stamp_duty.get('stamp_duty_amount', 0):,.2f}")
                    with col2:
                        st.metric("Registration Fee", f"₹{stamp_duty.get('registration_fee', 0):,.2f}")
                    
                    st.write(f"**Payment Procedure:** {stamp_duty.get('payment_procedure', '')}")
                    st.write(f"**Legal Reference:** {stamp_duty.get('legal_references', '')}")
                    
                    if 'applicable_exemptions' in stamp_duty and stamp_duty['applicable_exemptions']:
                        st.write("**Applicable Exemptions:**")
                        for exemption in stamp_duty['applicable_exemptions']:
                            st.write(f"• {exemption['type']} - {exemption['reduction_percentage']}% reduction")
                else:
                    st.warning("Stamp duty analysis not available.")


            # Document Improvement Tab - NEW
            with tabs[6]:
                st.subheader("Legal Terminology Standardization")
                
                if 'terminology_standardization' in analysis:
                    terminology = analysis['terminology_standardization']
                    
                    if 'non_standard_terms' in terminology and terminology['non_standard_terms']:
                        if 'standardization_impact' in terminology:
                            st.write(f"**Standardization Impact:** {terminology['standardization_impact']['description']}")
                        
                        with st.expander("Non-Standard Terms"):
                            for term in terminology['non_standard_terms']:
                                col1, col2 = st.columns([1, 2])
                                with col1:
                                    st.write(f"**{term['term']}**")
                                    st.write(f"Found {term['occurrences']} times")
                                with col2:
                                    st.write(f"Standard form: **{term['standard_form']}**")
                                    if 'legal_impact' in term:
                                        st.write(f"Impact: {term['legal_impact']['level'].title()}")
                    else:
                        st.success("No non-standard legal terminology found")
                else:
                    st.warning("Terminology standardization analysis not available.")
        
            st.subheader("Potential Objections Analysis")
            
            if 'potential_objections' in analysis:
                objections = analysis['potential_objections']
                
                if 'potential_objections' in objections and objections['potential_objections']:
                    st.write("**Potential objections that could be raised:**")
                    
                    # High risk objections
                    if 'highest_risk_areas' in objections and objections['highest_risk_areas']:
                        st.write("**High Risk Areas:**")
                        for obj in objections['highest_risk_areas']:
                            with st.error(f"**{obj['type']}**"):
                                st.write(f"Legal basis: {obj['legal_basis']}")
                                st.write(f"Potential argument: {obj['potential_argument']}")
                                st.markdown(f"**Amendment Suggestion:** {obj['suggested_amendment']}")
                    
                    # Show amendment recommendations
                    if 'amendment_recommendations' in objections:
                        with st.expander("Amendment Recommendations"):
                            for priority in objections['amendment_recommendations']:
                                st.write(f"**{priority['priority']} Amendments:**")
                                st.write(priority['description'])
                                for amendment in priority['amendments']:
                                    st.markdown(f"• {amendment}")
                else:
                    st.success("No significant potential objections identified")
            else:
                st.warning("Potential objections analysis not available.")
            
            # Force majeure analysis
            st.subheader("Force Majeure Analysis")
            
            if 'force_majeure_analysis' in analysis:
                force_majeure = analysis['force_majeure_analysis']
                
                if force_majeure.get('present', False):
                    # Calculate completeness visually
                    if 'comprehensiveness_score' in force_majeure:
                        completeness = force_majeure['comprehensiveness_score']
                        completeness_color = "green" if completeness > 0.8 else "orange" if completeness > 0.5 else "red"
                        st.markdown(f"<h3 style='color: {completeness_color}'>Comprehensiveness: {completeness:.0%}</h3>", unsafe_allow_html=True)
                    
                    # Show covered events
                    col1, col2 = st.columns(2)
                    with col1:
                        if 'covered_events' in force_majeure:
                            st.write("**Covered Events:**")
                            for event in force_majeure['covered_events']:
                                st.write(f"• {event.replace('_', ' ').title()}")
                    
                    with col2:
                        if 'missing_events' in force_majeure and force_majeure['missing_events']:
                            st.write("**Missing Events:**")
                            for event in force_majeure['missing_events']:
                                st.write(f"• {event.replace('_', ' ').title()}")
                    
                    # Show improvement suggestions
                    if 'improvement_suggestions' in force_majeure:
                        with st.expander("Improvement Suggestions"):
                            for suggestion in force_majeure['improvement_suggestions']:
                                st.write(f"• {suggestion}")
                else:
                    if 'recommendation' in force_majeure:
                        st.warning(force_majeure['recommendation'])
                    else:
                        st.warning("No force majeure clauses analyzed.")
            else:
                st.warning("Force majeure analysis not available.")

            

            with tabs[7]:
                st.subheader("Legal Language Complexity Analysis")
        
                if 'language_complexity' in analysis and 'error' not in analysis['language_complexity']:
                    # Get complexity data
                    complexity = analysis['language_complexity']
                    readability = complexity.get('readability_metrics', {})
                    jargon = complexity.get('jargon_analysis', {})
                    sentences = complexity.get('sentence_analysis', {})
                    simplification = complexity.get('simplification', {})
                    
                    # Display overall complexity category
                    if 'meta' in complexity and 'overall_complexity_category' in complexity['meta']:
                        category = complexity['meta']['overall_complexity_category']
                        category_color = {
                            'Very Complex': 'red',
                            'Complex': 'orange',
                            'Moderately Complex': 'yellow',
                            'Moderately Simple': 'lightgreen',
                            'Plain Language': 'green'
                        }.get(category.get('category', ''), 'gray')
                        
                        st.markdown(f"<h3 style='color: {category_color}'>{category.get('category', 'Unknown')} Language</h3>", unsafe_allow_html=True)
                        st.write(category.get('explanation', ''))
                    
                    # Display key readability metrics
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric(
                            "Reading Ease Score", 
                            f"{readability.get('flesch_reading_ease', 0):.1f}",
                            help="0-100 scale. Higher is easier to read."
                        )
                    with col2:
                        st.metric(
                            "Grade Level", 
                            f"{readability.get('flesch_kincaid_grade', 0):.1f}",
                            help="US grade level required to comprehend."
                        )
                    with col3:
                        st.metric(
                            "Avg. Sentence Length", 
                            f"{sentences.get('avg_sentence_length', 0):.1f}",
                            help="Average words per sentence."
                        )
                    
                    # Display visualizations if available
                    if 'visualizations' in complexity:
                        viz = complexity['visualizations']
                        
                        # Show readability gauge chart
                        if 'readability_gauge' in viz:
                            st.image(f"data:image/png;base64,{viz['readability_gauge']}")
                        
                        # Show sentence distribution chart
                        if 'sentence_distribution' in viz:
                            with st.expander("Sentence Length Distribution"):
                                st.image(f"data:image/png;base64,{viz['sentence_distribution']}")
                        
                        # Show word cloud
                        if 'word_cloud' in viz:
                            with st.expander("Common Terms Word Cloud"):
                                st.image(f"data:image/png;base64,{viz['word_cloud']}")
                        
                        # Show complexity comparison chart
                        if 'complexity_comparison' in viz:
                            with st.expander("Complexity Comparison"):
                                st.image(f"data:image/png;base64,{viz['complexity_comparison']}")
                    
                    # Display jargon analysis
                    st.subheader("Legal Jargon Analysis")



            # Translation Tab
            with tabs[8]:
                if target_language != 'en' and 'translated_content' in analysis:
                    st.subheader(f"Translated Content ({analysis['translated_content']['language']})")
                    st.write(analysis['translated_content']['translated_text'])

            # Export options
            st.markdown("---")
            st.subheader("Export Report")
            col1, col2 = st.columns(2)

            with col1:
                pdf_report = processor.generate_pdf_report(analysis, target_language)
                st.download_button(
                    "Download Complete Report (PDF)",
                    pdf_report,
                    file_name=f"legal_analysis_{datetime.now().strftime('%Y%m%d_%H%M')}.pdf",
                    mime="application/pdf"
                )

            with col2:
                # Export as JSON
                json_str = json.dumps(analysis, indent=2, ensure_ascii=False)
                st.download_button(
                    "Download Raw Data (JSON)",
                    json_str,
                    file_name=f"legal_analysis_{datetime.now().strftime('%Y%m%d_%H%M')}.json",
                    mime="application/json"
                )
            # Add will simulator promotion in main interface
            if not uploaded_file:
                st.markdown("---")
                st.subheader("Additional Legal Tools")
                
                col1, col2 = st.columns(2)
                with col1:
                    # Create a card-like container for the Will Simulator promo
                    with st.container():
                        st.markdown("### 📝 Will Scenario Simulator")
                        st.write("""
                        Analyze inheritance scenarios under Indian Law. Simulate asset distribution, 
                        identify potential challenges, and get recommendations for will creation.
                        """)
                        if st.button("Launch Will Simulator", key="main_will_button"):
                            st.session_state.show_will_simulator = True
                            st.rerun()

        else:
            st.error(analysis['error'])

if __name__ == "__main__":
    if 'show_will_simulator' not in st.session_state:
        st.session_state.show_will_simulator = False
    create_enhanced_ui()
