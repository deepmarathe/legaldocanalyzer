#will_scenario_ui.py
import streamlit as st
import json
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import base64

import io
import base64

from datetime import datetime
from io import BytesIO

# Add this to your imports at the top of the file
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image
from reportlab.lib.units import inch
from PIL import Image as PILImage
from will_scenario_simulator import WillScenarioSimulator



# # Set page configuration
# st.set_page_config(
#     page_title="Will Scenario Simulator",
#     page_icon="📜",
#     layout="wide",
#     initial_sidebar_state="expanded"
# )

def generate_pdf_report(results):
            """
            Generate a PDF report from the simulation results
            
            Args:
                results: The simulation results dictionary
                
            Returns:
                bytes: PDF report as bytes
            """
            buffer = io.BytesIO()
            doc = SimpleDocTemplate(buffer, pagesize=letter)
            styles = getSampleStyleSheet()
            elements = []
            
            # Create custom styles
            title_style = styles['Heading1']
            subtitle_style = styles['Heading2']
            normal_style = styles['Normal']
            
            # Add a custom style for notes and warnings
            note_style = ParagraphStyle(
                'NoteStyle', 
                parent=styles['Normal'],
                backColor=colors.lightgrey,
                borderColor=colors.black,
                borderWidth=1,
                borderPadding=5,
                borderRadius=2,
                spaceBefore=10,
                spaceAfter=10
            )
            
            warning_style = ParagraphStyle(
                'WarningStyle', 
                parent=styles['Normal'],
                textColor=colors.red,
                backColor=colors.lightpink,
                borderColor=colors.red,
                borderWidth=1,
                borderPadding=5,
                borderRadius=2,
                spaceBefore=10,
                spaceAfter=10
            )
            
            # Title
            elements.append(Paragraph("Will Scenario Simulation Report", title_style))
            elements.append(Spacer(1, 0.25*inch))
            elements.append(Paragraph(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M')}", normal_style))
            elements.append(Spacer(1, 0.25*inch))
            
            # Add applicable law section
            if 'applicable_law' in results:
                law = results['applicable_law']
                elements.append(Paragraph("Applicable Succession Law", subtitle_style))
                elements.append(Paragraph(f"<b>{law['name']}</b>", normal_style))
                elements.append(Paragraph(f"<b>Intestate Rule:</b> {law['intestate_rule']}", normal_style))
                elements.append(Paragraph(f"<b>Mandatory Share Requirement:</b> {'Yes' if law['mandatory_share'] else 'No'}", normal_style))
                elements.append(Paragraph(f"<b>Notes:</b> {law['notes']}", note_style))
                elements.append(Spacer(1, 0.25*inch))
            
            # Add Assets Summary
            if 'distribution_analysis' in results:
                analysis = results['distribution_analysis']
                elements.append(Paragraph("Assets Summary", subtitle_style))
                
                # Create a table for asset summary
                asset_data = [["Total Assets Value", f"₹{results['total_assets_value']:,}"],
                            ["Number of Assets", str(analysis['total_assets_count'])]]
                
                asset_table = Table(asset_data, colWidths=[2.5*inch, 2.5*inch])
                asset_table.setStyle(TableStyle([
                    ('BACKGROUND', (0, 0), (0, -1), colors.lightgrey),
                    ('GRID', (0, 0), (-1, -1), 1, colors.black),
                    ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
                    ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
                    ('PADDING', (0, 0), (-1, -1), 6),
                ]))
                elements.append(asset_table)
                elements.append(Spacer(1, 0.25*inch))
                
                # Add asset type breakdown if available
                if 'asset_types' in analysis and analysis['asset_types']:
                    elements.append(Paragraph("Asset Types", normal_style))
                    asset_types = analysis['asset_types']
                    
                    # Create table data
                    asset_type_data = [["Asset Type", "Value (₹)", "Percentage", "Dispute Risk"]]
                    for asset in asset_types:
                        asset_type_data.append([
                            asset['name'],
                            f"₹{asset['value']:,}",
                            f"{asset['percentage']:.2f}%",
                            asset['dispute_likelihood']
                        ])
                    
                    # Create table
                    asset_type_table = Table(asset_type_data, colWidths=[1.5*inch, 1.5*inch, 1*inch, 1*inch])
                    asset_type_table.setStyle(TableStyle([
                        ('BACKGROUND', (0, 0), (-1, 0), colors.lightgrey),
                        ('GRID', (0, 0), (-1, -1), 1, colors.black),
                        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
                        ('PADDING', (0, 0), (-1, -1), 6),
                    ]))
                    elements.append(asset_type_table)
                    elements.append(Spacer(1, 0.25*inch))
            
            # Add Beneficiary Distribution
            if 'distribution_analysis' in results and 'beneficiary_details' in results['distribution_analysis']:
                elements.append(Paragraph("Beneficiary Distribution", subtitle_style))
                ben_details = results['distribution_analysis']['beneficiary_details']
                
                if ben_details:
                    # Create table data
                    ben_data = [["Name", "Relation", "Percentage (%)", "Value (₹)", "Legal Heir"]]
                    for ben in ben_details:
                        ben_data.append([
                            ben['name'],
                            ben['relation'],
                            f"{ben['percentage']:.2f}%",
                            f"₹{ben['value']:,}",
                            "Yes" if ben['is_legal_heir'] else "No"
                        ])
                    
                    # Create table
                    ben_table = Table(ben_data, colWidths=[1.2*inch, 1*inch, 1*inch, 1.5*inch, 0.8*inch])
                    ben_table.setStyle(TableStyle([
                        ('BACKGROUND', (0, 0), (-1, 0), colors.lightgrey),
                        ('GRID', (0, 0), (-1, -1), 1, colors.black),
                        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
                        ('PADDING', (0, 0), (-1, -1), 6),
                    ]))
                    elements.append(ben_table)
                    elements.append(Spacer(1, 0.25*inch))
                    
                    # Add distribution metrics
                    elements.append(Paragraph("Distribution Metrics", normal_style))
                    metrics_data = [
                        ["Legal Heirs' Share", f"{results['distribution_analysis']['legal_heirs_share']}%"],
                        ["Immediate Family Share", f"{results['distribution_analysis']['immediate_family_share']}%"],
                        ["Non-Family Share", f"{results['distribution_analysis']['non_family_share']}%"],
                        ["Child Inequality Index", str(results['distribution_analysis']['child_inequality_index'])]
                    ]
                    
                    metrics_table = Table(metrics_data, colWidths=[2.5*inch, 2.5*inch])
                    metrics_table.setStyle(TableStyle([
                        ('BACKGROUND', (0, 0), (0, -1), colors.lightgrey),
                        ('GRID', (0, 0), (-1, -1), 1, colors.black),
                        ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
                        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
                        ('PADDING', (0, 0), (-1, -1), 6),
                    ]))
                    elements.append(metrics_table)
                    elements.append(Spacer(1, 0.25*inch))
            
            # Add visualizations if available
            if 'visualizations' in results and results['visualizations']:
                elements.append(Paragraph("Visualizations", subtitle_style))
                visualizations = results['visualizations']
                
                # Handle visualizations for PDF (convert base64 to PIL Image)
                for viz_key, viz_value in visualizations.items():
                    if viz_value:  # Check if visualization exists
                        try:
                            # Decode base64 to image
                            img_data = base64.b64decode(viz_value)
                            img = PILImage.open(io.BytesIO(img_data))
                            
                            # Save to temporary BytesIO
                            img_buffer = io.BytesIO()
                            img.save(img_buffer, format='PNG')
                            img_buffer.seek(0)
                            
                            # Create ReportLab Image object
                            img_obj = Image(img_buffer, width=6*inch, height=4*inch)
                            elements.append(Paragraph(f"{viz_key.replace('_', ' ').title()}", normal_style))
                            elements.append(img_obj)
                            elements.append(Spacer(1, 0.25*inch))
                        except Exception as e:
                            elements.append(Paragraph(f"Error including visualization: {str(e)}", warning_style))
            
            # Add potential challenges section
            if 'potential_challenges' in results:
                challenges = results['potential_challenges']
                elements.append(Paragraph("Potential Challenges", subtitle_style))
                
                # Overall risk
                risk = challenges.get('overall_risk', 'Unknown')
                elements.append(Paragraph(f"<b>Overall Risk:</b> {risk}", normal_style))
                elements.append(Spacer(1, 0.1*inch))
                
                # Add excluded legal heirs if any
                if 'excluded_legal_heirs' in challenges and challenges['excluded_legal_heirs']:
                    elements.append(Paragraph("Excluded Legal Heirs:", normal_style))
                    excluded_text = ""
                    for heir in challenges['excluded_legal_heirs']:
                        if heir['name']:
                            excluded_text += f"• {heir['name']} ({heir['relation']})<br/>"
                        else:
                            excluded_text += f"• {heir['relation']}<br/>"
                    elements.append(Paragraph(excluded_text, warning_style))
                
                # Add detailed challenges
                if 'challenges' in challenges and challenges['challenges']:
                    elements.append(Paragraph("Detailed Challenges:", normal_style))
                    for challenge in challenges['challenges']:
                        challenge_text = f"<b>{challenge.get('reason', 'Unknown Challenge')}</b> (Likelihood: {challenge.get('likelihood', 'Medium')})<br/>"
                        challenge_text += f"{challenge.get('description', '')}<br/>"
                        
                        if 'potential_challengers' in challenge:
                            challenge_text += "<b>Potential challengers:</b><br/>"
                            for challenger in challenge['potential_challengers']:
                                challenge_text += f"• {challenger}<br/>"
                        
                        if 'mitigation' in challenge:
                            challenge_text += f"<b>Mitigation:</b> {challenge['mitigation']}"
                        
                        elements.append(Paragraph(challenge_text, note_style))
                        elements.append(Spacer(1, 0.1*inch))
                
                elements.append(Spacer(1, 0.25*inch))
            
            # Add recommendations section
            if 'documentation_recommendations' in results:
                recommendations = results['documentation_recommendations']
                elements.append(Paragraph("Documentation Recommendations", subtitle_style))
                
                if recommendations:
                    # Group recommendations by type
                    rec_types = {
                        'essential': "Essential Requirements",
                        'challenge_mitigation': "Challenge Mitigation",
                        'asset_specific': "Asset-Specific Recommendations",
                        'procedural': "Procedural Requirements",
                        'age_related': "Age-Related Considerations"
                    }
                    
                    for rec_type, title in rec_types.items():
                        type_recs = [rec for rec in recommendations if rec.get('type') == rec_type]
                        
                        if type_recs:
                            elements.append(Paragraph(title, normal_style))
                            
                            for rec in type_recs:
                                importance = rec.get('importance', 'Medium')
                                rec_text = f"<b>{rec.get('title', 'Recommendation')}</b> (Importance: {importance})<br/>"
                                rec_text += f"{rec.get('description', '')}<br/>"
                                
                                if 'details' in rec:
                                    rec_text += f"<b>Details:</b> {rec['details']}"
                                
                                # Use note style for medium/low importance, warning style for critical/high
                                style = warning_style if importance in ['Critical', 'High'] else note_style
                                elements.append(Paragraph(rec_text, style))
                                elements.append(Spacer(1, 0.1*inch))
                    
                    elements.append(Spacer(1, 0.25*inch))
            
            # Add intestate comparison if available
            if 'intestate_scenario' in results:
                intestate = results['intestate_scenario']
                elements.append(Paragraph("Intestate Succession Comparison", subtitle_style))
                
                elements.append(Paragraph(f"<b>Applicable Law:</b> {intestate.get('law_name', 'Unknown')}", normal_style))
                elements.append(Paragraph(f"<b>Intestate Rule:</b> {intestate.get('intestate_rule', 'Unknown')}", normal_style))
                elements.append(Spacer(1, 0.1*inch))
                
                if 'distribution_details' in intestate and intestate['distribution_details']:
                    elements.append(Paragraph("Distribution Under Intestate Succession:", normal_style))
                    
                    intestate_dist = intestate['distribution_details']
                    intestate_data = [["Name", "Relation", "Percentage (%)", "Value (₹)"]]
                    
                    for ben in intestate_dist:
                        intestate_data.append([
                            ben['name'],
                            ben['relation'],
                            f"{ben['share_percentage']:.2f}%",
                            f"₹{ben['share_value']:,}"
                        ])
                    
                    intestate_table = Table(intestate_data, colWidths=[1.5*inch, 1.2*inch, 1.2*inch, 1.5*inch])
                    intestate_table.setStyle(TableStyle([
                        ('BACKGROUND', (0, 0), (-1, 0), colors.lightgrey),
                        ('GRID', (0, 0), (-1, -1), 1, colors.black),
                        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
                        ('PADDING', (0, 0), (-1, -1), 6),
                    ]))
                    elements.append(intestate_table)
            
            # Build the PDF
            doc.build(elements)
            
            # Get the value of the StringIO buffer
            pdf = buffer.getvalue()
            buffer.close()
            
            return pdf

def display_image(base64_image):
    """Display base64 encoded image"""
    return st.image(f"data:image/png;base64,{base64_image}")

def create_will_simulator_ui():
    """Create the UI for Will Scenario Simulator"""
    st.title("Will Scenario Simulator")
    st.write("Analyze inheritance scenarios under Indian Law and identify potential challenges")
    
    # Initialize simulator
    simulator = WillScenarioSimulator()
    
    # Sidebar settings
    with st.sidebar:
        st.header("Simulation Settings")
        
        # About section
        with st.expander("About Will Scenario Simulator"):
            st.write("""
            This tool helps analyze inheritance scenarios under Indian Law. 
            It simulates asset distribution, identifies potential challenges, and provides
            recommendations for will creation and execution.
            """)
        
        # Supported succession laws
        with st.expander("Supported Succession Laws"):
            st.write("The simulator supports the following succession laws:")
            for law_key, law in simulator.succession_laws.items():
                st.write(f"- **{law['name']}**")
            
        # Asset types explanation
        with st.expander("Asset Types"):
            st.write("The simulator supports the following asset types:")
            for asset_type, details in simulator.asset_types.items():
                st.write(f"- **{details['name']}**: {details['legal_considerations']}")
    
    # Main content area with tabs
    tabs = st.tabs([
        "📝 Input Data", 
        "📊 Results",
        "⚠️ Challenges",
        "📈 Visualizations",
        "📋 Recommendations",
        "⚖️ Intestate Comparison"
    ])
    
    # Tab 1: Input Data
    with tabs[0]:
        st.header("Input Your Will Scenario")
        
        # Personal Information
        st.subheader("Personal Information")
        col1, col2 = st.columns(2)
        with col1:
            name = st.text_input("Full Name")
            age = st.number_input("Age", min_value=18, max_value=120, value=45)
            gender = st.selectbox("Gender", ["Male", "Female", "Other"])
        
        with col2:
            religion = st.selectbox(
                "Religion", 
                ["Hindu", "Muslim", "Christian", "Parsi", "Jewish", "Buddhist", "Jain", "Sikh", "Other"]
            )
            marital_status = st.selectbox(
                "Marital Status", 
                ["Married", "Unmarried", "Divorced", "Widowed"]
            )
            domicile = st.text_input("Domicile", "India")
        
        # Family Information
        st.subheader("Family Information")
        col1, col2, col3 = st.columns(3)
        with col1:
            has_spouse = st.checkbox("Has Spouse")
            has_dependent_spouse = st.checkbox("Has Dependent Spouse", 
                                              value=has_spouse, 
                                              disabled=not has_spouse)
        with col2:
            has_children = st.checkbox("Has Children")
            has_minor_children = st.checkbox("Has Minor Children", 
                                            value=False, 
                                            disabled=not has_children)
        with col3:
            has_living_parents = st.checkbox("Has Living Parents")
            has_dependent_parents = st.checkbox("Has Dependent Parents", 
                                              value=False, 
                                              disabled=not has_living_parents)
        
        # Will Execution Details
        st.subheader("Will Execution Details")
        will_executed = st.checkbox("Will has been executed")
        if will_executed:
            col1, col2 = st.columns(2)
            with col1:
                properly_attested = st.checkbox("Properly Attested by Witnesses", value=True)
                interested_witnesses = st.checkbox("Witnesses are beneficiaries or interested parties", value=False)
            with col2:
                registered = st.checkbox("Will is Registered", value=False)
                sound_mind_certificate = st.checkbox("Has Sound Mind Certificate", value=False)
        
        # Assets
        st.subheader("Assets")
        st.write("Add the assets owned by the testator")
        
        assets = []
        num_assets = st.number_input("Number of Assets", min_value=1, max_value=20, value=3)
        
        for i in range(num_assets):
            with st.expander(f"Asset {i+1}"):
                asset_id = f"asset_{i+1}"
                asset_type = st.selectbox(
                    "Asset Type", 
                    list(simulator.asset_types.keys()),
                    format_func=lambda x: simulator.asset_types[x]['name'],
                    key=f"asset_type_{i}"
                )
                description = st.text_input("Description", key=f"desc_{i}")
                value = st.number_input("Value (₹)", min_value=0, value=1000000, key=f"val_{i}")
                
                assets.append({
                    "id": asset_id,
                    "type": asset_type,
                    "description": description,
                    "value": value
                })
        
        # Beneficiaries
        st.subheader("Beneficiaries")
        st.write("Add the beneficiaries to be included in the will")
        
        beneficiaries = []
        num_beneficiaries = st.number_input("Number of Beneficiaries", min_value=1, max_value=20, value=3)
        
        relation_options = [
            "Spouse", "Son", "Daughter", "Father", "Mother", 
            "Brother", "Sister", "Grandson", "Granddaughter", 
            "Uncle", "Aunt", "Nephew", "Niece", "Cousin", "Friend", "Other"
        ]
        
        for i in range(num_beneficiaries):
            with st.expander(f"Beneficiary {i+1}"):
                ben_id = f"ben_{i+1}"
                name = st.text_input("Name", key=f"ben_name_{i}")
                relation = st.selectbox("Relation", relation_options, key=f"ben_rel_{i}")
                age = st.number_input("Age", min_value=0, max_value=120, value=30, key=f"ben_age_{i}")
                
                beneficiaries.append({
                    "id": ben_id,
                    "name": name,
                    "relation": relation.lower(),
                    "age": age
                })
        
        # Distribution
        st.subheader("Asset Distribution")
        st.write("Specify how assets should be distributed among beneficiaries")
        
        distribution = []
        
        if assets and beneficiaries:
            for i, asset in enumerate(assets):
                with st.expander(f"Distribution for {asset['description']}"):
                    # Initialize leftover percentage
                    leftover = 100
                    asset_distribution = []
                    
                    # Add distribution for all beneficiaries except the last one
                    for j, ben in enumerate(beneficiaries[:-1]):
                        if leftover > 0:
                            percentage = st.slider(
                                f"Percentage for {ben['name']} ({ben['relation']})",
                                min_value=0.0,
                                max_value=float(leftover),
                                value=leftover / len(beneficiaries),
                                key=f"dist_{i}_{j}"
                            )
                            
                            if percentage > 0:
                                asset_distribution.append({
                                    "asset_id": asset["id"],
                                    "beneficiary_id": ben["id"],
                                    "percentage": percentage
                                })
                                leftover -= percentage
                    
                    # Last beneficiary automatically gets the rest
                    if leftover > 0 and beneficiaries:
                        last_ben = beneficiaries[-1]
                        st.write(f"{last_ben['name']} ({last_ben['relation']}) will get the remaining {leftover}%")
                        
                        asset_distribution.append({
                            "asset_id": asset["id"],
                            "beneficiary_id": last_ben["id"],
                            "percentage": leftover
                        })
                    
                    distribution.extend(asset_distribution)
        
        # Run Simulation Button
        if st.button("Run Simulation", key="run_simulation"):
            if not assets or not beneficiaries or not distribution:
                st.error("Please add at least one asset, one beneficiary, and complete the distribution.")
            else:
                # Create personal_info dictionary
                personal_info = {
                    "name": name,
                    "age": age,
                    "gender": gender.lower(),
                    "religion": religion,
                    "marital_status": marital_status.lower(),
                    "domicile": domicile,
                    "has_spouse": has_spouse,
                    "has_dependent_spouse": has_dependent_spouse,
                    "has_children": has_children,
                    "has_minor_children": has_minor_children,
                    "has_living_parents": has_living_parents,
                    "has_dependent_parents": has_dependent_parents,
                    "will_executed": will_executed
                }
                
                if will_executed:
                    personal_info["will_execution_details"] = {
                        "properly_attested": properly_attested,
                        "interested_witnesses": interested_witnesses,
                        "registered": registered,
                        "sound_mind_certificate": sound_mind_certificate
                    }
                
                # Run the simulation
                with st.spinner("Running will scenario simulation..."):
                    results = simulator.simulate_scenario(personal_info, assets, beneficiaries, distribution)
                
                # Store results in session state
                st.session_state.simulation_results = results
                
                # Show success message and prompt to view results
                if 'error' not in results:
                    st.success("Simulation completed successfully! Navigate to the Results tab to view the analysis.")
                else:
                    st.error(f"Simulation failed: {results['error']}")
    
    # Tab 2: Results
    with tabs[1]:
        if 'simulation_results' in st.session_state:
            results = st.session_state.simulation_results
            
            if 'error' not in results:
                st.header("Simulation Results")
                
                # Applicable Law
                st.subheader("Applicable Succession Law")
                law = results['applicable_law']
                st.write(f"**{law['name']}**")
                st.write(f"**Intestate Rule:** {law['intestate_rule']}")
                st.write(f"**Notes:** {law['notes']}")
                st.write(f"**Mandatory Share Requirement:** {'Yes' if law['mandatory_share'] else 'No'}")
                
                # Assets Summary
                st.subheader("Assets Summary")
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Total Assets Value", f"₹{results['total_assets_value']:,}")
                with col2:
                    st.metric("Number of Assets", results['distribution_analysis']['total_assets_count'])
                
                # Asset Types
                asset_types = results['distribution_analysis']['asset_types']
                if asset_types:
                    df_asset_types = pd.DataFrame(asset_types)
                    fig = px.bar(
                        df_asset_types, 
                        x='name', 
                        y='value',
                        color='dispute_likelihood',
                        title="Asset Distribution by Type",
                        labels={'name': 'Asset Type', 'value': 'Value (₹)'},
                        color_discrete_map={
                            'Very High': 'red',
                            'High': 'orange',
                            'Medium': 'yellow',
                            'Low': 'green'
                        }
                    )
                    st.plotly_chart(fig)
                
                # Beneficiary Distribution
                st.subheader("Beneficiary Distribution")
                ben_details = results['distribution_analysis']['beneficiary_details']
                if ben_details:
                    df_ben = pd.DataFrame(ben_details)
                    df_ben['rel_name'] = df_ben.apply(lambda x: f"{x['name']} ({x['relation']})", axis=1)
                    
                    fig = px.pie(
                        df_ben, 
                        values='value', 
                        names='rel_name',
                        title="Asset Distribution by Beneficiary",
                        hole=0.4
                    )
                    st.plotly_chart(fig)
                    
                    # Show beneficiary details table
                    st.write("Beneficiary Details:")
                    display_df = df_ben[['name', 'relation', 'percentage', 'value', 'is_legal_heir']]
                    display_df = display_df.rename(columns={
                        'name': 'Name',
                        'relation': 'Relation',
                        'percentage': 'Percentage (%)',
                        'value': 'Value (₹)',
                        'is_legal_heir': 'Legal Heir'
                    })
                    st.dataframe(display_df)
                
                # Distribution Analysis
                st.subheader("Distribution Analysis")
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Legal Heirs' Share", f"{results['distribution_analysis']['legal_heirs_share']}%")
                    st.metric("Immediate Family Share", f"{results['distribution_analysis']['immediate_family_share']}%")
                with col2:
                    st.metric("Non-Family Share", f"{results['distribution_analysis']['non_family_share']}%")
                    st.metric("Child Inequality Index", f"{results['distribution_analysis']['child_inequality_index']}")
                
                # If visualizations are present, display them
                if 'visualizations' in results and results['visualizations']:
                    st.subheader("Visualizations")
                    
                    if 'relationship_map' in results['visualizations']:
                        st.write("Beneficiary Relationship Map:")
                        display_image(results['visualizations']['relationship_map'])
            else:
                st.error(f"Simulation failed: {results['error']}")
        else:
            st.info("No simulation results yet. Please go to the Input Data tab and run a simulation.")
    
    # Tab 3: Challenges
    with tabs[2]:
        if 'simulation_results' in st.session_state:
            results = st.session_state.simulation_results
            
            if 'error' not in results and 'potential_challenges' in results:
                challenges = results['potential_challenges']
                
                st.header("Potential Will Challenges")
                
                # Overall challenge risk
                risk_color = {
                    'Very High': 'red',
                    'High': 'orange',
                    'Medium': 'yellow',
                    'Low': 'lightgreen',
                    'Very Low': 'green'
                }.get(challenges.get('overall_risk', 'Unknown'), 'gray')
                
                st.markdown(f"<h3 style='color: {risk_color}'>Overall Risk: {challenges.get('overall_risk', 'Unknown')}</h3>", unsafe_allow_html=True)
                
                # Challenge counts
                if 'challenge_type_counts' in challenges and challenges['challenge_type_counts']:
                    st.subheader("Challenge Types")
                    counts = challenges['challenge_type_counts']
                    fig = px.bar(
                        x=list(counts.keys()),
                        y=list(counts.values()),
                        labels={'x': 'Challenge Type', 'y': 'Count'},
                        title="Challenge Types Count"
                    )
                    st.plotly_chart(fig)
                
                # Legal Heirs
                st.subheader("Legal Heirs")
                if 'legal_heirs' in challenges and challenges['legal_heirs']:
                    for heir in challenges['legal_heirs']:
                        st.write(f"• {heir['name']} ({heir['relation']})")
                else:
                    st.write("No legal heirs identified.")
                
                # Excluded Legal Heirs
                if 'excluded_legal_heirs' in challenges and challenges['excluded_legal_heirs']:
                    st.subheader("Excluded Legal Heirs")
                    st.warning("The following legal heirs have been excluded from the will:")
                    for heir in challenges['excluded_legal_heirs']:
                        if heir['name']:
                            st.write(f"• {heir['name']} ({heir['relation']})")
                        else:
                            st.write(f"• {heir['relation']}")
                
                # Detailed challenges
                if 'challenges' in challenges and challenges['challenges']:
                    st.subheader("Detailed Challenges")
                    
                    for challenge in challenges['challenges']:
                        # Set container color based on likelihood
                        likelihood = challenge.get('likelihood', 'Medium')
                        container_func = {
                            'Very High': st.error,
                            'High': st.error,
                            'Medium': st.warning,
                            'Low': st.info
                        }.get(likelihood, st.info)
                        
                        with container_func(f"**{challenge.get('reason', 'Unknown Challenge')}** (Likelihood: {likelihood})"):
                            st.write(challenge.get('description', ''))
                            
                            if 'potential_challengers' in challenge:
                                st.write("**Potential challengers:**")
                                for challenger in challenge['potential_challengers']:
                                    st.write(f"• {challenger}")
                            
                            if 'mitigation' in challenge:
                                st.write(f"**Mitigation:** {challenge['mitigation']}")
            else:
                st.info("No challenge analysis available. Please run a simulation first.")
        else:
            st.info("No simulation results yet. Please go to the Input Data tab and run a simulation.")
    
    # Tab 4: Visualizations
    with tabs[3]:
        if 'simulation_results' in st.session_state:
            results = st.session_state.simulation_results
            
            if 'error' not in results and 'visualizations' in results:
                visualizations = results['visualizations']
                
                st.header("Visualizations")
                
                if visualizations:
                    # Beneficiary Distribution
                    if 'beneficiary_distribution' in visualizations:
                        st.subheader("Asset Distribution by Beneficiary")
                        display_image(visualizations['beneficiary_distribution'])
                    
                    # Asset Type Distribution
                    if 'asset_type_distribution' in visualizations:
                        st.subheader("Distribution by Asset Type")
                        display_image(visualizations['asset_type_distribution'])
                    
                    # Relationship Map
                    if 'relationship_map' in visualizations:
                        st.subheader("Beneficiary Relationship Map")
                        display_image(visualizations['relationship_map'])
                else:
                    st.info("No visualizations available for this simulation.")
            else:
                st.info("No visualizations available. Please run a simulation first.")
        else:
            st.info("No simulation results yet. Please go to the Input Data tab and run a simulation.")
    
    # Tab 5: Recommendations
    with tabs[4]:
        if 'simulation_results' in st.session_state:
            results = st.session_state.simulation_results
            
            if 'error' not in results and 'documentation_recommendations' in results:
                recommendations = results['documentation_recommendations']
                
                st.header("Documentation Recommendations")
                
                if recommendations:
                    # Group recommendations by type
                    rec_types = {
                        'essential': "Essential Requirements",
                        'challenge_mitigation': "Challenge Mitigation",
                        'asset_specific': "Asset-Specific Recommendations",
                        'procedural': "Procedural Requirements",
                        'age_related': "Age-Related Considerations"
                    }
                    
                    for rec_type, title in rec_types.items():
                        type_recs = [rec for rec in recommendations if rec.get('type') == rec_type]
                        
                        if type_recs:
                            st.subheader(title)
                            
                            for rec in type_recs:
                                # Set container color based on importance
                                importance = rec.get('importance', 'Medium')
                                container_func = {
                                    'Critical': st.error,
                                    'High': st.warning,
                                    'Medium': st.info,
                                    'Low': st.success
                                }.get(importance, st.info)
                                
                                with container_func(f"**{rec.get('title', 'Recommendation')}** (Importance: {importance})"):
                                    st.write(rec.get('description', ''))
                                    if 'details' in rec:
                                        st.write(f"**Details:** {rec['details']}")
                else:
                    st.info("No recommendations available for this simulation.")
            else:
                st.info("No recommendations available. Please run a simulation first.")
        else:
            st.info("No simulation results yet. Please go to the Input Data tab and run a simulation.")
    
    # Tab 6: Intestate Comparison
    with tabs[5]:
        if 'simulation_results' in st.session_state:
            results = st.session_state.simulation_results
            
            if 'error' not in results and 'intestate_scenario' in results:
                intestate = results['intestate_scenario']
                
                st.header("Intestate Succession Comparison")
                
                st.write(f"**Applicable Law:** {intestate.get('law_name', 'Unknown')}")
                st.write(f"**Intestate Rule:** {intestate.get('intestate_rule', 'Unknown')}")
                
                # Distribution details
                if 'distribution_details' in intestate and intestate['distribution_details']:
                    st.subheader("Distribution Under Intestate Succession")
                    
                    # Create dataframe for display
                    intestate_dist = intestate['distribution_details']
                    df_intestate = pd.DataFrame(intestate_dist)
                    
                    # Create pie chart
                    df_intestate['rel_name'] = df_intestate.apply(lambda x: f"{x['name']} ({x['relation']})", axis=1)
                    fig = px.pie(
                        df_intestate, 
                        values='share_value', 
                        names='rel_name',
                        title="Intestate Distribution",
                        hole=0.4
                    )
                    st.plotly_chart(fig)
                    
                    # Show intestate distribution table
                    st.write("Intestate Distribution Details:")
                    display_df = df_intestate[['name', 'relation', 'share_percentage', 'share_value']]
                    display_df = display_df.rename(columns={
                        'name': 'Name',
                        'relation': 'Relation',
                        'share_percentage': 'Percentage (%)',
                        'share_value': 'Value (₹)'
                    })
                    st.dataframe(display_df)
                    
                    # Compare with will distribution
                    if 'distribution_analysis' in results and 'beneficiary_details' in results['distribution_analysis']:
                        st.subheader("Will vs. Intestate Comparison")
                        
                        # Get will distribution
                        will_dist = results['distribution_analysis']['beneficiary_details']
                        
                        # Create comparison dataframe
                        comparison_data = []
                        all_beneficiaries = set()
                        
                        # Add will distribution data
                        will_map = {}
                        for ben in will_dist:
                            key = f"{ben['name']}|{ben['relation']}"
                            all_beneficiaries.add(key)
                            will_map[key] = {
                                'name': ben['name'],
                                'relation': ben['relation'],
                                'will_percentage': ben['percentage'],
                                'will_value': ben['value']
                            }
                        
                        # Add intestate distribution data
                        intestate_map = {}
                        for ben in intestate_dist:
                            key = f"{ben['name']}|{ben['relation']}"
                            all_beneficiaries.add(key)
                            intestate_map[key] = {
                                'name': ben['name'],
                                'relation': ben['relation'],
                                'intestate_percentage': ben['share_percentage'],
                                'intestate_value': ben['share_value']
                            }
                        
                        # Combine data
                        for key in all_beneficiaries:
                            entry = {}
                            if key in will_map:
                                entry.update(will_map[key])
                            else:
                                name, relation = key.split('|')
                                entry['name'] = name
                                entry['relation'] = relation
                                entry['will_percentage'] = 0
                                entry['will_value'] = 0
                            
                            if key in intestate_map:
                                entry.update({
                                    'intestate_percentage': intestate_map[key]['intestate_percentage'],
                                    'intestate_value': intestate_map[key]['intestate_value']
                                })
                            else:
                                entry['intestate_percentage'] = 0
                                entry['intestate_value'] = 0
                            
                            # Calculate difference
                            entry['value_difference'] = entry['will_value'] - entry['intestate_value']
                            entry['percentage_difference'] = entry['will_percentage'] - entry['intestate_percentage']
                            
                            comparison_data.append(entry)
                        
                        # Create comparison dataframe
                        df_comparison = pd.DataFrame(comparison_data)
                        
                        # Create bar chart comparison
                        fig = go.Figure()
                        
                        fig.add_trace(go.Bar(
                            x=df_comparison['name'],
                            y=df_comparison['will_percentage'],
                            name='Will Distribution',
                            marker_color='blue'
                        ))
                        
                        fig.add_trace(go.Bar(
                            x=df_comparison['name'],
                            y=df_comparison['intestate_percentage'],
                            name='Intestate Distribution',
                            marker_color='red'
                        ))
                        
                        fig.update_layout(
                            title='Will vs Intestate Percentage Comparison',
                            xaxis_title='Beneficiary',
                            yaxis_title='Percentage (%)',
                            barmode='group'
                        )
                        
                        st.plotly_chart(fig)
                        
                        # Show comparison table
                        st.write("Detailed Comparison:")
                        display_df = df_comparison[[
                            'name', 'relation', 
                            'will_percentage', 'intestate_percentage', 'percentage_difference',
                            'will_value', 'intestate_value', 'value_difference'
                        ]]
                        display_df = display_df.rename(columns={
                            'name': 'Name',
                            'relation': 'Relation',
                            'will_percentage': 'Will %',
                            'intestate_percentage': 'Intestate %',
                            'percentage_difference': 'Diff %',
                            'will_value': 'Will Value (₹)',
                            'intestate_value': 'Intestate Value (₹)',
                            'value_difference': 'Value Diff (₹)'
                        })
                        st.dataframe(display_df)
                else:
                    st.info("No intestate distribution details available.")
            else:
                st.info("No intestate scenario available. Please run a simulation first.")
        else:
            st.info("No simulation results yet. Please go to the Input Data tab and run a simulation.")
    
        # Generate report option
    st.markdown("---")
    if 'simulation_results' in st.session_state and 'error' not in st.session_state.simulation_results:
        st.subheader("Generate Report")
        col1, col2 = st.columns(2)
        with col1:
            # Option to download as JSON
            json_str = json.dumps(st.session_state.simulation_results, indent=2, ensure_ascii=False)
            st.download_button(
                "Download Results (JSON)",
                json_str,
                file_name=f"will_simulation_{datetime.now().strftime('%Y%m%d_%H%M')}.json",
                mime="application/json"
            )
        
        with col2:
            # Option to download as PDF
            pdf_report = generate_pdf_report(st.session_state.simulation_results)
            st.download_button(
                "Download Complete Report (PDF)",
                pdf_report,
                file_name=f"will_simulation_{datetime.now().strftime('%Y%m%d_%H%M')}.pdf",
                mime="application/pdf"
            )
            
        #TODO: Implement PDF report generation
        
if __name__ == "__main__":
    create_will_simulator_ui()