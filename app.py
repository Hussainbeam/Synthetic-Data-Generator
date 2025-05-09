from dotenv import load_dotenv
load_dotenv()

import os
openai=os.getenv("OPENAI_API_KEY")
import streamlit as st
import tempfile

import json
from deepeval.synthesizer import Synthesizer
from deepeval.synthesizer.config import StylingConfig
from deepeval.models.llms.openai_model import GPTModel

# Initialize GPT model with API key
gpt_model = GPTModel( _openai_api_key=openai)

st.set_page_config(page_title="DeepEval Synthetic Data Generator", layout="wide")

st.title("🧬 Synthetic Data Generator")

st.markdown("""
This app allows you to generate synthetic data using DeepEval's Synthesizer. You can either:
1. Generate data from scratch using a styling configuration
2. Generate data from uploaded documents
""")

# Sidebar for selection
generation_method = st.sidebar.radio(
    "Choose generation method:",
    ["Generate from styling configuration", "Generate from documents"]
)



if generation_method == "Generate from styling configuration":
    st.header("Generate Synthetic Data from Styling Configuration")
    
    col1, col2 = st.columns(2)
    
    with col1:
        input_format = st.text_area(
            "Input Format", 
            "Mimic queries a patient might share with a medical chatbot when seeking a diagnosis.",
            height=100
        )
        
        expected_output_format = st.text_area(
            "Expected Output Format", 
            "Ensure the output resembles a medical chatbot tasked with diagnosing a patient's illness. It should pose additional questions if details are inadequate or provide a diagnosis when input is sufficiently detailed.",
            height=150
        )
    
    with col2:
        task = st.text_input("Task", "Diagnosing patient symptoms via chatbot")
        scenario = st.text_input("Scenario", "Patients seeking diagnosis through a medical chatbot")
        num_goldens = st.slider("Number of examples to generate", 1, 50, 5)
    
    if st.button("Generate Data"):
        with st.spinner("Generating synthetic data..."):
            try:
                # Define the style for synthetic data
                styling_config = StylingConfig(
                    input_format=input_format,
                    expected_output_format=expected_output_format,
                    task=task,
                    scenario=scenario
                )
                
                # Initialize the Synthesizer with the styling configuration
                synthesizer = Synthesizer(styling_config=styling_config,
                                          model=gpt_model
                                        )
                
                # Generate synthetic goldens from scratch
                synthesizer.generate_goldens_from_scratch(num_goldens=num_goldens)
                
                # Access the generated synthetic data
                synthetic_goldens = synthesizer.synthetic_goldens
                
                # Display the synthetic goldens
                st.subheader("Generated Examples:")
                for i, golden in enumerate(synthetic_goldens):
                    with st.expander(f"Example {i+1}"):
                        st.markdown(f"```\n{golden.input}\n```")
                
                # Create JSON for download
                # Convert the synthetic goldens to a list of dictionaries
                json_data = []
                for golden in synthetic_goldens:
                    json_data.append({
                        "input": golden.input,
                    })
                
                # Convert to JSON string
                json_str = json.dumps(json_data, indent=2)
                
                # Add option to download as JSON
                st.download_button(
                    label="Download as JSON",
                    data=json_str,
                    file_name="synthetic_data.json",
                    mime="application/json"
                )
                
            except Exception as e:
                st.error(f"Error generating synthetic data: {str(e)}")

else:  # Generate from documents
    st.header("Generate Synthetic Data from Documents")
    
    uploaded_files = st.file_uploader("Upload documents (PDF, DOCX, TXT)", accept_multiple_files=True)
    
    if uploaded_files and st.button("Generate Data"):
        with st.spinner("Generating synthetic data from documents..."):
            try:
                with tempfile.TemporaryDirectory() as temp_dir:
                    # Save uploaded files to the temporary directory
                    file_paths = []
                    for uploaded_file in uploaded_files:
                        file_path = os.path.join(temp_dir, uploaded_file.name)
                        with open(file_path, "wb") as f:
                            f.write(uploaded_file.getvalue())
                        file_paths.append(file_path)
                        st.info(f"Processing: {uploaded_file.name}")
                    
                    # Initialize the Synthesizer
                    synthesizer = Synthesizer()
                    
                    # Generate synthetic goldens from documents
                    synthesizer.generate_goldens_from_docs(
                        document_paths=file_paths
                    )
                    
                    # Access the generated synthetic data
                    synthetic_goldens = synthesizer.synthetic_goldens
                    
                    # Display the synthetic goldens
                    st.subheader("Generated Examples:")
                    
                    if not synthetic_goldens:
                        st.warning("No synthetic data was generated. This might happen if the documents couldn't be processed properly.")
                    else:
                        for i, golden in enumerate(synthetic_goldens):
                            with st.expander(f"Example {i+1}"):
                                st.markdown("**Input:**")
                                st.markdown(f"```\n{golden.input}\n```")
                                st.markdown("**Expected Output:**")
                                st.markdown(f"```\n{golden.expected_output}\n```")
                                
                                # Display additional metadata if available
                                if hasattr(golden, 'context') and golden.context:
                                    st.markdown("**Context:**")
                                    st.markdown(f"```\n{golden.context}\n```")
                                
                                if hasattr(golden, 'additional_metadata') and golden.additional_metadata:
                                    st.markdown("**Additional Metadata:**")
                                    st.json(golden.additional_metadata)
                        
                        # Create JSON for download
                        # Convert the synthetic goldens to a list of dictionaries
                        json_data = []
                        for golden in synthetic_goldens:
                            data = {
                                "input": golden.input,
                                "expected_output": golden.expected_output
                            }
                            
                            # Add additional fields if they exist
                            if hasattr(golden, 'context') and golden.context:
                                data["context"] = golden.context
                            if hasattr(golden, 'additional_metadata') and golden.additional_metadata:
                                data["additional_metadata"] = golden.additional_metadata
                            if hasattr(golden, 'source_file') and golden.source_file:
                                data["source_file"] = golden.source_file
                                
                            json_data.append(data)
                        
                        # Convert to JSON string
                        json_str = json.dumps(json_data, indent=2)
                        
                        # Add option to download as JSON
                        st.download_button(
                            label="Download as JSON",
                            data=json_str,
                            file_name="synthetic_data_from_docs.json",
                            mime="application/json"
                        )
                    
            except Exception as e:
                st.error(f"Error generating synthetic data: {str(e)}")
                st.error("Make sure you have the required API key set if needed.")

st.sidebar.markdown("---")
st.sidebar.header("About")
st.sidebar.markdown("""
This application uses [DeepEval's Synthesizer](https://docs.confident-ai.com) to generate synthetic data for AI evaluation.

Key features:
- Generate data from custom style configurations
- Generate data from uploaded documents
- Customize number of examples to generate
- Download results in JSON format
""")

