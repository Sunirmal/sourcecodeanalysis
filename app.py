import streamlit as st
import os

# Define Python functions that will process the inputs
def process_combo(combo_input):
    return f"Combo box selected: {combo_input}"

def process_text(text_input):
    return f"Text box input: {text_input}"

def process_folder(files_uploaded):
    if len(files_uploaded) > 0:
        return f"Number of files uploaded: {len(files_uploaded)}"
    else:
        return "No files uploaded."

# Streamlit UI layout
st.set_page_config(page_title="Professional Streamlit App", page_icon=":guardsman:", layout="wide")

# Add custom CSS for a professional look
st.markdown("""
    <style>
        body {
            font-family: 'Helvetica Neue', Helvetica, Arial, sans-serif;
            background-color: #f9f9fb;
            color: #333;
        }
        
        /* Main header styling */
        .stTitle {
            font-size: 36px;
            font-weight: 700;
            text-align: center;
            color: #2C3E50;
        }
        
        /* Input fields styling */
        .stTextInput input, .stSelectbox select, .stTextArea textarea {
            border-radius: 8px;
            padding: 12px 16px;
            font-size: 16px;
            border: 1px solid #ddd;
            background-color: #fff;
            width: 100%;
            box-sizing: border-box;
        }
        
        /* Input fields on focus */
        .stTextInput input:focus, .stSelectbox select:focus, .stTextArea textarea:focus {
            border: 1px solid #4CAF50;
            box-shadow: 0 0 5px rgba(72, 202, 228, 0.3);
        }
        
        /* Button styling */
        .stButton > button {
            background-color: #4CAF50;
            color: white;
            font-size: 16px;
            padding: 12px 24px;
            border-radius: 5px;
            border: none;
            cursor: pointer;
            transition: background-color 0.3s ease-in-out;
        }
        
        .stButton > button:hover {
            background-color: #45a049;
        }
        
        /* Container styling for columns */
        .stColumn {
            margin: 20px 0;
        }

        /* Card-style containers for inputs/outputs */
        .card-container {
            background-color: #ffffff;
            padding: 20px;
            border-radius: 12px;
            box-shadow: 0px 4px 12px rgba(0, 0, 0, 0.1);
            margin-bottom: 20px;
        }
        
        /* Output box within card container */
        .output-box {
            margin-bottom: 15px;
            font-size: 16px;
            line-height: 1.6;
        }
        
        .output-box strong {
            font-weight: bold;
            color: #2C3E50;
        }
        
        /* Result container with border */
        .output-container {
            background-color: #f8f9fa;
            border-radius: 10px;
            padding: 20px;
            border: 2px solid #4CAF50;
            box-shadow: 0 4px 10px rgba(0, 0, 0, 0.1);
        }
        
        /* Professional page background */
        .stContainer {
            background-color: #f9f9fb;
            padding: 40px;
            border-radius: 12px;
            margin-top: 30px;
        }
        
        .stSubheader {
            color: #4CAF50;
            font-size: 24px;
            font-weight: 600;
            margin-bottom: 20px;
        }

    </style>
""", unsafe_allow_html=True)

# Title and Description
st.title("Streamlit Professional Application")
st.markdown("""
    <p style="font-size: 18px; text-align: center; color: #34495E;">
    A professional Streamlit application to demonstrate modern UI/UX elements, 
    file uploads, and input processing with a clean design.
    </p>
""", unsafe_allow_html=True)

# Create two columns: one for inputs (left) and one for outputs (right)
col1, col2 = st.columns([2, 3])  # Adjust the column width ratio

# Column 1: Controls (Left Aligned)
with col1:
    #st.subheader("Input Controls")
    #st.markdown("Please provide your details below:")

    # Use a form to group the inputs and align them properly
    with st.form(key="input_form"):
        # Combo box for selecting an option
        combo_options = ["APP1", "APP2"]
        combo_input = st.selectbox("Select App Id", combo_options)

        # Text box for user input
        repo_input = st.text_input("Enter Repo Url", "https://github.com/ram541619/CustomerOnboardFlow")

         # Text box for user input
        confluence_input = st.text_input("Enter Confluence Url", "https://github.com/ram541619/CustomerOnboardFlow")

        # Folder path input via file uploader (simulate folder selection)
        st.markdown("#### Select files from a folder:")
        files_uploaded = st.file_uploader("Upload Artifact FIles", accept_multiple_files=True, type=["txt", "csv", "pdf"])

        # Submit button inside the form
        submit_button = st.form_submit_button("Process")
        
        # If the button is pressed, process the inputs
        if submit_button:
            combo_result = process_combo(combo_input)
            text_result = process_text(repo_input)
            confluence_result = process_text(confluence_input)            
            folder_result = process_folder(files_uploaded)

# Column 2: Output (Right Aligned)
with col2:
    #st.subheader("Output Results")
    
    # Display the results from the functions inside the same container
    if 'combo_result' in locals():
        with st.container():
            st.markdown(f"<div class='output-container'>"
                        f"<div class='output-box'><strong>Combo Box Result:</strong><p>{combo_result}</p></div>"
                        f"<div class='output-box'><strong>Text Box Result:</strong><p>{text_result}</p></div>"
                        f"<div class='output-box'><strong>Folder Upload Result:</strong><p>{folder_result}</p></div>"
                        f"<div class='output-box'><strong>Confluece Result:</strong><p>{confluence_result}</p></div>"
                        f"</div>", unsafe_allow_html=True)
