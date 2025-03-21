#!/usr/bin/env python
# coding: utf-8

# In[ ]:


print("Python Settings (Ok)")


# In[ ]:


import os
import sys
import re
import uuid
import glob
import re
import json
from collections import defaultdict
from git import Repo
from langchain.text_splitter import Language
from langchain_community.document_loaders.generic import GenericLoader
from langchain_community.document_loaders.parsers.language.language_parser import LanguageParser
from langchain.text_splitter import RecursiveCharacterTextSplitter


# In[4]:


if len(sys.argv) < 2:
    print("Error: Please provide a GitHub repository URL as an argument.")
    sys.exit(1)

github_url = sys.argv[1]
print(f"GitHub URL: {github_url}")
repo_folder = f"test_repo_{uuid.uuid4().hex[:8]}"  # Creates a name like "test_repo_a1b2c3d4"
os.makedirs(repo_folder, exist_ok=True)


# In[5]:


repo_path = f"{repo_folder}/"
repo = Repo.clone_from(github_url, to_path=repo_path)


# In[6]:


loader = GenericLoader.from_filesystem(repo_path,
                                        glob = "**/*",
                                       suffixes=[".cs", ".java"],
                                       parser = LanguageParser(language=Language.PYTHON, parser_threshold=500)
)


# In[ ]:


documents = loader.load()
# Assuming you've already loaded documents with:
# documents = loader.load()

# Convert documents to text files
def save_documents_as_txt(documents, output_dir="extracted_texts"):
    """
    Save loaded documents as text files in the specified directory
    """
    import os

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    print(f"Saving {len(documents)} documents to {output_dir}/")

    for i, doc in enumerate(documents):
        # Create a safe filename from the source or use an index
        if hasattr(doc, 'metadata') and 'source' in doc.metadata:
            # Extract filename from source path
            source = doc.metadata['source']
            filename = os.path.basename(source).replace('.', '_')
        else:
            filename = f"document_{i}"

        # Ensure filename is unique
        filepath = os.path.join(output_dir, f"{filename}.txt")
        counter = 1
        while os.path.exists(filepath):
            filepath = os.path.join(output_dir, f"{filename}_{counter}.txt")
            counter += 1

        # Write document content to file
        with open(filepath, 'w', encoding='utf-8') as f:
            if hasattr(doc, 'page_content'):
                f.write(doc.page_content)
            else:
                f.write(str(doc))

    print(f"Successfully saved {len(documents)} documents to {output_dir}/")
    return output_dir

# Call the function to save documents as text files
output_directory = save_documents_as_txt(documents)

print(f"Documents converted to text files in: {output_directory}")


# In[ ]:


def consolidate_text_files(input_dir="extracted_texts", output_file="consolidated_repo.txt"):
    """
    Consolidate all text files from extracted_texts folder (including subdirectories)
    into a single text file.
    """
    # Ensure the input directory exists
    if not os.path.exists(input_dir):
        print(f"Error: Directory '{input_dir}' does not exist.")
        return None

    # Find all text files in the directory and its subdirectories
    text_files = glob.glob(os.path.join(input_dir, "**", "*.txt"), recursive=True)

    if not text_files:
        print(f"No text files found in '{input_dir}'.")
        return None

    print(f"Found {len(text_files)} text files to consolidate.")

    # Open the output file for writing
    with open(output_file, 'w', encoding='utf-8') as outfile:
        # Write a header with repository information
        outfile.write("# Consolidated Repository Content\n")
        outfile.write("Repository: https://uithub.com/ram541619/CustomerOnboardFlow\n\n")

        # Process each text file
        for file_path in sorted(text_files):
            # Get relative path for file identification
            rel_path = os.path.relpath(file_path, input_dir)

            # Write file separator and path information
            outfile.write(f"\n\n{'='*80}\n")
            outfile.write(f"FILE: {rel_path}\n")
            outfile.write(f"{'='*80}\n\n")

            # Read and write the content of the file
            try:
                with open(file_path, 'r', encoding='utf-8', errors='ignore') as infile:
                    content = infile.read()
                    outfile.write(content)
            except Exception as e:
                outfile.write(f"[Error reading file: {str(e)}]")

    print(f"Successfully consolidated all text files into '{output_file}'.")
    return output_file

# Call the function to consolidate all text files
consolidated_file = consolidate_text_files()

print(f"All repository content has been consolidated into: {consolidated_file}")


# In[ ]:


def analyze_consolidated_repo(file_path="consolidated_repo.txt"):
    """
    Analyze the consolidated repository file to extract:
    1. API endpoints
    2. Request/response body patterns
    3. Framework indicators
    """
    # Framework detection patterns
    framework_patterns = {
        'react': [r'import\s+React', r'from\s+[\'"]react[\'"]', r'ReactDOM', r'<.*Component'],
        'angular': [r'@Component', r'@NgModule', r'import.*@angular/core'],
        'vue': [r'new\s+Vue', r'createApp', r'Vue\.component'],
        'express': [r'express\(\)', r'app\.use\(', r'app\.(get|post|put|delete)'],
        'django': [r'from\s+django', r'urlpatterns', r'django\.urls'],
        'flask': [r'from\s+flask\s+import', r'app\s*=\s*Flask', r'@app\.route'],
        'spring': [r'@RestController', r'@RequestMapping', r'@SpringBootApplication'],
        'aspnet': [r'using\s+Microsoft\.AspNetCore', r'\[ApiController\]', r'\[Route\('],
        'laravel': [r'use\s+Illuminate', r'extends\s+Controller', r'Route::'],
    }

    # Endpoint detection patterns
    endpoint_patterns = {
        'rest_api': r'@(GetMapping|PostMapping|PutMapping|DeleteMapping|RequestMapping)\([\'"](.+?)[\'"]\)',
        'express': r'(app|router)\.(get|post|put|delete|patch)\([\'"](.+?)[\'"]\s*,',
        'flask': r'@app\.route\([\'"](.+?)[\'"]\)',
        'django': r'path\([\'"](.+?)[\'"]\s*,',
        'generic_url': r'(https?://[^\s\'"]+)',
        'swagger': r'paths:\s*\n(\s+/[^\n]+)',
        'openapi': r'[\'"]/[^\'"}]*[\'"]:\s*{',
    }

    # Request/response body patterns
    body_patterns = {
        'json_schema': r'schema\s*:\s*{([^}]+)}',
        'request_body': r'requestBody\s*:\s*{([^}]+)}',
        'response_body': r'responses\s*:\s*{([^}]+)}',
        'typescript_interface': r'interface\s+(\w+)\s*{([^}]+)}',
        'class_definition': r'class\s+(\w+).*{([^}]+)}',
    }

    # Results containers
    results = {
        'framework_scores': defaultdict(int),
        'endpoints': [],
        'request_response_models': []
    }

    # Current file being processed
    current_file = ""

    try:
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()

            # Split content by file markers
            file_sections = re.split(r'={80}\nFILE:\s*(.+?)\n={80}', content)

            # Process each file section
            for i in range(1, len(file_sections), 2):
                if i < len(file_sections):
                    current_file = file_sections[i]
                    file_content = file_sections[i+1] if i+1 < len(file_sections) else ""

                    # Detect framework indicators
                    for framework, patterns in framework_patterns.items():
                        for pattern in patterns:
                            matches = re.findall(pattern, file_content)
                            results['framework_scores'][framework] += len(matches)

                    # Detect endpoints
                    file_endpoints = []
                    for pattern_name, pattern in endpoint_patterns.items():
                        matches = re.findall(pattern, file_content)
                        if matches:
                            if isinstance(matches[0], tuple):
                                # Handle tuple results (like from express pattern)
                                for match in matches:
                                    # Get the last item in the tuple which is typically the endpoint
                                    endpoint = match[-1]
                                    http_method = match[1] if len(match) > 2 else "unknown"
                                    file_endpoints.append({
                                        'endpoint': endpoint,
                                        'method': http_method,
                                        'pattern_type': pattern_name
                                    })
                            else:
                                # Handle string results
                                for match in matches:
                                    file_endpoints.append({
                                        'endpoint': match,
                                        'method': "unknown",
                                        'pattern_type': pattern_name
                                    })

                    if file_endpoints:
                        results['endpoints'].append({
                            'file': current_file,
                            'endpoints': file_endpoints
                        })

                    # Detect request/response models
                    for pattern_name, pattern in body_patterns.items():
                        matches = re.findall(pattern, file_content)
                        if matches:
                            for match in matches:
                                if isinstance(match, tuple):
                                    name = match[0]
                                    body = match[1]
                                else:
                                    name = pattern_name
                                    body = match

                                results['request_response_models'].append({
                                    'file': current_file,
                                    'type': pattern_name,
                                    'name': name,
                                    'content': body.strip()
                                })

        # Determine the most likely framework
        if results['framework_scores']:
            most_likely_framework = max(results['framework_scores'].items(), key=lambda x: x[1])
            results['detected_framework'] = {
                'name': most_likely_framework[0],
                'confidence': most_likely_framework[1]
            }
        else:
            results['detected_framework'] = {
                'name': 'unknown',
                'confidence': 0
            }

        return results

    except Exception as e:
        print(f"Error analyzing repository: {str(e)}")
        return None

# Analyze the consolidated repository
analysis_results = analyze_consolidated_repo()

# Print summary of results
if analysis_results:
    print(f"Detected Framework: {analysis_results['detected_framework']['name']} (confidence: {analysis_results['detected_framework']['confidence']})")
    print(f"Found {sum(len(item['endpoints']) for item in analysis_results['endpoints'])} endpoints across {len(analysis_results['endpoints'])} files")
    print(f"Found {len(analysis_results['request_response_models'])} potential request/response models")

    # Save detailed results to JSON
    with open('api_analysis_results.json', 'w') as f:
        json.dump(analysis_results, f, indent=2)
    print("Detailed results saved to api_analysis_results.json")


# In[10]:


from langchain.llms import LlamaCpp
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
import json
import os

def analyze_with_llama(consolidated_file="consolidated_repo.txt", llama_model_path="models/llama-2-7b-chat.gguf"):
    """
    Use LLaMA to analyze the codebase for endpoints, request/response bodies, and framework detection
    """
    # Check if LLaMA model exists
    if not os.path.exists(llama_model_path):
        print(f"LLaMA model not found at {llama_model_path}. Please download it first.")
        return None

    # Initialize LLaMA model
    llm = LlamaCpp(
        model_path=llama_model_path,
        temperature=0.1,
        max_tokens=2000,
        top_p=0.95,
        n_ctx=4096  # Context window size
    )

    # Read the consolidated file in chunks due to context window limitations
    with open(consolidated_file, 'r', encoding='utf-8', errors='ignore') as f:
        content = f.read()

    # Split content into manageable chunks (e.g., by file)
    file_sections = re.split(r'={80}\nFILE:\s*(.+?)\n={80}', content)

    # Prepare results container
    llama_results = {
        'framework_analysis': {},
        'endpoints': [],
        'data_models': []
    }

    # Process each file with LLaMA
    for i in range(1, len(file_sections), 2):
        if i < len(file_sections):
            current_file = file_sections[i]
            file_content = file_sections[i+1] if i+1 < len(file_sections) else ""

            # Skip if file content is too large
            if len(file_content) > 3500:
                print(f"Skipping {current_file} - too large for context window")
                continue

            # Create prompt for framework detection
            framework_prompt = PromptTemplate(
                input_variables=["code"],
                template="""
                Analyze this code and identify the web framework being used. 
                Focus on imports, decorators, and patterns specific to frameworks like React, Angular, Vue, Express, Django, Flask, Spring, ASP.NET, or Laravel.

                Code:
                ```
                {code}
                ```

                Respond with a JSON object with these fields:
                1. "framework": the name of the detected framework
                2. "confidence": a number from 0-100 indicating confidence
                3. "evidence": key patterns that led to this conclusion
                """
            )

            # Create prompt for endpoint detection
            endpoint_prompt = PromptTemplate(
                input_variables=["code", "filename"],
                template="""
                Analyze this code and extract all API endpoints. 
                Look for route definitions, API controllers, and endpoint handlers.

                Filename: {filename}

                Code:
                ```
                {code}
                ```

                Respond with a JSON array of endpoints, where each endpoint has:
                1. "path": the URL path
                2. "method": HTTP method (GET, POST, etc.)
                3. "description": brief description of what the endpoint does
                4. "request_body": description of request parameters (if identifiable)
                5. "response_body": description of response format (if identifiable)
                """
            )

            # Execute LLaMA for framework detection
            try:
                framework_chain = LLMChain(llm=llm, prompt=framework_prompt)
                framework_response = framework_chain.run(code=file_content[:3500])

                # Parse JSON response
                try:
                    framework_data = json.loads(framework_response)
                    llama_results['framework_analysis'][current_file] = framework_data
                except json.JSONDecodeError:
                    print(f"Failed to parse framework analysis for {current_file}")
            except Exception as e:
                print(f"Error in framework analysis for {current_file}: {str(e)}")

            # Execute LLaMA for endpoint detection
            try:
                endpoint_chain = LLMChain(llm=llm, prompt=endpoint_prompt)
                endpoint_response = endpoint_chain.run(code=file_content[:3500], filename=current_file)

                # Parse JSON response
                try:
                    endpoint_data = json.loads(endpoint_response)
                    if endpoint_data:
                        llama_results['endpoints'].append({
                            'file': current_file,
                            'endpoints': endpoint_data
                        })
                except json.JSONDecodeError:
                    print(f"Failed to parse endpoint analysis for {current_file}")
            except Exception as e:
                print(f"Error in endpoint analysis for {current_file}: {str(e)}")

    # Determine overall framework based on individual file analyses
    framework_votes = defaultdict(int)
    for file_analysis in llama_results['framework_analysis'].values():
        if 'framework' in file_analysis and file_analysis['framework'] != 'unknown':
            framework_votes[file_analysis['framework']] += file_analysis.get('confidence', 50)

    if framework_votes:
        most_likely_framework = max(framework_votes.items(), key=lambda x: x[1])
        llama_results['detected_framework'] = {
            'name': most_likely_framework[0],
            'confidence': most_likely_framework[1] / sum(framework_votes.values()) * 100
        }
    else:
        llama_results['detected_framework'] = {
            'name': 'unknown',
            'confidence': 0
        }

    # Save results to JSON
    with open('llama_api_analysis.json', 'w') as f:
        json.dump(llama_results, f, indent=2)

    print(f"LLaMA analysis complete. Detected framework: {llama_results['detected_framework']['name']}")
    print(f"Found {sum(len(item['endpoints']) for item in llama_results['endpoints'])} endpoints")
    print("Detailed results saved to llama_api_analysis.json")

    return llama_results

# Uncomment to run LLaMA analysis (requires LLaMA model)
# llama_analysis = analyze_with_llama()


# In[11]:


def comprehensive_api_analysis(consolidated_file="consolidated_repo.txt", use_llama=False, llama_model_path="models/llama-2-7b-chat.gguf"):
    """
    Perform comprehensive API analysis using both pattern matching and optionally LLaMA
    """
    # Run pattern-based analysis
    pattern_results = analyze_consolidated_repo(consolidated_file)

    # Run LLaMA analysis if requested
    llama_results = None
    if use_llama:
        llama_results = analyze_with_llama(consolidated_file, llama_model_path)

    # Combine results
    combined_results = {
        'pattern_analysis': pattern_results,
        'llama_analysis': llama_results if llama_results else "Not performed"
    }

    # Determine final framework
    if llama_results and llama_results['detected_framework']['confidence'] > 50:
        final_framework = llama_results['detected_framework']['name']
    else:
        final_framework = pattern_results['detected_framework']['name']

    combined_results['final_framework'] = final_framework

    # Save combined results
    with open('combined_api_analysis.json', 'w') as f:
        json.dump(combined_results, f, indent=2)

    print(f"Comprehensive analysis complete.")
    print(f"Detected framework: {final_framework}")
    print("Detailed results save")

