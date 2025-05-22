import os
import json
import yaml
import openai
import concurrent.futures
from tqdm import tqdm
from queue import Queue
from threading import Lock

DIMENSION_EXPLANATION = {
    'disobey_task_specification': 'Failure to adhere to the speciﬁed constraints or requirements of a given task, leading to suboptimal or incorrect outcomes.',
    'disobey_role_specification': 'Failure to adhere to the deﬁned responsibilities and constraints of an assigned role, potentially leading to an agent behaving like another.',
    'incorrect_verification': 'Failure to adequately validate or cross-check crucial information or decisions during the iterations, potentially leading to errors or vulnerabilities in the system.',
    'step_repetition': 'Unnecessary reiteration of previously completed steps in a process, potentially causing delays or errors in task completion.',
    'no_or_incomplete_verification': '(partial) omission of proper checking or conﬁrmation of task outcomes or system outputs, potentially allowing errors or inconsistencies to propagate undetected.',
    'premature_termination': 'Ending a dialogue, interaction or task before all necessary information has been exchanged or objectives have been met, potentially resulting in incomplete or incorrect outcomes.'
}

# Load configuration
def load_config():
    config_path = os.path.join(os.path.dirname(__file__), 'config.yaml')
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

# Initialize OpenAI API key and base URL
config = load_config()
os.environ["OPENAI_API_KEY"] = config['openai']['api_key']
os.environ["OPENAI_BASE_URL"] = config['openai']['base_url']
openai.api_key = os.environ["OPENAI_API_KEY"]
openai.api_base = os.environ["OPENAI_BASE_URL"]

# Generic detection via OpenAI
def detect_dimension_with_openai(content, context, dimension_key, turn_idx, role_prompt, max_retries=3, retry_delay=1):
    """
    Detects if a specific dimension (failure mode) is present in the content
    given the context, using OpenAI's GPT model.

    Args:
        content (str): The current message to evaluate.
        context (list): A list of previous turns in the conversation.
                        Each turn is a dictionary with "role" and "content".
        dimension_key (str): The key for the dimension to check from DIMENSION_EXPLANATIONS.
        turn_idx (int): The index of the current turn being analyzed.
        max_retries (int): Maximum number of retry attempts for API calls.
        retry_delay (int): Delay in seconds between retries.

    Returns:
        tuple: (None, str) if max retries reached, otherwise (bool, str)
    """
    if not openai.api_key:
        print("OpenAI API key is not set. Please set the OPENAI_API_KEY environment variable.")
        return None, "API key not set"
        
    explanation = DIMENSION_EXPLANATION[dimension_key]
    
    # Keep track of all errors for better error reporting
    errors = []
    
    prompt = (
        f"**Instructions for AI Agent:**\n"
        f"1. You are provided with a 'Conversational Context', a 'Current Message to Evaluate', an 'Error Dimension to Check For' (by its key: '{dimension_key}'), and its 'Explanation of Error Dimension'.\n"
        f"2. Your task is to determine if the 'Current Message to Evaluate', when considered within its 'Conversational Context', *explicitly and unmistakably* exhibits the characteristics defined in the 'Explanation of Error Dimension' for '{dimension_key}'.\n"
        f"3. Base your decision *solely and strictly* on whether the message and context align with the provided 'Explanation of Error Dimension'. Do not infer beyond this explanation. Your evaluation should focus only on the specified dimension, ignoring other potential issues unless they directly relate to this dimension.\n"
        f"4. Output 'true' if this specific error dimension applies. Otherwise, output 'false'. No other output is permitted.\n\n"
        
        f"**Role of the current speaker:**\n{role_prompt}\n\n"
        f"**Conversational Context (last up to 3 turns):**\n{json.dumps(context, indent=2)}\n\n"
        f"**Message of the current speaker:**\n'{content}'\n\n"
        
        f"**Error Dimension to Check For:** {dimension_key}\n"
        f"**Explanation of Error Dimension ({dimension_key}):**\n{explanation}\n\n"
          f"Does the 'Current Message to Evaluate', in light of the 'Conversational Context', demonstrate the error dimension '{dimension_key}' as defined by its explanation above?\n"
        f"First provide your reasoning in one short sentence, then on a new line provide your answer as strictly 'true' or 'false':"
    )

    for attempt in range(max_retries):
        try:
            client = openai.OpenAI(
                api_key=openai.api_key,
                base_url=openai.api_base
            )
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                {
                    "role": "system",
                    "content": (
                    "You are a specialized AI agent focused on error detection in dialogues. "
                    "Your task is to meticulously analyze a 'Current Message to Evaluate' and its 'Conversational Context' "
                    "against a specific 'Error Dimension Explanation'. "
                    "First provide a short, one-sentence reasoning for your decision. "
                    "Then on a new line, output 'true' or 'false' to indicate if the error dimension is present."
                    )
                },
                {"role": "user", "content": prompt}
                ],
                temperature=0.0,
                max_tokens=100
            )
            full_response = response.choices[0].message.content.strip()
            lines = full_response.lower().split('\n')
            
            # Get the reasoning (first line) and decision (last line)
            reasoning = lines[0] if lines else ""
            decision = lines[-1] if lines else ""

            # Validate response format
            if not decision or decision not in ["true", "false"]:
                raise ValueError("Invalid response format from API")
            
            # Return tuple of decision and reasoning
            has_issue = decision == "true"
            return has_issue, reasoning
                
        except Exception as e:
            errors.append(f"Attempt {attempt + 1}: {str(e)}")
            if attempt < max_retries - 1:  # If not the last attempt
                print(f"Attempt {attempt + 1}/{max_retries} failed for {dimension_key}: {str(e)}. Retrying in {retry_delay} seconds...")
                import time
                time.sleep(retry_delay)
                continue
            else:
                error_msg = f"All {max_retries} attempts failed for OpenAI API ({dimension_key}). Errors:\n" + "\n".join(errors)
                print(error_msg)
                # Return None to indicate that this file should be skipped
                return None, error_msg
    
    # This line should never be reached due to the return statements and exception above
    return None, "Unexpected error in API call"


def analyze_dialogue(data, roles_prompt):
    """
    Analyzes the entire dialogue history for all roles.
    
    Args:
        data (dict): The dialogue data containing problem, roles, and history.
        
    Returns:
        dict: Analysis results in the specified format.
        None: If analysis should be skipped due to persistent API failures.
    """
    analysis = {}
    skip_file = False
    
    # Initialize analysis structure for each role
    for role in data['roles'].values():
        analysis[role] = {}
    
    # Process each turn in the dialogue history
    for turn_idx, turn in enumerate(data['history']):
        current_role = turn['name']
        
        # Get context (up to 3 previous turns)
        context = data['history'][max(0, turn_idx-3):turn_idx]
        
        # Initialize turn analysis for all roles
        for role in data['roles'].values():
            turn_key = f'turn{turn_idx + 1}'
            analysis[role][turn_key] = {"target": role == current_role}
            
            # Only perform detailed analysis for the current speaker
            if role == current_role:
                pattern = {}
                reasonings = {}
                role_prompt = roles_prompt[role]
                
                for dimension, explanation in DIMENSION_EXPLANATION.items():
                    # Detect if this dimension is present and get reasoning
                    has_issue, reasoning = detect_dimension_with_openai(turn['content'], context, dimension, turn_idx, role_prompt)
                    
                    # Check if we got an error response (has_issue is None)
                    if has_issue is None:
                        print(f"Skipping file due to persistent API failures in turn {turn_idx + 1}")
                        return None  # Signal to skip this file
                    
                    pattern[dimension] = has_issue
                    reasonings[dimension] = reasoning
                
                analysis[role][turn_key]['pattern'] = pattern
                analysis[role][turn_key]['reasonings'] = reasonings
    
    return {'annotation': analysis}

def process_file(args):
    """Process a single file with the given arguments."""
    file_path, new_file_path, task_type, pbar, pbar_lock = args
    try:
        # Skip if the file already exists in the annotated directory
        if os.path.exists(new_file_path):
            with pbar_lock:
                pbar.write(f"Skipping {file_path} - already exists in annotated")
            return

        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(new_file_path), exist_ok=True)

        # Read the JSON file
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Skip files that already have analysis
        if 'annotation' in data:
            with pbar_lock:
                pbar.write(f"Skipping {file_path} - already annotated")
            return
        
        with pbar_lock:
            pbar.write(f"Processing {file_path}...")
        
        role_prompt_path = f"roles_prompt/roles_prompt_{task_type}.json"
        with open(role_prompt_path, 'r', encoding='utf-8') as f:
            role_prompt = json.load(f)
        # Perform the analysis
        analysis_result = analyze_dialogue(data, role_prompt)
        
        # Check if we should skip this file due to persistent API failures
        if analysis_result is None:
            with pbar_lock:
                pbar.write(f"Skipping {file_path} - persistent API failures")
            return
        
        # Update the JSON with analysis results
        data.update(analysis_result)
        
        # Write to new location in annotated directory
        with open(new_file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        
        with pbar_lock:
            pbar.write(f"Completed annotation for {new_file_path}")
        
    except Exception as e:
        with pbar_lock:
            pbar.write(f"Error processing {file_path}: {str(e)}")
    finally:
        with pbar_lock:
            pbar.update(1)

def main():
    base_path = os.path.dirname(os.path.abspath(__file__))
    processed_dir = os.path.join(base_path, 'processed')
    annotated_dir = os.path.join(base_path, 'annotated')
    
    # Create annotated directory if it doesn't exist
    if not os.path.exists(annotated_dir):
        os.makedirs(annotated_dir)
    
    # Collect all files that need to be processed
    files_to_process = []
    for root, dirs, files in os.walk(processed_dir):
        for file in files:
            if file.endswith('.json'):
                file_path = os.path.join(root, file)
                
                task_type = os.path.relpath(file_path, processed_dir).split(os.sep)[0]
                rel_path = os.path.relpath(root, processed_dir)
                new_dir = os.path.join(annotated_dir, rel_path)
                new_file_path = os.path.join(new_dir, file)
                files_to_process.append((file_path, new_file_path, task_type))
    
    # Create progress bar and lock
    pbar = tqdm(total=len(files_to_process), desc="Processing files")
    pbar_lock = Lock()
    
    # Process files with ThreadPoolExecutor
    with concurrent.futures.ThreadPoolExecutor(max_workers=64) as executor:
        # Add progress bar and lock to each file's arguments
        args = [(fp, nfp, tasktype, pbar, pbar_lock) for fp, nfp, tasktype in files_to_process]
        # Submit all tasks
        executor.map(process_file, args)
    
    pbar.close()

if __name__ == "__main__":
    main()