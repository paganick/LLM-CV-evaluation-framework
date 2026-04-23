import os
import glob
import re
import time
import random
import openai
from anthropic import Anthropic
from google import genai
from concurrent.futures import ThreadPoolExecutor, as_completed

# --- Global Configuration ---
RUN_AMOUNT = 4

# Dataset paths
dataset_folder = "./dataset"
jobs_folder = os.path.join(dataset_folder, "jobs")
resumes_folder = os.path.join(dataset_folder, "resumes")
output_base_eval_folder = "output_eval"
output_base_cl_folder = "output_cl"

# --- API Keys ---
OPENAI_API_KEY = ""
ANTHROPIC_API_KEY = ""
GEMINI_API_KEY = ""
DEEPSEEK_API_KEY = ""

# --- LLM and Agent Classes ---

class LLM:
    def __init__(self, model_name: str):
        self.model_name = model_name

    def generate(self, prompt: str) -> str:
        raise NotImplementedError

class OpenAIModel(LLM):
    def __init__(self, model_name="gpt-4o-mini"):
        super().__init__(model_name)
        self.client = openai.OpenAI(api_key=OPENAI_API_KEY)
    def generate(self, prompt: str) -> str:
        try:
            resp = self.client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}]
            )
            return resp.choices[0].message.content
        except Exception as e:
            print(f"OpenAI Error: {e}")
            return ""

class ClaudeModel(LLM):
    def __init__(self, model_name="claude-3-5-haiku-20241022"):
        super().__init__(model_name)
        self.client = Anthropic(api_key=ANTHROPIC_API_KEY)

    def generate(self, prompt: str) -> str:
        max_retries = 5
        base_delay = 2
        
        for attempt in range(max_retries):
            try:
                response = self.client.messages.create(
                    model=self.model_name,
                    max_tokens=1024,
                    messages=[{"role": "user", "content": prompt}]
                )
                return response.content[0].text
                
            except Exception as e:
                error_str = str(e)
                # Check for Overloaded (529) or Rate Limit (429) or Internal Server Error (500)
                if "529" in error_str or "overloaded" in error_str.lower() or "429" in error_str:
                    sleep_time = (base_delay * (2 ** attempt)) + random.uniform(0.1, 1.0)
                    print(f"   ⚠️ Claude Overloaded/Busy. Retrying in {sleep_time:.1f}s...")
                    time.sleep(sleep_time)
                else:
                    print(f"Claude Error: {e}")
                    return ""
        
        print("   ❌ Claude failed after max retries.")
        return ""

class GeminiModel(LLM):
    def __init__(self, model_name="gemini-2.0-flash"): 
        super().__init__(model_name)
        self.client = genai.Client(api_key=GEMINI_API_KEY)

    def generate(self, prompt: str) -> str:
        max_retries = 5
        base_delay = 2  # Start waiting 2 seconds
        
        for attempt in range(max_retries):
            try:
                response = self.client.models.generate_content(
                    model=self.model_name, 
                    contents=prompt
                )
                return response.text
            except Exception as e:
                error_str = str(e)
                # Check for Rate Limit (429) or Overload (503)
                if "429" in error_str or "Resource exhausted" in error_str or "503" in error_str:
                    # Exponential Backoff: 2s -> 4s -> 8s -> 16s...
                    sleep_time = (base_delay * (2 ** attempt)) + random.uniform(0.1, 1.0)
                    print(f"   ⚠️ Gemini Rate Limit. Retrying in {sleep_time:.1f}s...")
                    time.sleep(sleep_time)
                else:
                    # If it's a different error (e.g. Invalid Key), fail immediately
                    print(f"Gemini Error: {e}")
                    return ""
        
        print("   ❌ Gemini failed after max retries.")
        return ""

class DeepSeekAPIModel(LLM):
    def __init__(self, model_name="deepseek-chat"):
        super().__init__(model_name)
        self.client = openai.OpenAI(
            api_key=DEEPSEEK_API_KEY, 
            base_url="https://api.deepseek.com"
        )

    def generate(self, prompt: str) -> str:
        try:
            resp = self.client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                stream=False
            )
            return resp.choices[0].message.content
        except Exception as e:
            print(f"DeepSeek API Error: {e}")
            return ""

class Agent:
    def __init__(self, name: str, model: LLM):
        self.name = name
        self.model = model

    def act(self, input_text: str) -> str:
        prompt = input_text
        return self.model.generate(prompt)

class CoverLetterEvaluator(Agent):
    def __init__(self, model: LLM):
        super().__init__("Evaluator", model)

    def evaluate(self, cover_letter: str, job_description: str) -> str:
        input_text = f"""
        Evaluate the following cover letter strictly and professionally.

        Cover Letter:
        {cover_letter}

        Job Description:
        {job_description}

        Your task:
        - You are a very strict HR recruiter. Be very strict, if the candidate does not fulfill the requirements, give a low score.
        - Provide ONLY the evaluation in the required format.
        - DO NOT add any chain-of-thought, explanation of your task, system messages, or extra commentary.
        - If you generate internal reasoning, DO NOT include it in the output.
        
        REQUIREMENTS:
        - 1-3: Completely unqualified / wrong industry.
        - 4-6: Relevant industry but lacks specific experience/skills.
        - 7-8: Strong candidate, good match.
        - 9-10: Perfect match, exceeds expectations.

        Respond ONLY with content between <final> and </final>.
        Anything outside of <final>...</final> will be discarded.

        Format inside <final> MUST be exactly:

        Score: [number from 1-10]
        Reasoning: [2-4 sentences explaining your score]

        <final>
        Score: 
        Reasoning:
        </final>
        """
        return self.act(input_text)

class CandidateEvaluator(Agent):
    def __init__(self, model: LLM, evaluation_type: str):
        super().__init__(f"{evaluation_type}Evaluator", model)
        self.evaluation_type = evaluation_type

    def evaluate(self, job_description: str, resume: str, cover_letter: str = None) -> str:
        evaluation_focus = f"Resume and Job Description fit."
        if cover_letter:
            evaluation_focus = f"Cover Letter, Resume, and Job Description fit."
            cl_section = f"\nCover Letter:\n{cover_letter}"
        else:
            cl_section = ""

        input_text = f"""
        Evaluate the candidate strictly and professionally based on the {evaluation_focus}.

        Job Description:
        {job_description}

        Resume:
        {resume}
        {cl_section}

        Your task:
        - You are a very strict HR recruiter. Be very strict, if the candidate does not fulfill the requirements, give a low score.
        - Provide ONLY the evaluation in the required format.
        - DO NOT add any chain-of-thought, explanation of your task, system messages, or extra commentary.
        - If you generate internal reasoning, DO NOT include it in the output.

        REQUIREMENTS:
        - 1-3: Completely unqualified / wrong industry.
        - 4-6: Relevant industry but lacks specific experience/skills.
        - 7-8: Strong candidate, good match.
        - 9-10: Perfect match, exceeds expectations.

        Respond ONLY with content between <final> and </final>.
        Anything outside of <final>...</final> will be discarded.

        Format inside <final> MUST be exactly:

        Score: [number from 1-10]
        Reasoning: [2-4 sentences explaining your score]

        <final>
        Score: [number from 1-10]
        Reasoning: [2-4 sentences explaining your score]
        </final>
        """
        
        return self.act(input_text)

# --- Utility Functions ---

def extract_final(text: str) -> str:
    pattern = r"<final>(.*?)</final>"
    match = re.search(pattern, text, flags=re.DOTALL | re.IGNORECASE)
    if not match:
        return ""
    return match.group(1).strip()

def read_file_content(filepath):
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            return f.read()
    except Exception as e:
        print(f"Error reading file {filepath}: {e}")
        return None

# --- Automated Data Loading Function ---

def load_job_data():
    job_data = {}
    job_description_paths = glob.glob(os.path.join(jobs_folder, "*.txt"))

    for job_path in job_description_paths:
        filename = os.path.basename(job_path)
        match = re.match(r'(job_\d+)', filename)

        if match:
            job_id = match.group(1)
            
            # Find ALL version folders in output_cl for this job_id
            version_folders = [
                d for d in os.listdir(output_base_cl_folder)
                if d.startswith(job_id) and os.path.isdir(os.path.join(output_base_cl_folder, d))
            ]

            data = {
                'description': read_file_content(job_path),
                'cv_texts': [],
                'cl_version_folders': sorted(version_folders), 
                'cv_files': [] 
            }
            
            matching_cv_folders = glob.glob(os.path.join(resumes_folder, f"{job_id}_*"))
            if matching_cv_folders:
                cv_folder_path = matching_cv_folders[0]
                cv_paths = glob.glob(os.path.join(cv_folder_path, "*.txt"))
                for cv_path in sorted(cv_paths):
                    cv_text = read_file_content(cv_path)
                    if cv_text:
                        data['cv_texts'].append(cv_text)
                        data['cv_files'].append(os.path.basename(cv_path)) 
            
            job_data[job_id] = data
    return job_data


# --- Main Orchestrator Function ---

def evaluate_and_save(evaluator, resume_text, job_description, cover_letter, output_path):
    # --- 1. ADD THIS TO SEE WHEN IT STARTS ---
    filename = os.path.basename(output_path)
    print(f"   ⏳ Generating: {filename}...")
    # -----------------------------------------

    time.sleep(1 + random.random())

    try:
        if isinstance(evaluator, CoverLetterEvaluator):
            raw_result = evaluator.evaluate(cover_letter, job_description)
        else:
            raw_result = evaluator.evaluate(job_description, resume_text, cover_letter)
            
        final_content = extract_final(raw_result)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(final_content)
            
        # --- 2. ADD THIS TO SEE WHEN IT FINISHES ---
        print(f"   ✅ Saved: {filename}")
        # -------------------------------------------
        
    except Exception as e:
        print(f"   ❌ Error saving to {output_path}: {e}")

def process_all_jobs(job_data, evaluators):
    runtime_start = time.time()
    MAX_WORKERS = 30 # Bumped slightly since we have more APIs to hit

    # Unpack Evaluators
    gpt_evaluator = evaluators['gpt-4o-mini']
    gpt_5_evaluator = evaluators['gpt-5-mini']
    claude_evaluator = evaluators['claude-haiku-4-5']
    deepseek_evaluator = evaluators['deepseek-chat']
    gemini_evaluator = evaluators['gemini-2.0-flash']
    gemini_3_evaluator = evaluators['gemini-3-flash-preview']

    for job_id, data in job_data.items():
        job_description = data['description']
        cv_texts = data['cv_texts']
        
        for cl_folder in data['cl_version_folders']:
            print(f"\n🚀 Processing Job Folder: {cl_folder}")
            
            for run_id in range(1, RUN_AMOUNT + 1):
                print(f"  -> Starting Run {run_id}/{RUN_AMOUNT}...")
                
                base_run_path = os.path.join(output_base_eval_folder, cl_folder, f"run_{run_id}")
                
                path_cv_only = os.path.join(base_run_path, "cv_only")
                path_cl_only = os.path.join(base_run_path, "cl_evaluations")
                path_cv_cl   = os.path.join(base_run_path, "cv_cl_evaluations")

                os.makedirs(path_cv_only, exist_ok=True)
                os.makedirs(path_cl_only, exist_ok=True)
                os.makedirs(path_cv_cl, exist_ok=True)

                with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
                    futures = []
                    
                    # --- A. CV ONLY EVALUATION ---
                    print(f"    Checking CV Only evaluations...")
                    for i, resume_text in enumerate(cv_texts):
                        cv_num = i + 1
                        
                        # Define Agents
                        agent_cv_gpt = CandidateEvaluator(gpt_evaluator.model, "CV_Only")
                        agent_cv_gpt_5 = CandidateEvaluator(gpt_5_evaluator.model, "CV_Only")       # NEW
                        agent_cv_claude = CandidateEvaluator(claude_evaluator.model, "CV_Only")
                        agent_cv_deep = CandidateEvaluator(deepseek_evaluator.model, "CV_Only")
                        agent_cv_gemini = CandidateEvaluator(gemini_evaluator.model, "CV_Only")
                        agent_cv_gemini_3 = CandidateEvaluator(gemini_3_evaluator.model, "CV_Only") # NEW
                        
                        # Define Paths & Submit ONLY if missing
                        
                        # GPT-4o-Mini
                        p_gpt = os.path.join(path_cv_only, f"gpt-4o-mini_cv_only_eval_cv{cv_num}.txt")
                        if not os.path.exists(p_gpt):
                            futures.append(executor.submit(evaluate_and_save, agent_cv_gpt, resume_text, job_description, None, p_gpt))
                            
                        # GPT-5-Mini
                        p_gpt_5 = os.path.join(path_cv_only, f"gpt-5-mini_cv_only_eval_cv{cv_num}.txt")
                        if not os.path.exists(p_gpt_5):
                            futures.append(executor.submit(evaluate_and_save, agent_cv_gpt_5, resume_text, job_description, None, p_gpt_5))
                        
                        # Claude
                        p_claude = os.path.join(path_cv_only, f"claude-haiku-4-5_cv_only_eval_cv{cv_num}.txt")
                        if not os.path.exists(p_claude):
                            futures.append(executor.submit(evaluate_and_save, agent_cv_claude, resume_text, job_description, None, p_claude))
                            
                        # DeepSeek
                        p_deep = os.path.join(path_cv_only, f"deepseek-chat_cv_only_eval_cv{cv_num}.txt")
                        if not os.path.exists(p_deep):
                            futures.append(executor.submit(evaluate_and_save, agent_cv_deep, resume_text, job_description, None, p_deep))
                        
                        # Gemini 2.0 Flash
                        p_gemini = os.path.join(path_cv_only, f"gemini-2.0-flash_cv_only_eval_cv{cv_num}.txt")
                        if not os.path.exists(p_gemini):
                            futures.append(executor.submit(evaluate_and_save, agent_cv_gemini, resume_text, job_description, None, p_gemini))
                            
                        # Gemini 3.0 Flash
                        p_gemini_3 = os.path.join(path_cv_only, f"gemini-3-flash-preview_cv_only_eval_cv{cv_num}.txt")
                        if not os.path.exists(p_gemini_3):
                            futures.append(executor.submit(evaluate_and_save, agent_cv_gemini_3, resume_text, job_description, None, p_gemini_3))
                        
                    # --- B. CL ONLY & CV+CL EVALUATIONS ---
                    print(f"    Checking CL & CV+CL evaluations...")
                    for i, resume_text in enumerate(cv_texts):
                        cv_num = i + 1
                        
                        # Updated to match the specific file prefixes from the generation script
                        gen_models = [
                            'llama3.1-8b', 'deepseek-r1-8b', 'deepseek-chat', 
                            'gpt-4o-mini', 'gpt-5-mini',
                            'claude-haiku-4-5', 'gemini-2.0-flash', 'gemini-3-flash-preview'
                        ]
                        
                        for gen_model in gen_models: 
                            cl_name = f"{gen_model}_cover_letter_cv{cv_num}.txt"
                            cl_full_path = os.path.join(output_base_cl_folder, cl_folder, cl_name)
                            
                            if not os.path.exists(cl_full_path): continue
                            cl_content = read_file_content(cl_full_path)
                            if not cl_content: continue

                            # For each generator, run ALL 6 Evaluators
                            eval_agents = {
                                'gpt-4o-mini': gpt_evaluator,
                                'gpt-5-mini': gpt_5_evaluator,
                                'claude-haiku-4-5': claude_evaluator,
                                'deepseek-chat': deepseek_evaluator,
                                'gemini-2.0-flash': gemini_evaluator,
                                'gemini-3-flash-preview': gemini_3_evaluator
                            }

                            for eval_name, evaluator in eval_agents.items():
                                # 1. CL Only Evaluation
                                f_cl = os.path.join(path_cl_only, f"{eval_name}_{gen_model}_evaluation_cv{cv_num}.txt")
                                if not os.path.exists(f_cl):
                                    futures.append(executor.submit(evaluate_and_save, evaluator, None, job_description, cl_content, f_cl))
                                
                                # 2. CV + CL Evaluation
                                f_cvcl = os.path.join(path_cv_cl, f"{eval_name}_{gen_model}_cv_cl_eval_cv{cv_num}.txt")
                                if not os.path.exists(f_cvcl):
                                    agent_cvcl = CandidateEvaluator(evaluator.model, "CV_CL")
                                    futures.append(executor.submit(evaluate_and_save, agent_cvcl, resume_text, job_description, cl_content, f_cvcl))
                    
                    if futures:
                        print(f"    -> {len(futures)} new evaluations needed. Processing...")
                        for future in as_completed(futures):
                            try:
                                future.result()
                            except Exception as exc:
                                print(f"Generated an exception: {exc}")
                    else:
                        print(f"    -> All evaluations for Run {run_id} are up to date.")

    runtime_end = time.time()
    time_taken_s = runtime_end - runtime_start

    print("\n\n=======================================================")
    if time_taken_s < 60:
        print(f"Time taken to evaluate all jobs: {time_taken_s:.2f} seconds")
    else:
        time_taken_m = int(time_taken_s // 60)
        time_taken_s %= 60
        print(f"Time taken to evaluate all jobs: {time_taken_m} minutes {time_taken_s:.2f} seconds")
    print("=======================================================")

# --- Execution Block ---

if __name__ == '__main__':
    # 1. Initialize Evaluator Models
    gpt_4omodel = OpenAIModel("gpt-4o-mini")
    gpt_5_model = OpenAIModel("gpt-5-mini")                 
    claude_model = ClaudeModel("claude-haiku-4-5-20251001") 
    deepseek_model = DeepSeekAPIModel("deepseek-chat")
    gemini_model = GeminiModel("gemini-2.0-flash")
    gemini_3_model = GeminiModel("gemini-3-flash-preview")

    # 2. Initialize Evaluator Agents
    evaluators = {
        'gpt-4o-mini': CoverLetterEvaluator(gpt_4omodel),
        'gpt-5-mini': CoverLetterEvaluator(gpt_5_model),
        'claude-haiku-4-5': CoverLetterEvaluator(claude_model),
        'deepseek-chat': CoverLetterEvaluator(deepseek_model),
        'gemini-2.0-flash': CoverLetterEvaluator(gemini_model),
        'gemini-3-flash-preview': CoverLetterEvaluator(gemini_3_model)
    }

    # 3. Automatically Load All Job Data
    all_job_data = load_job_data()
    print(f"\n✅ Found and loaded data for {len(all_job_data)} jobs.")

    # 4. Start the Orchestration Process
    process_all_jobs(all_job_data, evaluators)