import openai
from anthropic import Anthropic
from ollama import Client as OllamaClient
from google import genai
import os
import re
import time
import random
import glob
from concurrent.futures import ThreadPoolExecutor, as_completed

# ==========================================
# 1. API KEY CONFIGURATION
# ==========================================
OPENAI_API_KEY = ""
ANTHROPIC_API_KEY = ""
GEMINI_API_KEY = ""
DEEPSEEK_API_KEY = ""

# PARALLELISM SETTINGS
MAX_WORKERS = 25

# ==========================================
# 2. LLM Class Definitions
# ==========================================
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
    def __init__(self, model_name="claude-3-7-sonnet-20250219"):
        super().__init__(model_name)
        self.client = Anthropic(api_key=ANTHROPIC_API_KEY)
    def generate(self, prompt: str) -> str:
        try:
            response = self.client.messages.create(
                model=self.model_name,
                max_tokens=4096,
                messages=[{"role": "user", "content": prompt}]
            )
            return response.content[0].text
        except Exception as e:
            print(f"Claude Error: {e}")
            return ""

class OllamaModel(LLM):
    def __init__(self, model_name="llama3.1:latest"):
        super().__init__(model_name)
        try: self.client = OllamaClient()
        except: self.client = None
    def generate(self, prompt: str) -> str:
        if not self.client: return "Ollama Client Error"
        try:
            resp = self.client.chat(model=self.model_name, messages=[{"role": "user", "content": prompt}])
            return resp["message"]["content"]
        except Exception as e:
            print(f"Ollama Error: {e}")
            return ""

class GeminiModel(LLM):
    def __init__(self, model_name="gemini-2.0-flash"): 
        super().__init__(model_name)
        self.client = genai.Client(api_key=GEMINI_API_KEY)
    def generate(self, prompt: str) -> str:
        max_retries = 5
        base_delay = 2
        for attempt in range(max_retries):
            try:
                response = self.client.models.generate_content(model=self.model_name, contents=prompt)
                return response.text
            except Exception as e:
                error_str = str(e)
                if "429" in error_str or "Resource exhausted" in error_str or "503" in error_str:
                    sleep_time = (base_delay * (2 ** attempt)) + random.uniform(0.1, 1.0)
                    print(f"   ⚠️ Gemini Rate Limit. Retrying in {sleep_time:.1f}s...")
                    time.sleep(sleep_time)
                else:
                    print(f"Gemini Error: {e}")
                    return ""
        return ""

class DeepSeekAPIModel(LLM):
    def __init__(self, model_name="deepseek-chat"):
        super().__init__(model_name)
        self.client = openai.OpenAI(api_key=DEEPSEEK_API_KEY, base_url="https://api.deepseek.com")
    def generate(self, prompt: str) -> str:
        try:
            resp = self.client.chat.completions.create(
                model=self.model_name, messages=[{"role": "user", "content": prompt}], stream=False
            )
            return resp.choices[0].message.content
        except Exception as e:
            print(f"DeepSeek API Error: {e}")
            return ""

# ==========================================
# 3. Agent Definitions
# ==========================================
class Agent:
    def __init__(self, name: str, role: str, model: LLM):
        self.name = name
        self.role = role
        self.model = model
    def act(self, input_text: str) -> str:
        prompt = f"Role: {self.role}\n\nTask: {input_text}"
        return self.model.generate(prompt)

class CoverLetterWriter(Agent):
    def __init__(self, model: LLM):
        role = "You are a professional career coach who writes tailored, persuasive cover letters."
        super().__init__("Writer", role, model)
    def write_letter(self, job_description: str, cv_text: str) -> str:
        random_word_count = random.randint(350, 400)
        input_text = f"""
        Write a {random_word_count} word cover letter for this job description:
        {job_description}

        Using the information given in this CV:
        {cv_text}

        Output: **DO NOT include any additional commentary, signs or symbols.** Make sure that all the information is correct, double check if all the information is accurate with the information in the CV. 
        If you are unsure about any information, leave it out. 
        Make sure the cover letter is professional and concise. 
        Start the cover letter with "Dear Hiring Manager" and end it with "Sincerely, Alexis".
        DO NOT add any contact information.
        """
        return self.act(input_text)

# ==========================================
# 4. Model Initialization
# ==========================================

# --- OpenAI Models ---
gpt_4o_mini_writer_model = OpenAIModel("gpt-4o-mini")
gpt_5_mini_writer_model = OpenAIModel("gpt-5-mini")

# --- Gemini Models ---
gemini_flash_writer_model = GeminiModel("gemini-2.0-flash")
#gemini_pro_writer_model = GeminiModel("gemini-3.1-pro-preview")
gemini_3_flash_writer_model = GeminiModel("gemini-3-flash-preview")

# --- Other Models ---
llama_writer_model = OllamaModel("llama3.1:latest")
deep_local_writer_model = OllamaModel("deepseek-r1:latest")
deep_api_writer_model = DeepSeekAPIModel("deepseek-chat")
claude_writer_model = ClaudeModel("claude-haiku-4-5-20251001")

# --- Create Agents ---
# Dictionary mapping: { "file_prefix": Agent_Instance }
writers = {
    "gemini-2.0-flash_cover_letter": CoverLetterWriter(gemini_flash_writer_model),
    "gemini-3-flash-preview_cover_letter":   CoverLetterWriter(gemini_3_flash_writer_model),
    "llama3.1-8b_cover_letter":         CoverLetterWriter(llama_writer_model),
    "deepseek-r1-8b_cover_letter":      CoverLetterWriter(deep_local_writer_model),
    "deepseek-chat_cover_letter":    CoverLetterWriter(deep_api_writer_model),
    "gpt-4o-mini_cover_letter":      CoverLetterWriter(gpt_4o_mini_writer_model),
    "gpt-5-mini_cover_letter":           CoverLetterWriter(gpt_5_mini_writer_model),
    "claude-haiku-4-5_cover_letter":     CoverLetterWriter(claude_writer_model)
}

# ==========================================
# 5. Parallel Generation Logic
# ==========================================
def process_single_task(task_args):
    """
    Worker function to process a single cover letter generation.
    """
    writer_key, writer_agent, job_desc, cv_text, cv_id_str, output_folder = task_args
    
    filename = f"{writer_key}_{cv_id_str}.txt"
    filepath = os.path.join(output_folder, filename)

    # Check if exists before doing work
    if os.path.exists(filepath):
        return f"Skipped (Exists): {filename}"

    try:
        # Generate
        content = writer_agent.write_letter(job_desc, cv_text)

        # Cleanup DeepSeek thoughts
        if "deepseek" in writer_key:
            content = re.sub(r'<think>.*?</think>', '', content, flags=re.DOTALL).strip()

        # Save
        if content:
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(content)
            return f"✅ Generated: {filename}"
        else:
            return f"⚠️ Empty Content: {filename}"

    except Exception as e:
        return f"❌ Error {filename}: {str(e)}"

def generate_cover_letters_parallel(cv_texts, job_description, output_folder_name):
    output_cl_folder = f"./output_cl/{output_folder_name}"
    os.makedirs(output_cl_folder, exist_ok=True)

    start_time = time.time()
    all_tasks = []

    # Prepare all tasks
    for i in range(len(cv_texts)):
        cv_text = cv_texts[i]
        cv_id_str = f"cv{i+1}"
        
        for writer_key, writer_agent in writers.items():
            # Bundle arguments for the worker
            task = (writer_key, writer_agent, job_description, cv_text, cv_id_str, output_cl_folder)
            all_tasks.append(task)

    print(f"   🚀 Queuing {len(all_tasks)} generation tasks with {MAX_WORKERS} workers...")

    # Execute in parallel
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = [executor.submit(process_single_task, t) for t in all_tasks]
        
        # Monitor progress
        completed = 0
        total = len(futures)
        
        for future in as_completed(futures):
            result = future.result()
            completed += 1
            # Only print every 5th result to reduce clutter, or print errors/successes
            if "Error" in result or "Empty" in result or completed % 5 == 0:
                print(f"   [{completed}/{total}] {result}")

    end_time = time.time()
    print(f"   ⏱️ Batch complete in {end_time - start_time:.2f} seconds.")

# ==========================================
# 6. Data Loading & Execution
# ==========================================
dataset_folder = "./dataset"
jobs_folder = os.path.join(dataset_folder, "jobs")
resumes_folder = os.path.join(dataset_folder, "resumes")
job_data = {}

def read_file_content(filepath):
    try:
        with open(filepath, 'r', encoding='utf-8') as f: return f.read()
    except Exception as e: return None

# Discovery Logic
job_description_paths = glob.glob(os.path.join(jobs_folder, "*.txt"))
for job_path in job_description_paths:
    filename = os.path.basename(job_path)
    match = re.match(r'(job_\d+)', filename)

    if match:
        job_id = match.group(1) 
        job_full_name = filename.replace('.txt', '')
        
        job_data[job_id] = {
            'description': read_file_content(job_path),
            'cv_texts': [],
            'output_folder_name': job_full_name
        }

        matching_cv_folders = glob.glob(os.path.join(resumes_folder, f"{job_id}_*"))
        if matching_cv_folders:
            cv_folder_path = matching_cv_folders[0]
            cv_paths = glob.glob(os.path.join(cv_folder_path, "*.txt"))
            # Sort to ensure cv1, cv2, cv3 order
            # (Simple sort might do cv1, cv10, cv2... natural sort is better but simple sort is okay here)
            for cv_path in sorted(cv_paths): 
                cv_text = read_file_content(cv_path)
                if cv_text: job_data[job_id]['cv_texts'].append(cv_text)

# Execution
print("\n--- Starting Parallel Generation ---")
for job_id, data in job_data.items():
    print(f"\n📂 Processing Job: {data['output_folder_name']}")
    generate_cover_letters_parallel(
        cv_texts=data['cv_texts'], 
        job_description=data['description'], 
        output_folder_name=data['output_folder_name']
    )
print("\n--- All Done! ---")