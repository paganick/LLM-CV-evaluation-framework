import os
import re

# Base directory where evaluation folders are located
BASE_DIR = "./output_eval"

# The EXACT strings currently at the START of the filenames
EVAL_MAP = {
    "gpt": "gpt-4o-mini",
    "gemini": "gemini-2.0-flash",
    "claude": "claude-haiku-4-5",
    "claude-haiku": "claude-haiku-4-5",        # <-- ADD THIS LINE
    "claude-haiku-haiku": "claude-haiku-4-5",
    "claude-haiku-haiku-4-5": "claude-haiku-4-5",        # <-- ADD THIS LINE
    "deep_api": "deepseek-chat",
    "gpt-5-mini": "gpt-5-mini",                  # NEW
    "gemini-3-flash-preview": "gemini-3-flash-preview", # NEW
    "deepseek-chat": "deepseek-chat",          # FIXED from "deep_api
    "gemini-2.0-flash": "gemini-2.0-flash",    # FIXED from "gemini"
    "gpt-4o-mini": "gpt-4o-mini"               # FIXED from "gpt"
}

# The EXACT strings currently in the MIDDLE of the filenames
# I have added both the short and long versions to guarantee a match
WRITER_MAP = {
    "gpt-4o-mini": "gpt-4o-mini",
    "gpt": "gpt-4o-mini",
    "gemini": "gemini-2.0-flash",
    "gemini-2.0-flash": "gemini-2.0-flash",
    "claude-haiku-haiku": "claude-haiku-4-5",
    "claude-haiku": "claude-haiku-4-5",
    "llama3.1": "llama3.1-8b",        # Fixed from "llama"
    "llama": "llama3.1-8b",
    "deepseek-r1": "deepseek-r1-8b",  
    "deep_local": "deepseek-r1-8b",
    "deepseek-chat": "deepseek-chat", # Fixed from "deep_api"
    "deep_api": "deepseek-chat",
    "gpt-5-mini": "gpt-5-mini",        # NEW
    "gemini-3-flash-preview": "gemini-3-flash-preview" # NEW
}

def rename_all_evaluations():
    print(f"📂 Scanning {BASE_DIR}...")
    
    # Sort keys by length descending so "gpt-4o-mini" is checked before "gpt"
    # re.escape ensures dots like in "llama3.1" don't break the regex
    eval_keys = "|".join([re.escape(k) for k in sorted(EVAL_MAP.keys(), key=len, reverse=True)])
    writer_keys = "|".join([re.escape(k) for k in sorted(WRITER_MAP.keys(), key=len, reverse=True)])
    
    # PATTERN 1: Files with BOTH Evaluator and Writer 
    pattern_dual = re.compile(
        rf"^({eval_keys})_({writer_keys})_(evaluation_cv\d+\.txt|cv_cl_eval_cv\d+\.txt)$"
    )

    # PATTERN 2: Files with ONLY Evaluator (CV Only files)
    pattern_single = re.compile(
        rf"^({eval_keys})_(cv_only_eval_cv\d+\.txt)$"
    )

    count_dual = 0
    count_single = 0

    for root, dirs, files in os.walk(BASE_DIR):
        for filename in files:
            
            # Check for Pattern 1 (Dual Models)
            match_dual = pattern_dual.match(filename)
            if match_dual:
                old_eval = match_dual.group(1)
                old_writer = match_dual.group(2)
                suffix = match_dual.group(3)
                
                new_eval = EVAL_MAP[old_eval]
                new_writer = WRITER_MAP[old_writer]
                
                new_filename = f"{new_eval}_{new_writer}_{suffix}"
                
                # Execute Rename
                if filename != new_filename:
                    os.rename(os.path.join(root, filename), os.path.join(root, new_filename))
                    print(f"✅ Renamed (Dual):   {filename} -> {new_filename}")
                    count_dual += 1
                continue # Skip to next file

            # Check for Pattern 2 (Single Model - CV Only)
            match_single = pattern_single.match(filename)
            if match_single:
                old_eval = match_single.group(1)
                suffix = match_single.group(2)
                
                new_eval = EVAL_MAP[old_eval]
                
                new_filename = f"{new_eval}_{suffix}"
                
                # Execute Rename
                if filename != new_filename:
                    os.rename(os.path.join(root, filename), os.path.join(root, new_filename))
                    print(f"✅ Renamed (Single): {filename} -> {new_filename}")
                    count_single += 1

    print(f"\n✨ Done! Renamed {count_dual} Dual files and {count_single} Single files.")

if __name__ == "__main__":
    rename_all_evaluations()