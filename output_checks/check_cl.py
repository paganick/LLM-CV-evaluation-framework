import os

# ==========================
# CONFIGURATION
# ==========================
BASE_DIR = "./output_cl"  # The directory to scan

def check_for_empty_files(base_dir):
    print(f"🔍 Scanning '{base_dir}' for empty or whitespace-only files...\n")
    
    empty_files = []
    total_files = 0
    
    if not os.path.exists(base_dir):
        print(f"❌ Error: Directory '{base_dir}' does not exist.")
        return

    for root, dirs, files in os.walk(base_dir):
        for file in files:
            if file.endswith(".txt"):
                total_files += 1
                file_path = os.path.join(root, file)
                
                try:
                    # Check 1: Is file size 0?
                    if os.path.getsize(file_path) == 0:
                        empty_files.append((file_path, "0 Bytes"))
                        continue
                        
                    # Check 2: Is content just whitespace/newlines?
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                        if not content.strip():
                            empty_files.append((file_path, "Whitespace Only"))

                except Exception as e:
                    print(f"⚠️ Error reading {file_path}: {e}")

    # --- REPORT ---
    if empty_files:
        print(f"❌ Found {len(empty_files)} empty files out of {total_files} total scanned:\n")
        for path, reason in empty_files:
            print(f"  - [{reason}] {path}")
            
        print("\n💡 Recommendation: You should delete these files and re-run the generation script.")
    else:
        print(f"✅ Success! Scanned {total_files} files. No empty cover letters found.")

if __name__ == "__main__":
    check_for_empty_files(BASE_DIR)