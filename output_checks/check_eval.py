import os
import re

# ==========================
# CONFIGURATION
# ==========================
BASE_DIR = "./output_eval"  # The directory to scan

def check_eval_files(base_dir):
    print(f"🔍 Scanning '{base_dir}' for empty or invalid evaluation files...\n")
    
    problem_files = []
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
                        problem_files.append((file_path, "0 Bytes (Empty File)"))
                        continue
                        
                    # Check 2: Content Validation
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read().strip()
                        
                        if not content:
                            problem_files.append((file_path, "Whitespace Only"))
                            continue
                            
                        # Check 3: Does it contain the required XML tags?
                        if "Score:" not in content:
                            problem_files.append((file_path, "Missing Score"))

                except Exception as e:
                    print(f"⚠️ Error reading {file_path}: {e}")

    # --- REPORT ---
    if problem_files:
        print(f"❌ Found {len(problem_files)} problematic files out of {total_files} scanned:\n")
        
        # Limit output if there are too many errors
        for i, (path, reason) in enumerate(problem_files):
            if i < 20:
                print(f"  - [{reason}] {path}")
            else:
                remaining = len(problem_files) - 20
                print(f"  ... and {remaining} more.")
                break
            
        print("\n💡 Recommendation: Delete these files so the script re-generates them.")
        
        # --- USER INPUT DELETION ---
        choice = input("\nDo you want to DELETE these files now? (y/n): ")
        if choice.lower() == 'y':
            print("\nDeleting files...")
            count = 0
            for path, _ in problem_files:
                try:
                    os.remove(path)
                    print(f"  Deleted: {path}")
                    count += 1
                except Exception as e:
                    print(f"  Failed to delete {path}: {e}")
            print(f"\n✅ Deleted {count} files. Run your evaluation script again to regenerate them.")
        else:
            print("\nOperation cancelled. No files were deleted.")

    else:
        print(f"✅ Success! Scanned {total_files} evaluation files. All valid.")

if __name__ == "__main__":
    check_eval_files(BASE_DIR)