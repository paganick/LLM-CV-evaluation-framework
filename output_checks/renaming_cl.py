import os

# Base directory where output folders are located
BASE_DIR = "./output_cl"

# The specific prefix change you requested
OLD_PREFIX = "llama3.1"
NEW_PREFIX = "llama3.1-8b"

def rename_gemini_files():
    print(f"📂 Scanning {BASE_DIR}...")
    count = 0
    
    # Walk through all job subfolders
    for root, dirs, files in os.walk(BASE_DIR):
        for filename in files:
            
            # Check if the file starts with the old prefix
            if filename.startswith(OLD_PREFIX):
                
                old_path = os.path.join(root, filename)
                
                # Replace the old prefix with the new one (only the first occurrence)
                new_filename = filename.replace(OLD_PREFIX, NEW_PREFIX, 1)
                new_path = os.path.join(root, new_filename)
                
                try:
                    os.rename(old_path, new_path)
                    print(f"✅ Renamed: {filename} -> {new_filename}")
                    count += 1
                except Exception as e:
                    print(f"❌ Error renaming {filename}: {e}")

    print(f"\n✨ Done! Renamed {count} files.")

if __name__ == "__main__":
    rename_gemini_files()