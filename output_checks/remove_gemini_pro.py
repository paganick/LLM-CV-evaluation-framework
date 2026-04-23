import os

# Base directory where evaluation folders are located
BASE_DIR = "./output_eval"

# The specific writer string we want to find and delete
# We include the underscores to ensure it's exactly in the middle 
# (e.g., "gpt-5-mini_gemini-3.1-pro-preview_evaluation...")
TARGET_STRING = "gemini_3_"

def delete_gemini_preview_evals():
    print(f"📂 Scanning {BASE_DIR} for files to delete...")
    count = 0
    
    # Walk through all job subfolders
    for root, dirs, files in os.walk(BASE_DIR):
        for filename in files:
            
            # Check if our target string is inside the filename
            if TARGET_STRING in filename:
                filepath = os.path.join(root, filename)
                
                try:
                    os.remove(filepath)
                    print(f"🗑️ Deleted: {filename}")
                    count += 1
                except Exception as e:
                    print(f"❌ Error deleting {filename}: {e}")

    print(f"\n✨ Done! Successfully deleted {count} evaluation files.")

if __name__ == "__main__":
    delete_gemini_preview_evals()