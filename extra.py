import os

search_root = "data"  # or wherever you cloned the GitHub data repo
required_files = ["tds-course-content.jsonl", "tds-forum-posts.jsonl"]

for root, dirs, files in os.walk(search_root):
    for file in files:
        if file in required_files:
            full_path = os.path.join(root, file)
            print(f"✅ Found: {file} at {full_path}")
