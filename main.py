import json
import os
import urllib.request
import tempfile
from video_utils import get_dummy_captions

def process_track2_tasks():
    input_path = "/input/tasks.json"
    output_path = "/output/results.json"
    
    # Read the evaluation tasks
    try:
        with open(input_path, 'r') as f:
            tasks = json.load(f)
    except FileNotFoundError:
        print(f"Error: {input_path} not found.")
        return

    results = []
    
    for task in tasks:
        task_id = task["task_id"]
        video_url = task["video_url"]
        
        print(f"Processing Task ID: {task_id}")
        
        # Download the video securely to a temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as temp_video:
            urllib.request.urlretrieve(video_url, temp_video.name)
            
            # Pass the downloaded file to your existing pipeline
            with open(temp_video.name, 'rb') as video_file:
                # LINE DELETED HERE - It already has a .name attribute!
                captions = get_dummy_captions(video_file)
        
        # Clean up the downloaded file to save space
        os.remove(temp_video.name)
        
        # Format output strictly to the required schema
        results.append({
            "task_id": task_id,
            "captions": {
                "formal": captions.get("formal", "Error generating formal caption."),
                "sarcastic": captions.get("sarcastic", "Error generating sarcastic caption."),
                "humorous_tech": captions.get("humorous_tech", "Error generating tech caption."),
                "humorous_non_tech": captions.get("humorous_non_tech", "Error generating non-tech caption.")
            }
        })
        
    # Write the final results before the container exits
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=4)
        
    print("All tasks processed successfully.")

if __name__ == "__main__":
    process_track2_tasks()