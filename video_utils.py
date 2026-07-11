import cv2
import tempfile
import base64
import os
import re
from dotenv import load_dotenv
from fireworks.client import Fireworks 
import json

load_dotenv()

def extract_sampled_frames(uploaded_file, num_uniform=8, num_motion=8, max_dim=768):
    """
    Hybrid Extraction (Memory-Optimized for Dedicated Instances):
    1. Extracts 'num_uniform' evenly spaced frames for story context.
    2. Scans for 'num_motion' frames with the highest movement (action).
    3. Merges, sorts chronologically, and encodes them at a higher resolution (768px).
    """
    base64_frames = []
    uploaded_file.seek(0)
    
    with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as temp_video:
        temp_video.write(uploaded_file.read())
        video_path = temp_video.name
        
    try:
        video = cv2.VideoCapture(video_path)
        total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
        
        if total_frames == 0:
            return []
            
        # STEP 1: Fast motion scanning (Pass 1)
        motion_scores = []
        video.set(cv2.CAP_PROP_POS_FRAMES, 0)
        ret, prev_frame = video.read()
        
        if ret:
            prev_gray = cv2.cvtColor(cv2.resize(prev_frame, (256, 256)), cv2.COLOR_BGR2GRAY)
            
            for i in range(1, total_frames, 2): 
                video.set(cv2.CAP_PROP_POS_FRAMES, i)
                ret, frame = video.read()
                if not ret:
                    break
                    
                curr_gray = cv2.cvtColor(cv2.resize(frame, (256, 256)), cv2.COLOR_BGR2GRAY)
                score = cv2.sumElems(cv2.absdiff(prev_gray, curr_gray))[0]
                motion_scores.append((i, score))
                prev_gray = curr_gray

        motion_scores.sort(key=lambda x: x[1], reverse=True)
        motion_indices = {x[0] for x in motion_scores[:num_motion]}
        
        step = max(total_frames // num_uniform, 1)
        uniform_indices = {min(i * step, total_frames - 1) for i in range(num_uniform)}
        
        final_indices = sorted(list(motion_indices.union(uniform_indices)))
        
        for idx in final_indices:
            video.set(cv2.CAP_PROP_POS_FRAMES, idx)
            success, frame = video.read()
            
            if success:
                height, width = frame.shape[:2]
                if max(height, width) > max_dim:
                    scale = max_dim / max(height, width)
                    frame = cv2.resize(frame, (int(width * scale), int(height * scale)))
                
                _, buffer = cv2.imencode('.jpg', frame)
                base64_str = base64.b64encode(buffer).decode('utf-8')
                base64_frames.append(base64_str)
                
    finally:
        video.release()
        if os.path.exists(video_path):
            os.remove(video_path)
            
    return base64_frames

def extract_json_payload(response: str) -> dict:
    """Extracts and parses the JSON block inside the <OUTPUT> tags."""
    fallback_dict = {
        "formal": "⚠️ Data missing or model failed to generate this persona.",
        "sarcastic": "⚠️ Data missing or model failed to generate this persona.",
        "humorous_tech": "⚠️ Data missing or model failed to generate this persona.",
        "humorous_non_tech": "⚠️ Data missing or model failed to generate this persona."
    }
    
    if not response or not response.strip():
        return fallback_dict
    
    try:
        match = re.search(r'<OUTPUT>(.*?)</OUTPUT>', response, flags=re.DOTALL | re.IGNORECASE)
        json_str = match.group(1).strip() if match else response.strip()
        
        json_str = re.sub(r'^```json\s*|```$', '', json_str, flags=re.MULTILINE).strip()
        parsed_json = json.loads(json_str)
        
        for key in fallback_dict.keys():
            if key in parsed_json:
                fallback_dict[key] = parsed_json[key]
                
        return fallback_dict
        
    except Exception as e:
        return {
            "formal": "⚠️ Error processing structured payload response.",
            "sarcastic": "⚠️ Fallback activated.",
            "humorous_tech": f"Exception: {str(e)}",
            "humorous_non_tech": f"Raw Response Received: {response[:100]}..."
        }

def get_dummy_captions(uploaded_file) -> dict:
    api_key_status = os.getenv("FIREWORKS_API_KEY")
    dedicated_model_path = os.getenv("FIREWORKS_DEDICATED_MODEL")
    
    if not dedicated_model_path:
        return {k: "⚠️ Error: FIREWORKS_DEDICATED_MODEL not found in .env." for k in ["formal", "sarcastic", "humorous_tech", "humorous_non_tech"]}
    
    if api_key_status and not api_key_status.startswith("fw_your_actual"):
        # ✅ FIX: Lowered uniform and motion targets to 8 each (16 total frames) to prevent private server VRAM OOM crashes
        extracted_frames = extract_sampled_frames(uploaded_file, num_uniform=8, num_motion=8)
        if not extracted_frames:
            return {k: "⚠️ Error: Frames could not be extracted." for k in ["formal", "sarcastic", "humorous_tech", "humorous_non_tech"]}
        
        video_name = uploaded_file.name
        frame_count = len(extracted_frames)

        system_instruction = (
            f"You are an expert multi-agent video analysis system. You will receive {frame_count} chronological frames from '{video_name}'.\n"
            "Your task is to analyze the entire video and generate 4 distinct caption variants matching specific personas.\n\n"
            "CRITICAL: Do not narrate events chronologically. Keep outputs concise (1-3 sentences per persona).\n"
            "Account for sudden actions, dynamic tension, or structural twists at the end of the clip.\n\n"
            "You MUST output your response as a single, valid JSON object wrapped inside <OUTPUT> and </OUTPUT> tags.\n"
            "Keep internal reasoning incredibly brief so you do not hit token length limits. Output the JSON object immediately."
        )

        user_prompt = (
            "Analyze the provided video frames and populate the following JSON schema. "
            "Every caption MUST be strictly 1 to 3 sentences long and must strictly adhere to the specific grounding rules for that persona:\n\n"
            "{\n"
            '  "formal": "👔 **Formal Caption:** [Rules: Write a highly clinical, objective, and analytical summary. Focus strictly on the technical protocol, thermodynamics, physical mechanics, or precise workflow dynamics observed in the frames. Do not use generic corporate filler; use precise domain-specific terminology relevant to the video\'s subject matter.]",\n'
            '  "sarcastic": "😒 **Sarcastic Caption:** [Rules: Deliver a dry, cynical, or mocking jab. Ground the sarcasm directly in the specific tools, extreme precision, or unnecessary complexity of the actions taking place. Do not use generic jokes; mock the exact situation shown.]",\n'
            '  "humorous_tech": "💻 **Tech Caption:** [Rules: Translate the video\'s core action entirely into software engineering, IT infrastructure, or DevOps metaphors. Map specific steps or numbers to concepts like data parsing, caching, latency, compilation, deployment loops, or bug tracking. Must read like a genuine engineering incident or ticket.]",\n'
            '  "humorous_non_tech": "🍿 **Reaction Caption:** [Rules: Write an enthusiastic, high-energy internet-style reaction comment typical of TikTok, Reels, or YouTube Shorts. Use popular slang, highly expressive punctuation, and context-relevant emojis. Reflect the immediate viral aesthetic of the visual hook.]"\n'
            "}\n\n"
            "CRITICAL: Return ONLY the valid JSON object wrapped inside <OUTPUT> and </OUTPUT> tags. Ensure the JSON is complete and fully formed."
        )

        client = Fireworks(api_key=api_key_status)
        
        content_payload = []
        for idx, base64_img in enumerate(extracted_frames):
            content_payload.append({"type": "text", "text": f"Frame {idx+1}:"})
            content_payload.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{base64_img}"}
            })
        content_payload.append({"type": "text", "text": user_prompt})

        try:
            response = client.chat.completions.create(
                model=dedicated_model_path,  
                messages=[
                    {"role": "system", "content": system_instruction},
                    {"role": "user", "content": content_payload}
                ],
                temperature=0.2, 
                max_tokens=1500  # ✅ FIX: Safe boundary for private single-GPU model contexts
            )
            
            return extract_json_payload(response.choices[0].message.content)
            
        except Exception as e:
            return {k: f"⚠️ API Error: {str(e)}" for k in ["formal", "sarcastic", "humorous_tech", "humorous_non_tech"]}
    
    return {
        "formal": "👔 *[OFFLINE DUMMY MODE]* API Key placeholder active.",
        "sarcastic": "😒 *[OFFLINE DUMMY MODE]* Setup complete.",
        "humorous_tech": "💻 *[OFFLINE DUMMY MODE]* Key required.",
        "humorous_non_tech": "🍿 *[OFFLINE DUMMY MODE]* Silence reigns."
    }