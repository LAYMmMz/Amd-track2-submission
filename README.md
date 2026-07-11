# Amd-track2-submission

# AI Video Persona Captioner - Track 2 Submission

## 🚀 Project Overview
This project processes video clips and automatically generates highly contextual captions in four distinct personas: Formal, Sarcastic, Humorous Tech, and Humorous Non-Tech. 

## 🧠 Infrastructure & Compute
* **Backend Engine:** Containerized Python application using OpenCV for optimized, motion-based frame extraction (capped at 16 frames to prevent VRAM overflow).
* **AI Model:** Powered by the **Fireworks AI API** via a dedicated model deployment to ensure lightning-fast processing and zero timeouts.
* **Frontend Demo:** Hosted on Vercel for visual demonstration purposes.

## 📁 Repository Structure
* `/backend-docker`: Contains the exact code baked into my submitted Docker image.
* `/frontend-vercel`: Contains the source code for the visual web interface used in the video pitch.

## ⚙️ Automated Judging Details
* **Docker Image Tag:** `jhonanthoric/amd-track2-agent:latest`
* The container successfully writes strictly formatted JSON to `/output/results.json` without requiring any local files or private secrets at runtime.
