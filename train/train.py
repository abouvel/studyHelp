from ultralytics import YOLO
from wakepy import keep
import os
import requests
import torch

# Configuration
WEBHOOK_URL = "https://discordapp.com/api/webhooks/1452013162667704533/k5Aruxca7YYTPtb2yS2LMzF22DVHuLU18vQjDgSUG5exw3eICO37nfJ41BFxqBOkVXz0"
script_dir = os.path.dirname(os.path.abspath(__file__))

def send_discord_embed(title, description, color, fields=None):
    """Sends a formatted embed to Discord"""
    payload = {
        "embeds": [{
            "title": title,
            "description": description,
            "color": color,
            "fields": fields or [],
            "footer": {"text": "YOLO Study Strava Trainer"}
        }]
    }
    requests.post(WEBHOOK_URL, json=payload)

def train_model():
    # Clear GPU cache before starting to prevent instant OOM crash
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    with keep.presenting():
        model = YOLO("yolo11n.pt")

        try:
            # START TRAINING
            results = model.train(
                data="fpi_det.yaml",
                epochs=150,
                imgsz=640,
                patience=30,
                
                # --- STABILITY & SPEED ---
                batch=-1,          # Auto-batch: Finds the best fit for your 4GB VRAM
                device=0,          # RTX 3050
                workers=4,         # Lowered to 4 for better system stability
                half=True,         # FP16 speed boost
                amp=True,          
                optimizer='AdamW', 
                close_mosaic=10,   
                
                # --- PROJECT STORAGE ---
                project=os.path.join(script_dir, "f"),
                name="study_strava_run",
                exist_ok=True,
                cache=False        # Disable RAM cache to prevent system freeze
            )

            # --- PREPARE SUCCESS NOTIFICATION ---
            # Extracting metrics from the results object
            # map50_95 = results.results_dict.get('metrics/mAP50-95(B)', 0)
            # map50 = results.results_dict.get('metrics/mAP50(B)', 0)
            # epochs_run = len(results.fitness) # Actual epochs run before patience stop

            # fields = [
            #     {"name": "Accuracy (mAP50-95)", "value": f"{map50_95:.4f}", "inline": True},
            #     {"name": "Accuracy (mAP50)", "value": f"{map50:.4f}", "inline": True},
            #     {"name": "Epochs Completed", "value": str(epochs_run), "inline": True}
            # ]

            send_discord_embed(
                "✅ Training Complete!", 
                "The model finished successfully. Ready for Go app development!", 
                65280, # Green
            )

        except Exception as e:
            # --- PREPARE ERROR NOTIFICATION ---
            send_discord_embed(
                "❌ Training Crashed", 
                f"**Error Message:**\n```{str(e)}```", 
                16711680 # Red
            )

if __name__ == "__main__":
    train_model()