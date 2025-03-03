# 
import os
import gradio as gr
from datetime import datetime
import shutil
import re
import torch
import sys
# Dynamically determine the root project directory
ROOT_DIR = os.path.abspath(os.path.dirname(__file__))

# Add necessary directories to `sys.path` to resolve imports
sys.path.append(ROOT_DIR)
sys.path.append(os.path.join(ROOT_DIR, "acceleration", "t2v_turbo"))
sys.path.append(os.path.join(ROOT_DIR, "acceleration", "t2v_turbo", "scheduler"))
sys.path.append(os.path.join(ROOT_DIR, "acceleration", "t2v_turbo", "pipeline"))
sys.path.append(os.path.join(ROOT_DIR, "acceleration", "t2v_turbo", "utils"))

# Import after modifying sys.path
from acceleration.t2v_turbo.inference_merge import InferencePipeline

# Initialize the Inference Pipeline
pipeline = InferencePipeline()

# Directory where Gradio serves files
OUTPUT_DIR = "./output"
os.makedirs(OUTPUT_DIR, exist_ok=True)  # Ensure accessible folder exists

def sanitize_filename(text):
    """Removes special characters and replaces spaces with underscores."""
    return re.sub(r'[^\w\s-]', '', text).replace(" ", "_")

def run_inference(prompt, model_version, precision, steps, frames, fps, seed):
    # Define save path based on model selection
    save_path = "results/07B" if model_version == "7B" else "results/09B"
    os.makedirs(save_path, exist_ok=True)

    # Sanitize filename
    safe_prompt = sanitize_filename(prompt)[:30]  
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    final_video_path = os.path.join(OUTPUT_DIR, f"{safe_prompt}_{timestamp}.mp4")

    # Call the function from InferencePipeline
    generated_video_path, _ = pipeline.generate_video(
        prompt=prompt,
        model_version=model_version,  
        precision=precision,
        seed=seed,
        guidance_scale=7.5,
        num_inference_steps=steps,
        num_frames=frames,
        fps=fps,
        save_path=save_path
    )

    if generated_video_path:
        shutil.move(generated_video_path, final_video_path)
        return final_video_path
    else:
        return "Error: Video generation failed."

# Gradio UI
def gradio_interface():
    with gr.Blocks(css="body { background-color: black; color: white; }") as demo:
        gr.HTML("""
        <div style='text-align: center;'>
            <img src='https://upload.wikimedia.org/wikipedia/commons/6/6a/AMD_Logo.png' width='150'/>
            <h1>AMD Hummingbird T2V Video Generation</h1>
            <p>Generate high-quality videos from text prompts using AMD's efficient T2V model.</p>
        </div>
        """)

        with gr.Row():
            with gr.Column():
                prompt = gr.Textbox(
                    label="Prompt",
                    placeholder="Enter your video description",
                    value="A cute happy Corgi playing in park, sunset, animated style.",
                    lines=3
                )
                model_version = gr.Dropdown(["7B", "9B"], value="9B", label="Model Version")
                precision = gr.Dropdown(choices=[16, 32], value=16, label="Model Precision")
                steps = gr.Slider(4, 8, value=8, step=1, label="Steps")
                frames = gr.Slider(10, 50, value=26, step=1, label="Number of Frames")
                fps = gr.Slider(5, 30, value=8, step=1, label="Frames Per Second")
                seed = gr.Slider(0, 100, value=0, step=1, label="Random Seed")
                generate_button = gr.Button("Generate Video")

            with gr.Column():
                video_output = gr.Video(label="Generated Video", format="mp4")

        generate_button.click(run_inference, inputs=[prompt, model_version, precision, steps, frames, fps, seed], outputs=video_output)

    return demo

demo = gradio_interface()
if __name__ == "__main__":
    demo.launch(share=True)

