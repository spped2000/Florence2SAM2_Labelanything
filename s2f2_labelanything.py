import gradio as gr
import os
import subprocess
import glob
import shutil
from IPython.display import display, HTML

UPLOAD_FOLDER = 'uploads'
OUTPUT_FOLDER = 'outputs'
SAVE_FOLDER = 'saved_results'
IMAGES_SUBFOLDER = 'images'
LABELS_SUBFOLDER = 'labels'

def process_images(images, text_input):
    # Clear previous uploads and outputs
    for folder in [UPLOAD_FOLDER, OUTPUT_FOLDER]:
        if os.path.exists(folder):
            shutil.rmtree(folder)
        os.makedirs(folder)
    
    # Save uploaded images
    image_paths = []
    for i, image in enumerate(images):
        file_extension = os.path.splitext(image.name)[1]
        image_path = os.path.join(UPLOAD_FOLDER, f"image_{i}{file_extension}")
        shutil.copy(image.name, image_path)
        image_paths.append(image_path)
    
    # Run the auto-labeling script
    cmd = [
        'python', 's2f2_labelanything.py',
        '--pipeline', 'open_vocabulary_detection_segmentation',
        '--image_dir', UPLOAD_FOLDER,
        '--output_dir', OUTPUT_FOLDER,
        '--text_input', text_input
    ]
    subprocess.run(cmd)
    
    # Collect output files
    output_files = glob.glob(os.path.join(OUTPUT_FOLDER, '*'))
    
    # Prepare results
    image_results = []
    box_coordinates = ""
    for file in output_files:
        if file.endswith(('.png', '.jpg', '.jpeg')):
            image_results.append((file, os.path.basename(file)))
        elif file.endswith('_box.txt'):
            with open(file, 'r') as f:
                content = f.read()
            box_coordinates += f"Bounding boxes for {os.path.basename(file)}:\n{content}\n\n"
    
    return image_results, box_coordinates

def save_results(save_button, use_box_coordinates):
    if not save_button:
        return "Click the Save button to save results."
    
    # Create main save folder and subfolders
    os.makedirs(os.path.join(SAVE_FOLDER, IMAGES_SUBFOLDER), exist_ok=True)
    os.makedirs(os.path.join(SAVE_FOLDER, LABELS_SUBFOLDER), exist_ok=True)
    
    # Copy original images to images subfolder
    for image in glob.glob(os.path.join(UPLOAD_FOLDER, '*')):
        shutil.copy(image, os.path.join(SAVE_FOLDER, IMAGES_SUBFOLDER))
    
    # Copy label files to labels subfolder based on user selection
    label_type = "box" if use_box_coordinates else "mask"
    for label_file in glob.glob(os.path.join(OUTPUT_FOLDER, f'*_{label_type}.txt')):
        new_name = os.path.basename(label_file).replace(f'_{label_type}', '')
        shutil.copy(label_file, os.path.join(SAVE_FOLDER, LABELS_SUBFOLDER, new_name))
    
    return f"Results saved in {SAVE_FOLDER} folder. Images in '{IMAGES_SUBFOLDER}' subfolder and labels in '{LABELS_SUBFOLDER}' subfolder with {'box' if use_box_coordinates else 'mask'} coordinates."

with gr.Blocks() as iface:
    gr.Markdown("# Auto-Label Image Processor")
    gr.Markdown("Upload images and provide text input for open vocabulary detection and segmentation.")
    
    with gr.Row():
        input_images = gr.File(file_count="multiple", label="Upload Images")
        text_input = gr.Textbox(label="Text Input (e.g., 'green basket')")
    
    process_button = gr.Button("Process Images")
    
    with gr.Row():
        output_gallery = gr.Gallery(label="Processed Images", show_label=True)
        output_text = gr.Textbox(label="Bounding Box Coordinates", show_label=True)
    
    with gr.Row():
        use_box_coordinates = gr.Checkbox(label="Use bounding box coordinates (uncheck for mask coordinates)", value=True)
        save_button = gr.Button("Save Results")
    
    save_output = gr.Textbox(label="Save Status")
    
    process_button.click(
        process_images,
        inputs=[input_images, text_input],
        outputs=[output_gallery, output_text]
    )
    
    save_button.click(
        save_results,
        inputs=[save_button, use_box_coordinates],
        outputs=[save_output]
    )

# Launch the interface
iface.launch(share=True)

# Display the public URL
public_url = iface.share_url
display(HTML(f'<p>Public URL: <a href="{public_url}" target="_blank">{public_url}</a></p>'))
