import os
import cv2
import torch
import argparse
import numpy as np
import supervision as sv
from PIL import Image
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
from transformers import AutoProcessor, AutoModelForCausalLM
from utils.supervision_utils import CUSTOM_COLOR_MAP
import glob

"""
Define Some Hyperparam
"""

TASK_PROMPT = {
    "caption": "<CAPTION>",
    "detailed_caption": "<DETAILED_CAPTION>",
    "more_detailed_caption": "<MORE_DETAILED_CAPTION",
    "object_detection": "<OD>",
    "dense_region_caption": "<DENSE_REGION_CAPTION>",
    "region_proposal": "<REGION_PROPOSAL>",
    "phrase_grounding": "<CAPTION_TO_PHRASE_GROUNDING>",
    "referring_expression_segmentation": "<REFERRING_EXPRESSION_SEGMENTATION>",
    "region_to_segmentation": "<REGION_TO_SEGMENTATION>",
    "open_vocabulary_detection": "<OPEN_VOCABULARY_DETECTION>",
    "region_to_category": "<REGION_TO_CATEGORY>",
    "region_to_description": "<REGION_TO_DESCRIPTION>",
    "ocr": "<OCR>",
    "ocr_with_region": "<OCR_WITH_REGION>",
}

OUTPUT_DIR = "./outputs"

if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR, exist_ok=True)

"""
Init Florence-2 and SAM 2 Model
"""

FLORENCE2_MODEL_ID = "microsoft/Florence-2-large"
SAM2_CHECKPOINT = "./checkpoints/sam2_hiera_large.pt"
SAM2_CONFIG = "sam2_hiera_l.yaml"

# environment settings
# use bfloat16
torch.autocast(device_type="cuda", dtype=torch.bfloat16).__enter__()

if torch.cuda.get_device_properties(0).major >= 8:
    # turn on tfloat32 for Ampere GPUs (https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

device = "cuda:0" if torch.cuda.is_available() else "cpu"
torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

# build florence-2
florence2_model = AutoModelForCausalLM.from_pretrained(FLORENCE2_MODEL_ID, trust_remote_code=True, torch_dtype='auto').eval().to(device)
florence2_processor = AutoProcessor.from_pretrained(FLORENCE2_MODEL_ID, trust_remote_code=True)

# build sam 2
sam2_model = build_sam2(SAM2_CONFIG, SAM2_CHECKPOINT, device=device)
sam2_predictor = SAM2ImagePredictor(sam2_model)

def run_florence2(task_prompt, text_input, model, processor, image):
    assert model is not None, "You should pass the init florence-2 model here"
    assert processor is not None, "You should set florence-2 processor here"

    device = model.device

    if text_input is None:
        prompt = task_prompt
    else:
        prompt = task_prompt + text_input
    
    inputs = processor(text=prompt, images=image, return_tensors="pt").to(device, torch.float16)
    generated_ids = model.generate(
      input_ids=inputs["input_ids"].to(device),
      pixel_values=inputs["pixel_values"].to(device),
      max_new_tokens=1024,
      early_stopping=False,
      do_sample=False,
      num_beams=3,
    )
    generated_text = processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
    parsed_answer = processor.post_process_generation(
        generated_text, 
        task=task_prompt, 
        image_size=(image.width, image.height)
    )
    return parsed_answer


"""
We support a set of pipelines built by Florence-2 + SAM 2
"""

"""
Pipeline-1: Object Detection + Segmentation
"""
def object_detection_and_segmentation(
    florence2_model,
    florence2_processor,
    sam2_predictor,
    image_path,
    task_prompt="<OD>",
    text_input=None,
    output_dir=OUTPUT_DIR
):
    assert text_input is None, "Text input should not be none when calling object detection pipeline."
    # run florence-2 object detection in demo
    image = Image.open(image_path).convert("RGB")
    results = run_florence2(task_prompt, text_input, florence2_model, florence2_processor, image)
    
    """ Florence-2 Object Detection Output Format
    {'<OD>': 
        {
            'bboxes': 
                [
                    [33.599998474121094, 159.59999084472656, 596.7999877929688, 371.7599792480469], 
                    [454.0799865722656, 96.23999786376953, 580.7999877929688, 261.8399963378906], 
                    [224.95999145507812, 86.15999603271484, 333.7599792480469, 164.39999389648438], 
                    [449.5999755859375, 276.239990234375, 554.5599975585938, 370.3199768066406], 
                    [91.19999694824219, 280.0799865722656, 198.0800018310547, 370.3199768066406]
                ], 
            'labels': ['car', 'door', 'door', 'wheel', 'wheel']
        }
    }
    """
    results = results[task_prompt]
    # parse florence-2 detection results
    input_boxes = np.array(results["bboxes"])
    class_names = results["labels"]
    class_ids = np.array(list(range(len(class_names))))
    
    # predict mask with SAM 2
    sam2_predictor.set_image(np.array(image))
    masks, scores, logits = sam2_predictor.predict(
        point_coords=None,
        point_labels=None,
        box=input_boxes,
        multimask_output=False,
    )
    
    if masks.ndim == 4:
        masks = masks.squeeze(1)
    
    # specify labels
    labels = [
        f"{class_name}" for class_name in class_names
    ]
    
    # visualization results
    img = cv2.imread(image_path)
    detections = sv.Detections(
        xyxy=input_boxes,
        mask=masks.astype(bool),
        class_id=class_ids
    )
    
    box_annotator = sv.BoxAnnotator()
    annotated_frame = box_annotator.annotate(scene=img.copy(), detections=detections)
    
    label_annotator = sv.LabelAnnotator()
    annotated_frame = label_annotator.annotate(scene=annotated_frame, detections=detections, labels=labels)
    cv2.imwrite(os.path.join(output_dir, "grounded_sam2_florence2_det_annotated_image.jpg"), annotated_frame)
    
    mask_annotator = sv.MaskAnnotator()
    annotated_frame = mask_annotator.annotate(scene=annotated_frame, detections=detections)
    cv2.imwrite(os.path.join(output_dir, "grounded_sam2_florence2_det_image_with_mask.jpg"), annotated_frame)

    print(f'Successfully save annotated image to "{output_dir}"')


"""
Pipeline 6: Open-Vocabulary Detection + Segmentation
"""
def open_vocabulary_detection_and_segmentation(
    florence2_model,
    florence2_processor,
    sam2_predictor,
    image_path,
    task_prompt="<OPEN_VOCABULARY_DETECTION>",
    text_input=None,
    output_dir="./outputs"
):
    # Get the base name of the image file (e.g., "t_good.png")
    image_name = os.path.splitext(os.path.basename(image_path))[0]
    
    # Run Florence-2 object detection
    image = Image.open(image_path).convert("RGB")
    img_width, img_height = image.size
    results = run_florence2(task_prompt, text_input, florence2_model, florence2_processor, image)
    
    results = results[task_prompt]
    input_boxes = np.array(results["bboxes"])
    class_names = results["bboxes_labels"]
    class_ids = np.array(list(range(len(class_names))))
    
    # Predict mask with SAM 2
    sam2_predictor.set_image(np.array(image))
    masks, scores, logits = sam2_predictor.predict(
        point_coords=None,
        point_labels=None,
        box=input_boxes,
        multimask_output=False,
    )
    
    if masks.ndim == 4:
        masks = masks.squeeze(1)
    
    labels = [f"{class_name}" for class_name in class_names]
    
    # Visualization results
    img = cv2.imread(image_path)
    detections = sv.Detections(
        xyxy=input_boxes,
        mask=masks.astype(bool),
        class_id=class_ids
    )
    
    box_annotator = sv.BoxAnnotator()
    annotated_frame = box_annotator.annotate(scene=img.copy(), detections=detections)
    
    label_annotator = sv.LabelAnnotator()
    annotated_frame = label_annotator.annotate(scene=annotated_frame, detections=detections, labels=labels)
    
    # Save the annotated image with a unique name based on the original image
    annotated_image_path = os.path.join(output_dir, f"{image_name}_open_vocabulary_detection.jpg")
    cv2.imwrite(annotated_image_path, annotated_frame)
    
    mask_annotator = sv.MaskAnnotator()
    annotated_frame = mask_annotator.annotate(scene=annotated_frame, detections=detections)
    
    # Save the image with masks with a unique name based on the original image
    masked_image_path = os.path.join(output_dir, f"{image_name}_open_vocabulary_detection_with_mask.jpg")
    cv2.imwrite(masked_image_path, annotated_frame)

    print(f'Successfully saved annotated images as "{annotated_image_path}" and "{masked_image_path}"')
    
    # Save mask coordinates in YOLO-like format to mask.txt
    mask_filename = os.path.join(output_dir, f"{image_name}_mask.txt")
    with open(mask_filename, "w") as f:
        for i, mask in enumerate(masks):
            mask_points = mask.nonzero()  # Get mask points
            polygon = []
            for y, x in zip(*mask_points):
                normalized_x = x / img_width  # Normalize x coordinate
                normalized_y = y / img_height  # Normalize y coordinate
                polygon.append(f"{normalized_x} {normalized_y}")
            
            class_id = class_ids[i]
            f.write(f"{class_id} " + " ".join(polygon) + "\n")
    
    print(f'Successfully saved mask coordinates to "{mask_filename}"')

    # Save bounding box coordinates in YOLOv8 format to box.txt
    box_filename = os.path.join(output_dir, f"{image_name}_box.txt")
    with open(box_filename, "w") as f:
        for i, box in enumerate(input_boxes):
            x_min, y_min, x_max, y_max = box
            # Calculate center coordinates and dimensions
            center_x = (x_min + x_max) / 2.0 / img_width
            center_y = (y_min + y_max) / 2.0 / img_height
            width = (x_max - x_min) / img_width
            height = (y_max - y_min) / img_height
            
            class_id = class_ids[i]
            f.write(f"{class_id} {center_x} {center_y} {width} {height}\n")
    
    print(f'Successfully saved bounding box coordinates to "{box_filename}"')


def get_unique_filename(directory, filename):
    """
    Generate a unique file name by appending a number if a file with the same name exists.
    """
    base, ext = os.path.splitext(filename)
    counter = 1
    new_filename = filename
    
    while os.path.exists(os.path.join(directory, new_filename)):
        new_filename = f"{base}_{counter}{ext}"
        counter += 1
        
    return new_filename

if __name__ == "__main__":

    parser = argparse.ArgumentParser("Grounded SAM 2 Florence-2 Demos", add_help=True)
    parser.add_argument("--image_dir", type=str, required=True, help="path to the directory containing image files")
    parser.add_argument("--pipeline", type=str, required=True, help="pipeline to use for processing images")
    parser.add_argument("--output_dir", type=str, required=True, help="path to the directory where the processed images will be saved")
    parser.add_argument("--text_input", type=str, default=None, required=False, help="text input for specific pipelines")
    args = parser.parse_args()

    IMAGE_DIR = args.image_dir
    PIPELINE = args.pipeline
    OUTPUT_DIR = args.output_dir
    INPUT_TEXT = args.text_input

    # Ensure the output directory exists
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Get a list of all image files in the directory
    image_files = glob.glob(os.path.join(IMAGE_DIR, "*.png")) + glob.glob(os.path.join(IMAGE_DIR, "*.jpg"))

    for image_path in image_files:
        print(f"Processing image: {image_path}")
        
        # Extract the image name and ensure it doesn't overwrite existing files
        image_name = os.path.basename(image_path)
        unique_image_name = get_unique_filename(OUTPUT_DIR, image_name)
        
        if PIPELINE == "object_detection_segmentation":
            object_detection_and_segmentation(
                florence2_model=florence2_model,
                florence2_processor=florence2_processor,
                sam2_predictor=sam2_predictor,
                image_path=image_path,
                output_dir=OUTPUT_DIR
            )

        elif PIPELINE == "open_vocabulary_detection_segmentation":
            open_vocabulary_detection_and_segmentation(
                florence2_model=florence2_model,
                florence2_processor=florence2_processor,
                sam2_predictor=sam2_predictor,
                image_path=image_path,
                text_input=INPUT_TEXT,
                output_dir=OUTPUT_DIR
            )
        else:
            raise NotImplementedError(f"Pipeline: {PIPELINE} is not implemented at this time")

        # After processing, rename the output files if needed
        original_output_path = os.path.join(OUTPUT_DIR, image_name)
        unique_output_path = os.path.join(OUTPUT_DIR, unique_image_name)

        # If the original output path exists, rename it to the unique one
        if os.path.exists(original_output_path):
            os.rename(original_output_path, unique_output_path)

    print(f"Processing complete. All images saved to {OUTPUT_DIR}.")
