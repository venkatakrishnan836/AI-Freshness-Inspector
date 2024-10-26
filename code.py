import os
import torch
from PIL import Image
from transformers import MllamaForConditionalGeneration, AutoProcessor

MODEL_ID = "meta-llama/Llama-3.2-11B-Vision-Instruct"
DEVICE_MAP = "auto"
TORCH_DTYPE = torch.bfloat16

processor = AutoProcessor.from_pretrained(MODEL_ID)
model = MllamaForConditionalGeneration.from_pretrained(
    MODEL_ID,
    torch_dtype=TORCH_DTYPE,
    device_map=DEVICE_MAP,
)

PROMPT_TEMPLATE = """
ANALYZE THE PROVIDED IMAGE OF A FOOD PRODUCT AND PROVIDE A CONCISE OUTPUT IN THE FOLLOWING FORMAT:
"Product: [SPECIFIC PRODUCT NAME], Freshness: [FRESHNESS CATEGORY] (shelf life: [NUMBER] days)"

CATEGORIES:
- Fresh (no visible signs of spoilage)
- Moderately Fresh (minor signs of spoilage, still consumable)
- Rotten (noticeable signs of spoilage, approaching expiration ,significant signs of spoilage, not suitable for consumption)

CONSIDER THE FOLLOWING FACTORS:
- Visible signs of mold, slime, or fungal growth
- Discoloration, unnatural color, or unusual texture
- Presence of pests, insects, or rodent activity
- Unpleasant or strong odors
- Soft, soggy, or brittle texture

AVOID:
- Explanations about image or analysis
- Descriptions of analysis or image
- Unclear or ambiguous responses
- Unnecessary details

INCLUDE:
- Clear categorization
- Specific shelf life
- Product name
"""
def process_image(image_path):
    image = Image.open(image_path)
    messages = [
        {
            "role": "user", 
            "content": [
                {"type": "image"},
                {"type": "text", "text": PROMPT_TEMPLATE}
            ]
        }
    ]

    input_text = processor.apply_chat_template(messages, add_generation_prompt=True)
    inputs = processor(
        image,
        input_text,
        add_special_tokens=False,
        return_tensors="pt"
    ).to(model.device)

    output = model.generate(**inputs, max_new_tokens=100)
    generated_text = processor.decode(output[0], skip_special_tokens=True).strip()

    prompt_length = len(processor.decode(inputs["input_ids"][0], skip_special_tokens=True).strip())
    output_text = generated_text[prompt_length:].strip()
    print(output_text)
    parts = output_text.split(', ')

    if len(parts) == 2:
        product = parts[0].split(': ')[1]
        freshness_shelf_life = parts[1].split(' (')
        freshness = freshness_shelf_life[0]
        shelf_life = freshness_shelf_life[1].replace(' days)', '')

        return {
        "Product": product,
        "Freshness": freshness,
        "Shelf Life": shelf_life
                }
    else:
        return {"Product": "Unknown", "Freshness": "Unknown", "Shelf Life": "0"}

def process_images_from_folder(folder_path):
    for image_filename in os.listdir(folder_path):
        image_path = os.path.join(folder_path, image_filename)
        output = process_image(image_path)
        print(f"Output for {image_filename}:")
        print(f"Product: {output['Product']}")
        print(f"Freshness: {output['Freshness']}")
        print(f"Shelf Life: {output['Shelf Life']}\n")

# Specify the folder path containing the images
folder_path = "Foler path"
process_images_from_folder(folder_path)
