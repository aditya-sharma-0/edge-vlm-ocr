import torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForImageTextToText, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model
from trl import SFTTrainer, SFTConfig
from datasets import Dataset

# 1. Setup Model & Processor (Same as before)
model_id = "HuggingFaceTB/SmolVLM-256M-Instruct"

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)

processor = AutoProcessor.from_pretrained(model_id)
model = AutoModelForImageTextToText.from_pretrained(
    model_id,
    quantization_config=bnb_config,
    device_map="auto"
)
model.gradient_checkpointing_enable()

lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    bias="none",
)
model = get_peft_model(model, lora_config)

print("Model Loaded. Preparing Dummy Data...")

# 2. Create Dummy Dataset for the Backward Pass Test
# We generate a blank 256x256 image and format it exactly how SmolVLM expects it.
dummy_image = Image.new('RGB', (256, 256), color='black')

# SmolVLM requires a specific chat template format
messages = [
    {
        "role": "user", 
        "content": [
            {"type": "image"}, 
            {"type": "text", "text": "Locate the text 'CAUTION' in this image."}
        ]
    },
    {
        "role": "assistant", 
        "content": [
            {"type": "text", "text": "<ref>CAUTION</ref><box>[[250, 100, 300, 150]]</box>"}
        ]
    }
]

# Apply the processor's chat template
formatted_text = processor.apply_chat_template(messages, add_generation_prompt=False)

# Create a Hugging Face Dataset object
dummy_dataset = Dataset.from_dict({
    "text": [formatted_text] * 10,  # Duplicate it 10 times
    "images": [[dummy_image]] * 10
})

def format_data(example):
    # Removed max_length and truncation so the massive 1000+ token images don't get chopped in half
    return processor(text=example["text"], images=example["images"][0])

print("Tokenizing Data...")
processed_dataset = dummy_dataset.map(format_data, remove_columns=["text", "images"])

# 3. The Ultimate VRAM Test (Training Execution)
print("Starting Training Loop Test...")
training_args = SFTConfig(
    output_dir="./vram-test-output",
    per_device_train_batch_size=1,
    gradient_accumulation_steps=4,
    optim="paged_adamw_8bit",
    logging_steps=1,
    max_steps=5,         # Just 5 steps to verify gradients don't crash the GPU
    learning_rate=2e-4,
    fp16=True,
    dataset_text_field="input_ids", # We already tokenized it
)

trainer = SFTTrainer(
    model=model,
    train_dataset=processed_dataset,
    args=training_args,
)

# Fire the engines
trainer.train()
print("SUCCESS! Backward Pass complete. Your RTX 4050 is ready for real data.")