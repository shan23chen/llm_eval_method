import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer, TrainingArguments, Trainer
from datasets import load_dataset
from peft import LoraConfig, get_peft_model

# transformers parameters
model_name = "meta-llama/Llama-2-7b-hf"

# Load model and tokenizer
model = AutoModelForSequenceClassification.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Cast layer norms to float32 for stability during training
for param in model.parameters():
    param.requires_grad = False  # freeze the model
    if param.ndim == 1:
        param.data = param.data.to(torch.float32)

# Apply LoRA adapters
config = LoraConfig(
    r=64,  # Rank of the adapters
    lora_alpha=32,  # Scaling factor
    target_modules=["q_proj", "v_proj", "out_proj", "fc1", "fc2"],  # Layers to apply adapters
    lora_dropout=0.01,
    bias="none",
    task_type="SEQUENCE_CLASSIFICATION",
)
model = get_peft_model(model, config)

# Load data
data = load_dataset("some_dataset")
data = data.map(lambda samples: tokenizer(samples["text"], padding=True, truncation=True), batched=True)

# Training
training_args = TrainingArguments(
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    warmup_steps=10,
    max_steps=20,
    learning_rate=3e-4,
    logging_steps=1,
    output_dir="outputs",
    # Additional arguments may be required here for your specific task
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=data["train"],
    # The data collator should match your task, e.g., DataCollatorWithPadding for sequence classification
    data_collator=transformers.DataCollatorWithPadding(tokenizer),
)

model.config.use_cache = False  # Disable caching for training
trainer.train()

# Inference for sequence classification
batch = tokenizer(["Your sequence here"], return_tensors="pt")
model.eval()
with torch.no_grad():
    outputs = model(**batch)
    predictions = torch.argmax(outputs.logits, dim=-1)

print("Predicted class:", predictions.item())
