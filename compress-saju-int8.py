import argparse
from datasets import load_dataset, load_from_disk
from transformers import AutoTokenizer
from llmcompressor.modifiers.quantization import GPTQModifier
from llmcompressor.transformers import SparseAutoModelForCausalLM, oneshot
from llmcompressor.transformers.compression.helpers import calculate_offload_device_map
import torch

# Parse command line arguments
parser = argparse.ArgumentParser(description="Quantize a model with GPTQ.")
parser.add_argument("--model", type=str, required=True, help="Path to the model directory.")
args = parser.parse_args()

# 1) Select model and load it.
MODEL_ID = args.model
# device_map = calculate_offload_device_map(MODEL_ID, reserve_for_hessians=True, num_gpus=2)
# print(f"{device_map=}")
model = SparseAutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    device_map="auto",
    torch_dtype="auto",
)
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

# Select number of samples. 512 samples is a good place to start.
NUM_CALIBRATION_SAMPLES = 1024
MAX_SEQUENCE_LENGTH = 2048

# Load dataset and preprocess.
DATASET_ID = "/home/chang/t9/Models/calib_data_gemma/dataset/train"
DATASET_SPLIT = "train_sft"
ds = load_from_disk(DATASET_ID)
ds = ds.shuffle(seed=42).select(range(NUM_CALIBRATION_SAMPLES))

def preprocess(example):
    return {
        "text": example["text"]
    }


ds1 = ds.map(preprocess)

DATASET_ID = "HuggingFaceH4/ultrachat_200k"
ds = load_dataset(DATASET_ID, split=DATASET_SPLIT)
ds = ds.shuffle(seed=42).select(range(NUM_CALIBRATION_SAMPLES//8))

def preprocess2(example):
    return {
        "text": tokenizer.apply_chat_template(
            example["messages"],
            tokenize=False,
        )
    }

ds2 = ds.map(preprocess2)

from datasets import concatenate_datasets
ds = concatenate_datasets([ds1, ds2])
ds = ds.shuffle(seed=43).select(range(len(ds)))

# Tokenize inputs.
def tokenize(sample):
    return tokenizer(
        sample["text"],
        padding=False,
        max_length=MAX_SEQUENCE_LENGTH,
        truncation=True,
        add_special_tokens=False,
    )


ds = ds.map(tokenize, remove_columns=ds.column_names)

# 3) Select quantization algorithms. In this case, we:
#   * quantize the weights to int8 with GPTQ (static per channel)
#   * quantize the activations to int8 (dynamic per token)
recipe = GPTQModifier(targets="Linear", scheme="W8A8", ignore=["lm_head"])

# 4) Apply quantization and save to disk compressed.
oneshot(
    model=model,
    dataset=ds,
    recipe=recipe,
    max_seq_length=MAX_SEQUENCE_LENGTH,
    num_calibration_samples=len(ds),
    output_dir=MODEL_ID + "/int8",
)

# Confirm generations of the quantized model look sane.
print("========== SAMPLE GENERATION ==============")
input_ids = tokenizer("Hello my name is", return_tensors="pt").input_ids.to("cuda")
output = model.generate(input_ids, max_new_tokens=20)
print(tokenizer.decode(output[0]))
print("==========================================")



