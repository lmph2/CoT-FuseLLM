import argparse
import sys
sys.path.append("/home/lmph2/rds/hpc-work/LLM_fusion/FuseLLM")
from typing import Dict, List
import torch
import torch.nn.functional as F
import time
from src.utils.common import load_tokenizer_and_model
from src.utils.others import (
    IGNORE_TOKEN_ID,
    AttrDict,
    dict_to_list,
    get_logger,
    get_tokenizer,
    release_model_and_tensor,
)

def parse_args():
    parser = argparse.ArgumentParser(
        description="Forward for each teacher model to get logits of each token."
    )
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        required=True,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="The input data dir. Should contain the training files.",
    )
    parser.add_argument(
        "--dataset_save_dir",
        type=str,
        required=True,
        help="The local dir to save processed data.",
    )
    parser.add_argument(
        "--dataset_sample_prop",
        type=float,
        default=None,
        help="The prop to sample dataset. Debugging only.",
    )
    parser.add_argument(
        "--dataset_split_num",
        type=int,
        default=None,
        help="The number to split dataset.",
    )
    parser.add_argument(
        "--dataset_index", type=int, default=None, help="The index of current dataset."
    )
    parser.add_argument("--cache_dir", type=str, default=None, help="The cache dir.")
    parser.add_argument(
        "--model_max_length", type=int, default=2048, help="The model max length."
    )
    parser.add_argument(
        "--training_mode", type=str, default="full", help="full or qlora."
    )
    parser.add_argument(
        "--load_in_half", type=str, default="none", help="none or fp16 or bf16."
    )
    parser.add_argument("--batch_size", type=int, default=8, help="The batch size.")
    parser.add_argument(
        "--preprocessing_num_workers",
        type=int,
        default=80,
        help="The number of processes to do data loading.",
    )
    parser.add_argument(
        "--top_k_logits", type=int, default=10, help="The number of logit for saving."
    )
    parser.add_argument(
        "--save_per_token_metric", action="store_true", help="Save per token metric."
    )
    parser.add_argument("--no_assert", default=True, action="store_true", help="Delete the assert.")
    args = parser.parse_args()
    return args

args = parse_args()
model_args = {
        "model_name_or_path": args.model_name_or_path,
        "cache_dir": args.cache_dir,
        "model_max_length": args.model_max_length,
        "training_mode": args.training_mode,
        "use_flash_attn": False
    }

def generate_input_ids():
    # Define the shape
    batch_size = 8
    sequence_length = 2048

    # Generate random input ids
    # Assuming the values range between 1 and 29571 as seen in the example
    max_value = 29571
    input_ids = torch.randint(1, max_value + 1, (batch_size, sequence_length), device='cuda:0')

    # Optional: Introduce padding by setting some values to 0
    # For simplicity, we assume a 50% chance that a token is padding (value 0)
    padding_mask = torch.randint(0, 2, (batch_size, sequence_length), device='cuda:0').bool()
    input_ids[padding_mask] = 0

    return input_ids

def generate_attention_mask(input_ids):
    # The attention mask will be the same shape as input_ids
    attention_mask = (input_ids != 0).int()
    return attention_mask

# Generate the tensor
input_ids_tensor = generate_input_ids()
attention_mask_tensor = generate_attention_mask(input_ids_tensor)

_, model = load_tokenizer_and_model(AttrDict(model_args))
model = model.to(dtype=torch.bfloat16)
if torch.cuda.is_available():
    model = model.cuda()
model.eval()

torch.cuda.empty_cache()

start_time = time.time()
input_ids = input_ids_tensor.cuda()
attention_mask = attention_mask_tensor.cuda()
end_time = time.time()
print(f"Time taken for inputs and masks to cuda: {end_time - start_time} seconds")

start_time = time.time()
with torch.no_grad():
    logits = model(
        input_ids=input_ids, attention_mask=attention_mask
    ).logits.to(torch.float16)
end_time = time.time()
print(f"Time taken for calculating logits: {end_time - start_time} seconds")




# # Assuming logits and targets are already defined as below:
# start_time0 = time.time()
# logits = torch.randn(8, 2048, 32000, device='cuda:0', dtype=torch.float16)
# targets = torch.randint(0, 32000, (8, 2047), device='cuda:0')
# end_time0 = time.time()
# print(f"Time taken for generation: {end_time0 - start_time0} seconds")

# Transform logits as per your description
# start_time1= time.time()
# logits = logits[..., :-1, :].contiguous().view(-1, logits.size(-1))

# Reshape targets to match the logits shape
# targets = targets.view(-1)
# end_time1 = time.time()
# print(f"Time taken for transformations: {end_time1 - start_time1} seconds")

# Measure the time taken for cross entropy computation
# start_time = time.time()
# loss = F.cross_entropy(logits[..., :-1, :].contiguous().view(-1, logits.size(-1)), targets)
# end_time = time.time()


# print(f"Time taken for cross entropy: {end_time - start_time} seconds")