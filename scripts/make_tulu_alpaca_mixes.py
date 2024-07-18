import json
import argparse
import os
import random
from datasets import load_dataset

random.seed(52009)

parser = argparse.ArgumentParser()
parser.add_argument("--alpaca_vif_data", type=str, required=True)
parser.add_argument("--output_dir", type=str, required=True)
args = parser.parse_args()

tulu_mix = load_dataset("allenai/tulu-v2-sft-mixture", split="train")
alpaca_gpt4 = load_dataset("vicgalle/alpaca-gpt4", split="train")

vif_alpaca = []
id_ = 0
for line in open(args.alpaca_vif_data):
    datum = json.loads(line)
    vif_alpaca.append(
        {
            "dataset": "vif_gpt4_alpaca",
            "id": f"vif_gpt4_alpaca_{id_}",
            "messages": datum["messages"],
        }
    )
    id_ += 1

print(f"Read VIF Alpaca data of size: {len(vif_alpaca)}")

tulu_outputs = set(x["messages"][1]["content"].strip() for x in tulu_mix if x["dataset"] == "gpt4_alpaca")
extra_alpaca = []
no_input_prefix = "Below is an instruction that describes a task. Write a response that appropriately completes the request."
with_input_prefix = "Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request."
id_ = 0
alpaca_data = list(alpaca_gpt4)
random.shuffle(alpaca_data)
for datum in alpaca_gpt4:
    if datum["output"].strip() not in tulu_outputs:
        if datum["input"]:
            prompt = f"{with_input_prefix}\n\n### Instruction:\n{datum['instruction']}\n\n### Input:\n{datum['input']}\n\n### Response:"
        else:
            prompt = f"{no_input_prefix}\n\n### Instruction:\n{datum['instruction']}\n\n### Response:"

        messages = [
            {
                "role": "user",
                "content": prompt,
            },
            {
                "role": "assistant",
                "content": datum["output"]
            }
        ]
        extra_alpaca.append(
            {
                "dataset": "extra_gpt4_alpaca",
                "id": f"extra_gpt4_alpaca_{id_}",
                "messages": messages,
            }
        )
        id_ += 1
    if id_ == len(vif_alpaca) - 1:
        break


if not os.path.exists(args.output_dir):
    os.makedirs(args.output_dir)
tulu_data = list(tulu_mix)
outfile = os.path.join(args.output_dir, "tulu_and_random_alpaca_data.jsonl")
with open(outfile, "w") as f:
    for datum in tulu_data + extra_alpaca:
        print(json.dumps(datum), file=f)

outfile = os.path.join(args.output_dir, "tulu_and_vif_alpaca_data.jsonl")
with open(outfile, "w") as f:
    for datum in tulu_data + vif_alpaca:
        print(json.dumps(datum), file=f)



