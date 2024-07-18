import json
import argparse
import random
from collections import defaultdict

random.seed(20934)

parser = argparse.ArgumentParser()
parser.add_argument("--input_files", type=str, nargs="+", required=True)
parser.add_argument("--output_file", type=str, required=True)
args = parser.parse_args()

indexed_data = defaultdict(list)
for input_file in args.input_files:
    for line in open(input_file):
        datum = json.loads(line)
        if "(empty)" in datum["constraint_prompt"]:
            continue
        indexed_data[datum["index"]].append(datum)

num_constraints = [len(v) for v in indexed_data.values()]
print(f"Data points with constraints: {len(indexed_data)}")
print(f"Num constraints per datum: {sum(num_constraints) / len(num_constraints)}")
with open(args.output_file, "w") as outfile:
    for instances in indexed_data.values():
        if len(instances) == 1:
            index = 0
        else:
            index = random.randint(0, len(instances) - 1)

        print(json.dumps(instances[index]), file=outfile)