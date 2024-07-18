"""Full extraction on examples"""
import os
import itertools
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Union
import dill
import json
from collie.extract_constraints import FullExtractor
from collie.extractor_utils import TextChunker, ConstraintExtractor
from collie.instructions_extractor import InstructionsLoader
from collie.constraint_renderer import ConstraintRenderer
from sent_constraints import SENT_CONSTRAINTS
from para_constraints import PARA_CONSTRAINTS
from word_constraints import WORD_CONSTRAINTS
from passage_constraints import PASSAGE_CONSTRAINTS

#DATASET_NAME = "allenai/WildChat"
#OUTPUT_DIR = "wildchat_data"

#DATASET_NAME = "allenai/tulu-v2-sft-mixture"
#OUTPUT_DIR = "tulu2_data"

DATASET_NAME = "vicgalle/alpaca-gpt4"
OUTPUT_DIR = "alpaca_gpt4_data"

if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

textloader = InstructionsLoader(
    dataset_name=DATASET_NAME,
    randomize=True
)

def preprocess_messages(messages):
    for message in messages:
        if message["role"] == "assistant":
            return message["content"]

null_chunker = TextChunker(
    paragraph_delim="\n",
    randomize=True,
    preprocessor=preprocess_messages,
)

passage_chunker = TextChunker(
    paragraph_delim="\n",
    randomize=True,
    preprocessor=preprocess_messages,
    chunk_by_passage=True,
)

sentence_chunker = TextChunker(
    paragraph_delim="\n",
    randomize=True,
    preprocessor=preprocess_messages,
    chunk_by_sentence=True,
)

word_extractor = FullExtractor(
    chunker = null_chunker,
    loader = textloader,
)

passage_extractor = FullExtractor(
    chunker = passage_chunker,
    loader = textloader,
)

sentence_extractor = FullExtractor(
    chunker = sentence_chunker,
    loader = textloader,
)

def extract_all(
    outdir:str, 
    extractors: Dict[str, FullExtractor],
    constraints: Dict[str, Union[List[ConstraintExtractor], ConstraintExtractor]],
    max_passage:int=30000,
    max_seq_per_passage:int=100,
    ex_per_constraint:int=5000,
    suffix:str = "",
    conj=True
):
    results = defaultdict(dict)
    for (source, extractor), (constr_name, constraint) in itertools.product(
        extractors.items(), constraints.items()
    ):
        extractor.extract(constraint, conjunction=conj, max_documents=max_passage, max_seq_per_document=max_seq_per_passage)
        results[source][constr_name] = extractor.get_constraints(total_examples=ex_per_constraint, conjunction=conj)
        # extractor.inspect_results(f"temp/{source}{suffix}_dump.txt")
    
    for source, r in results.items():
        output_path = Path(outdir).joinpath(f"{source}{suffix}.jsonl")
        print(f"Writing {output_path}")
        num_data = 0
        with output_path.open(mode="w") as f:
            for constraint_id, extracted_data in r.items():
                for datum in extracted_data:
                    renderer = ConstraintRenderer(constraint=datum["constraint"], check_value=datum["targets"])
                    constraint_prompt = renderer.prompt
                    messages = datum["metadata"]["messages"]
                    updated_instruction = "\n".join([datum["metadata"]["prompt_prefix"], constraint_prompt, messages[0]["content"]])
                    messages[0]["content"] = updated_instruction
                    output_datum = {
                        "constraint_id": constraint_id,
                        "constraint_prompt": constraint_prompt,
                        "passing_text": datum["example"],
                        "messages": datum["metadata"]["messages"],
                        "index": datum["metadata"]["iter_index"],
                    }
                    print(json.dumps(output_datum), file=f)
                    num_data += 1
        print(f"Wrote {num_data} data points")


print("Extracting sentence level constraint data")
extract_all(
    OUTPUT_DIR,
    extractors={"sent": sentence_extractor},
    constraints=SENT_CONSTRAINTS,
    suffix="_sent",
)

print("Extracting paragraph level constraint data")
extract_all(
    OUTPUT_DIR,
    extractors={"word": word_extractor},
    constraints=PARA_CONSTRAINTS,
    suffix="_para",
)

# print("Extracting word level constraint data")
# extract_all(
#     OUTPUT_DIR,
#     extractors={"word": word_extractor},
#     constraints=WORD_CONSTRAINTS,
#     suffix="_word",
#     max_seq_per_passage=None,
# )

print("Extracting doc level constraint data")
extract_all(
    OUTPUT_DIR,
    extractors={"passage": passage_extractor},
    constraints=PASSAGE_CONSTRAINTS,
    suffix="_passage",
    max_seq_per_passage=None,
)