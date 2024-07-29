"""Full extraction on examples"""
import os
import itertools
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Union
import json
import argparse
from collie.extract_constraints import FullExtractor
from collie.extractor_utils import TextChunker, ConstraintExtractor
from collie.instructions_extractor import InstructionsLoader
from collie.constraint_renderer import ConstraintRenderer
import alpaca_constraints


def extract_all(
    outdir:str, 
    extractors: Dict[str, FullExtractor],
    constraints: Dict[str, Union[List[ConstraintExtractor], ConstraintExtractor]],
    max_passage:int=500,
    max_seq_per_passage:int=100,
    ex_per_constraint:int=100,
    suffix:str = "",
    conj=True
):
    results = defaultdict(dict)
    for (source, extractor), (constr_name, constraint) in itertools.product(
        extractors.items(), constraints.items()
    ):
        print(f"Extracting data satisfying constraint: {constr_name}")
        extractor.extract(constraint, conjunction=conj, max_documents=max_passage, max_seq_per_document=max_seq_per_passage)
        results[source][constr_name] = extractor.get_constraints(total_examples=ex_per_constraint, conjunction=conj)
    
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

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="vicgalle/alpaca-gpt4")
    parser.add_argument("--output_dir", type=str, default="alpaca_gpt4_data")
    parser.add_argument(
        "--num_words_for_presence",
        type=int,
        default=100,
        help="To define constraints that look for presence or absence of specific words, how many words should we sample from the dataset?",
    )
    args = parser.parse_args()

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    text_loader = InstructionsLoader(
        dataset_name=args.dataset,
        randomize=True
    )

    def preprocess_messages(messages):
        for message in messages:
            if message["role"] == "assistant":
                return message["content"]

    chunker = TextChunker(
        paragraph_delim="\n\n\n\n",
        randomize=True,
        preprocessor=preprocess_messages,
    )

    extractor = FullExtractor(
        chunker = chunker,
        loader = text_loader,
    )

    print("Extracting length constraint data")
    extract_all(
        args.output_dir,
        extractors={"passage": extractor},
        constraints=alpaca_constraints.LENGTH_CONSTRAINTS,
        suffix="_length",
        max_passage=30000,
        ex_per_constraint=200,
        max_seq_per_passage=None,
    )

    print("Extracting position constraint data")
    extract_all(
        args.output_dir,
        extractors={"passage": extractor},
        constraints=alpaca_constraints.POSITION_CONSTRAINTS,
        suffix="_position",
        max_passage=30000,
        ex_per_constraint=1000,
        max_seq_per_passage=None,
    )

    print(f"Extracting length and position constraint data")
    extract_all(
        args.output_dir,
        extractors={"passage": extractor},
        constraints=alpaca_constraints.LENGTH_AND_POSITION_CONSTRAINTS,
        suffix="_length_position",
        max_passage=30000,
        ex_per_constraint=1000,
        max_seq_per_passage=None,
    )

    word_frequencies = text_loader.get_token_frequencies()
    print("Extracting word presence constraint data")
    extract_all(
        args.output_dir,
        extractors={"passage": extractor},
        constraints=alpaca_constraints.get_word_presence_constraints(word_frequencies=word_frequencies),
        suffix="_word_presence",
        max_passage=10000,
        ex_per_constraint=500,
        max_seq_per_passage=None,
    )