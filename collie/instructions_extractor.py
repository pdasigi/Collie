import random
import copy
from datasets import load_dataset
from collie.extractor_utils import TextLoader
from tqdm import tqdm
from collections import defaultdict
from nltk import word_tokenize

class InstructionsLoader(TextLoader):
    def __init__(self,
                 dataset_name: str = "allenai/tulu-v2-sft-mixture",
                 randomize: bool = False,
                 **kwargs) -> None:
        dataset = load_dataset(dataset_name, split="train", **kwargs)
        print("Filtering dataset")
        if "tulu" in dataset_name:
            self.dataset = self.filter_tulu(dataset)
        elif "WildChat" in dataset_name:
            self.dataset = self.filter_wildchat(dataset)
        elif "alpaca" in dataset_name:
            self.dataset = self.filter_alpaca_gpt4(dataset)
        print(f"Dataset size: before filtering - {len(dataset)}; after filtering - {len(self.dataset)}")
        self.randomize = randomize

    def filter_tulu(self, dataset):
        filtered_dataset = []
        for datum in tqdm(dataset):
            if "messages" in datum:
                messages = datum["messages"]
            # Ignore conversations that are not single-turn
            if len(messages) != 2:
                continue
            filtered_dataset.append(datum)
        return filtered_dataset

    def filter_wildchat(self, dataset):
        filtered_dataset = []
        for datum in tqdm(dataset):
            # Ignore non-English conversations
            if "language" in datum and datum["language"] != "English":
                continue
            if "toxic" in datum and datum["toxic"]:
                continue
            messages = datum.pop("conversation")
            datum["messages"] = messages
            # Ignore conversations that are not single-turn
            if len(messages) != 2:
                continue
            filtered_dataset.append(datum)
        return filtered_dataset

    def filter_alpaca_gpt4(self, dataset):
        filtered_dataset = []
        tulu_mix = load_dataset("allenai/tulu-v2-sft-mixture", split="train")
        tulu_alpaca_responses = set([d["messages"][1]["content"].strip() for d in tulu_mix if d["dataset"] == "gpt4_alpaca"])
        prefix_without_input = "Below is an instruction that describes a task. Write a response that appropriately completes the request."
        prefix_with_input = "Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request."
        for datum in tqdm(dataset):
            # Removing those instances already in the Tulu mix
            if datum["output"].strip() in tulu_alpaca_responses:
                continue
            processed_datum = {}
            if datum["input"]:
                processed_datum["prompt_prefix"] = prefix_with_input
                user_prompt = f"Instruction: {datum['instruction']}\nInput: {datum['input']}"
            else:
                processed_datum["prompt_prefix"] = prefix_without_input
                user_prompt = f"Instruction: {datum['instruction']}"
            messages = [
                {
                    "role": "user",
                    "content": user_prompt
                },
                {
                    "role": "assistant",
                    "content": datum["output"]
                }
            ]
            processed_datum["messages"] = messages
            filtered_dataset.append(processed_datum)
        return filtered_dataset

    def __iter__(self):
        self.indices = list(range(len(self.dataset)))
        if self.randomize:
            random.shuffle(self.indices)
        def metadata_iter():
            for idx in self.indices:
                datadict = copy.deepcopy(self.dataset[idx])
                datadict["iter_index"] = idx
                yield datadict["messages"], datadict 
        self._metadata_iter = metadata_iter()
        return self

    def __next__(self):
        messages, metadata = next(self._metadata_iter)
        return messages, metadata
    
    def __len__(self):
        return len(self.dataset)

    def get_token_frequencies(self):
        # Returns a dict with token -> number of documents containing that token
        word_frequencies = defaultdict(int)
        for d in self.dataset:
            response = d["messages"][1]["content"]
            tokens = word_tokenize(response)
            for token in set(tokens):
                word_frequencies[token.lower()] += 1
        return dict(word_frequencies)


if __name__ == "__main__":
    from constraints import (
        TargetLevel,
        Count,
        Relation,
        Position,
        Reduction
    )
    from extractor_utils import TextChunker, ConstraintExtractor
    from extract_constraints import FullExtractor

    textloader = InstructionsLoader(
        dataset_name="allenai/WildChat",
        randomize=True
    )

    def preprocess_messages(messages):
        for message in messages:
            if message["role"] == "assistant":
                return message["content"]

    chunker = TextChunker(
        paragraph_delim="\n",
        randomize=True,
        preprocessor=preprocess_messages,
        chunk_by_passage=True,
    )

    extractor = FullExtractor(
        chunker = chunker,
        loader = textloader,
    )

    # First constraint
    constr_extractor = ConstraintExtractor(
        init_range = {
            "target_level": [TargetLevel("word")],
            "transformation": [Count()],
            "relation": [Relation("==")]
        },
        target_range = list(range(5,15))
    )

    extractor.extract(constr_extractor, max_documents=10)
    extractor.print_examples(num=3)

    # Second constraint
    constr_extractor = ConstraintExtractor(
        init_range = {
            "target_level": [TargetLevel("word")],
            "transformation": [Position([0, 1, 3]), Position([3, 5, 9])],
            "relation": [Relation("==")],
            "reduction": [Reduction("all")]
        },
        target_range = None
    )

    extractor.extract(constr_extractor, max_documents=10)
    extractor.print_examples(num=3)