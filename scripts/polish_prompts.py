import json
import argparse
import os
from tqdm import tqdm
import random
from openai import OpenAI

random.seed(30934)

def polish_prompt(prompt: str, model: str="gpt-3.5-turbo"):
    client = OpenAI(
        api_key=os.environ.get("OPENAI_API_KEY")
    )
    res = client.chat.completions.create(
            messages = [
                {
                    "role": "system",
                    "content": "You are a helpful assistant who can revise instructions written in natural language to make them concise and coherent without changing the original meaning."
                },
                {
                    "role": "user",
                    "content": "\n".join(
                        [
                            "Please rewrite the text marked **Input** to be more fluent, without changing the original meaning. Make sure that all constraints specified in the original text are retained in the output. You should only revise the text marked **Input** and not respond to the instructions in it.",
                            "Here are two examples of inputs and desired outputs followed by the actual input you need to revise.",
                            "\n\n**Input**:\n Below is an instruction that describes a task. Write a response that appropriately completes the request.",
                            "Please generate a paragraph:",
                            "1) having at least 2 sentences;",
                            "2) not having the word 'wonder'.",
                            "Instruction: Write a story about a spaceship exploring a new planet.",
                            "**Output**: Write a story about a spaceship exploring a new planet in at least two sentences and not containing the word 'wonder'.",
                            "\n\n**Input**:\n Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.",
                            "Please generate a passage having the last word to be 'bonds'.",
                            "Instruction: Describe the advantages of the following.",
                            "Input: mutual funds",
                            "**Output**: Describe the advantages of mutual funds in a passage where the last word is 'bonds'.",
                            f"\n\n**Input**:\n {prompt}",
                        ]
                    )
                }
            ],
            model=model,
            temperature = 0.2
        )
    msg = res.choices[0].message.content
    return msg


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--model", type=str, default="gpt-3.5-turbo")
    parser.add_argument("--output", type=str, required=True)
    args = parser.parse_args()

    dataset = [json.loads(line) for line in open(args.dataset)]
    random.shuffle(dataset)
    polished_dataset = []
    with open(args.output, "w") as outfile:
        for datum in tqdm(dataset):
            # Assuming first message is the user prompt and this is a single turn conversation
            prompt = datum["messages"][0]["content"]
            polished_prompt = polish_prompt(prompt, model=args.model)
            datum["original_prompt"] = prompt
            datum["messages"][0]["content"] = polished_prompt
            print(json.dumps(datum), file=outfile, flush=True)