from loop import Loop
from oai_api import GPT
import os
from dotenv import load_dotenv
import argparse

load_dotenv()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--max_iteration-count", type=int, default=1)
    parser.add_argument("-p", "--initial-prompt", type=str)
    parser.add_argument("-o", "--output-dir", type=str, default="./data")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    gpt = GPT(os.environ["OAI"])
    loop = Loop(
        args.max_iteration_count, args.initial_prompt, gpt, output_dir=args.output_dir
    )
    loop.run()
