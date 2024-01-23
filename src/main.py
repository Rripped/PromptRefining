from loop import Loop
from oai_api import GPT
import os
from dotenv import load_dotenv
import argparse
import json

load_dotenv()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--max_iteration-count", type=int, default=1)
    parser.add_argument("-p", "--initial-prompt", type=str)
    parser.add_argument("-o", "--output-dir", type=str, default="./data")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    system_messages = json.load(open("./src/system_messages.json"))

    gpt = GPT(os.environ["OAI"], image_model="dall-e-3")
    loop = Loop(
        args.max_iteration_count,
        args.initial_prompt,
        temperature=0.4,
        gpt=gpt,
        output_dir=args.output_dir,
        image_system_message=system_messages["image_system_message"][6],
        prompt_system_message=system_messages["prompt_system_message"][5],
    )
    loop.run()
