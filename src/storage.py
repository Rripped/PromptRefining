import requests
import shutil
import urllib.request
import os
import sys
import json


class Storage:
    def __init__(
        self,
        path,
        max_iteration_count,
        initial_prompt,
        multimodal_model,
        image_model,
        prompt_model,
    ) -> None:
        self.max_iteration_count = max_iteration_count
        self.initial_prompt = initial_prompt
        self.multimodal_model = multimodal_model
        self.image_model = image_model
        self.prompt_model = prompt_model
        self.i = 0

        self.init_dir(path)
        self.store_config()

    def init_dir(self, path, name=None):
        if name and not os.path.exists(os.path.join(path, name)):
            self.dir = os.path.join(path, name)
            os.makedirs(self.dir)
            return
        elif name and os.path.exists(os.path.join(path, name)):
            dir = os.path.join(path, name)
            sys.exit(f"{dir} already exists")

        cut_prompt = self.initial_prompt.replace(" ", "_")[:12]

        dir = os.path.join(
            path,
            f"{cut_prompt}_{self.max_iteration_count}_{self.multimodal_model}_{self.image_model}_{self.prompt_model}_v0",
        )

        if not os.path.exists(dir):
            self.dir = dir
            os.makedirs(self.dir)
            return
        else:
            while os.path.exists(dir):
                version = int(dir.split("_")[-1][1:])
                version += 1
                dir = dir[:-1] + str(version)

            self.dir = dir
            os.makedirs(self.dir)
            return

    def store_config(self):
        # store config as a json file

        config = {
            "max_iteration_count": self.max_iteration_count,
            "initial_prompt": self.initial_prompt,
            "multimodal_model": self.multimodal_model,
            "image_model": self.image_model,
            "prompt_model": self.prompt_model,
        }

        with open(f"{self.dir}/config.json", "w") as f:
            json.dump(config, f)

    def init_data(self, image_url, text_embedding):
        res = requests.get(image_url)
        with open(os.path.join(self.dir, f"{self.i}_image.png"), "wb") as f:
            f.write(res.content)

        with open(f"{self.dir}/history.md", "a") as f:
            f.write(f'# Prompt "{self.initial_prompt}"\n\n')
            # embed image in markdown
            f.write(f"![{self.i}_image.png]({self.i}_image.png)\n\n")

        with open(os.path.join(self.dir, f"{self.i}_text_embedding.json"), "w") as f:
            json.dump(text_embedding, f)

        self.i += 1

    def store_next_iteration(self, image_url, differences, prompt, text_embedding):
        # store image in for current iteration in dir
        res = requests.get(image_url)
        with open(os.path.join(self.dir, f"{self.i}_image.png"), "wb") as f:
            f.write(res.content)

        # store differences and prompt per iteration as markdown including the images

        with open(os.path.join(self.dir, f"history.md"), "a") as f:
            f.write(f"\n## Iteration {self.i}\n\n")
            # embed image in markdown
            f.write(f"![{self.i}_image.png]({self.i}_image.png)\n")
            # write differences
            f.write(f"\n{differences}\n\n")
            # embed prompt in markdown
            f.write(f"### Prompt\n\n{prompt}")

        # store text embedding per iteration as json
        with open(os.path.join(self.dir, f"{self.i}_text_embedding.json"), "w") as f:
            json.dump(text_embedding, f)

        self.i += 1
