import requests
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
        image_system_message,
        prompt_system_message,
    ) -> None:
        self.max_iteration_count = max_iteration_count
        self.initial_prompt = initial_prompt
        self.multimodal_model = multimodal_model
        self.image_model = image_model
        self.prompt_model = prompt_model
        self.image_system_message = image_system_message
        self.prompt_system_message = prompt_system_message
        self.init_dir(path)
        self.history = os.path.join(self.dir, f"history.md")
        self.i = 0
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
        config = {
            "max_iteration_count": self.max_iteration_count,
            "initial_prompt": self.initial_prompt,
            "multimodal_model": self.multimodal_model,
            "image_model": self.image_model,
            "prompt_model": self.prompt_model,
            "image_system_message": self.image_system_message,
            "prompt_system_message": self.prompt_system_message,
        }

        with open(f"{self.dir}/config.json", "w") as f:
            json.dump(config, f)

    def init_data(self, image_url):
        res = requests.get(image_url)
        with open(os.path.join(self.dir, f"{self.i}_image.png"), "wb") as f:
            f.write(res.content)

        with open(f"{self.dir}/history.md", "a") as f:
            f.write(f'# Prompt "{self.initial_prompt}"\n\n')
            # embed image in markdown
            f.write(f"![{self.i}_image.png]({self.i}_image.png)\n\n")
        self._increase_iteration_count()

    def store_next_iteration(
        self,
        image_url,
        differences,
        prompt,
        prompt_embeddings,
        difference_embeddings,
        messages,
    ):
        self._write_iteration_count()
        self._write_differences(differences)
        self._write_prompt(prompt)
        self._write_image(image_url)
        self._write_messages(messages)
        self._write_prompt_embeddings(prompt_embeddings)
        self._write_difference_embeddings(difference_embeddings)
        self._increase_iteration_count()

    def _increase_iteration_count(self):
        self.i += 1

    def _write_iteration_count(self):
        with open(self.history, "a") as f:
            f.write(f"\n## Iteration {self.i}\n\n")

    def _write_image(self, image_url):
        image = requests.get(image_url).content
        with open(os.path.join(self.dir, f"{self.i}_image.png"), "wb") as f:
            f.write(image)
        with open(self.history, "a") as f:
            f.write(f"### Image\n\n![{self.i}_image.png]({self.i}_image.png)\n\n")

    def _write_differences(self, differences):
        with open(self.history, "a") as f:
            f.write(f"### Differences\n\n{differences}\n\n")

    def _write_prompt(self, prompt):
        with open(self.history, "a") as f:
            f.write(f"### Prompt\n\n{prompt}\n\n")

    def _write_messages(self, messages):
        with open(os.path.join(self.dir, f"{self.i}_messages.json"), "w") as f:
            f.write(json.dumps(messages, indent=4))

    def _write_prompt_embeddings(self, prompt_embeddings):
        with open(os.path.join(self.dir, "prompt_embeddings.json"), "w") as f:
            f.write(json.dumps(prompt_embeddings, indent=4))

    def _write_difference_embeddings(self, difference_embeddings):
        with open(os.path.join(self.dir, "difference_embeddings.json"), "w") as f:
            f.write(json.dumps(difference_embeddings, indent=4))
