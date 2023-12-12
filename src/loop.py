from oai_api import GPT
import requests
import shutil
import urllib.request
import os
import sys
import json

from storage import Storage


class Loop:
    def __init__(
        self,
        max_iteration_count,
        initial_prompt,
        temperature,
        gpt: GPT,
        output_dir,
        img_sysmsg,
        prompt_sysmsg,
    ) -> None:
        self.max_iteration_count = max_iteration_count
        self.initial_prompt = initial_prompt
        self.gpt = gpt
        self.temperature = temperature
        self.embeddings = []
        self.storage = Storage(
            output_dir,
            max_iteration_count,
            initial_prompt,
            gpt.multimodal_model,
            gpt.image_model,
            gpt.prompt_model,
        )
        self.img_sysmsg = img_sysmsg
        self.prompt_sysmsg = prompt_sysmsg

    def run(self):
        messages = [
            {"role": "user", "content": ""},
            {"role": "assistant", "content": self.initial_prompt},
        ]
        image = self.gpt.generate_image(self.initial_prompt)

        initial_embedding = self.gpt.get_embeddings(self.initial_prompt)  # embedding
        self.embeddings += initial_embedding

        # init history storage with initial image
        self.storage.init_data(self.gpt.get_image_url(image), initial_embedding)

        # load json of sysmsg

        for i in range(self.max_iteration_count):
            url = self.gpt.get_image_url(image)
            print(f"Iteration {i + 1}:")
            differences = self.gpt.detect_differences(
                self.initial_prompt,
                url,
                system_message=self.img_sysmsg,
                max_tokens=150,
                temperature=self.temperature,
            )  # differences

            print(f"Differences: {differences}")

            if "DONE" in differences:
                break

            messages += [{"role": "user", "content": differences}]
            new_prompt = self.gpt.generate_prompt_from_differences(
                messages, self.temperature, system_message=self.prompt_sysmsg
            )  # prompt
            embedding = self.gpt.get_embeddings(new_prompt)  # embedding
            self.embeddings += embedding
            messages += [{"role": "assistant", "content": new_prompt}]
            print(f"New prompt: {new_prompt}")
            image = self.gpt.generate_image(new_prompt)  # image

            self.storage.store_next_iteration(
                self.gpt.get_image_url(image), differences, new_prompt, embedding
            )

        return i
