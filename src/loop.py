from oai_api import GPT
from storage import Storage


class Loop:
    def __init__(
        self,
        max_iteration_count,
        initial_prompt,
        temperature,
        gpt: GPT,
        output_dir,
        image_system_message,
        prompt_system_message,
    ) -> None:
        self.max_iteration_count = max_iteration_count
        self.initial_prompt = initial_prompt
        self.gpt = gpt
        self.temperature = temperature
        self.prompt_embeddings = []
        self.difference_embeddings = []
        self.storage = Storage(
            output_dir,
            max_iteration_count,
            initial_prompt,
            gpt.multimodal_model,
            gpt.image_model,
            gpt.prompt_model,
            image_system_message,
            prompt_system_message,
            temperature,
        )
        self.img_sysmsg = image_system_message
        self.prompt_sysmsg = prompt_system_message

    def run(self):
        messages = [
            {"role": "user", "content": ""},
            {"role": "assistant", "content": self.initial_prompt},
        ]
        image = self.gpt.generate_image(self.initial_prompt)
        prompt_embeddings = self.gpt.get_embeddings(self.initial_prompt)
        self.prompt_embeddings.append(prompt_embeddings)
        self.storage.init_data(self.gpt.get_image_url(image))

        for i in range(self.max_iteration_count):
            url = self.gpt.get_image_url(image)
            print(f"Iteration {i + 1}:")
            differences = self.gpt.detect_differences(
                self.initial_prompt,
                url,
                system_message=self.img_sysmsg,
                max_tokens=200,
                temperature=self.temperature,
            )
            difference_embeddings = self.gpt.get_embeddings(differences)
            self.difference_embeddings.append(difference_embeddings)
            print(f"Differences: {differences}")
            messages += [{"role": "user", "content": differences}]
            if "DONE" in differences:
                break
            new_prompt = self.gpt.generate_prompt_from_differences(
                messages, self.temperature, system_message=self.prompt_sysmsg
            )
            prompt_embeddings = self.gpt.get_embeddings(new_prompt)
            self.prompt_embeddings.append(prompt_embeddings)
            messages += [{"role": "assistant", "content": new_prompt}]
            print(f"New prompt: {new_prompt}")
            if "DONE" in new_prompt:
                break
            image = self.gpt.generate_image(new_prompt)
            self.storage.store_next_iteration(
                self.gpt.get_image_url(image),
                differences,
                new_prompt,
                self.prompt_embeddings,
                self.difference_embeddings,
                messages,
            )

        return i
