import subprocess
import time
import argparse


class Llava:
    def __init__(
        self,
        sysmsg: str = "",
        model_path: str = "liuhaotian/llava-v1.5-7b",
        load_8bit: bool = False,
    ):
        self.sysmsg = sysmsg
        self.model_path = model_path
        self.load_xbit = "--load-8bit" if load_8bit else "--load-4bit"
        self.process = None

    def init_worker(self, prompt, image_path, temperature=0.2):
        command = [
            "python",
            "-m",
            "llava.serve.cli",
            "--model-path",
            self.model_path,
            "--image-file",
            image_path,
            "--temperature",
            str(temperature),
            self.load_xbit,
        ]

        process = subprocess.Popen(
            command,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )

        i = 0
        while i < 100:
            output = process.stdout.readline()
            if "USER:" in output:
                self.process = process
                return
            time.sleep(0.1)  # short delay to avoid overwhelming the CPU
            i += 1

        # throw None if process is not ready after 30 seconds
        print("Error: process not ready after 30 seconds, process is None")
        return

    def prompt_llava(self, prompt):
        stdout, stderr = self.process.communicate(prompt, timeout=10)

        if self.process.returncode != 0:
            # tbd: handle errors if necessary
            print(f"Error: {stderr}")
            return None

        return stdout


if __name__ == "__main__":
    llava = Llava()

    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--prompt", type=str)
    parser.add_argument("-i", "--image", type=str)
    parser.add_argument("-t", "--temperature", type=float, default=0.2)
    args = parser.parse_args()

    # python src/llava.py -p "what do you see here?" -i ./data/mouse\ with_5_gpt-4-vision-preview_dall-e-2_gpt-4-1106-preview_v0/0_image.png -t 0.2
    llava.init_worker(args.prompt, args.image, args.temperature)
    print(llava.prompt_llava(args.prompt))
