# Data

You can find our most interesting runs here, separated by:

## Detached Runs

In these runs we have detached the initial prompt from the loop to observe the model's behavior in recognizing and generating the image and prompts to hold the scene stable.
We observed interesting concept drifts as the model increases salient features to the extreme over time. E.g. for "realistic portrait of an old man" the eyes color changes to strong blue and the skin becomes extremely wrinkled. Additionally the beard becomes a mustache which curls stronger over the iterations.

We conducted these runs to get a better understanding of the model's blind spots and focus points in image recognition and generation.

## Successful runs

In these runs the image adapted accurately to the initial prompt through iterative prompt improvement in our feedback loop. We observed some minor misinterpretations by GPT-4V but the model was able to correct them in the next iterations.

## Failed runs

In these runs the image alignment failed due to misinterpretation of the image's differences to the prompt. In most cases the model terminated too early, ignoring obvious differences, especially in the case of object counts the recognition fails.

## Stable Diffusion

This folder contains some example images generated, based on the prompts used in other runs. It mostly demonstrates the lower quality in image generation compared to DALL-E 3.
