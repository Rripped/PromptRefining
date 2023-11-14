# PromptRefining

Project Foundation Models WT23 Uni Stuttgart

1. *By creating a feedback loop between ChatGPT-4V and a Text-to-Image Model, we aim to improve the precision of the image generated by an original prompt.*
2. *By evaluating the alignment iteratively, we can deduce blindspots in current TTI models and (multimodal) LLMs like GPT-4V.* 
3. *By detaching the comparison to the original prompt in the loop, we can visualize converging patterns of the TTI models and dig deeper into perceptual blindspots of GPT-4V.* 

## Research Questions/Goals

1. Feedback Loop Enhancement of TTI Models and ChatGPT-4V: "Does the integration of a feedback loop between ChatGPT-4V and a Text-to-Image (TTI) model enhance the precision of images generated based on an initial prompt?" 
2. Detecting Blindspots in Image Interpretation: "What are the blindspots of ChatGPT-4V in Image Recognition?"
3. Detecting Blindspots in Image Generation and Prompt Interpretation: "What are the blindspots of current TTI models, when interpreting prompts and generating images?"

## Methods to answer the Research Questions

Run the feedback loop with simple initial prompts, analyze iterations and divergent/convergent features.

Check the differences between iterations in the embedding space.

Run the loop, but instead of trying to align to an initial prompt, focus on holding first image stable (describe -> prompt -> image -> describe -> prompt -> image...).

Analyze convergent patterns and regularities.

## How to evaluate Image precision?

Intuition: If a prompt says "a cheese and a mouse", the image should not contain additional specifica (e.g. mouse with clothes, a cheeseplate...)

To measure the precision, subjective manual analysis, combined with embedding analysis is used.

## Related Work

- [ChatGPT Asks, BLIP-2 Answers: Automatic Questioning Towards Enriched Visual Descriptions](https://www.semanticscholar.org/paper/ChatGPT-Asks%2C-BLIP-2-Answers%3A-Automatic-Questioning-Zhu-Chen/69cfdc8df16ae63b7acba4ac6f727f78b86893c3)
- [HuggingGPT: Solving AI Tasks with ChatGPT and its Friends in Hugging Face](https://arxiv.org/abs/2303.17580)
- [LLM Lies: Hallucinations are not Bugs, but Features as Adversarial Examples](https://arxiv.org/abs/2310.01469)
- [Knowledge Injection to Counter Large Language Model (LLM) Hallucination](https://link.springer.com/chapter/10.1007/978-3-031-43458-7_34)
- [The Internal State of an LLM Knows When It’s Lying](https://arxiv.org/pdf/2304.13734.pdf)
- [ChatGPT and Simple Linguistic Inferences: Blind Spots and Blinds](https://arxiv.org/abs/2305.14785)
