import torch
import gc
from diffusers import DiffusionPipeline
from xformers.ops import MemoryEfficientAttentionFlashAttentionOp

# Clear CUDA cache
torch.cuda.empty_cache()


pipe = DiffusionPipeline.from_pretrained(
    "stabilityai/stable-diffusion-2-1", torch_dtype=torch.float16
)
pipe = pipe.to("cuda")
# pipe.enable_xformers_memory_efficient_attention(
#     attention_op=MemoryEfficientAttentionFlashAttentionOp
# )
# # Workaround for not accepting attention shape using VAE for Flash Attention
# pipe.vae.enable_xformers_memory_efficient_attention(attention_op=None)

generated_image = pipe("An image of a squirrel in Picasso style").images[0]
generated_image.save("generated_image.jpg")

del generated_image
torch.cuda.empty_cache()
gc.collect()
