import torch
from diffusers import StableDiffusionPipeline
import matplotlib.pyplot as plt

# Check if GPU is available
device = "cuda" if torch.cuda.is_available() else "cpu"

# Load the Stable Diffusion model
pipe = StableDiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4")
pipe.to(device)

# Function to generate an image
def generate_image(user_prompt):
    image = pipe(user_prompt).images[0]  # Generate image
    return image

# User prompt
prompt = "Ironman fighting with batman in space"
generated_image = generate_image(prompt)

# Display the generated image
plt.imshow(generated_image)
plt.axis("off")
plt.show()
