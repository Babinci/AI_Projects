from aura_sr import AuraSR
from PIL import Image

# Load the AuraSR model
aura_sr = AuraSR.from_pretrained("fal/AuraSR-v2")

# Define the image path
image_path = "input.jpg"

# Load the image without resizing
image = Image.open(image_path)

# Convert RGBA to RGB if the image has an alpha channel
if image.mode == 'RGBA':
    # Convert to RGB by removing the alpha channel or compositing against a white background
    background = Image.new('RGB', image.size, (255, 255, 255))
    background.paste(image, mask=image.split()[3])  # Use alpha channel as mask
    image = background
elif image.mode != 'RGB':
    # Handle other color modes by converting to RGB
    image = image.convert('RGB')

# Upscale the image
upscaled_image = aura_sr.upscale_4x_overlapped(image)

# Save the upscaled image
upscaled_image.save("image_out.png")