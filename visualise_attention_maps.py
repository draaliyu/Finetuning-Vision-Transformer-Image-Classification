import numpy as np
import matplotlib.pyplot as plt
from vit_keras import vit, utils, visualize

# Load the Vision Transformer model
img_size = 384
imagenet_classes = utils.get_imagenet_classes()
vit_model = vit.vit_l16(
    image_size=img_size,
    activation='sigmoid',
    pretrained=True,
    include_top=True,
    pretrained_top=True
)

# Define the image URL and load the image
image_url = 'example.jpg'
image = utils.read(image_url, img_size)

# Compute the attention map
att_map = visualize.attention_map(model=vit_model, image=image)

# Make a prediction
processed_image = vit.preprocess_inputs(image)[np.newaxis]
prediction = vit_model.predict(processed_image)[0]
predicted_class = imagenet_classes[prediction.argmax()]
print('Prediction:', predicted_class)

# Plot the original image and the attention map
fig, (ax_orig, ax_att) = plt.subplots(ncols=2, figsize=(12, 6))
ax_orig.axis('off')
ax_att.axis('off')
ax_orig.set_title('Original')
ax_att.set_title('Attention Map')
ax_orig.imshow(image)
ax_att.imshow(att_map)
plt.show()
