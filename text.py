import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import os

def input(img_path):
    img = Image.open(img_path).convert('L').resize((8, 8))
    img_array = np.array(img).astype(np.float32)
    processed_img = 16 - (img_array * 16.0/255.0 )
    
    # 3. Display
    plt.figure(figsize=(4,4))
    plt.imshow(processed_img , cmap='gray') 
    plt.show()

my_image = "/home/krish/Downloads/Flask/Screenshot From 2026-04-07 11-20-57.png"

input(my_image)