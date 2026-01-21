import numpy as np
import random
from datetime import datetime, timedelta
from PIL import Image, ImageDraw, ImageFont

def get_muted_colors():
    tans = [(160, 140, 115), (145, 120, 105), (175, 155, 135)]
    peaches = [(190, 140, 110), (180, 130, 105), (205, 150, 125)]
    return random.choice(tans), random.choice(peaches)

def create_pillow_gradient(width, height):
    c1, c2 = get_muted_colors()
    mask = Image.linear_gradient('L') 
    channels = []
    for i in range(3):
        diff = c2[i] - c1[i]
        lut = [int(c1[i] + (diff * (x / 255.0))) for x in range(256)]
        channels.append(mask.point(lut))
    return Image.merge("RGB", channels).resize((width, height)).convert("RGBA")

def generate_image_and_label():
    """Returns both the image array and the date string (the label)."""
    start_date = datetime(2009, 4, 1)
    end_date = datetime(2010, 1, 31)
    target_date = start_date + timedelta(days=random.randrange((end_date - start_date).days))
    
    is_weekend = target_date.weekday() >= 5
    bg_text = "WEEKEND" if is_weekend else "WEEKDAY"
    
    month_day = f"{str(target_date.month).zfill(2)}/{str(target_date.day).zfill(2)}"
    
    weekday = target_date.strftime("%a").capitalize()
    time_of_day = random.choice(["After School", "Evening", "Morning", "Afternoon"])

    width, height = 750, 350
    img = create_pillow_gradient(width, height)
    draw = ImageDraw.Draw(img)

    try:
        font_date = ImageFont.truetype("ariblk.ttf", 130)
        font_bg = ImageFont.truetype("ariblk.ttf", 160)
        font_sub = ImageFont.truetype("ariblk.ttf", 50)
    except:
        font_date = ImageFont.load_default(size=130)
        font_bg = ImageFont.load_default(size=160)
        font_sub = ImageFont.load_default(size=50)

    draw.text((-20, 20), bg_text, font=font_bg, fill=(40, 120, 255, 70))
    draw.polygon([(0, 40), (width, 0), (width, 180), (0, 220)], fill=(30, 144, 255, 110))
    draw.text((60, 45), month_day, font=font_date, fill=(255, 255, 255))
    draw.text((180, 165), time_of_day, font=font_sub, fill=(150, 255, 255))

    return np.array(img.convert("RGB")), month_day

def save_database(n_images=2000, filename="image_dataset.npz"):
    images_list = []
    labels_list = []
    
    print(f"Generating {n_images} images and labels...")
    
    for i in range(n_images):
        img_array, date_label = generate_image_and_label()
        images_list.append(img_array)
        labels_list.append(date_label)
        
        if i % 500 == 0 and i > 0:
            print(f"Progress: {i}/{n_images}...")
    
    np.savez_compressed(
        filename, 
        images=np.array(images_list), 
        labels=np.array(labels_list)
    )
    print(f"Done, saved {n_images} samples to {filename}")

if __name__ == "__main__":
    save_database(20000)