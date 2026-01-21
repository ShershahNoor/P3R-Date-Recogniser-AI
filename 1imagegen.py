import random
from datetime import datetime, timedelta
from PIL import Image, ImageDraw, ImageFont

def get_muted_colors():
    tans = [
        (160, 140, 115),
        (145, 120, 105), 
        (175, 155, 135)  
    ]
    peaches = [
        (190, 140, 110), 
        (180, 130, 105), 
        (205, 150, 125)  
    ]
    return random.choice(tans), random.choice(peaches)

def create_pillow_gradient(width, height):
    c1, c2 = get_muted_colors()
    
    generate = Image.linear_gradient('L') 
    
    channels = []
    for i in range(3):
        diff = c2[i] - c1[i]
        lut = [int(c1[i] + (diff * (x / 255.0))) for x in range(256)]
        channels.append(generate.point(lut))

    gradient = Image.merge("RGB", channels).resize((width, height))
    return gradient.convert("RGBA")

def generate_persona_date_v4():
    start_date = datetime(2009, 4, 1)
    end_date = datetime(2010, 1, 31)
    target_date = start_date + timedelta(days=random.randrange((end_date - start_date).days))
    
    is_weekend = target_date.weekday() >= 5
    bg_text = "WEEKEND" if is_weekend else "WEEKDAY"
    month_day = f"{target_date.month}/{target_date.day}"
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

    date_pos = (60, 45)
    for ox, oy in [(-1, -1), (1, -1), (-1, 1), (1, 1)]:
        draw.text((date_pos[0]+ox, date_pos[1]+oy), month_day, font=font_date, fill=(255, 255, 255))
    draw.text(date_pos, month_day, font=font_date, fill=(255, 255, 255))

    date_bbox = draw.textbbox(date_pos, month_day, font=font_date)
    tri_right = date_bbox[2] + 110
    tri_width = 100
    tri_points = [
        (tri_right - tri_width, 60), 
        (tri_right, 60),             
        (tri_right - (tri_width // 2), 140)
    ]
    draw.polygon(tri_points, fill=(10, 10, 20)) 
    
    wday_bbox = draw.textbbox((0, 0), weekday, font=font_sub)
    wday_w = wday_bbox[2] - wday_bbox[0]
    draw.text((tri_right - wday_w - 5, 65), weekday, font=font_sub, fill=(255, 255, 255))

    time_x, time_y = 180, 165
    cyan = (150, 255, 255)
    for dx in range(-4, 5):
        for dy in range(-4, 5):
            draw.text((time_x + dx, time_y + dy), time_of_day, font=font_sub, fill=(0, 0, 0))
    draw.text((time_x, time_y), time_of_day, font=font_sub, fill=cyan)

    filename = f"persona_v4_{target_date.month}_{target_date.day}.png"
    img.save(filename)
    img.show()

if __name__ == "__main__":
    generate_persona_date_v4()