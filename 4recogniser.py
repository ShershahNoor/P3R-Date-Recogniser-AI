import torch
import torch.nn as nn
from PIL import Image
import numpy as np

class DateCNN(nn.Module):
    def __init__(self):
        super(DateCNN, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.fc_layers = nn.Sequential(
            nn.Linear(64 * 7 * 7, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 11)
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)
        return self.fc_layers(x)

def predict_date(image_path, model_path="date_model.pth"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model = DateCNN().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    
    chars = "0123456789/"
    img = Image.open(image_path).convert('L')
    prediction = ""
    
    with torch.no_grad():
        for i in range(5):
            x_start = 60 + (i * 45) 
            patch = img.crop((x_start, 45, x_start + 50, 180)).resize((28, 28))
            
            tensor = torch.FloatTensor(np.array(patch)).unsqueeze(0).unsqueeze(0) / 255.0
            tensor = tensor.to(device)
            
            output = model(tensor)
            char_idx = torch.argmax(output, dim=1).item()
            prediction += chars[char_idx]
            
    return prediction

if __name__ == "__main__":
    image_to_check = "test_image.png" 
    
    try:
        guessed_date = predict_date(image_to_check)
        print(f"--- RECOGNITION COMPLETE ---")
        print(f"AI Guessed Date: {guessed_date}")
    except FileNotFoundError:
        print(f"Error: Could not find '{image_to_check}' or 'date_model.pth'")