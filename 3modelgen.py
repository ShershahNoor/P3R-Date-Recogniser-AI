import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from PIL import Image
from torch.utils.data import DataLoader, Dataset

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

class DigitDataset(Dataset):
    def __init__(self, npz_path):
        data = np.load(npz_path)
        self.images = data['images']
        self.labels = data['labels']
        self.chars = "0123456789/"
        self.char_to_idx = {c: i for i, c in enumerate(self.chars)}

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        full_img = Image.fromarray(self.images[idx]).convert('L')
        
        label_str = self.labels[idx]
        char_idx_to_pick = np.random.randint(0, len(label_str))
        target_char = label_str[char_idx_to_pick]
        
        x_start = 60 + (char_idx_to_pick * 45) 
        digit_patch = full_img.crop((x_start, 45, x_start + 50, 180)).resize((28, 28))
        
        tensor = torch.FloatTensor(np.array(digit_patch)) / 255.0
        return tensor.unsqueeze(0), self.char_to_idx[target_char]

def run_ai_pipeline(npz_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DateCNN().to(device)
    
    dataset = DigitDataset(npz_path)
    loader = DataLoader(dataset, batch_size=16, shuffle=True)
    optimizer = optim.Adam(model.parameters(), lr=0.0005)
    criterion = nn.CrossEntropyLoss()

    print(f"Training on {len(dataset)} images...")
    model.train()
    for epoch in range(100): 
        total_loss = 0
        for imgs, lbls in loader:
            imgs, lbls = imgs.to(device), lbls.to(device)
            optimizer.zero_grad()
            output = model(imgs)
            loss = criterion(output, lbls)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1} Complete. Loss: {total_loss/len(loader):.4f}")

    torch.save(model.state_dict(), "date_model.pth")
    return model

def predict_date(image_path, model):
    """Takes a single PNG file and returns the predicted date string."""
    model.eval()
    chars = "0123456789/"
    img = Image.open(image_path).convert('L')
    prediction = ""
    
    with torch.no_grad():
        for i in range(5):
            x_start = 60 + (i * 45)
            patch = img.crop((x_start, 45, x_start + 50, 180)).resize((28, 28))
            tensor = torch.FloatTensor(np.array(patch)).unsqueeze(0).unsqueeze(0) / 255.0
            
            output = model(tensor.to(next(model.parameters()).device))
            char_idx = torch.argmax(output, dim=1).item()
            prediction += chars[char_idx]
            
    return prediction

if __name__ == "__main__":
    trained_model = run_ai_pipeline("image_dataset.npz")
