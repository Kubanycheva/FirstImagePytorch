from fastapi import FastAPI, HTTException, UploadFile, File
import io
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import uvicorn

class ImageCNN(nn.Module):
  def __init__(self):
    super().__init__()
    self.conv = nn.Sequential(
        nn.Conv2d(1, 16, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(2),
    )
    self.fc = nn.Sequential(
        nn.Flatten(),
        nn.Linear(16 * 14 * 14, 64),
        nn.ReLU(),
        nn.Linear(64, 10)
    )

  def forward(self, x):
    x = self.conv(x)
    x = self.fc(x)
    return x



model = ImageCNN()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.load_state_dict(torch.load('model.pth', map_location=device))
model.to(device)
model.eval()


check_image_app = FastAPI()

transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((28, 28)),
    transforms.ToTensor(),
])


@check_image_app.post('/predict/')
async def predict(file: UploadFile = File(...)):
    try:
        image_data = await file.read()
        img = Image.open(io.BytesIO(image_data))
        img_tensor = transform(img).unsqueeze(0).to(device)

        with torch.no_grad():
            y_pred = model(img_tensor)
            predicted = torch.argmax(y_pred, dim=1).item()
        return {'answer_class': predicted}

    except Exception as e:
        return {'Error': str(e)}

if __name__ == '__main__':
    uvicorn.run(check_image_app, host='127.0.0.1', port=8000)