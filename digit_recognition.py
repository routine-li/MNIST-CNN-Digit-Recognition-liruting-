import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFilter
import tkinter as tk
import numpy as np
import easyocr
import time

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

reader = easyocr.Reader(['en'], gpu=False)

batch_size = 64
learning_rate = 0.001
num_epochs = 12
loss_history = []

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))])

train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.dropout = nn.Dropout(0.25)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = x.view(-1, 64 * 7 * 7)
        x = self.dropout(self.relu(self.fc1(x)))
        x = self.fc2(x)
        return x

model = CNN()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

print("开始训练...")
for epoch in range(num_epochs):
    running_loss = 0.0
    for train_images, train_labels in train_loader:
        outputs = model(train_images)
        loss = criterion(outputs, train_labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    avg_loss = running_loss / len(train_loader)
    loss_history.append(avg_loss)
    print(f'第 {epoch+1} 轮, 训练损失: {avg_loss:.4f}')

model.eval()
correct = 0
total = 0
with torch.no_grad():
    for test_images, test_labels in test_loader:
        outputs = model(test_images)
        _, predicted = torch.max(outputs.data, 1)
        total += test_labels.size(0)
        correct += (predicted == test_labels).sum().item()
acc = 100 * correct / total
print(f'测试集准确率: {acc:.2f}%')

plt.figure()
plt.plot(range(1, num_epochs+1), loss_history)
plt.title('训练损失变化曲线')
plt.xlabel('轮数')
plt.ylabel('损失值')
plt.grid(True)
plt.show()
plt.close('all')
time.sleep(0.5)

def predict_ten_images():
    class_images = {}
    for test_images, test_labels in test_loader:
        for img, label in zip(test_images, test_labels):
            label = label.item()
            if label not in class_images:
                class_images[label] = img
            if len(class_images) == 10:
                break
        if len(class_images) == 10:
            break
    sorted_labels = sorted(class_images.keys())
    sorted_images = [class_images[l] for l in sorted_labels]
    with torch.no_grad():
        _, predicted = torch.max(model(torch.stack(sorted_images)), 1)
    fig, axes = plt.subplots(2,5,figsize=(12,6))
    axes = axes.flatten()
    for i in range(10):
        axes[i].imshow(sorted_images[i].squeeze(), cmap='gray')
        axes[i].set_title(f'真:{sorted_labels[i]}\n预:{predicted[i].item()}')
        axes[i].axis('off')
    plt.show()
    plt.close('all')
    time.sleep(0.5)

def predict_one_by_one():
    for t in range(10):
        img = None
        for test_images, test_labels in test_loader:
            for im, l in zip(test_images, test_labels):
                if l.item() == t:
                    img = im
                    break
            if img is not None:
                break
        p = model(img.unsqueeze(0)).argmax().item()
        plt.figure()
        plt.imshow(img.squeeze(), cmap='gray')
        plt.title(f'真实:{t} 预测:{p}')
        plt.axis('off')
        plt.show()
        plt.close('all')
        time.sleep(0.3)

predict_ten_images()
predict_one_by_one()

def preprocess_for_recognition(img):
    img = img.filter(ImageFilter.MaxFilter(3))
    img = img.convert("L")
    img = img.point(lambda x: 255 if x > 10 else 0)
    return img.convert("RGB")

def main_gui():
    window = tk.Tk()
    window.title('实时手写数字识别')
    window.geometry('300x420')

    canvas = tk.Canvas(window, width=280, height=280, bg='black')
    canvas.pack(pady=5)

    label_result = tk.Label(window, text='模型预测：', font=('Arial', 16))
    label_result.pack(pady=5)

    img = Image.new('L', (280, 280), 0)
    draw = ImageDraw.Draw(img)
    last_x = None
    last_y = None
    pen_width = 18

    def paint(e):
        nonlocal last_x, last_y
        x, y = e.x, e.y
        if last_x is not None and last_y is not None:
            draw.line((last_x, last_y, x, y), fill=255, width=pen_width)
            canvas.create_line(last_x, last_y, x, y, fill='white', width=pen_width, capstyle=tk.ROUND)
        last_x = x
        last_y = y

    def reset(e):
        nonlocal last_x, last_y
        last_x = None
        last_y = None

    def run_predict():
        processed_img = preprocess_for_recognition(img)
        img_np = np.array(processed_img)
        res = reader.readtext(img_np, detail=0, allowlist="0123456789")
        if res:
            label_result.config(text=f'模型预测：{"".join(res)}')
        else:
            label_result.config(text='模型预测：未识别')

    def clear_canvas():
        canvas.delete("all")
        nonlocal img, draw
        img = Image.new('L', (280, 280), 0)
        draw = ImageDraw.Draw(img)
        label_result.config(text='模型预测：')

    tk.Button(window, text="开始预测", command=run_predict).pack(pady=3)
    tk.Button(window, text="清除画布", command=clear_canvas).pack(pady=3)

    canvas.bind("<B1-Motion>", paint)
    canvas.bind("<ButtonRelease-1>", reset)

    window.mainloop()

if __name__ == "__main__":
    main_gui()