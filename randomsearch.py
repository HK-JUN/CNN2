import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from PIL import Image
import matplotlib.pyplot as plt
import dataset_test as dt
import cnn2_model as cnn2
import random

class BinocularDataset(Dataset):
    def __init__(self, data, labels, transform=None):
        self.data = data
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        left_path, right_path = self.data[idx]
        left_img = Image.open(left_path).convert('L')
        right_img = Image.open(right_path).convert('L')

        if self.transform:
            left_img = self.transform(left_img)
            right_img = self.transform(right_img)

        return (left_img, right_img), self.labels[idx]


def random_search_hyperparameters(num_trials, param_grid, device):
    best_params = None
    best_valid_loss = float('inf')

    for trial in range(num_trials):
        # 랜덤 하이퍼파라미터 선택
        current_params = {k: random.choice(v) for k, v in param_grid.items()}
        print(f"Trial {trial+1}: Testing with Params = {current_params}")
        
        valid_loss = train_and_evaluate(current_params, device)
        
        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            best_params = current_params
        print(f"Trial {trial+1}: Valid Loss = {valid_loss:.4f}, Params = {current_params}")
        
    return best_params

def train_and_evaluate(params, device):
    dataset_path = 'dataset/modelnet2d/'
    class_set =  ['chair', 'car', 'lamp', 'airplane', 'person']
    transform = transforms.Compose([
        transforms.Resize((48, 48)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])
    dataset = dt.get_data_from_file(class_set, dataset_path)
    train_dataset, valid_dataset, test_dataset = dt.train_test_split(dataset)

    train_data, train_labels = dt.split_data_label(train_dataset)
    test_data, test_labels = dt.split_data_label(test_dataset)
    valid_data, valid_labels = dt.split_data_label(valid_dataset)

    train_dataset = BinocularDataset(train_data, train_labels, transform=transform)
    valid_dataset = BinocularDataset(valid_data, valid_labels, transform=transform)
    test_dataset = BinocularDataset(test_data, test_labels, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=params['BATCH_SIZE'], shuffle=True, num_workers=4)
    valid_loader = DataLoader(valid_dataset, batch_size=params['BATCH_SIZE'], shuffle=False, num_workers=4)

    # 모델, 손실 함수, 옵티마이저 설정
    model = cnn2.CNN2(num_classes=len(class_set)).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=params['lr'])

    # 훈련 부분
    model.train()
    for epoch in range(params['NUM_EPOCHS']):
        for data in train_loader:
            (inputsL, inputsR), labels = data
            inputsL, inputsR, labels = inputsL.to(device), inputsR.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputsL, inputsR)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
        print(f"epoch {epoch} finished")
    # 검증 부분
    model.eval()
    valid_loss = 0.0
    with torch.no_grad():
        for data in valid_loader:
            (inputsL, inputsR), labels = data
            inputsL, inputsR, labels = inputsL.to(device), inputsR.to(device), labels.to(device)
            outputs = model(inputsL, inputsR)
            loss = criterion(outputs, labels)
            valid_loss += loss.item()
    
    return valid_loss / len(valid_loader)

def main():
    # 하이퍼파라미터 범위 설정
    param_grid = {
        'BATCH_SIZE': [32,64, 128, 256],
        'NUM_EPOCHS': [10, 20, 30, 50, 70, 100],
        'lr': [0.001, 0.01, 0.0001]
    }

    # 장치 설정
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    best_params = random_search_hyperparameters(10, param_grid, device)
    print("Best Parameters:", best_params)
if __name__ == "__main__":
    main()
