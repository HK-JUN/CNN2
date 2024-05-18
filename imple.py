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
dataset_path = 'dataset/modelnet2d/'
class_set =  ['chair', 'car', 'lamp', 'airplane', 'person']

BATCH_SIZE = 32
IMG_SIZE = 48
NUM_CHANNEL = 1  # Assuming grayscale images
NUM_EPOCHS = 100
LEARNING_RATE = 0.001
torch.autograd.set_detect_anomaly(True)
# Custom Dataset class
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

def main():
    # Transformations
    data_transforms = {
    'train': transforms.Compose([
        transforms.RandomHorizontalFlip(),  # 무작위로 수평 뒤집기
        transforms.RandomVerticalFlip(),    # 무작위로 수직 뒤집기
        transforms.RandomRotation(15),      # -15도에서 15도 사이 무작위 회전
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),  # 색상 변형 적용
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),  # 무작위 변형(평행 이동)
        transforms.ToTensor(),  # 이미지를 PyTorch 텐서로 변환
        transforms.Normalize(mean=[0.5], std=[0.5])  # 정규화
    ]),
    'valid': transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ]),
}

    dataset = dt.get_data_from_file(class_set, dataset_path)
    train_dataset, valid_dataset, test_dataset = dt.train_test_split(dataset)

    train_data, train_labels = dt.split_data_label(train_dataset)
    test_data, test_labels = dt.split_data_label(test_dataset)
    valid_data, valid_labels = dt.split_data_label(valid_dataset)

    train_dataset = BinocularDataset(train_data, train_labels, transform=data_transforms['train'])
    valid_dataset = BinocularDataset(valid_data, valid_labels, transform=data_transforms['valid'])
    test_dataset = BinocularDataset(test_data, test_labels, transform=data_transforms['valid'])

    # DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    valid_loader = DataLoader(valid_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

    print("Train Dataset: {}".format(len(train_dataset)))
    print("Test Dataset: {}".format(len(test_dataset)))
    print("Valid Dataset: {}".format(len(valid_dataset)))
    num_classes = len(class_set)
    print("Number of Classes: {}".format(num_classes))


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = cnn2.CNN2(num_classes=len(class_set)).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    train_losses = []  # 훈련 손실 저장 리스트
    valid_losses = []  # 검증 손실 저장 리스트
    # 훈련 루프
    
    for epoch in range(NUM_EPOCHS):
        model.train()
        running_loss = 0.0
        #print(train_loader)
        itemnum = 0
        for data in train_loader:
            (inputsL, inputsR), labels = data
            inputsL, inputsR, labels = inputsL.to(device), inputsR.to(device), labels.to(device)
            #print(f"l shape: {inputsL.shape} and R shape: {inputsR.shape}")
            # 옵티마이저 초기화
            optimizer.zero_grad()
            
            # 순전파 + 역전파 + 최적화
            outputs = model(inputsL, inputsR)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            torch.cuda.empty_cache()
            running_loss += loss.item()
            print(f"epoch {epoch+1}, batch {itemnum+1}/{len(train_loader)} finished,Train Loss: {loss.item()}")
            itemnum = itemnum +1

        model.eval()
        valid_loss = 0.0
        with torch.no_grad():
            for data in valid_loader:
                (inputsL, inputsR), labels = data
                inputsL, inputsR, labels = inputsL.to(device), inputsR.to(device), labels.to(device)
                outputs = model(inputsL, inputsR)
                loss = criterion(outputs, labels)
                valid_loss += loss.item()
        train_losses.append(running_loss / len(train_loader))  # 이번 에폭의 훈련 손실 저장
        valid_losses.append(valid_loss / len(valid_loader))  # 이번 에폭의 검증 손실 저장
        
        print(f"Epoch {epoch+1}, Train Loss: {running_loss / len(train_loader)}, Valid Loss: {valid_loss / len(valid_loader)}")

    print("training finished! saving the model...")
    PATH = f"saved_model/tran_model_e{epoch}_b{BATCH_SIZE}_l{LEARNING_RATE}.pt"

    # 모델의 상태 저장
    torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': loss,
                }, PATH)
    plt.figure()
    plt.plot(train_losses, label='Train Loss', color='blue')
    plt.plot(valid_losses, label='Valid Loss', color='red')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.savefig(f'result_plot/tran_plot_e{epoch+1}_b{BATCH_SIZE}_l{LEARNING_RATE}.png')
if __name__ == "__main__":
    main()
