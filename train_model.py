import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from PIL import Image
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
import dataset_test as dt
#import cnn2_model as cnn2
import deeper_model as cnn2

# Constants
dataset_path = 'dataset/modelnet2d/'
class_set =  ['chair', 'car', 'lamp', 'airplane', 'person']
BATCH_SIZE = 32
IMG_SIZE = 48
NUM_CHANNEL = 1  # Assuming grayscale images
NUM_EPOCHS = 10
LEARNING_RATE = 0.01
K_FOLDS = 3  # Number of folds for k-fold CV

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
    # Data Preparation
    transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])
    raw_dataset = dt.get_data_from_file(class_set, dataset_path)
    train_data, valid_data, test_data = dt.train_test_split(raw_dataset)
    print("Train Dataset: {}".format(len(train_data)))
    print("Test Dataset: {}".format(len(test_data)))
    print("Valid Dataset: {}".format(len(valid_data)))
    # train과 valid 데이터셋 합치기
    combined_train_valid = train_data + valid_data

    # 데이터와 레이블 분리
    data, labels = dt.split_data_label(combined_train_valid)
    print(f"Total data size: {len(data)}, Labels size: {len(labels)}")
    kf = KFold(n_splits=K_FOLDS, shuffle=True)
    all_train_losses = []
    all_valid_losses = []
    
    best_valid_loss = float('inf')
    best_model_info = None
    # K-Fold Cross-Validation
    for fold, (train_idx, valid_idx) in enumerate(kf.split(data)):
        print(f"Checking index range for fold {fold}:")
        print(f"  Train index range: min {min(train_idx)}, max {max(train_idx)}")
        print(f"  Valid index range: min {min(valid_idx)}, max {max(valid_idx)}")
        fold_train_losses = []
        fold_valid_losses = []
        train_dataset = BinocularDataset([data[i] for i in train_idx], [labels[i] for i in train_idx], transform=transform)
        valid_dataset = BinocularDataset([data[i] for i in valid_idx], [labels[i] for i in valid_idx], transform=transform)
        # DataLoader 설정
        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,num_workers=4)
        valid_loader = DataLoader(valid_dataset, batch_size=BATCH_SIZE, shuffle=False,num_workers=4)
        # Model setup
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = cnn2.CNN2(num_classes=len(class_set)).to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

        # Training loop
        for epoch in range(NUM_EPOCHS):
            model.train()
            running_loss = 0.0
            itemnum = 0
            for tdata in train_loader:
                (inputsL, inputsR), tlabels = tdata
                inputsL, inputsR, tlabels = inputsL.to(device), inputsR.to(device), tlabels.to(device)
                optimizer.zero_grad()
                outputs = model(inputsL, inputsR)
                loss = criterion(outputs, tlabels)
                loss.backward()
                optimizer.step()
                torch.cuda.empty_cache()
                running_loss += loss.item()
                print(f"fold[{fold}]:epoch {epoch+1}, batch {itemnum+1}/{len(train_loader)} finished,Train Loss: {loss.item()}")
                itemnum = itemnum +1
            # Validation loop
            model.eval()
            valid_loss = 0.0
            with torch.no_grad():
                for vdata in valid_loader:
                    (inputsL, inputsR), vlabels = vdata
                    inputsL, inputsR, vlabels = inputsL.to(device), inputsR.to(device), vlabels.to(device)
                    outputs = model(inputsL, inputsR)
                    loss = criterion(outputs, vlabels)
                    valid_loss += loss.item()
            fold_train_losses.append(running_loss / len(train_loader))
            fold_valid_losses.append(valid_loss / len(valid_loader))

            # Log results
            print(f"Fold {fold+1}, Epoch {epoch+1}, Train Loss: {running_loss / len(train_loader)}, Valid Loss: {valid_loss / len(valid_loader)}")
            if valid_loss < best_valid_loss:
                best_valid_loss = valid_loss
                best_model_info = {
                    'fold': fold,
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'valid_loss': valid_loss
                }
            if best_model_info:
                torch.save(best_model_info, f'saved_model/best_model_fold{fold}_e{epoch}_b{BATCH_SIZE}_l{LEARNING_RATE}.pth')
                print(f"Best model saved from fold {best_model_info['fold']+1} at epoch {best_model_info['epoch']+1} with valid loss {best_model_info['valid_loss']:.4f}")
        all_train_losses.append(fold_train_losses)
        all_valid_losses.append(fold_valid_losses)

    # Results
    print("Training complete. Saving fold results...")
    # Save the fold results for further analysis or averaging
    if best_model_info:
        torch.save(best_model_info, f'saved_model/best_model_e{NUM_EPOCHS}_b{BATCH_SIZE}_l{LEARNING_RATE}.pth')
        print(f"Best model saved from fold {best_model_info['fold']+1} at epoch {best_model_info['epoch']+1} with valid loss {best_model_info['valid_loss']:.4f}")
    avg_train_losses = np.mean(all_train_losses, axis=0)
    avg_valid_losses = np.mean(all_valid_losses, axis=0)
    plt.figure(figsize=(10, 5))
    plt.plot(avg_train_losses, label='Average Train Loss', color='blue')
    plt.plot(avg_valid_losses, label='Average Valid Loss', color='red')
    plt.title('Training and Validation Loss Across Folds')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig(f'result_plot/tran_plot_e{NUM_EPOCHS}_b{BATCH_SIZE}_l{LEARNING_RATE}.png')
if __name__ == "__main__":
    main()
