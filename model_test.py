import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from PIL import Image
import dataset_test as dt
import deeper_model as cnn2
from torchmetrics import Accuracy, F1Score
class_set = ['chair', 'car', 'lamp', 'airplane', 'person']
class BinocularDataset(torch.utils.data.Dataset):
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

def evaluate_model(model, test_loader, device):
    model.eval()
    test_accuracy = Accuracy(num_classes=len(class_set), average='macro', task='multiclass').to(device)
    test_f1 = F1Score(num_classes=len(class_set), average='macro', task='multiclass').to(device)
    with torch.no_grad():
        for data in test_loader:
            (inputsL, inputsR), labels = data
            inputsL, inputsR, labels = inputsL.to(device), inputsR.to(device), labels.to(device)
            outputs = model(inputsL, inputsR)
            test_accuracy.update(outputs, labels)
            test_f1.update(outputs, labels)

    accuracy = test_accuracy.compute()
    f1_score = test_f1.compute()
    print(f"Test Accuracy: {accuracy}, Test F1 Score: {f1_score}")
    return accuracy, f1_score

def main():
    EPOCH = 99
    IMG_SIZE = 48  # Assuming image size is 48x48
    BATCH_SIZE = 128  # Batch size for evaluation
    LEARNING_RATE = 0.01
    
    dataset_path = 'dataset/modelnet2d/'  # Path to dataset

    # Transformation for the images
    transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])

    # Load dataset and prepare test data
    dataset = dt.get_data_from_file(class_set, dataset_path)
    _, _, test_dataset = dt.train_test_split(dataset)
    test_data, test_labels = dt.split_data_label(test_dataset)
    test_dataset = BinocularDataset(test_data, test_labels, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

    # Setup device and load model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = cnn2.CNN2(num_classes=len(class_set)).to(device)
    #model_path = f"saved_model/model_e{EPOCH}_b{BATCH_SIZE}_l{LEARNING_RATE}.pt"  # Specify the correct path
    model_path = "saved_model/best_model_fold1_e6_b32_l0.01.pth"
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])

    # Evaluate the model
    evaluate_model(model, test_loader, device)

if __name__ == "__main__":
    main()
