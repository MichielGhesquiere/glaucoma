# %%
# Standard libraries
import json
import os
import pickle
import random

# Data manipulation and linear algebra
import numpy as np
import pandas as pd

# Machine Learning and Neural Networks
import sklearn
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision import datasets, models, transforms
from torchvision.transforms import (Compose, Normalize, RandomAffine, RandomApply, RandomHorizontalFlip,
                                    RandomResizedCrop, RandomRotation, Resize, ToTensor)
from torch.utils.data import DataLoader, Dataset, Subset

# Image processing
from PIL import Image
import cv2
import timm
from transformers import ViTForImageClassification, ViTImageProcessor

# Visualization libraries
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

# %%
# load in the D:\glaucoma_datasets\SMDG-19\metadata - standardized.csv as a pandas dataframe
import pandas as pd

# load in the metadata
df = pd.read_csv(r'D:\glaucoma_datasets\SMDG-19\metadata - standardized.csv')
df.head()

# %%
# print the columns
print(df.columns)

# %%
# plot the 'types' column (glaucoma labels)
sns.countplot(x='types', data=df)
plt.title('Distribution of Glaucoma Labels')
plt.xlabel('Glaucoma Labels')
plt.ylabel('Count')
plt.show()
# print the unique values in the 'types' column
print(df['types'].unique())
# drop all the -1 types values from the df
df = df[df['types'] != -1]
# print the unique values in the 'types' column
print(df['types'].unique())

# %%
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import pandas as pd
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import os
import json
from datetime import datetime

# %%
# Set random seed for reproducibility
torch.manual_seed(42)

# Load and prepare your DataFrame
base_path = r'D:\glaucoma_datasets\SMDG-19' 

# Remove rows with NaN in required columns and "Not Visible" values
df = df.dropna(subset=['fundus', 'fundus_oc_seg', 'fundus_od_seg'])
df = df[df['fundus_oc_seg'] != 'Not Visible']
df = df[df['fundus_od_seg'] != 'Not Visible']

# Split data
train_df, val_df = train_test_split(df, test_size=0.2, random_state=42)

# Print dataset sizes
print(f'Train samples: {len(train_df)}')
print(f'Validation samples: {len(val_df)}')

# Define transforms
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
])

# %%
# plot summury statistics of following columns: gender, age, eye, iop, vcdr,  'expert1_grade', 'expert2_grade', 'expert3_grade', 'expert4_grade', 'expert5_grade''cdr_avg', 'cdr_expert1','cdr_expert2', 'cdr_expert3', 'cdr_expert4
import matplotlib.pyplot as plt
import seaborn as sns

# Set of relevant columns
cols = ['gender', 'age', 'eye', 'iop', 'vcdr',
        'expert1_grade', 'expert2_grade', 'expert3_grade', 'expert4_grade', 'expert5_grade',
        'cdr_avg', 'cdr_expert1', 'cdr_expert2', 'cdr_expert3', 'cdr_expert4']

# Total number of rows in the dataset
total_rows = len(df)

# Set plot style
sns.set(style="whitegrid")
plt.figure(figsize=(16, 20))

# Loop through columns and plot each one
for i, col in enumerate(cols, 1):
    plt.subplot(5, 3, i)
    non_null_count = df[col].notnull().sum()
    title = f"{col} ({non_null_count}/{total_rows} filled)"
    
    if df[col].dtype == 'object' or df[col].nunique() < 10:
        # Categorical: countplot
        sns.countplot(data=df, x=col, order=df[col].dropna().value_counts().index)
        plt.xticks(rotation=45)
    else:
        # Numerical: histogram
        sns.histplot(df[col].dropna(), kde=True, bins=20)
    
    plt.title(title)
    plt.tight_layout()

plt.suptitle("Summary Statistics of Glaucoma Dataset (Filled Counts Included)", fontsize=16, y=1.02)
plt.show()

# %%
df_airogs = pd.read_csv(r'D:\glaucoma_datasets\AIROGS\train_labels.csv')
df_airogs.head()

# path to the images is D:\glaucoma_datasets\AIROGS\img + challenge_id + .jpg
df_airogs['path'] = df_airogs['challenge_id'].apply(lambda x: os.path.join(r'D:\glaucoma_datasets\AIROGS\img', str(x) + '.jpg'))
# check the class distribution
df_airogs['class'].value_counts().plot(kind='bar')

# %%
# PAPILLA dataset: already included in SMDG-19

# %%
class GlaucomaDataset(Dataset):
    def __init__(self, root_dir, csv_file, image_dir='Images', transform=None, max_images=None):
        self.root_dir = root_dir
        self.csv_file = csv_file
        self.image_dir = image_dir
        self.transform = transform
        self.max_images = max_images
        
        if csv_file == "index.json":
            with open(os.path.join(root_dir, csv_file), 'r') as f:
                self.labels_dict = json.load(f)
            self.image_filenames = list(self.labels_dict.keys())
        elif csv_file == "metadata - standardized.csv":
            self.labels_df = pd.read_csv(os.path.join(root_dir, csv_file)) 
            # Filter DataFrame based on labels
            self.labels_df = self.labels_df[self.labels_df['types'].isin([0, 1])]
            
            
            # Filter image_filenames based on labels
            self.image_filenames = [f for f in os.listdir(os.path.join(root_dir, image_dir)) if f.endswith('.png')]
            self.image_filenames = [f for f in self.image_filenames if f[:-4] in self.labels_df['names'].values]

 
        else:
            self.labels_df = pd.read_csv(os.path.join(root_dir, csv_file))
            self.image_filenames = [f for f in os.listdir(os.path.join(root_dir, image_dir)) if f.endswith('.jpg')]
            
        
        self.image_filenames = self.image_filenames[:max_images] if max_images is not None else self.image_filenames

        print(f'Successfully loaded dataset with {len(self.image_filenames)} images.')

    def __len__(self):
        return len(self.image_filenames)

    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir, self.image_dir, self.image_filenames[idx])
        img = cv2.imread(img_name)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert to RGB
        img = Image.fromarray(img)  # Convert numpy array to PIL Image

        if self.transform:
            img = self.transform(img)
        if self.csv_file == 'metadata - standardized.csv':
            label = self.labels_df.loc[self.labels_df['names']+".png" == self.image_filenames[idx], 'types'].values[0]        
               
            
        label_tensor = torch.tensor(label, dtype=torch.long)

        
        return img, label_tensor

# %%
train_transforms = Compose([
    Resize((512,512)),  # Slightly larger to allow for random crops
    RandomResizedCrop((312,312), scale=(0.8, 1.0)),  # Random scaling
    RandomHorizontalFlip(),
    RandomApply([RandomRotation(10)], p=0.5),
    RandomApply([RandomAffine(degrees=0, scale=(0.9, 1.1))], p=0.5),  # Random scaling
    Resize((312,312)),
    
    ToTensor(),
   Normalize(mean=[0.7538, 0.4848, 0.3553], std=[0.2417, 0.1874, 0.2076]),
])


# Define validation transformations without augmentation
val_transforms = Compose([
    Resize((312,312)),
    ToTensor(),
   Normalize(mean=[0.7538, 0.4848, 0.3553], std=[0.2417, 0.1874, 0.2076]),
])

# %%
batch_size=32
num_classes = 2

smdg_csv_file='metadata - standardized.csv'
smdg_root_dir = r'D:\glaucoma_datasets\SMDG-19'
smdg_image_dir = r'full-fundus\full-fundus'

# check the number of files in the smdg_image_dir
smdg_image_files = os.listdir(os.path.join(smdg_root_dir, smdg_image_dir))
print(f"Number of files in {smdg_image_dir}: {len(smdg_image_files)}")


# Initialize the datasets with appropriate transformations
smdg_train_dataset = GlaucomaDataset(root_dir=smdg_root_dir, image_dir=smdg_image_dir,
                                csv_file=smdg_csv_file, transform=train_transforms)




smdg_val_dataset = GlaucomaDataset(root_dir=smdg_root_dir, image_dir=smdg_image_dir, 
                                      csv_file=smdg_csv_file, transform=val_transforms)



# Split dataset indices
smdg_train_indices, smdg_val_indices = train_test_split(range(len(smdg_train_dataset)), test_size=0.3, random_state=42)

# Create Subset for train and validation datasets
smdg_train_dataset = Subset(smdg_train_dataset, smdg_train_indices)
smdg_val_dataset = Subset(smdg_val_dataset, smdg_val_indices)



smdg_train_loader = DataLoader(smdg_train_dataset, batch_size=batch_size,
                              shuffle=False, num_workers=os.cpu_count(), pin_memory=True)
smdg_val_loader = DataLoader(smdg_val_dataset, batch_size=batch_size,
                              shuffle=False, num_workers=os.cpu_count(), pin_memory=True)

print("loaded smdg dataaset")

# %%
train_loaders=[smdg_train_loader]

val_loaders=[smdg_val_loader]

dataset_name=["SMDG"]

# print the number of images in the train and validation datasets
print(f"Number of images in the train dataset: {len(smdg_train_dataset)}")
print(f"Number of images in the validation dataset: {len(smdg_val_dataset)}")

# %%
import matplotlib.pyplot as plt
import numpy as np

def show_images(dataset, num_images=5, title="Dataset Samples"):
    # Create a figure with num_images rows and 1 column
    fig, axes = plt.subplots(num_images, 1, figsize=(8, 4*num_images))
    
    # Randomly select indices
    indices = np.random.choice(len(dataset), num_images, replace=False)
    
    for idx, ax in zip(indices, axes):
        image, label = dataset[idx]
        
        # Convert the image tensor to numpy and transpose to (H,W,C)
        img_np = image.permute(1, 2, 0).numpy()
        
        # Normalize image for display if needed
        img_np = (img_np - img_np.min()) / (img_np.max() - img_np.min())
        
        ax.imshow(img_np)
        ax.axis('off')
        ax.set_title(f'Label: {"Glaucoma" if label == 1 else "Normal"}')
    
    plt.suptitle(title)
    plt.tight_layout()
    plt.show()

# Display training samples
print("Training Dataset Samples:")
show_images(smdg_train_dataset, title="Training Dataset Samples")

# Display validation samples
print("\nValidation Dataset Samples:")
show_images(smdg_val_dataset, title="Validation Dataset Samples")

# %%
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.models import resnet50
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm
import os
from datetime import datetime
import matplotlib.pyplot as plt

def plot_training_history(history):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10))
    
    # Plot losses
    ax1.plot(history['train_loss'], label='Train Loss')
    ax1.plot(history['val_loss'], label='Val Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.set_title('Training and Validation Loss')
    
    # Plot accuracies
    ax2.plot(history['train_acc'], label='Train Acc')
    ax2.plot(history['val_acc'], label='Val Acc')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy (%)')
    ax2.legend()
    ax2.set_title('Training and Validation Accuracy')
    
    plt.tight_layout()
    plt.show()

def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, 
                num_epochs=50, device='cuda'):
    
    # Create directory for checkpoints
    os.makedirs('checkpoints', exist_ok=True)
    
    best_val_loss = float('inf')
    history = {
        'train_loss': [], 'train_acc': [],
        'val_loss': [], 'val_acc': []
    }
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        running_loss = 0.0
        train_correct = 0
        train_total = 0
        
        train_loop = tqdm(train_loader, desc=f'Epoch [{epoch+1}/{num_epochs}]')
        for inputs, labels in train_loop:
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()
            
            # Update progress bar
            train_loop.set_postfix({
                'loss': loss.item(), 
                'acc': 100 * train_correct / train_total
            })
        
        train_loss = running_loss / len(train_loader)
        train_acc = 100 * train_correct / train_total
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
        
        val_loss = val_loss / len(val_loader)
        val_acc = 100 * val_correct / val_total
        
        # Learning rate scheduling
        scheduler.step(val_loss)
        
        # Update history
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        
        # Save checkpoint if validation loss improved
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
            }, f'checkpoints/best_model.pth')
            print(f'Saved new best model with validation loss: {val_loss:.4f}')
        
        print(f'\nEpoch {epoch+1}/{num_epochs}:')
        print(f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%')
        print(f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%')
        print(f'Learning Rate: {optimizer.param_groups[0]["lr"]}')
        print('-' * 60)
    
    # Plot training history
    plot_training_history(history)
    return model, history

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Initialize model
model = resnet50(weights='ResNet50_Weights.DEFAULT')
model.fc = nn.Linear(model.fc.in_features, num_classes)  # Modify last layer
model = model.to(device)

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5, verbose=True)

# Train the model
model, history = train_model(model, smdg_train_loader, smdg_val_loader, criterion, 
                           optimizer, scheduler, num_epochs=10, device=device)


