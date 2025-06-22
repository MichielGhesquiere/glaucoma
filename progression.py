# %%
import pandas as pd
import os
import matplotlib.pyplot as plt
import numpy as np
import cv2
import numpy as np

# %%
os.environ['KMP_DUPLICATE_LIB_OK']='True'

# read in the data
df = pd.read_excel('C:\\Users\\Michi\\OneDrive - KU Leuven\\phd\\GRAPE\\VF and clinical information.xlsx')
# Drop unnamed columns
df = df.loc[:, ~df.columns.str.contains('^Unnamed')]

# Drop rows with missing Subject Number
df = df.dropna(subset=['Subject Number'])

# Reset index after dropping rows
df = df.reset_index(drop=True)
df.head()

# %%
import os
import re

def get_image_paths(image_dir):
    image_paths = []
    pattern = re.compile(r'(\d+)_(OD|OS)_(\d+)\.jpg')

    for root, dirs, files in os.walk(image_dir):
        for filename in files:
            match = pattern.match(filename)
            if match:
                subject_number = int(match.group(1))
                laterality = match.group(2)
                visit_number = int(match.group(3))
                image_path = os.path.join(root, filename)
                image_paths.append({
                    'Subject Number': subject_number,
                    'Laterality': laterality,
                    'Visit Number': visit_number,
                    'Image Path': image_path
                })
    return image_paths

# Example usage
image_dir = r'C:\Users\Michi\OneDrive - KU Leuven\phd\GRAPE\CFPs\CFPs'
image_data = get_image_paths(image_dir)
image_df = pd.DataFrame(image_data)


# %%
# Ensure Subject Number is of the same type
df['Subject Number'] = df['Subject Number'].astype(int)

# Merge on Subject Number and Laterality
merged_df = pd.merge(image_df, df, on=['Subject Number', 'Laterality'], how='left')

# %%
# Check for missing progression status
missing_progression = merged_df['Progression Status'].isnull().sum()
print(f'Missing progression status for {missing_progression} images.')

# Fill missing progression statuses if appropriate or drop
merged_df = merged_df.dropna(subset=['Progression Status'])

# %%
from PVBM.DiscSegmenter import DiscSegmenter
segmenter = DiscSegmenter()

optic_disc = segmenter.segment(image_path=str(merged_df['Image Path'].iloc[0]))

plt.imshow(optic_disc)
plt.title("optic disc")
plt.axis('off')
plt.show()
plt.imshow(cv2.imread(str(merged_df['Image Path'].iloc[0])))
plt.title("Retinal Image")
plt.axis('off')
plt.show()

# %%
#Compute the center and radius of the optic disc, as well as the Region of Interest (ROI) on which the Geometrical VBMs will be computed
#And the zones ABC on which the Central Retinal Equivalent will be computed
center, radius, roi, zones_ABC = segmenter.post_processing(segmentation=optic_disc, max_roi_size=600) 
# plot the image
plt.imshow(cv2.imread(str(merged_df['Image Path'].iloc[1])))
plt.imshow(zones_ABC/255, alpha = 0.5)
# plot the center of the optic disc
plt.scatter(center[0], center[1], c='r')
plt.title("Zones A, B and C")
plt.axis('off')
plt.show()

# %%
import torch
from torch.utils.data import Dataset
from PIL import Image

class FundusSequenceDataset(Dataset):
    def __init__(self, df, transform=None):
        self.df = df
        self.transform = transform
        self.subjects = df['Subject Number'].unique()
        
    def __len__(self):
        return len(self.subjects)
    
    def __getitem__(self, idx):
        subject = self.subjects[idx]
        subject_data = self.df[self.df['Subject Number'] == subject]
        
        # Sort images by Visit Number
        subject_data = subject_data.sort_values('Visit Number')
        
        images = []
        for img_path in subject_data['Image Path']:
            image = Image.open(img_path).convert('RGB')
            if self.transform:
                image = self.transform(image)
            images.append(image)
        
        # Stack images into a tensor of shape (sequence_length, channels, height, width)
        images = torch.stack(images)
        
        # Get the progression status (assuming it's the same for all visits)
        progression_status = subject_data['Progression Status'].iloc[0]
        progression_status = torch.tensor(progression_status, dtype=torch.long)
        
        return images, progression_status


# %%
from torchvision import transforms

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    # Add any other transforms you need
])


# %%
from torch.utils.data import DataLoader

dataset = FundusSequenceDataset(merged_df, transform=transform)
dataloader = DataLoader(dataset, batch_size=1, shuffle=True)

# %%
def visualize_subject_sequence_laterality(subject_number, df, laterality, 
                                        show_disc_detection=True):
    """
    Visualize image sequence for a subject with optic disc overlay and zoom
    """
    # Filter by both subject number and laterality
    subject_data = df[(df['Subject Number'] == subject_number) & 
                     (df['Laterality'] == laterality)]
    subject_data = subject_data.sort_values('Visit Number')
    
    if len(subject_data) == 0:
        print(f"No images found for subject {subject_number}, {laterality}")
        return
    
    # Initialize segmenter
    segmenter = DiscSegmenter()
    
# Prepare visualization rows
    images = []
    disc_overlay_images = []
    zoomed_images_plain = []  # New list for zoomed images without overlay
    zoomed_images_overlay = []  # Renamed for clarity
    visit_numbers = []
    
    for idx, row in subject_data.iterrows():
        # Original image
        image = cv2.imread(row['Image Path'])
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        images.append(image_rgb)
        visit_numbers.append(row['Visit Number'])
        
        # Optic disc segmentation
        if show_disc_detection:
            # Get optic disc segmentation and post-processing results
            optic_disc = segmenter.segment(image_path=row['Image Path'])
            center, radius, roi, _ = segmenter.post_processing(
                segmentation=optic_disc, max_roi_size=600
            )
            
            # Create overlay image
            overlay_img = image_rgb.copy()
            optic_disc_array = np.array(optic_disc)
            mask = np.zeros_like(image_rgb)
            mask[optic_disc_array > 0] = [255, 0, 0]  # Red color for the disc
            overlay_img = cv2.addWeighted(overlay_img, 0.7, mask, 0.3, 0)
            disc_overlay_images.append(overlay_img)
            
            # Create zoomed region
            padding = int(radius * 0.8)  # 80% of radius as padding
            y_min = max(0, int(center[1] - radius - padding))
            y_max = min(image_rgb.shape[0], int(center[1] + radius + padding))
            x_min = max(0, int(center[0] - radius - padding))
            x_max = min(image_rgb.shape[1], int(center[0] + radius + padding))
            
            # Store plain zoomed region
            zoomed_plain = image_rgb[y_min:y_max, x_min:x_max].copy()
            zoomed_images_plain.append(zoomed_plain)
            
            # Create zoomed region with overlay
            zoom_mask = mask[y_min:y_max, x_min:x_max]
            zoomed_overlay = cv2.addWeighted(zoomed_plain, 0.7, zoom_mask, 0.3, 0)
            zoomed_images_overlay.append(zoomed_overlay)
    
    progression_status = subject_data['Progression Status'].iloc[0]
    
    # Create subplots with four rows
    rows = 4 if show_disc_detection else 1
    fig, axes = plt.subplots(rows, len(images), figsize=(15, 5*rows))
    
    # Handle axes dimensionality
    if rows == 1:
        if len(images) == 1:
            axes = np.array([[axes]])
        else:
            axes = axes[np.newaxis, :]
    elif len(images) == 1:
        axes = axes[:, np.newaxis]
    
    # Plot original images
    for ax, img, visit in zip(axes[0], images, visit_numbers):
        ax.imshow(img)
        ax.set_title(f'Visit {visit}')
        ax.axis('off')
    axes[0][0].set_ylabel('Original')
    
    if show_disc_detection:
        # Plot images with disc overlay
        for ax, img in zip(axes[1], disc_overlay_images):
            ax.imshow(img)
            ax.axis('off')
        axes[1][0].set_ylabel('Optic Disc')
        
        # Plot plain zoomed images
        for ax, img in zip(axes[2], zoomed_images_plain):
            ax.imshow(img)
            ax.axis('off')
        axes[2][0].set_ylabel('Zoomed Region')
        
        # Plot zoomed images with overlay
        for ax, img in zip(axes[3], zoomed_images_overlay):
            ax.imshow(img)
            ax.axis('off')
        axes[3][0].set_ylabel('Zoomed + Disc')
    
    plt.suptitle(f'Subject {subject_number} - {laterality} - Progression Status: {progression_status}')
    plt.tight_layout()
    plt.show()

# Visualize for a few subjects, both OD and OS
for subject in merged_df['Subject Number'].unique()[:3]:
    for lat in ['OD', 'OS']:
        visualize_subject_sequence_laterality(subject, merged_df, lat)

# %%
import numpy as np
import cv2
import matplotlib.pyplot as plt
import pandas as pd
from scipy.ndimage import binary_fill_holes
from skimage.measure import regionprops
import seaborn as sns

class GlaucomaMetrics:
    """
    Extract and calculate clinically relevant metrics from optic cup and disc segmentations
    """
    def __init__(self, pixel_to_mm_ratio=None):
        """
        Initialize with optional pixel to millimeter conversion ratio for real-world measurements
        pixel_to_mm_ratio: conversion factor if available from image metadata
        """
        self.pixel_to_mm_ratio = pixel_to_mm_ratio
        
    def extract_metrics(self, disc_mask, cup_mask):
        """
        Extract comprehensive metrics from disc and cup masks
        
        Parameters:
        - disc_mask: Binary mask of optic disc segmentation
        - cup_mask: Binary mask of optic cup segmentation
        
        Returns:
        - Dictionary of calculated metrics
        """
        # Ensure masks are binary
        disc_mask = (disc_mask > 0).astype(np.uint8)
        cup_mask = (cup_mask > 0).astype(np.uint8)
        
        # Fill any holes in the masks
        disc_mask = binary_fill_holes(disc_mask).astype(np.uint8)
        cup_mask = binary_fill_holes(cup_mask).astype(np.uint8)
        
        # Get region properties
        disc_props = regionprops(disc_mask)[0] if np.sum(disc_mask) > 0 else None
        cup_props = regionprops(cup_mask)[0] if np.sum(cup_mask) > 0 else None
        
        if disc_props is None or cup_props is None:
            return None
        
        # Basic area measurements
        disc_area = np.sum(disc_mask)
        cup_area = np.sum(cup_mask)
        rim_area = disc_area - cup_area
        
        # Calculate area ratio
        area_ratio = cup_area / disc_area if disc_area > 0 else 0
        
        # Find contours for shape analysis
        disc_contours, _ = cv2.findContours(disc_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        cup_contours, _ = cv2.findContours(cup_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        
        # Get vertical and horizontal diameters
        metrics = {}
        
        if len(disc_contours) > 0 and len(cup_contours) > 0:
            # Get bounding boxes
            disc_x, disc_y, disc_w, disc_h = cv2.boundingRect(disc_contours[0])
            cup_x, cup_y, cup_w, cup_h = cv2.boundingRect(cup_contours[0])
            
            # Calculate vertical cup-to-disc ratio (VCDR)
            vcdr = cup_h / disc_h if disc_h > 0 else 0
            
            # Calculate horizontal cup-to-disc ratio (HCDR)
            hcdr = cup_w / disc_w if disc_w > 0 else 0
            
            # Find centers
            disc_center = disc_props.centroid
            cup_center = cup_props.centroid
            
            # Calculate cup displacement from center of disc
            displacement_x = abs(cup_center[1] - disc_center[1])
            displacement_y = abs(cup_center[0] - disc_center[0])
            displacement = np.sqrt(displacement_x**2 + displacement_y**2)
            
            # Calculate rim-to-disc ratio in different regions (ISNT rule)
            # Create four quadrants around the disc centroid
            h, w = disc_mask.shape
            y_center, x_center = int(disc_center[0]), int(disc_center[1])
            
            # Create quadrant masks
            superior_mask = np.zeros((h, w), dtype=np.uint8)
            superior_mask[:y_center, :] = 1
            
            inferior_mask = np.zeros((h, w), dtype=np.uint8)
            inferior_mask[y_center:, :] = 1
            
            nasal_mask = np.zeros((h, w), dtype=np.uint8)
            # Nasal is left for OD, right for OS - assuming a standard convention
            # This would need adjustment based on the specific image orientation
            nasal_mask[:, :x_center] = 1
            
            temporal_mask = np.zeros((h, w), dtype=np.uint8)
            temporal_mask[:, x_center:] = 1
            
            # Calculate rim area in each quadrant
            rim_mask = disc_mask - cup_mask
            superior_rim = np.sum(rim_mask * superior_mask)
            inferior_rim = np.sum(rim_mask * inferior_mask)
            nasal_rim = np.sum(rim_mask * nasal_mask)
            temporal_rim = np.sum(rim_mask * temporal_mask)
            
            # Calculate rim-to-disc ratio in each quadrant
            superior_disc = np.sum(disc_mask * superior_mask)
            inferior_disc = np.sum(disc_mask * inferior_mask)
            nasal_disc = np.sum(disc_mask * nasal_mask)
            temporal_disc = np.sum(disc_mask * temporal_mask)
            
            superior_ratio = superior_rim / superior_disc if superior_disc > 0 else 0
            inferior_ratio = inferior_rim / inferior_disc if inferior_disc > 0 else 0
            nasal_ratio = nasal_rim / nasal_disc if nasal_disc > 0 else 0
            temporal_ratio = temporal_rim / temporal_disc if temporal_disc > 0 else 0
            
            # Approximate volume metrics (if we assume height correlates with brightness)
            # Note: This is an approximation. Real volume needs OCT or stereoscopic images
            cup_volume_proxy = cup_area * cup_props.mean_intensity
            disc_volume_proxy = disc_area * disc_props.mean_intensity
            
            # Store all metrics
            metrics = {
                'disc_area': disc_area,
                'cup_area': cup_area,
                'rim_area': rim_area,
                'area_ratio': area_ratio,
                'vcdr': vcdr,
                'hcdr': hcdr,
                'cup_displacement': displacement,
                'superior_rim_ratio': superior_ratio,
                'inferior_rim_ratio': inferior_ratio,
                'nasal_rim_ratio': nasal_ratio,
                'temporal_rim_ratio': temporal_ratio,
                'isnt_violation': not (nasal_ratio > superior_ratio > inferior_ratio > temporal_ratio),
                'cup_volume_proxy': cup_volume_proxy,
                'disc_volume_proxy': disc_volume_proxy
            }
            
            # Convert to real-world measurements if ratio is available
            if self.pixel_to_mm_ratio is not None:
                metrics['disc_area_mm2'] = disc_area * (self.pixel_to_mm_ratio ** 2)
                metrics['cup_area_mm2'] = cup_area * (self.pixel_to_mm_ratio ** 2)
                metrics['rim_area_mm2'] = rim_area * (self.pixel_to_mm_ratio ** 2)
        
        return metrics
    
    def visualize_ratio_measurements(self, image, disc_mask, cup_mask):
        """
        Create visualization of cup-to-disc ratio measurements
        """
        # Ensure masks are binary
        disc_mask = (disc_mask > 0).astype(np.uint8)
        cup_mask = (cup_mask > 0).astype(np.uint8)
        
        # Convert image to RGB if it's grayscale
        if len(image.shape) == 2:
            image_rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        elif image.shape[2] == 3:
            image_rgb = image.copy()
        else:
            raise ValueError("Unsupported image format")
        
        # Find contours
        disc_contours, _ = cv2.findContours(disc_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        cup_contours, _ = cv2.findContours(cup_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        
        if len(disc_contours) == 0 or len(cup_contours) == 0:
            return image_rgb
            
        # Create overlay image
        overlay = image_rgb.copy()
        
        # Draw disc and cup contours
        cv2.drawContours(overlay, disc_contours, -1, (0, 255, 0), 2)  # Green for disc
        cv2.drawContours(overlay, cup_contours, -1, (255, 0, 0), 2)   # Red for cup
        
        # Get bounding rectangles
        disc_rect = cv2.minAreaRect(disc_contours[0])
        cup_rect = cv2.minAreaRect(cup_contours[0])
        
        # Get the vertices of the rotated rectangles
        disc_box = cv2.boxPoints(disc_rect).astype(np.int32)
        cup_box = cv2.boxPoints(cup_rect).astype(np.int32)
        
        # Draw the rectangles
        cv2.drawContours(overlay, [disc_box], 0, (0, 255, 0), 1)
        cv2.drawContours(overlay, [cup_box], 0, (255, 0, 0), 1)
        
        # Get vertical and horizontal lines for VCDR and HCDR measurement
        disc_x, disc_y, disc_w, disc_h = cv2.boundingRect(disc_contours[0])
        cup_x, cup_y, cup_w, cup_h = cv2.boundingRect(cup_contours[0])
        
        # Draw vertical measurement lines
        cv2.line(overlay, (disc_x + disc_w//2, disc_y), 
                (disc_x + disc_w//2, disc_y + disc_h), (0, 255, 255), 2)  # Yellow for disc
        cv2.line(overlay, (cup_x + cup_w//2, cup_y), 
                (cup_x + cup_w//2, cup_y + cup_h), (255, 255, 0), 2)      # Cyan for cup
        
        # Draw horizontal measurement lines
        cv2.line(overlay, (disc_x, disc_y + disc_h//2), 
                (disc_x + disc_w, disc_y + disc_h//2), (0, 255, 255), 2)  # Yellow for disc
        cv2.line(overlay, (cup_x, cup_y + cup_h//2), 
                (cup_x + cup_w, cup_y + cup_h//2), (255, 255, 0), 2)      # Cyan for cup
        
        # Calculate ratios
        vcdr = cup_h / disc_h if disc_h > 0 else 0
        hcdr = cup_w / disc_w if disc_w > 0 else 0
        
        # Display ratios on image
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(overlay, f"VCDR: {vcdr:.2f}", (10, 30), font, 0.7, (255, 255, 255), 2)
        cv2.putText(overlay, f"HCDR: {hcdr:.2f}", (10, 60), font, 0.7, (255, 255, 255), 2)
        
        # Blend original image with overlay
        alpha = 0.7
        return cv2.addWeighted(image_rgb, alpha, overlay, 1-alpha, 0)
    
    def analyze_progression(self, metrics_series):
        """
        Analyze a time series of metrics for evidence of progression
        
        Parameters:
        - metrics_series: List of metrics dictionaries in chronological order
        
        Returns:
        - Dictionary of progression metrics
        """
        if not metrics_series or len(metrics_series) < 2:
            return {"error": "Need at least two time points for progression analysis"}
        
        # Convert to DataFrame for easier analysis
        df = pd.DataFrame(metrics_series)
        
        # Calculate changes between first and last visit
        first_metrics = metrics_series[0]
        last_metrics = metrics_series[-1]
        
        # Key metrics to track
        progression_indicators = {
            'vcdr_change': last_metrics['vcdr'] - first_metrics['vcdr'],
            'vcdr_percent_change': ((last_metrics['vcdr'] - first_metrics['vcdr']) / first_metrics['vcdr']) * 100 if first_metrics['vcdr'] > 0 else 0,
            'rim_area_change': last_metrics['rim_area'] - first_metrics['rim_area'],
            'rim_area_percent_change': ((last_metrics['rim_area'] - first_metrics['rim_area']) / first_metrics['rim_area']) * 100 if first_metrics['rim_area'] > 0 else 0,
            'area_ratio_change': last_metrics['area_ratio'] - first_metrics['area_ratio'],
            'superior_rim_change': last_metrics['superior_rim_ratio'] - first_metrics['superior_rim_ratio'],
            'inferior_rim_change': last_metrics['inferior_rim_ratio'] - first_metrics['inferior_rim_ratio'],
            'isnt_violated_now': last_metrics['isnt_violation'],
            'isnt_newly_violated': last_metrics['isnt_violation'] and not first_metrics['isnt_violation']
        }
        
        # Calculate rates of change (per year if timestamps available)
        # For now, just divide by number of intervals
        intervals = len(metrics_series) - 1
        for key in ['vcdr_change', 'rim_area_change', 'area_ratio_change',
                    'superior_rim_change', 'inferior_rim_change']:
            progression_indicators[f'{key}_rate'] = progression_indicators[key] / intervals
        
        # Determine if progression is likely based on thresholds
        # These thresholds should be adjusted based on clinical guidelines
        progression_likely = (
            (progression_indicators['vcdr_change'] > 0.05) or
            (progression_indicators['rim_area_percent_change'] < -10) or
            progression_indicators['isnt_newly_violated']
        )
        
        progression_indicators['progression_likely'] = progression_likely
        
        return progression_indicators
    
    def plot_metrics_over_time(self, metrics_series, visit_numbers=None):
        """
        Plot key metrics over time to visualize progression
        """
        if not metrics_series or len(metrics_series) < 2:
            print("Need at least two time points to plot progression")
            return
            
        # Default visit numbers if not provided
        if visit_numbers is None:
            visit_numbers = list(range(1, len(metrics_series) + 1))
            
        # Convert to DataFrame
        data = []
        for i, metrics in enumerate(metrics_series):
            metrics_copy = metrics.copy()
            metrics_copy['visit'] = visit_numbers[i]
            data.append(metrics_copy)
            
        df = pd.DataFrame(data)
        
        # Create plot
        fig, axs = plt.subplots(2, 2, figsize=(14, 10))
        
        # Plot VCDR and HCDR
        axs[0, 0].plot(df['visit'], df['vcdr'], 'o-', label='Vertical CDR')
        axs[0, 0].plot(df['visit'], df['hcdr'], 's-', label='Horizontal CDR')
        axs[0, 0].set_ylabel('Cup-to-Disc Ratio')
        axs[0, 0].set_title('Cup-to-Disc Ratio Progression')
        axs[0, 0].legend()
        axs[0, 0].grid(True, alpha=0.3)
        
        # Plot rim area
        axs[0, 1].plot(df['visit'], df['rim_area'], 'o-')
        axs[0, 1].set_ylabel('Rim Area (pixels)')
        axs[0, 1].set_title('Rim Area Progression')
        axs[0, 1].grid(True, alpha=0.3)
        
        # Plot ISNT ratios
        axs[1, 0].plot(df['visit'], df['superior_rim_ratio'], 'o-', label='Superior')
        axs[1, 0].plot(df['visit'], df['inferior_rim_ratio'], 's-', label='Inferior')
        axs[1, 0].plot(df['visit'], df['nasal_rim_ratio'], '^-', label='Nasal')
        axs[1, 0].plot(df['visit'], df['temporal_rim_ratio'], 'd-', label='Temporal')
        axs[1, 0].set_ylabel('Rim-to-Disc Ratio')
        axs[1, 0].set_title('ISNT Rule Quadrants')
        axs[1, 0].legend()
        axs[1, 0].grid(True, alpha=0.3)
        
        # Plot area ratio (another representation of cup-to-disc)
        axs[1, 1].plot(df['visit'], df['area_ratio'], 'o-')
        axs[1, 1].set_ylabel('Cup-to-Disc Area Ratio')
        axs[1, 1].set_title('Area Ratio Progression')
        axs[1, 1].grid(True, alpha=0.3)
        
        # Format all subplots
        for ax in axs.flat:
            ax.set_xlabel('Visit Number')
            ax.set_xticks(df['visit'])
        
        plt.tight_layout()
        plt.show()
        
        return fig

# Function to enhance the visualization with metrics
def visualize_subject_sequence_with_metrics(subject_number, df, laterality, 
                                           cup_disc_model=None):
    """
    Visualize image sequence for a subject with both segmentations and metrics
    
    Parameters:
    - subject_number: Subject ID number
    - df: DataFrame with image paths and visit information
    - laterality: 'OD' or 'OS'
    - cup_disc_model: Trained model for joint cup and disc segmentation
    """
    # Filter by both subject number and laterality
    subject_data = df[(df['Subject Number'] == subject_number) & 
                     (df['Laterality'] == laterality)]
    subject_data = subject_data.sort_values('Visit Number')
    
    if len(subject_data) == 0:
        print(f"No images found for subject {subject_number}, {laterality}")
        return
    
    # Initialize metrics calculator
    metrics_calculator = GlaucomaMetrics()
    
    # Prepare visualization rows
    images = []
    segmentation_images = []
    metrics_images = []
    visit_numbers = []
    metrics_list = []
    
    for idx, row in subject_data.iterrows():
        # Original image
        image = cv2.imread(row['Image Path'])
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        images.append(image_rgb)
        visit_numbers.append(row['Visit Number'])
        
        # Get segmentations using the joint model
        # This is placeholder code - you'll need to adapt based on your model
        if cup_disc_model:
            # Preprocess image
            img_tensor = preprocess_for_model(image)  # You'll need to implement this
            
            # Run prediction
            with torch.no_grad():
                cup_pred, disc_pred = cup_disc_model(img_tensor)
                
            # Convert predictions to binary masks
            cup_mask = (cup_pred.squeeze().cpu().numpy() > 0.5).astype(np.uint8)
            disc_mask = (disc_pred.squeeze().cpu().numpy() > 0.5).astype(np.uint8)
        else:
            # Mock segmentation for demonstration
            # In practice, you would use actual segmentations from your model
            h, w = image.shape[:2]
            disc_center = (w//2, h//2)
            disc_radius = min(h, w) // 8
            cup_radius = disc_radius // 2
            
            # Create mock masks
            disc_mask = np.zeros((h, w), dtype=np.uint8)
            cv2.circle(disc_mask, disc_center, disc_radius, 1, -1)
            
            cup_mask = np.zeros((h, w), dtype=np.uint8)
            cv2.circle(cup_mask, disc_center, cup_radius, 1, -1)
        
        # Create segmentation overlay
        overlay = image_rgb.copy()
        disc_overlay = np.zeros_like(image_rgb)
        cup_overlay = np.zeros_like(image_rgb)
        
        disc_overlay[disc_mask > 0] = [0, 255, 0]  # Green for disc
        cup_overlay[cup_mask > 0] = [255, 0, 0]    # Red for cup
        
        overlay = cv2.addWeighted(overlay, 0.7, disc_overlay, 0.3, 0)
        overlay = cv2.addWeighted(overlay, 0.7, cup_overlay, 0.3, 0)
        segmentation_images.append(overlay)
        
        # Calculate metrics
        metrics = metrics_calculator.extract_metrics(disc_mask, cup_mask)
        metrics_list.append(metrics)
        
        # Create metrics visualization
        metrics_vis = metrics_calculator.visualize_ratio_measurements(image_rgb, disc_mask, cup_mask)
        metrics_images.append(metrics_vis)
    
    # Analyze progression
    progression_results = metrics_calculator.analyze_progression(metrics_list)
    progression_status = subject_data['Progression Status'].iloc[0]
    
    # Create subplots with three rows
    fig, axes = plt.subplots(3, len(images), figsize=(15, 12))
    
    # Handle axes dimensionality for single-image case
    if len(images) == 1:
        axes = axes[:, np.newaxis]
    
    # Plot original images
    for ax, img, visit in zip(axes[0], images, visit_numbers):
        ax.imshow(img)
        ax.set_title(f'Visit {visit}')
        ax.axis('off')
    axes[0][0].set_ylabel('Original')
    
    # Plot segmentation images
    for ax, img in zip(axes[1], segmentation_images):
        ax.imshow(img)
        ax.axis('off')
    axes[1][0].set_ylabel('Segmentation')
    
    # Plot metrics images
    for ax, img in zip(axes[2], metrics_images):
        ax.imshow(img)
        ax.axis('off')
    axes[2][0].set_ylabel('Metrics')
    
    # Add progression information to title
    model_assessment = "Progression detected" if progression_results['progression_likely'] else "No progression detected"
    plt.suptitle(f'Subject {subject_number} - {laterality} - '
                f'Clinical Status: {progression_status} - '
                f'Model Assessment: {model_assessment}')
    plt.tight_layout()
    plt.subplots_adjust(top=0.92)  # Make room for the suptitle
    
    # Plot metrics trends in a separate figure
    metrics_fig = metrics_calculator.plot_metrics_over_time(metrics_list, visit_numbers)
    
    return fig, metrics_fig, progression_results

# Function to build a machine learning dataset for progression prediction
def build_progression_features(df, cup_disc_model):
    """
    Extract features from all subjects and build a dataset for progression prediction
    
    Parameters:
    - df: DataFrame with image information and clinical progression status
    - cup_disc_model: Trained model for cup and disc segmentation
    
    Returns:
    - Features DataFrame ready for machine learning
    """
    metrics_calculator = GlaucomaMetrics()
    progression_features = []
    
    # Group by subject and laterality
    for (subject, lat), subject_data in df.groupby(['Subject Number', 'Laterality']):
        subject_data = subject_data.sort_values('Visit Number')
        
        # Need at least two visits
        if len(subject_data) < 2:
            continue
            
        # Get progression status from data
        progression_status = subject_data['Progression Status'].iloc[0]
        
        # Get metrics for each visit
        metrics_list = []
        for idx, row in subject_data.iterrows():
            # Load image
            image = cv2.imread(row['Image Path'])
            
            # Get cup and disc segmentations (as before)
            # Placeholder code - replace with actual model predictions
            if cup_disc_model:
                # Use your model here
                img_tensor = preprocess_for_model(image)
                with torch.no_grad():
                    cup_mask, disc_mask = cup_disc_model(img_tensor)
                    cup_mask = cup_mask.squeeze().cpu().numpy() > 0.5
                    disc_mask = disc_mask.squeeze().cpu().numpy() > 0.5
            else:
                # Mock data
                h, w = image.shape[:2]
                disc_mask = np.zeros((h, w), dtype=np.uint8)
                cup_mask = np.zeros((h, w), dtype=np.uint8)
                
                # Vary the masks slightly based on visit number to simulate progression
                disc_radius = min(h, w) // 8
                cup_radius = disc_radius // 2
                
                if progression_status == 'Progressing':
                    # For progressing cases, increase cup size with visit
                    visit_num = row['Visit Number']
                    cup_radius += int(visit_num * disc_radius * 0.05)
                
                cv2.circle(disc_mask, (w//2, h//2), disc_radius, 1, -1)
                cv2.circle(cup_mask, (w//2, h//2), cup_radius, 1, -1)
            
            # Calculate metrics
            metrics = metrics_calculator.extract_metrics(disc_mask, cup_mask)
            if metrics:
                metrics_list.append(metrics)
        
        # Skip if we couldn't get metrics
        if len(metrics_list) < 2:
            continue
            
        # Calculate progression metrics
        progression_metrics = metrics_calculator.analyze_progression(metrics_list)
        
        # Add subject information and ground truth
        feature_row = {
            'subject_id': subject,
            'laterality': lat,
            'visits': len(subject_data),
            'progression_status': 1 if progression_status == 'Progressing' else 0
        }
        
        # Add all progression metrics as features
        for key, value in progression_metrics.items():
            if key != 'error' and key != 'progression_likely':
                feature_row[key] = value
                
        # Add baseline (first visit) metrics
        for key, value in metrics_list[0].items():
            feature_row[f'baseline_{key}'] = value
            
        # Add final (last visit) metrics
        for key, value in metrics_list[-1].items():
            feature_row[f'final_{key}'] = value
            
        progression_features.append(feature_row)
    
    # Convert to DataFrame
    return pd.DataFrame(progression_features)

# Progression prediction model training function 
def train_progression_model(features_df):
    """
    Train a model to predict glaucoma progression based on extracted features
    
    Parameters:
    - features_df: DataFrame with features and progression status
    
    Returns:
    - Trained model and evaluation metrics
    """
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
    
    # Drop non-feature columns
    X = features_df.drop(['subject_id', 'laterality', 'progression_status'], axis=1)
    y = features_df['progression_status']
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train_scaled, y_train)
    
    # Evaluate
    y_pred = model.predict(X_test_scaled)
    y_prob = model.predict_proba(X_test_scaled)[:, 1]
    
    # Print classification report
    print("Classification Report:")
    print(classification_report(y_test, y_pred))
    
    # Calculate and plot ROC curve
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    roc_auc = auc(fpr, tpr)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve for Glaucoma Progression Prediction')
    plt.legend(loc="lower right")
    plt.grid(alpha=0.3)
    plt.show()
    
    # Get feature importance
    feature_importance = pd.DataFrame({
        'Feature': X.columns,
        'Importance': model.feature_importances_
    }).sort_values('Importance', ascending=False)
    
    # Plot top 15 features
    plt.figure(figsize=(10, 8))
    sns.barplot(data=feature_importance.head(15), x='Importance', y='Feature')
    plt.title('Top 15 Features for Progression Prediction')
    plt.xlabel('Importance')
    plt.tight_layout()
    plt.show()
    
    return {
        'model': model,
        'scaler': scaler,
        'report': classification_report(y_test, y_pred, output_dict=True),
        'confusion_matrix': confusion_matrix(y_test, y_pred),
        'roc_auc': roc_auc,
        'feature_importance': feature_importance
    }


