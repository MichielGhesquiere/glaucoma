# Multi-Source Domain Adaptation for Glaucoma Classification:

## Abstract

This repository presents a framework for multi-source domain adaptation in automated glaucoma detection from fundus photography. Our approach addresses the critical challenge of domain shift across different imaging devices, demographics, and clinical settings by leveraging Vision Foundation Models (VFM) pre-trained with self-supervised learning and implementing advanced domain adaptation techniques.

## 1. Introduction to Glaucoma

### 1.1 Clinical Background

Glaucoma is a group of progressive optic neuropathies characterized by irreversible retinal ganglion cell death and visual field defects. It represents the leading cause of irreversible blindness worldwide, affecting over 76 million people globally. The insidious nature of glaucoma, often progressing asymptomatically in early stages, makes early detection crucial for preventing vision loss.

### 1.2 Fundus Photography in Glaucoma Detection

Fundus photography provides a non-invasive method for examining the optic nerve head, allowing clinicians to assess:

- **Cup-to-Disc Ratio (CDR)**: The ratio between the optic cup and optic disc areas, a key indicator of glaucomatous damage
- **Neuroretinal Rim**: Thinning or notching patterns indicative of neural tissue loss
- **Retinal Nerve Fiber Layer (RNFL)**: Changes in thickness and appearance
- **Vascular Changes**: Alterations in retinal blood vessel patterns

### 1.3 Challenges in Automated Glaucoma Detection

Traditional machine learning approaches face significant challenges in clinical deployment:

1. **Domain Shift**: Performance degradation when models encounter data from different sources
2. **Device Heterogeneity**: Variations in imaging equipment, protocols, and image quality
3. **Demographic Bias**: Unequal performance across different population groups
4. **Limited Generalizability**: Poor performance on out-of-distribution test sets

## 2. Open-Source Datasets

Our framework leverages multiple publicly available datasets to ensure robust training and comprehensive evaluation:

### 2.1 SMDG-19

### 2.2 CHAKSU Multi-Camera Dataset

- **Size**: 1,644 fundus images
- **Cameras**: Three different fundus cameras (Bosch, Forus, Remidio)
- **Population**: Indian cohort
- **Annotations**: Majority-vote consensus from multiple experts
- **Key Features**: Explicit camera type labeling for domain-specific analysis
- **Use Case**: Multi-source training and out-of-distribution evaluation

### 2.3 AIROGS (Artificial Intelligence for RObust Glaucoma Screening)

- **Size**: 101,442 fundus images (training set)
- **Population**: Diverse European cohort
- **Quality**: Mix of high and moderate quality images
- **Labels**: Binary glaucoma classification
- **Key Features**: Large-scale dataset for robust pre-training
- **Use Case**: Large-scale pre-training and data augmentation

### 2.4 GRAPE (Glaucoma Risk Assessment and Progression Evaluation)

- **Size**: Longitudinal dataset with multiple visits per patient
- **Focus**: Glaucoma progression prediction
- **Data**: Fundus images with visual field measurements
- **Key Features**: Temporal progression labels for advanced modeling
- **Use Case**: Progression prediction and longitudinal analysis

#### PAPILLA
- **Size**: 488 fundus images
- **Origin**: Subset of SMDG-19 with specific identifiers
- **Use Case**: Independent external validation

#### ACRIMA (Automatic Classification of Retinal Images for Glaucoma)
- **Size**: 705 fundus images
- **Population**: Spanish cohort
- **Use Case**: Cross-population generalization testing

#### HYGD (Harvard Glaucoma Dataset)
- **Size**: 1,000+ fundus images
- **Population**: US-based cohort
- **Use Case**: Geographic generalization assessment

#### OIA-ODIR (Ocular Imaging for AI - Online Diabetic Retinopathy)
- **Size**: Test subset for glaucoma classification
- **Use Case**: Multi-disease classification robustness

## 3. Vision Foundation Model (VFM) Architecture

### 3.1 Foundation Model Overview

Our approach is built upon Vision Foundation Models (VFMs), specifically designed for medical imaging applications. VFMs represent a paradigm shift from traditional supervised learning by leveraging self-supervised pre-training on large-scale unlabeled datasets.

### 3.2 Self-Supervised Learning (SSL) Pre-training

The VFM employed in this work utilizes advanced SSL techniques:

#### 3.2.1 Pre-training Methodology
- **Architecture**: Vision Transformer (ViT) Base with Patch16 configuration
- **Pre-training Data**: Large corpus of unlabeled fundus images
- **SSL Objective**: Masked image modeling and contrastive learning
- **Feature Learning**: Robust representation learning without manual annotations

#### 3.2.2 Advantages of SSL for Medical Imaging
1. **Reduced Annotation Dependency**: Leverages abundant unlabeled medical images
2. **Rich Feature Representations**: Learns generalizable visual patterns
3. **Transfer Learning**: Strong initialization for downstream tasks
4. **Domain Robustness**: Better handling of domain variations

### 3.3 Model Architecture Details

```
VFM Architecture:
├── Input: 224×224 RGB fundus images
├── Patch Embedding: 16×16 patches → 768-dimensional vectors
├── Positional Encoding: Learnable position embeddings
├── Transformer Encoder: 12 layers, 12 attention heads
├── Feature Extraction: 768-dimensional global representations
└── Classification Head: Linear layer with dropout regularization
```

### 3.4 Fine-tuning Strategies

Our framework implements three sophisticated fine-tuning approaches:

#### 3.4.1 Linear Probing
- **Frozen Backbone**: Pre-trained features remain fixed
- **Trainable Component**: Only classification head
- **Purpose**: Baseline assessment of feature quality
- **Duration**: 5-15 epochs

#### 3.4.2 Gradual Unfreezing
- **Progressive Adaptation**: Layer-by-layer unfreezing
- **Strategy**: Classification head → top layers → all layers
- **Benefits**: Prevents catastrophic forgetting
- **Monitoring**: Validation-based unfreezing decisions

#### 3.4.3 Layer-wise Learning Rate Decay (LLRD)
- **Differential Learning Rates**: Lower rates for deeper layers
- **Decay Factor**: 0.85 (configurable)
- **Rationale**: Preserve low-level features while adapting high-level representations
- **Implementation**: Custom optimizer with layer-specific learning rates

## 4. Domain Adaptation Techniques

Our framework implements a comprehensive suite of domain adaptation methods to address distribution shift and improve generalization:

### 4.1 Domain Adversarial Neural Networks (DANN)

#### 4.1.1 Theoretical Foundation
DANN implements the principle of domain-invariant feature learning through adversarial training:

```
min_θf,θy max_θd L_y(D_s, θf, θy) - λ L_d(D_s ∪ D_t, θf, θd)
```

Where:
- `θf`: Feature extractor parameters
- `θy`: Label classifier parameters  
- `θd`: Domain classifier parameters
- `L_y`: Classification loss on source domain
- `L_d`: Domain classification loss
- `λ`: Domain adaptation weight

#### 4.1.2 Implementation Details

**Gradient Reversal Layer (GRL)**:
```python
class GradientReversalFunction(Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)
    
    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.neg() * ctx.alpha, None
```

**Domain Classifier Architecture**:
```
Domain Classifier:
├── Input: 768-dimensional features from VFM
├── Hidden Layer 1: Linear(768, 256) + ReLU + Dropout(0.2)
├── Hidden Layer 2: Linear(256, 128) + ReLU + Dropout(0.2)
└── Output: Linear(128, num_domains)
```

#### 4.1.3 Training Dynamics
- **Adversarial Weight Schedule**: `λ = 2/(1+exp(-10*p)) - 1` where `p` is training progress
- **Balanced Optimization**: Simultaneous minimization of classification loss and maximization of domain confusion
- **Convergence Monitoring**: Domain classifier accuracy should approach random chance

### 4.2 MixStyle Domain Generalization

#### 4.2.1 Mechanism
MixStyle addresses domain shift by mixing feature statistics across different domains during training:

```python
def mixstyle_forward(self, x):
    if not self.training or torch.rand(1) > self.p:
        return x
    
    B = x.size(0)
    mu = x.mean(dim=[2, 3], keepdim=True)
    var = x.var(dim=[2, 3], keepdim=True)
    sig = (var + self.eps).sqrt()
    
    # Mix statistics
    perm = torch.randperm(B)
    lam = Beta(self.alpha, self.alpha).sample()
    
    mu_mix = lam * mu + (1 - lam) * mu[perm]
    sig_mix = lam * sig + (1 - lam) * sig[perm]
    
    return sig_mix * (x - mu) / sig + mu_mix
```

#### 4.2.2 Benefits
- **Style Augmentation**: Increases diversity of feature statistics
- **Domain Invariance**: Reduces dependence on domain-specific style information
- **Simple Integration**: Easy to incorporate into existing architectures
- **Complementary to DANN**: Can be used alongside adversarial training

### 4.3 Stochastic Weight Averaging (SWA)

#### 4.3.1 Mathematical Foundation
SWA computes running averages of model weights during training:

```
θ_SWA = (1/n) * Σ(i=1 to n) θ_i
```

#### 4.3.2 Implementation
```python
class StochasticWeightAveraging:
    def update(self, epoch):
        if epoch >= self.swa_start and (epoch - self.swa_start) % self.swa_freq == 0:
            if self.swa_n == 0:
                # Initialize SWA state
                for name, param in self.model.named_parameters():
                    self.swa_state[name] = param.data.clone()
            else:
                # Update running average
                for name, param in self.model.named_parameters():
                    self.swa_state[name] += (param.data - self.swa_state[name]) / (self.swa_n + 1)
            self.swa_n += 1
```

#### 4.3.3 Advantages
- **Improved Generalization**: Reduces overfitting through weight averaging
- **Stable Convergence**: Smoother loss landscapes
- **Better Calibration**: Improved confidence estimation

### 4.4 Test-Time Adaptation (TTA)

#### 4.4.1 Entropy Minimization
TTA adapts model parameters at inference time using unlabeled test samples:

```python
def test_time_adapt(self, x):
    # Forward pass
    logits = self.model(x)
    probs = F.softmax(logits, dim=1)
    
    # Entropy loss
    entropy = -torch.sum(probs * torch.log(probs + 1e-8), dim=1)
    loss = entropy.mean()
    
    # Backward pass and parameter update
    loss.backward()
    self.optimizer.step()
    
    return logits
```

#### 4.4.2 Benefits
- **Online Adaptation**: Adapts to target domain during inference
- **Unsupervised**: Requires no labeled target data
- **Complementary**: Works alongside other DA techniques

### 4.5 Multi-Source Training Strategy

#### 4.5.1 Leave-One-Dataset-Out (LODO) Evaluation
- **Training**: Combine multiple source datasets
- **Testing**: Evaluate on held-out target dataset
- **Metrics**: Domain-specific performance assessment

#### 4.5.2 Weighted Sampling
- **Balanced Representation**: Equal sampling from each source domain
- **Class Balance**: Maintain class distribution across domains
- **Implementation**: Custom data loader with domain-aware sampling

## 5. Experimental Framework

### 5.1 Training Configuration

#### 5.1.1 Model Setup
- **Base Architecture**: ViT-Base-Patch16-224
- **Pre-trained Weights**: VFM_Fundus_weights.pth
- **Input Resolution**: 224×224 pixels
- **Batch Size**: 32 (with gradient accumulation if needed)

#### 5.1.2 Optimization
- **Optimizer**: AdamW with weight decay (1e-4)
- **Learning Rate**: 1e-5 for VFM (with LLRD)
- **Scheduler**: ReduceLROnPlateau with patience
- **Early Stopping**: 10 epochs patience on validation loss

#### 5.1.3 Data Augmentation
- **Standard Augmentation**: RandomResizedCrop, RandomHorizontalFlip, ColorJitter
- **Advanced Techniques**: MixUp, CutMix, RandAugment
- **Normalization**: ImageNet statistics adapted for fundus images

### 5.2 Evaluation Metrics

#### 5.2.1 Classification Performance
- **Area Under Curve (AUC)**: Primary performance metric
- **Sensitivity at 95% Specificity**: Clinically relevant threshold
- **Balanced Accuracy**: Accounts for class imbalance
- **F1-Score**: Harmonic mean of precision and recall

#### 5.2.2 Calibration Assessment
- **Expected Calibration Error (ECE)**: Reliability of confidence estimates
- **Brier Score**: Proper scoring rule for probabilistic predictions
- **Reliability Diagrams**: Visual assessment of calibration quality

#### 5.2.3 Fairness Metrics
- **Demographic Parity**: Equal positive prediction rates across groups
- **Equalized Odds**: Equal TPR and FPR across demographic groups
- **Underdiagnosis Disparity**: Differences in false negative rates
