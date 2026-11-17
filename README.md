# BiAlign-MIL
## Bidirectional Cross-Scale Context Enhancement for Whole Slide Image Classification

BiAlign-MIL is a bidirectional cross-scale context enhancement framework for weakly supervised whole slide image (WSI) classification. It explicitly models spatial correspondences between high- and low-magnification patches, enabling regional context aggregation and instance-conditioned feedback to enhance high-magnification representations.

The framework establishes a parent–child relationship between low-magnification regions and their corresponding high-magnification instances. Through cross-attention-based regional aggregation and gated residual feedback, BiAlign-MIL integrates local morphological information with broader tissue-level context while preserving the original high-magnification instance structure.

The original implementation class name, `BiAlign-MIL`, is retained to ensure compatibility with existing imports and checkpoint parameter naming.

---

## 1. Method Overview

BiAlign-MIL consists of three main stages:

1. **Spatial Parent–Child Alignment:** Establish spatial correspondences between high-magnification instances and their associated low-magnification parent regions.
2. **Bidirectional Cross-Scale Interaction:** Aggregate high-magnification instance information into regional representations and feed the resulting context back to individual instances through instance-conditioned gating.
3. **Slide-Level Classification:** Integrate the enhanced high-magnification representations using the multi-scale aggregation backbone to generate slide-level predictions.

### 1.1 Regional Context Aggregation

For each valid low-magnification parent region, its feature representation serves as the query, while the corresponding high-magnification child instances provide the keys and values.

A four-head cross-attention mechanism aggregates local morphological information into a regional context representation:

\[
u_j=e_j^L+
\operatorname{MHA}
(e_j^L,E_j^H,E_j^H)
\]

where:
- \(e_j^L\) denotes the low-magnification embedding of parent region \(j\).
- \(E_j^H\) represents the high-magnification embeddings belonging to the same parent region.
- \(u_j\) denotes the resulting regional context representation.

### 1.2 Instance-Conditioned Context Feedback

The aggregated regional context is subsequently returned to its corresponding high-magnification instances.

An instance-conditioned gating mechanism determines the feedback intensity for each child instance:

\[
g_i=\sigma\left(
\operatorname{MLP}
([e_i^H;u_j;m_i])
\right)
\]

The enhanced high-magnification feature is calculated as:

\[
\tilde h_i=h_i+
m_i\cdot g_i\cdot W_{\mathrm{out}}u_j
\]

where:
- \(h_i\) is the original high-magnification feature.
- \(g_i\) is the instance-specific gating coefficient.
- \(m_i\) is the valid spatial-pairing mask.
- \(W_{\mathrm{out}}\) projects the regional context into the original feature space.
- \(\tilde h_i\) is the enhanced instance representation.

Instances without valid spatial correspondences retain their original features.

This design enables instances within the same parent region to share regional context while receiving different feedback strengths according to their individual representations.

---

## 2. Model Architecture

The implementation follows a two-branch architecture initialized from the same high-magnification checkpoint.

### High-Magnification Reference Branch

- Loads the pretrained high-magnification checkpoint.
- Freezes all model parameters.
- Remains in evaluation mode.
- Produces reference predictions.

### Multi-Scale Enhancement Branch

- Initializes from the same high-magnification checkpoint.
- Uses the frozen `_fc1` module to extract high-magnification embeddings.
- Projects low-magnification features into a 384-dimensional embedding space.
- Applies spatially grouped cross-attention and instance-conditioned residual feedback.
- Integrates enhanced representations through the multi-scale aggregation backbone.

The trainable components include `layer1`, `_fc3`, `ctm`, and `norm` in the multi-scale branch, together with the corresponding cross-scale fusion modules.

The enhanced features are written back according to their original instance indices before slide-level aggregation.

---

## 3. Repository Structure

```text
BiAlign-MIL/
│
├── README.md
├── config.py
├── data.py
├── run_v1.py
├── verify_package.py
│
└── models/
    ├── __init__.py
    └── model_bialign_ms.py
```

The implementation is organized into model architecture, configuration, data processing, execution, and verification modules.

| File | Description |
|---|---|
| `models/model_bialign_ms.py` | Core BiAlign-MIL cross-scale model implementation |
| `config.py` | Model and training configuration |
| `data.py` | Feature loading and spatial parent–child correspondence processing |
| `run_v1.py` | Training and inference entry point |
| `verify_package.py` | Source integrity, syntax, and data correspondence verification |

---

## 4. Input Representation

BiAlign-MIL processes one WSI bag per forward pass.

The input consists of concatenated high- and low-magnification features and their corresponding spatial coordinates.

### 4.1 Feature Tensor

\[
X\in\mathbb{R}^{N\times1536}
\]

The feature tensor is organized as:

- First 768 dimensions: high-magnification features.
- Last 768 dimensions: low-magnification features.

### 4.2 Spatial Coordinates

\[
C\in\mathbb{R}^{N\times5}
\]

Each coordinate vector contains:

| Index | Description |
|---|---|
| 0 | High-magnification x-coordinate |
| 1 | High-magnification y-coordinate |
| 2 | Low-magnification x-coordinate |
| 3 | Low-magnification y-coordinate |
| 4 | Low-magnification pairing validity indicator |

The validity indicator determines whether an instance participates in cross-scale interaction.

### 4.3 Slide-Level Labels

\[
y\in\{0,1\}
\]

The model is trained using slide-level supervision for binary classification, without requiring instance-level annotations.

---

## 5. Experimental Configuration

The principal configuration of BiAlign-MIL V1 is summarized below.

| Parameter | Configuration |
|---|---|
| High feature dimension | 768 |
| Low feature dimension | 768 |
| Embedding dimension | 384 |
| Number of classes | 2 |
| Number of attention heads | 4 |
| Backbone layers | 2 |
| Number of clusters | 5 |
| Optimizer | Adam |
| Fusion module learning rate | 2e-4 |
| Unfrozen backbone learning rate | 2e-5 |
| Weight decay | 1e-5 |
| Maximum training epochs | 300 |
| Minimum epochs | 20 |
| Early stopping patience | 20 |
| Batch size | 1 WSI per iteration |
| Output projection initialization | Zero initialization |
| Final gating layer bias | -2.0 |

The output projection is initialized with zero weights and zero bias, preserving the original high-magnification representations at initialization.

The final gating layer uses a bias initialization of -2.0 to control the initial strength of contextual residual feedback.

During cross-scale training, the model employs slide-level cross-entropy loss.

---

## 6. Environment and Dependencies

The implementation is based on the PyTorch deep learning framework.

The principal dependencies include:

- Python
- PyTorch
- NumPy
- scikit-learn
- einops
- nystrom-attention
- mmcv

The backbone implementation and pretrained checkpoints should be used with their corresponding dependency configurations to maintain model architecture and parameter compatibility.

For environment management, the installed package versions can be recorded using:

```bash
python -m pip freeze > environment-original.txt
```

---

## 7. Usage

### Step 1. Prepare the Input Data

Prepare the high- and low-magnification feature representations and their spatial coordinates according to the input format described in Section 4.

### Step 2. Configure the Experiment

Set the model parameters, data paths, checkpoint paths, and training hyperparameters in:

```text
config.py
```

### Step 3. Verify the Implementation

Run the verification script:

```bash
python verify_package.py
```

### Step 4. Execute Training and Inference

Use the main execution script:

```bash
python run_v1.py
```

The execution configuration is managed through the project configuration module.

---

## 8. Datasets

BiAlign-MIL is evaluated on two publicly available whole slide image datasets:

- **CAMELYON16:** Lymph node metastasis classification.
- **TCGA-NSCLC:** Lung cancer subtype classification.

Both tasks are formulated as weakly supervised slide-level classification problems.

The model uses high- and low-magnification feature representations with spatial correspondence information to perform cross-scale contextual enhancement.

Dataset access and preparation should follow the respective official distribution policies.

---

## 9. Implementation Details

Several implementation choices are retained to ensure consistency with the original model architecture:

- The original model class name `BiAlign-MIL` is preserved.
- The high-magnification reference branch remains frozen.
- The multi-scale branch is initialized from the corresponding high-magnification checkpoint.
- Cross-scale interaction is performed only for valid spatially paired instances.
- Parent-level context is shared among child instances belonging to the same spatial region.
- Instance-conditioned gating independently controls contextual feedback.
- Enhanced features preserve their original instance ordering before aggregation.
- Checkpoint loading uses strict parameter matching to maintain architectural consistency.

These choices establish a consistent implementation of the bidirectional cross-scale enhancement mechanism.

---

## 10. Acknowledgements

We acknowledge the authors and developers of the publicly available datasets, pretrained feature encoders, and multiple instance learning frameworks that support this research.

---

## 11. Contact

For questions regarding the implementation or research, please contact the corresponding author.

**Project:** BiAlign-MIL  
**Repository:** https://github.com/YokiMa02/BiAlign-MIL

**Paper Title:** Bidirectional Cross-Scale Context Enhancement for Whole Slide Image Classification