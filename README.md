# OpenVLA TFDS Configuration

## Related Repositories

- **RLDS Repository**: https://github.com/kpertsch/rlds_dataset_builder/tree/main
- **OpenVLA Repository**: https://github.com/openvla/openvla/tree/main

## Data Generation

To generate data, you have two options:

1. **Custom Example**: Look at `save_data_example.py` to see a custom implementation example
2. **RLDS Example**: Run `create_example_data.py` provided by RLDS inside the `example_dataset` folder


# OpenVLA TFDS Setup Example

This repository demonstrates how to create a TensorFlow Dataset (TFDS) for OpenVLA by showing the modifications required to adapt from a generic example dataset to a domain-specific dataset (custom_reach1k).

## Key Differences Between Example and Custom Reach1K

### 1. **Class Name Changes**

**Example Dataset:**
```python
class ExampleDataset(tfds.core.GeneratorBasedBuilder):
```

**Custom Reach1K:**
```python
class Customreach1k(tfds.core.GeneratorBasedBuilder):
```

### 2. **Image Resolution Modifications**

**Example Dataset:**
```python
'image': tfds.features.Image(
    shape=(64, 64, 3),  # Small 64x64 images
    dtype=np.uint8,
    encoding_format='png',
    doc='Main camera RGB observation.',
),
```

**Custom Reach1K:**
```python
'image': tfds.features.Image(
    shape=(224, 224, 3),  # Larger 224x224 images for better visual quality (openvla expected size)
    dtype=np.uint8,
    encoding_format='png',
    doc='Main camera RGB observation.',
),
```

### 3. **Feature Schema Simplification**

The custom_reach1k dataset removes some observation features to focus on the essential components:

**Example Dataset (includes all features):**
```python
'observation': tfds.features.FeaturesDict({
    'image': tfds.features.Image(...),
    'wrist_image': tfds.features.Image(...),  # ✅ Included
    'state': tfds.features.Tensor(...),       # ✅ Included
}),
```

**Custom Reach1K (simplified):**
```python
'observation': tfds.features.FeaturesDict({
    'image': tfds.features.Image(
        shape=(224, 224, 3),
        dtype=np.uint8,
        encoding_format='png',
        doc='Main camera RGB observation.',
    ),
    # 'wrist_image': tfds.features.Image(...),  # ❌ Commented out
    # 'state': tfds.features.Tensor(...),       # ❌ Commented out
}),
```

### 4. **Action Space Dimensionality**

The action space is reduced to match the specific robot configuration:

**Example Dataset:**
```python
'action': tfds.features.Tensor(
    shape=(10,),  # 10-dimensional action space
    dtype=np.float32,
    doc='Robot action, consists of [7x joint velocities, '
        '2x gripper velocities, 1x terminate episode].',
),
```

**Custom Reach1K:**
```python
'action': tfds.features.Tensor(
    shape=(7,),   # 7-dimensional action space
    dtype=np.float32,
    doc='Robot action, consists of [7x joint velocities, '
        '2x gripper velocities, 1x terminate episode].',
),
```

### 5. **Data File Path Patterns**

The file naming convention is updated to match the specific dataset:

**Example Dataset:**
```python
return {
    'train': self._generate_examples(path='data/train/episode_*.npy'),
    'val': self._generate_examples(path='data/val/episode_*.npy'),
}
```

**Custom Reach1K:**
```python
return {
    'train': self._generate_examples(path='data/train/reach_episode_*.npy'),  # 'reach_' prefix
    'val': self._generate_examples(path='data/val/reach_episode_*.npy'),      # 'reach_' prefix
}
```


This approach ensures compatibility with OpenVLA Dataset and Shape.

### 6. **Building the Dataset**

Once your data is set up correctly, you can build the TFDS dataset using:

```bash
tfds build
```

**Important**: You must run this command from inside the `custom_reach1k` folder for it to work properly.




## Required Changes to OpenVLA Scripts

To integrate your custom dataset with OpenVLA, you need to modify three key files in the OpenVLA codebase:

### 1. **transforms.py** - Dataset Standardization Transform

**File**: `openvla/prismatic/vla/datasets/rlds/oxe/transforms.py`

Add your custom transform function and register it in the transforms registry:

```python
#######################################
#ADDED New Dataset Here
def custom_reach1k_dataset_transform(trajectory: dict) -> dict:
    """
    Standardizes customReach1k_data trajectories.
    Assumes each step has keys: 'image', 'action', 'language_instruction'.
    """
    # If your data is already in the correct format, you may just need to pass it through:
    return trajectory
#######################################

# === Registry ===
OXE_STANDARDIZATION_TRANSFORMS = {
    "customreach1k": custom_reach1k_dataset_transform,  # ✅ ADD THIS LINE
    # ...existing transforms...
}
```


### 2. **configs.py** - Dataset Configuration

**File**: `openvla/prismatic/vla/datasets/rlds/oxe/configs.py`

Add configuration for your dataset specifying the observation keys and state encoding:

```python
# Add your dataset configuration
OXE_DATASET_CONFIGS = {
    "customreach1k": {
        "image_obs_keys": {"primary": "image", "secondary": None, "wrist": None},
        "depth_obs_keys": {"primary": None, "secondary": None, "wrist": None},
        "state_obs_keys": {"primary": None, "secondary": None, "wrist": None},
        "state_encoding": StateEncoding.POS_QUAT,  
        "action_encoding": ActionEncoding.EEF_POS,  
    },
    # ...existing configs...
}
```


### 3. **finetune.py** - Training Script Environment Setup

**File**: `openvla/vla-scripts/finetune.py`

Add the path to your dataset builder directory:

```python
######################################
#ADD path for dataset builder
import os
os.environ['TFDS_DATASETS'] = '~/[path to folder]/rlds_dataset_builder'


import custom_reach1k.custom_reach1k_dataset_builder
######################################
```


This integration allows OpenVLA to seamlessly work with your custom dataset while maintaining compatibility with the existing training pipeline.

Look at **File**: `openvla.sh` for example and needed export path for dataset setup