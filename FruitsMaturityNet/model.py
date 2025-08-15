import torch
import torch.nn as nn
import timm
from typing import Dict, List, cast

class FruitClassifier(nn.Module):
    """
    EfficientNet-B0 backbone with two classification heads for fruit type and ripeness.
    """
    def __init__(self, num_fruit_types: int, num_ripeness_levels: int):
        super(FruitClassifier, self).__init__()
        self.num_fruit_types = num_fruit_types
        self.num_ripeness_levels = num_ripeness_levels

        # Load EfficientNet-B0 backbone
        self.backbone = timm.create_model('efficientnet_b0', pretrained=True)
        # Remove the original classifier head
        self.backbone.classifier = nn.Identity()

        # Get the number of features from the backbone's output
        # This is typically the last layer before the original classifier
        # For EfficientNet-B0, it's the `num_features` attribute
        num_features = cast(int, self.backbone.num_features)

        # Head for fruit type classification
        self.fruit_type_head = nn.Linear(num_features, num_fruit_types)

        # Head for ripeness classification
        self.ripeness_head = nn.Linear(num_features, num_ripeness_levels)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Forward pass through the model.

        Args:
            x (torch.Tensor): Input image tensor.

        Returns:
            Dict[str, torch.Tensor]: Dictionary containing logits for fruit type and ripeness.
        """
        features = self.backbone(x)
        fruit_type_logits = self.fruit_type_head(features)
        ripeness_logits = self.ripeness_head(features)
        return {
            "fruit_type_logits": fruit_type_logits,
            "ripeness_logits": ripeness_logits
        }

# Example usage and label mappings (these should be consistent across miner/validator)
# In a real scenario, these would be loaded from a config or generated during data prep.
FRUIT_TYPE_MAP: Dict[int, str] = {
    0: "Apple", 1: "Banana", 2: "Orange",
}
FRUIT_TYPE_REV_MAP: Dict[str, int] = {v: k for k, v in FRUIT_TYPE_MAP.items()}

# For ripeness, we'll use a simplified binary classification for demonstration.
# The actual labels will come from manual feedback during online improvement.
RIPENESS_MAP: Dict[int, str] = {
    0: "Fresh",
    1: "Rotten", # This will be the target for manual feedback
}
RIPENESS_REV_MAP: Dict[str, int] = {v: k for k, v in RIPENESS_MAP.items()}

NUM_FRUIT_TYPES = len(FRUIT_TYPE_MAP)
NUM_RIPENESS_LEVELS = len(RIPENESS_MAP)
