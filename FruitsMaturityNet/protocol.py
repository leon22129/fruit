import bittensor as bt
from typing import Optional, List, Dict, Any
import base64

class FruitPrediction(bt.Synapse):
    """
    A Bittensor synapse for requesting fruit type and ripeness predictions.
    Miners receive an image and return predictions.
    """
    image: str = ""  # Base64 encoded image string
    request_id: str = "" # Unique ID for the request, used for caching and feedback

    # Output fields (filled by the miner)
    fruit_type_prediction: Optional[str] = None
    ripeness_prediction: Optional[str] = None

    # Optional: Confidence scores for predictions
    fruit_type_confidence: Optional[float] = None
    ripeness_confidence: Optional[float] = None

    def deserialize(self) -> Dict[str, Any]:
        """
        Deserialize the synapse to a dictionary containing all the response information.
        This is useful for logging and processing the miner's response.
        """
        return {
            "fruit_type_prediction": self.fruit_type_prediction,
            "ripeness_prediction": self.ripeness_prediction,
            "fruit_type_confidence": self.fruit_type_confidence,
            "ripeness_confidence": self.ripeness_confidence,
            "request_id": self.request_id,
        }

    def encode_image(self, image_bytes: bytes) -> None:
        """Encodes image bytes to base64 string."""
        self.image = base64.b64encode(image_bytes).decode('utf-8')

    def decode_image(self) -> Optional[bytes]:
        """Decodes base64 image string to bytes."""
        if self.image:
            return base64.b64decode(self.image)
        return None
    
    @classmethod
    def from_dict(cls, data: dict):
        """Create a FruitPrediction object from a dictionary."""
        obj = cls()
        obj.request_id = data.get("request_id", "")
        obj.fruit_type_prediction = data.get("fruit_type_prediction")
        obj.ripeness_prediction = data.get("ripeness_prediction")
        obj.fruit_type_confidence = data.get("fruit_type_confidence")
        obj.ripeness_confidence = data.get("ripeness_confidence")
        return obj

class FeedbackSynapse(bt.Synapse):
    """
    A Bittensor synapse for validators to send manual feedback (ground truth) to miners.
    Miners receive this feedback to improve their models.
    """
    request_id: str = "" # The ID of the original prediction request
    true_fruit_type: str = ""
    true_ripeness: str = ""

    def deserialize(self) -> Dict[str, Any]:
        """
        Deserialize the synapse to a dictionary containing the feedback information.
        """
        return {
            "request_id": self.request_id,
            "true_fruit_type": self.true_fruit_type,
            "true_ripeness": self.true_ripeness,
        }
