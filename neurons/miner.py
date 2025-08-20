from FruitsMaturityNet.base.miner import BaseMinerNeuron
import bittensor as bt
import argparse
import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torch.optim as optim
from torch import Tensor
from PIL import Image
import io
import os
import time
import threading
from collections import OrderedDict, deque
from typing import cast, Tuple

from FruitsMaturityNet.protocol import FruitPrediction, FeedbackSynapse
from FruitsMaturityNet.model import FruitClassifier, FRUIT_TYPE_MAP, RIPENESS_MAP, NUM_FRUIT_TYPES, NUM_RIPENESS_LEVELS
from FruitsMaturityNet.dataset import FruitDataset, FeedbackDataset


class Miner(BaseMinerNeuron):
    def __init__(self, config=None):
        super(Miner, self).__init__(config=config)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        bt.logging.info(f"Using device: {self.device}")

        # Model
        self.model = FruitClassifier(NUM_FRUIT_TYPES, NUM_RIPENESS_LEVELS).to(self.device)
        self.model_lock = threading.Lock()

        # Paths
        self.dataset_root = os.path.join(os.getcwd(), "dataset")
        self.feedback_dataset_root = os.path.join(os.getcwd(), "data", "feedback_dataset")
        os.makedirs(self.feedback_dataset_root, exist_ok=True)
        self.model_path = os.path.join(os.getcwd(), "fruit_model.pth")

        # Buffers
        self.feedback_buffer: deque[Tuple[bytes, str, str]] = deque(maxlen=1000)
        self.image_cache = OrderedDict()
        self.cache_max_size = 1000

        # Transform
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        # Load or offline train
        if os.path.exists(self.model_path):
            self.load_model()
        else:
            bt.logging.info("No saved model found. Training offline...")
            self.offline_train()
            self.save_model()

        # Background fine-tuning
        self.training_thread = threading.Thread(target=self.background_loop, daemon=True)
        self.training_thread.start()

        self.axon = bt.axon(wallet=self.wallet, port=self.config.axon.port)
        # Attach endpoints
        self.axon.attach(
            forward_fn=self.predict_fruit
        ).attach(
            forward_fn=self.receive_feedback
        )

    # ----------------------------
    # Save / Load model
    # ----------------------------
    def save_model(self):
        torch.save(self.model.state_dict(), self.model_path)
        bt.logging.info(f"Model saved to {self.model_path}")

    def load_model(self):
        self.model.load_state_dict(torch.load(self.model_path, map_location=self.device))
        bt.logging.info(f"Loaded model from {self.model_path}")

    # ----------------------------
    # Offline training
    # ----------------------------
    def offline_train(self, epochs: int = 5):
        dataset = FruitDataset(root_dir=self.dataset_root, train=True, transform=self.transform)
        dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
        self.train_model(self.model, dataloader, epochs=epochs, lr=1e-4)

    def train_model(self, model, dataloader, epochs=1, lr=1e-4):
        optimizer = optim.Adam(model.parameters(), lr=lr)
        model.train()
        for epoch in range(epochs):
            total_loss = 0
            for batch in dataloader:
                images = batch["image"].to(self.device)
                ft_labels = batch["fruit_type_label"].to(self.device)
                rp_labels = batch["ripeness_label"].to(self.device)

                optimizer.zero_grad()
                outputs = model(images)
                loss_ft = F.cross_entropy(outputs["fruit_type_logits"], ft_labels)
                loss_rp = F.cross_entropy(outputs["ripeness_logits"], rp_labels)
                loss = loss_ft + loss_rp
                loss.backward()
                optimizer.step()
                total_loss += loss.item()

            bt.logging.info(f"Epoch {epoch+1}/{epochs} - Loss: {total_loss/len(dataloader):.4f}")
            print(f"Epoch {epoch+1}/{epochs} - Loss: {total_loss/len(dataloader):.4f}")

    # ----------------------------
    # Background loop
    # ----------------------------
    def background_loop(self):
        while True:
            if len(self.feedback_buffer) >= 5:
                bt.logging.info(f"Fine-tuning with {len(self.feedback_buffer)} feedback samples...")
                feedback_list = list(self.feedback_buffer)
                self.feedback_buffer.clear()

                valid_feedback = []

                # Save feedback to permanent dataset
                for idx, (img_bytes, ft, rp) in enumerate(feedback_list):
                    if not ft or not rp:
                        bt.logging.warning(f"Skipping invalid feedback: {ft}, {rp}")
                        continue

                    # save feedback to permanent dataset
                    img_name = f"{time.time()}_{idx}.jpg"
                    label_file = img_name.replace(".jpg", ".txt")
                    with open(os.path.join(self.feedback_dataset_root, img_name), "wb") as f:
                        f.write(img_bytes)
                    with open(os.path.join(self.feedback_dataset_root, label_file), "w") as f:
                        f.write(f"{ft},{rp}")

                    valid_feedback.append((img_bytes, ft, rp))

                if valid_feedback:
                    # Fine-tune
                    feedback_dataset = FeedbackDataset([
                        {"image_bytes": img, "true_fruit_type": ft, "true_ripeness": rp}
                        for img, ft, rp in feedback_list
                    ], transform=self.transform)
                    loader = DataLoader(feedback_dataset, batch_size=16, shuffle=True)

                    with self.model_lock:
                        self.train_model(self.model, loader, epochs=1, lr=1e-5)
                        self.save_model()

                    bt.logging.info("Fine-tuning complete. Weights saved.")
                else:
                    bt.logging.warning("No valid feedback to fine-tune.")

            time.sleep(5)

    # ----------------------------
    # Prediction
    # ----------------------------
    def predict_fruit(self, synapse: FruitPrediction) -> FruitPrediction:
        try:
            img_bytes = synapse.decode_image()
            if not img_bytes:
                synapse.fruit_type_prediction = "Error: No image"
                synapse.ripeness_prediction = "Error: No image"
                bt.logging.error(f"Error: No image data found in synapse {synapse.request_id}")
                return synapse

            # Cache for feedback
            self.image_cache[synapse.request_id] = img_bytes
            if len(self.image_cache) > self.cache_max_size:
                self.image_cache.popitem(last=False)

            image = Image.open(io.BytesIO(img_bytes)).convert('RGB')
            tensor = cast(Tensor, self.transform(image)).unsqueeze(0).to(self.device)

            with self.model_lock:
                self.model.eval()
                with torch.no_grad():
                    outputs = self.model(tensor)
                    ft_id = torch.argmax(outputs["fruit_type_logits"], dim=1).item()
                    rp_id = torch.argmax(outputs["ripeness_logits"], dim=1).item()
                    synapse.fruit_type_prediction = FRUIT_TYPE_MAP.get(ft_id, "Unknown")
                    synapse.ripeness_prediction = RIPENESS_MAP.get(rp_id, "Unknown")

        except Exception as e:
            bt.logging.error(f"Prediction error: {e}")
        bt.logging.info(f"Response: ReqNo: {synapse.request_id}, Fruit: {synapse.fruit_type_prediction}, Ripeness: {synapse.ripeness_prediction}")
        return synapse

    # ----------------------------
    # Feedback
    # ----------------------------
    def receive_feedback(self, synapse: FeedbackSynapse) -> FeedbackSynapse:
        img_bytes = self.image_cache.get(synapse.request_id)
        if img_bytes:
            self.feedback_buffer.append((img_bytes, synapse.true_fruit_type, synapse.true_ripeness))
            bt.logging.info(f"Feedback added. Buffer size: {len(self.feedback_buffer)}, Request ID: {synapse.request_id} - Fruit: {synapse.true_fruit_type}, Ripeness: {synapse.true_ripeness}")
        else:
            bt.logging.warning(f"No image found for feedback {synapse.request_id}")
        return synapse


if __name__ == "__main__":
    with Miner() as miner:
        bt.logging.info(f"Running miner on subnet {miner.config.netuid}")
        last_block = 0
        while True:
            block = getattr(miner, "cached_block", None)
            metrics = getattr(miner, "cached_metrics", None)

            if block and block % 5 == 0 and block > last_block and metrics:
                log = (
                    f"Block: {block} | "
                    f"Stake:{metrics['stake']:.02f} | "
                    f"Rank:{metrics['rank']:.04f} | "
                    f"Trust:{metrics['trust']:.04f} | "
                    f"Consensus:{metrics['consensus']:.04f} | "
                    f"Incentive:{metrics['incentive']:.04f} | "
                    f"Emission:{metrics['emission']:.04f}"
                )
                bt.logging.info(log)
                last_block = block

            time.sleep(5)