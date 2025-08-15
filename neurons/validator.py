import bittensor as bt
import torch
import time
import os
import json
import uuid
from datetime import datetime
from collections import deque
from typing import List, Dict, Any, Tuple
from torch.utils.data import DataLoader
import torch.nn.functional as F
import io
import torchvision.transforms as transforms # Added for image conversion

# Import model, protocol, and dataset from the FruitsMaturityNet
from FruitsMaturityNet.protocol import FruitPrediction, FeedbackSynapse # Added FeedbackSynapse
from FruitsMaturityNet.model import FRUIT_TYPE_MAP, RIPENESS_MAP, FRUIT_TYPE_REV_MAP, RIPENESS_REV_MAP
from FruitsMaturityNet.dataset import FruitDataset # Only FruitDataset needed for validator
from FruitsMaturityNet.validator import forward

class Validator:
    def __init__(self):
        self.config = self.get_config()
        self.setup_logging()
        self.wallet = bt.wallet(config=self.config)
        self.subtensor = bt.subtensor(config=self.config)
        self.dendrite = bt.dendrite(wallet=self.wallet)
        self.metagraph = self.subtensor.metagraph(self.config.netuid)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Ensure the feedback directory exists
        self.feedback_dir = os.path.join(os.getcwd(), "feedback")
        os.makedirs(self.feedback_dir, exist_ok=True)
        bt.logging.info(f"Feedback will be logged to: {self.feedback_dir}")

        # Load dataset for sending requests
        self.dataset_root = os.path.join(os.getcwd(), "dataset")
        self.fruit_dataset = FruitDataset(root_dir=self.dataset_root, train=False) # Use test set for validation
        self.dataloader = DataLoader(self.fruit_dataset, batch_size=1, shuffle=True)
        self.data_iterator = iter(self.dataloader)

        # Store recent requests for feedback
        self.pending_feedback: deque[Dict[str, Any]] = deque(maxlen=100)

        # Initialize scores for miners
        self.scores = torch.zeros(self.metagraph.n, dtype=torch.float32).to(self.device)
        self.alpha = 0.9 # Exponential moving average alpha for scores

    def get_config(self):
        parser = bt.cli.config_parser()
        parser.add_argument('--neuron.name', type=str, default='fruit_validator', help='Name of the validator.')
        parser.add_argument('--neuron.min_axon_size', type=int, default=1, help='Minimum number of axons to query.')
        parser.add_argument('--neuron.feedback_prompt_interval', type=int, default=60, help='Interval to prompt for manual feedback (seconds).')
        config = bt.config(parser)
        return config

    def setup_logging(self):
        bt.logging(config=self.config, logging_dir=self.config.neuron.name)

    def get_random_image_data(self) -> Tuple[bytes, str, str]:
        """Gets a random image and its true labels from the dataset."""
        try:
            sample = next(self.data_iterator)
        except StopIteration:
            # Reset iterator if all samples are consumed
            self.data_iterator = iter(self.dataloader)
            sample = next(self.data_iterator)

        image_tensor = sample["image"].squeeze(0) # Remove batch dimension
        fruit_type_label_id = sample["fruit_type_label"].item()
        ripeness_label_id = sample["ripeness_label"].item() # This will be the default 0 initially

        # Convert tensor to PIL Image and then to bytes
        to_pil = transforms.ToPILImage()
        image_pil = to_pil(image_tensor)
        img_byte_arr = io.BytesIO()
        image_pil.save(img_byte_arr, format='JPEG')
        image_bytes = img_byte_arr.getvalue()

        true_fruit_type = FRUIT_TYPE_MAP.get(fruit_type_label_id, "Unknown")
        true_ripeness = RIPENESS_MAP.get(ripeness_label_id, "Normal") # Default to Normal

        return image_bytes, true_fruit_type, true_ripeness

    def log_feedback_request(self, request_id: str, image_bytes: bytes, true_fruit_type: str, true_ripeness: str, miner_responses_info: Dict[int, Dict[str, Any]]):
        """Logs a request to the feedback folder for manual review."""
        feedback_entry = {
            "request_id": request_id,
            "timestamp": datetime.now().isoformat(),
            "true_fruit_type_dataset": true_fruit_type,
            "true_ripeness_dataset": true_ripeness,
            "miner_responses": miner_responses_info,
            "feedback_provided": False,
            "true_fruit_type_manual": None,
            "true_ripeness_manual": None,
        }
        self.pending_feedback.append({"feedback_entry": feedback_entry, "image_bytes": image_bytes})
        bt.logging.info(f"Logged request {request_id} for manual feedback.")

    def process_manual_feedback(self):
        """Prompts the user for manual feedback and sends FeedbackSynapse to miners."""
        if not self.pending_feedback:
            return

        bt.logging.info("\n--- Manual Feedback Prompt ---")
        bt.logging.info(f"You have {len(self.pending_feedback)} pending feedback items.")

        feedback_item = self.pending_feedback.popleft()
        entry = feedback_item["feedback_entry"]
        image_bytes = feedback_item["image_bytes"]

        bt.logging.info(f"\nRequest ID: {entry['request_id']}")
        bt.logging.info(f"Original Dataset Fruit Type: {entry['true_fruit_type_dataset']}")
        bt.logging.info(f"Original Dataset Ripeness: {entry['true_ripeness_dataset']}")
        bt.logging.info(f"Miner Responses: {json.dumps(entry['miner_responses'], indent=2)}")

        temp_image_path = os.path.join(self.feedback_dir, f"review_{entry['request_id']}.jpg")
        with open(temp_image_path, "wb") as f:
            f.write(image_bytes)
        bt.logging.info(f"Image saved for review at: {temp_image_path}")

        true_fruit_type_input = input("Enter CORRECT Fruit Type (e.g., Apple, Banana): ").strip()
        true_ripeness_input = input("Enter CORRECT Ripeness (Normal, Ripe/Overripe): ").strip()

        if not true_fruit_type_input or not true_ripeness_input:
            bt.logging.warning("Feedback skipped due to empty input.")
            return

        if true_fruit_type_input not in FRUIT_TYPE_REV_MAP or true_ripeness_input not in RIPENESS_REV_MAP:
            bt.logging.warning("Invalid input. Skipping feedback.")
            return

        entry["true_fruit_type_manual"] = true_fruit_type_input
        entry["true_ripeness_manual"] = true_ripeness_input
        entry["feedback_provided"] = True
        entry["feedback_provided_at"] = datetime.now().isoformat()

        feedback_file_path = os.path.join(self.feedback_dir, f"feedback_{entry['request_id']}.json")
        with open(feedback_file_path, "w") as f:
            json.dump(entry, f, indent=4)
        bt.logging.info(f"Feedback saved to {feedback_file_path}")

        os.remove(temp_image_path)

        # Send feedback to miners
        responding_miner_uids = list(entry["miner_responses"].keys())
        feedback_synapse = FeedbackSynapse(
            request_id=entry["request_id"],
            true_fruit_type=true_fruit_type_input,
            true_ripeness=true_ripeness_input
        )

        target_axons = []
        for uid in responding_miner_uids:
            if uid < len(self.metagraph.axons) and self.metagraph.axons[uid].is_serving:
                target_axons.append(self.metagraph.axons[uid])

        if target_axons:
            self.dendrite.query(target_axons, feedback_synapse, timeout=5, deserialize=False)

    def calculate_rewards(self, responses: List[FruitPrediction], true_fruit_type: str, true_ripeness: str) -> torch.Tensor:
        """
        Calculates rewards for miners based on their predictions and true labels.
        Rewards are higher for correct predictions.
        """
        rewards = torch.zeros(len(responses), dtype=torch.float32).to(self.device)
        true_fruit_type_id = FRUIT_TYPE_REV_MAP.get(true_fruit_type, -1)
        true_ripeness_id = RIPENESS_REV_MAP.get(true_ripeness, -1)

        if true_fruit_type_id == -1 or true_ripeness_id == -1:
            bt.logging.warning(f"Invalid true labels for reward calculation: {true_fruit_type}, {true_ripeness}. Returning zero rewards.")
            return rewards

        for i, response in enumerate(responses):
            if response.fruit_type_prediction is None or response.ripeness_prediction is None:
                continue # No prediction, no reward

            miner_fruit_type_id = FRUIT_TYPE_REV_MAP.get(response.fruit_type_prediction, -1)
            miner_ripeness_id = RIPENESS_REV_MAP.get(response.ripeness_prediction, -1)

            fruit_type_correct = (miner_fruit_type_id == true_fruit_type_id)
            ripeness_correct = (miner_ripeness_id == true_ripeness_id)

            # Simple reward scheme: 0.5 for each correct prediction
            reward = 0.0
            if fruit_type_correct:
                reward += 0.5
            if ripeness_correct:
                reward += 0.5

            rewards[i] = reward

        return rewards

    def update_scores(self, rewards: torch.FloatTensor, uids: torch.LongTensor):
        """Updates the validator's internal scores based on rewards."""
        # Ensure scores tensor is correctly sized
        if self.scores.shape[0] != self.metagraph.n:
            self.scores = torch.zeros(self.metagraph.n, dtype=torch.float32).to(self.device)

        # Apply rewards to the corresponding UIDs
        for i, uid in enumerate(uids):
            if uid < len(self.scores):
                self.scores[uid] = self.alpha * self.scores[uid] + (1 - self.alpha) * rewards[i]

        # Normalize scores
        if torch.sum(self.scores) > 0:
            self.scores = F.normalize(self.scores, p=1, dim=0)
        
        bt.logging.info(f"Updated scores: {self.scores}")

    def run(self):
        if not self.wallet.hotkey_file:
            bt.logging.error("Hotkey file not found. Please create one.")
            return

        if not self.subtensor.is_hotkey_registered(netuid=self.config.netuid, hotkey_ss58=self.wallet.hotkey.ss58_address):
            bt.logging.error(f"Hotkey {self.wallet.hotkey.ss58_address} is not registered on netuid {self.config.netuid}.")
            return

        bt.logging.info(f"Validator started on network: {self.config.subtensor.chain_endpoint} with netuid: {self.config.netuid}")

        step = 0
        last_feedback_prompt_time = time.time()

        while True:
            try:
                # Sync metagraph
                self.metagraph.sync(subtensor=self.subtensor)
                
                forward(self)

                # Set weights on the chain
                weights = self.scores.clone().to('cpu')
                uids_to_set = torch.arange(self.metagraph.n).to('cpu')

                if torch.sum(weights) > 0:
                    self.subtensor.set_weights(
                        wallet=self.wallet,
                        netuid=self.config.netuid,
                        uids=uids_to_set,
                        weights=weights,
                        wait_for_inclusion=False,
                        wait_for_finalization=True,
                        version_key=1
                    )
                    bt.logging.info(f"Set weights: {weights}")

                # Process manual feedback periodically
                if time.time() - last_feedback_prompt_time > self.config.neuron.feedback_prompt_interval:
                    self.process_manual_feedback()
                    last_feedback_prompt_time = time.time()

                step += 1
                time.sleep(bt.__blocktime__)

            except Exception as e:
                bt.logging.error(f"Validator main loop error: {e}")
                time.sleep(30)

if __name__ == "__main__":
    validator = Validator()
    validator.run()
