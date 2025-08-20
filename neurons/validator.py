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
from FruitsMaturityNet.base.validator import BaseValidatorNeuron

class Validator(BaseValidatorNeuron):
    def __init__(self, config=None):
        super(Validator, self).__init__(config=config)

        if self.config.neuron.feedback_prompt_interval is None:
            self.config.neuron.feedback_prompt_interval = 60.0

        bt.logging.info("load_state()")
        self.load_state()

        self.wallet = bt.wallet(config=self.config)
        self.subtensor = bt.subtensor(config=self.config)
        self.dendrite = bt.dendrite(wallet=self.wallet)
        self.metagraph = self.subtensor.metagraph(self.config.netuid)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if isinstance(self.scores, torch.Tensor):
            self.scores = self.scores.cpu().float()

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

    def setup_logging(self):
        bt.logging(config=self.config, logging_dir=self.config.neuron.name)

    async def forward(self):
        """
        Validator forward pass. Consists of:
        - Generating the query
        - Querying the miners
        - Getting the responses
        - Rewarding the miners
        - Updating the scores
        """
        # TODO(developer): Rewrite this function based on your protocol definition.
        return await forward(self)

if __name__ == "__main__":
    with Validator() as validator:
        bt.logging.info("Running validator on subnet %d"%validator.config.netuid)
        last_block = 0
        while True:

            # bt.logging.info("Validator running...", time.time())
            time.sleep(5)
