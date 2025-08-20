import bittensor as bt
import torch
import time
import os
from datetime import datetime
from collections import deque
from typing import List, Dict, Any, Tuple
from torch.utils.data import DataLoader
import torch.nn.functional as F
import io
import torchvision.transforms as transforms # Added for image conversion

# Import model, protocol, and dataset from the FruitsMaturityNet
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

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if isinstance(self.scores, torch.Tensor):
            self.scores = self.scores.cpu().float()

        # Load dataset for sending requests
        self.dataset_root = os.path.join(os.getcwd(), "dataset")
        self.fruit_dataset = FruitDataset(root_dir=self.dataset_root, train=False) # Use test set for validation
        self.dataloader = DataLoader(self.fruit_dataset, batch_size=1, shuffle=True)
        self.data_iterator = iter(self.dataloader)

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
