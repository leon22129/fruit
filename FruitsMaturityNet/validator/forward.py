# Copyright © 2023 Bittensor. All rights reserved.

from FruitsMaturityNet.validator.reward import get_rewards
import bittensor as bt
import torch
import uuid
import random
import io
import torchvision.transforms as transforms
from typing import List, Tuple, Dict, Any
from torch.utils.data import DataLoader

from FruitsMaturityNet.protocol import FruitPrediction
from FruitsMaturityNet.model import FRUIT_TYPE_MAP, RIPENESS_MAP
from FruitsMaturityNet.dataset import FruitDataset


def get_random_image_data(dataloader, data_iterator) -> Tuple[bytes, str, str, Any]:
    """Gets a random image and its true labels from the dataset."""
    try:
        sample = next(data_iterator)
    except StopIteration:
        # Reset iterator if all samples are consumed
        data_iterator = iter(dataloader)
        sample = next(data_iterator)

    image_tensor = sample["image"].squeeze(0)  # Remove batch dimension
    fruit_type_label_id = sample["fruit_type_label"].item()
    ripeness_label_id = sample["ripeness_label"].item()

    # Convert tensor to PIL Image and then to bytes
    to_pil = transforms.ToPILImage()
    image_pil = to_pil(image_tensor)
    img_byte_arr = io.BytesIO()
    image_pil.save(img_byte_arr, format='JPEG')
    image_bytes = img_byte_arr.getvalue()

    true_fruit_type = FRUIT_TYPE_MAP.get(fruit_type_label_id, "Unknown")
    true_ripeness = RIPENESS_MAP.get(ripeness_label_id, "Normal")

    return image_bytes, true_fruit_type, true_ripeness, data_iterator


def forward(self):
    """
    The forward function is called by the validator every time step.
    
    It is responsible for querying the network and scoring the responses.
    
    Args:
        self (:obj:`bittensor.neuron.Neuron`): The neuron object which contains all the necessary state for the validator.
    
    """
    # Get all active miners
    miner_uids = self.metagraph.uids.to(self.device)
    
    # Filter for active miners
    active_miners = []
    for uid in miner_uids:
        if self.metagraph.axons[uid].is_serving:
            active_miners.append(uid.item())

    if len(active_miners) < self.config.neuron.min_axon_size:
        bt.logging.warning(f"Not enough active miners ({len(active_miners)}/{self.config.neuron.min_axon_size}). Waiting...")
        return

    # Select a random subset of miners to query
    query_uids = torch.tensor(random.sample(active_miners, min(len(active_miners), 10)))
    query_axons = [self.metagraph.axons[uid] for uid in query_uids]

    # Prepare the request
    request_id = str(uuid.uuid4())
    image_bytes, true_fruit_type, true_ripeness, self.data_iterator = get_random_image_data(
        self.dataloader, self.data_iterator
    )
    
    synapse = FruitPrediction(request_id=request_id)
    synapse.encode_image(image_bytes)

    bt.logging.info(f"Sending request {request_id} to {len(query_axons)} miners. Original labels: {true_fruit_type}, {true_ripeness}")

    # Send the request to miners
    responses: List[FruitPrediction] = self.dendrite.query(
        query_axons,
        synapse,
        timeout=12,
        deserialize=False,
    )

    # Process responses and collect valid ones
    miner_responses_info = {}
    valid_responses = []
    valid_uids = []
    
    for i, response in enumerate(responses):
        if response.fruit_type_prediction is not None and response.ripeness_prediction is not None:
            valid_responses.append(response)
            valid_uids.append(query_uids[i])
            miner_responses_info[query_uids[i].item()] = response.deserialize()
        else:
            bt.logging.warning(f"Miner {query_uids[i].item()} returned invalid response")

    if not valid_responses:
        bt.logging.warning("No valid responses received from miners.")
        return

    # Log the request for manual feedback
    self.log_feedback_request(request_id, image_bytes, true_fruit_type, true_ripeness, miner_responses_info)

    # Calculate rewards using the reward function
    rewards = get_rewards(
        self,
        query_uids=torch.LongTensor(valid_uids),
        responses=valid_responses,
        true_fruit_type=true_fruit_type,
        true_ripeness=true_ripeness,
    )

    bt.logging.info(f"Scored responses: {rewards}")
    
    # Update the scores based on the rewards
    self.update_scores(rewards, torch.tensor(valid_uids))
