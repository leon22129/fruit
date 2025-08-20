# Copyright © 2023 Bittensor. All rights reserved.

import torch
import bittensor as bt
from typing import List

from FruitsMaturityNet.protocol import FruitPrediction
from FruitsMaturityNet.model import FRUIT_TYPE_REV_MAP, RIPENESS_REV_MAP


def reward(response: FruitPrediction, true_fruit_type: str, true_ripeness: str) -> float:
    """
    Reward the miner response to the fruit prediction request. This method returns a reward
    value for the miner, which is used to update the miner's score.

    Returns:
    - float: The reward value for the miner.
    """
    if response.fruit_type_prediction is None or response.ripeness_prediction is None:
        return 0.0

    true_fruit_type_id = FRUIT_TYPE_REV_MAP.get(true_fruit_type, -1)
    true_ripeness_id = RIPENESS_REV_MAP.get(true_ripeness, -1)

    if true_fruit_type_id == -1 or true_ripeness_id == -1:
        bt.logging.warning(f"Invalid true labels for reward calculation: {true_fruit_type}, {true_ripeness}")
        return 0.0

    miner_fruit_type_id = FRUIT_TYPE_REV_MAP.get(response.fruit_type_prediction, -1)
    miner_ripeness_id = RIPENESS_REV_MAP.get(response.ripeness_prediction, -1)

    fruit_type_correct = (miner_fruit_type_id == true_fruit_type_id)
    ripeness_correct = (miner_ripeness_id == true_ripeness_id)

    # Simple reward scheme: 0.5 for each correct prediction
    reward_value = 0.0
    if fruit_type_correct:
        reward_value += 0.5
    if ripeness_correct:
        reward_value += 0.5

    return reward_value


def get_rewards(
    self,
    responses,
    true_fruit_type: str,
    true_ripeness: str,
) -> torch.Tensor:
    """
    Returns a tensor of rewards for the given query and responses.

    Args:
    - query_uids (torch.LongTensor): The uids of the miners that were queried.
    - responses (List[FruitPrediction]): The responses from the miners.
    - true_fruit_type (str): The true fruit type label.
    - true_ripeness (str): The true ripeness label.

    Returns:
    - torch.FloatTensor: The rewards for the given query and responses.
    """
    # Get all the reward results by iteratively calling your reward() function.
    return torch.FloatTensor(
        [reward(response, true_fruit_type, true_ripeness) for response in responses]
    ).to(self.device)
