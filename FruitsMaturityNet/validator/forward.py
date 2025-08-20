import random
import uuid
import bittensor as bt
import torch
from typing import List
from FruitsMaturityNet.protocol import FruitPrediction, FeedbackSynapse
from FruitsMaturityNet.validator.reward import get_rewards

# keep a generator or list of image paths globally in the validator
if not hasattr(bt, "_test_image_iterator"):
    import os
    test_root = "dataset/test"
    all_image_paths = []

    for class_dir in os.listdir(test_root):
        full_class_path = os.path.join(test_root, class_dir)
        if not os.path.isdir(full_class_path):
            continue
        for file in os.listdir(full_class_path):
            full_path = os.path.join(full_class_path, file)
            all_image_paths.append((full_path, class_dir))

    # Shuffle for randomness
    random.shuffle(all_image_paths)
    bt._test_image_iterator = iter(all_image_paths)

def check_uid_availability(
    metagraph: "bt.metagraph.Metagraph", uid: int, vpermit_tao_limit: int
) -> bool:
    """Check if uid is available. The UID should be available if it is serving and has less than vpermit_tao_limit stake
    Args:
        metagraph (:obj: bt.metagraph.Metagraph): Metagraph object
        uid (int): uid to be checked
        vpermit_tao_limit (int): Validator permit tao limit
    Returns:
        bool: True if uid is available, False otherwise
    """
    # TODO: Check if is_serving is still relevant
    # Filter non serving axons.
    if not metagraph.axons[uid].is_serving:
        return False
    # Filter validator permit > 1024 stake.
    if metagraph.validator_permit[uid]:
        if metagraph.S[uid] > vpermit_tao_limit:
            return False
    # Available otherwise.
    return True

def get_miner_uids(
    self, exclude: List[int] = None
) -> torch.LongTensor:
    """Returns k available random uids from the metagraph.
    Args:
        k (int): Number of uids to return.
        exclude (List[int]): List of uids to exclude from the random sampling.
    Returns:
        uids (torch.LongTensor): Randomly sampled available uids.
    Notes:
        If `k` is larger than the number of available `uids`, set `k` to the number of available `uids`.
    """
    candidate_uids = []
    avail_uids = []

    for uid in range(self.metagraph.n.item()):
        uid_is_available = check_uid_availability(
            self.metagraph, uid, self.config.neuron.vpermit_tao_limit
        )
        uid_is_not_excluded = exclude is None or uid not in exclude

        if uid_is_available:
            avail_uids.append(uid)
            if uid_is_not_excluded:
                candidate_uids.append(uid)

    uids = torch.tensor(candidate_uids)
    return uids

async def forward(self):
    try:
        # Get next image
        img_path, class_dir = next(bt._test_image_iterator)
    except StopIteration:
        bt.logging.info("No more test images left!")
        return

    # Extract labels from directory name
    raw_label = class_dir.lower()
    state = "fresh" if "Fresh" in raw_label else "Rotten"
    if "apple" in raw_label:
        fruit = "Apple"
    elif "banana" in raw_label:
        fruit = "Banana"
    elif "orange" in raw_label:
        fruit = "Orange"
    else:
        fruit = "unknown"

    labels = [state, fruit]
    bt.logging.info(f"Sending image {img_path} with labels {labels}")

    try:
        # Load image and encode
        img_bytes = open(img_path, "rb").read()
        synapse = FruitPrediction(
            request_id=str(uuid.uuid4())
        )
        synapse.encode_image(img_bytes)

        miner_uids = get_miner_uids(self)

        bt.logging.info(f"valid miners {miner_uids}")

        # Send through dendrite
        responses = await self.dendrite(
            axons=[self.metagraph.axons[uid] for uid in miner_uids],
            synapse=synapse
        )

        responses_objs: List[FruitPrediction] = [
            FruitPrediction.from_dict(r) if isinstance(r, dict) else r
            for r in responses
        ]

        bt.logging.info(f"Received responses for {img_path}: {responses}")

        # Send feedback
        feedback_synapse = FeedbackSynapse(
            request_id=synapse.request_id,
            # true_fruit_type=fruit,
            # true_ripeness=state
        )

        await self.dendrite(
            axons=[self.metagraph.axons[uid] for uid in miner_uids],
            synapse=feedback_synapse
        )

        rewards = get_rewards(self, responses_objs, fruit, state)

        # Ensure self.scores is CPU NumPy array
        if isinstance(self.scores, torch.Tensor):
            self.scores = self.scores.detach().cpu().numpy()

        # Convert rewards to CPU NumPy array
        rewards_np = rewards.detach().cpu().numpy()

        # Convert miner UIDs to CPU NumPy array
        miner_uids_np = torch.LongTensor(miner_uids).cpu().numpy()

        # Update scores
        self.update_scores(rewards_np, miner_uids_np)

    except Exception as e:
        bt.logging.error(f"Error forwarding {img_path}: {e}")

