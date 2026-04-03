import torch
import torch.nn.functional as F
import numpy as np
from typing import List, Sequence, Tuple, Union

if __package__:
    from .model import ESM2PLLScorer
else:  # pragma: no cover
    from model import ESM2PLLScorer


def _diff_positions(winner: str, loser: str) -> List[int]:
    if len(winner) != len(loser):
        raise ValueError("Winner and loser must have the same sequence length.")
    return [idx for idx, (w_char, l_char) in enumerate(zip(winner, loser)) if w_char != l_char]


def _as_pair_batch(pairs: Union[Tuple[str, str], Sequence[Tuple[str, str]]]) -> List[Tuple[str, str]]:
    if isinstance(pairs, tuple) and len(pairs) == 2 and isinstance(pairs[0], str):
        return [pairs]
    return list(pairs)


def dpo_loss(
    pair: Union[Tuple[str, str], Sequence[Tuple[str, str]]],
    beta: float, 
    scorer: ESM2PLLScorer, 
    reference: ESM2PLLScorer,
    policy_use_grad: bool = True,
) -> torch.Tensor:
    """Compute mean DPO loss over one pair or a batch of pairs."""
    pair_batch = _as_pair_batch(pair)
    losses: List[torch.Tensor] = []

    for winner, loser in pair_batch:
        diff_positions = _diff_positions(winner, loser)
        if len(diff_positions) == 0:
            continue
        
        w_masked_pll = scorer.masked_pseudo_log_likelihood([winner], diff_positions, use_grad=policy_use_grad).squeeze(0)
        l_masked_pll = scorer.masked_pseudo_log_likelihood([loser], diff_positions, use_grad=policy_use_grad).squeeze(0)

        ref_w_masked_pll = reference.masked_pseudo_log_likelihood([winner], diff_positions, use_grad=False).squeeze(0)
        ref_l_masked_pll = reference.masked_pseudo_log_likelihood([loser], diff_positions, use_grad=False).squeeze(0)

        delta_score = w_masked_pll - l_masked_pll
        delta_ref_score = ref_w_masked_pll - ref_l_masked_pll
        losses.append(-F.logsigmoid(beta * (delta_score - delta_ref_score)))

    if not losses:
        raise ValueError("No valid non-identical winner-loser pairs in batch.")

    return torch.stack(losses).mean()

def implicit_reward(
    sequence: str, 
    masked_positions: np.ndarray, 
    beta: float, 
    scorer: ESM2PLLScorer, 
    reference: ESM2PLLScorer
) -> torch.Tensor:
    """Compute the implicit reward for a single masked sequence."""
    # compute masked PLL score for the sequence at the masked positions
    positions = [int(pos) for pos in masked_positions]
    masked_pll = scorer.masked_pseudo_log_likelihood([sequence], positions, use_grad=False).squeeze(0)
    ref_masked_pll = reference.masked_pseudo_log_likelihood([sequence], positions, use_grad=False).squeeze(0)
    # reward only matters in masked positions
    reward = beta * (masked_pll - ref_masked_pll)   
    return reward

def reward_accuracy(
    pair: Tuple[str, str], 
    masked_positions: np.ndarray, 
    beta: float, 
    scorer: ESM2PLLScorer, 
    reference: ESM2PLLScorer
) -> bool:
    """Determine if the reward correctly ranks the winner above the loser."""
    winner, loser = pair
    winner_reward = implicit_reward(winner, masked_positions, beta, scorer, reference)
    loser_reward = implicit_reward(loser, masked_positions, beta, scorer, reference)
    return bool((winner_reward > loser_reward).item())

def reward_margin(
    pair: Tuple[str, str], 
    masked_positions: np.ndarray, 
    beta: float, 
    scorer: ESM2PLLScorer, 
    reference: ESM2PLLScorer
) -> torch.Tensor:
    """Compute the reward margin between winner and loser."""
    winner, loser = pair
    winner_reward = implicit_reward(winner, masked_positions, beta, scorer, reference)
    loser_reward = implicit_reward(loser, masked_positions, beta, scorer, reference)
    margin = winner_reward - loser_reward
    return margin

def implicit_KL_divergence(
    sequence: str, 
    scorer: ESM2PLLScorer, 
    reference: ESM2PLLScorer
) -> torch.Tensor:
    """Compute the implicit KL divergence for a single sequence."""
    masked_pll = scorer.pseudo_log_likelihood([sequence], use_grad=False).squeeze(0)
    ref_masked_pll = reference.pseudo_log_likelihood([sequence], use_grad=False).squeeze(0)
    kl_divergence = (masked_pll - ref_masked_pll)
    return kl_divergence