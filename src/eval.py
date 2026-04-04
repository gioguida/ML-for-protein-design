import torch
from typing import Sequence

if __package__:
    from .model import ESM2PLLScorer
else:  # pragma: no cover
    from model import ESM2PLLScorer


def sequence_perplexity(
    sequences: Sequence[str], 
    scorer: ESM2PLLScorer,
    cdr_only: bool = True
) -> torch.Tensor:
    """
    Computes the perplexity for a batch of sequences based on Pseudo-Log-Likelihood (PLL).
    Returns a tensor of shape [batch_size] containing perplexity scores.
    """
    if len(sequences) == 0:
        raise ValueError("sequences must not be empty")

    # Get the Pseudo-Log-Likelihood for the batch
    pll_scores = scorer.pseudo_log_likelihood(sequences, cdr_only=cdr_only, use_grad=False)
    
    # N is the length of the scored portion. 
    # For ESM2PLLScorer cdr_only, this implies all sequences in the batch have the same CDR length
    N = len(sequences[0])  
    
    # Calculate Perplexity: exp(-PLL / N)
    perplexity = torch.exp(-pll_scores / N)
    return perplexity
