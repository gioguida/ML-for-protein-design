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

    seq_lengths = {len(seq) for seq in sequences}
    if len(seq_lengths) != 1:
        raise ValueError("All sequences in a batch must have the same length.")

    # Get the Pseudo-Log-Likelihood for the batch
    pll_scores = scorer.pseudo_log_likelihood(sequences, cdr_only=cdr_only, use_grad=False)

    # N is the number of positions scored by PLL.
    if cdr_only:
        N = float(next(iter(seq_lengths)))
    else:
        N = float(scorer.tokenize_sequences([sequences[0]]).shape[1])

    if N <= 0:
        raise ValueError("Number of scored positions must be positive.")
    
    # Calculate Perplexity: exp(-PLL / N)
    perplexity = torch.exp(-pll_scores / N)
    return perplexity
