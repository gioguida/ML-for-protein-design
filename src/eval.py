import math
import torch
from typing import Dict, List, Sequence

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


def corpus_perplexity(
    sequences: Sequence[str],
    scorer: ESM2PLLScorer,
    cdr_only: bool = True,
) -> float:
    """Compute corpus-level perplexity as exp(total_nll / total_scored_tokens).

    Unlike mean per-sequence perplexity, this matches the common MLM-style
    evaluation where loss is averaged before exponentiation.
    """
    if len(sequences) == 0:
        raise ValueError("sequences must not be empty")

    total_pll = 0.0
    total_scored_tokens = 0

    if cdr_only:
        by_len: Dict[int, List[str]] = {}
        for seq in sequences:
            by_len.setdefault(len(seq), []).append(seq)

        for cdr_len, seq_group in by_len.items():
            if cdr_len <= 0:
                continue
            pll_scores = scorer.pseudo_log_likelihood(seq_group, cdr_only=True, use_grad=False)
            total_pll += float(pll_scores.sum().item())
            total_scored_tokens += int(cdr_len) * len(seq_group)
    else:
        seq_lengths = {len(seq) for seq in sequences}
        if len(seq_lengths) != 1:
            raise ValueError("All sequences must have the same length when cdr_only=False.")
        pll_scores = scorer.pseudo_log_likelihood(sequences, cdr_only=False, use_grad=False)
        tokens_per_sequence = int(scorer.tokenize_sequences([sequences[0]]).shape[1])
        total_pll += float(pll_scores.sum().item())
        total_scored_tokens += tokens_per_sequence * len(sequences)

    if total_scored_tokens <= 0:
        raise ValueError("No valid tokens were evaluated for perplexity.")

    avg_nll = -total_pll / float(total_scored_tokens)
    return float(math.exp(avg_nll))
