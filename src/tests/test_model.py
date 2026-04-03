"""Probe ESM2 token logits and masked-position true-token log-probabilities."""

import torch

from esme import ESM2
from esme.alphabet import Alphabet, tokenize

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
ESM_MODEL_PATH = "/cluster/project/krause/flohmann/mgm/oracle_assets/esm2_8m.safetensors"


def add_context(cdr: str):
    left_context = "EVQLQESGGGLVQPGESLRLSCVGSGSSFGESTLSYYAVSWVRQAPGKGLEWLSIINAGGGDIDYADSVEGRFTISRDNSKETLYLQMTNLRVEDTGVYYCAK"
    right_context = "WGQGTMVTVSSASTKGPSVFPLAPSSKSTSGGTAALGCLVKDYFPEPVTVSWNSGALTSGVHTFPAVLQSSGLYSLSSVVTVPSSSLGTQTYICNVNHKPSNTKVDKRVEPKSC"

    return left_context + cdr + right_context


def load_esm():
    """Load the pretrained ESM2 model on the selected device."""
    esm = ESM2.from_pretrained(ESM_MODEL_PATH, device=DEVICE)
    esm.eval()
    return esm


def get_mask_token_idx(alphabet: Alphabet) -> int:
    """Resolve mask token index across possible Alphabet API variants."""
    for attr in ("mask_idx", "mask_index", "mask_token_id"):
        if hasattr(alphabet, attr):
            return int(getattr(alphabet, attr))

    for attr in ("tok_to_idx", "token_to_idx", "stoi"):
        mapping = getattr(alphabet, attr, None)
        if isinstance(mapping, dict) and "<mask>" in mapping:
            return int(mapping["<mask>"])

    raise AttributeError("Could not find mask token index in Alphabet.")


def main():
    esm = load_esm()
    alphabet = Alphabet()
    mask_token_idx = get_mask_token_idx(alphabet)

    # Example CDR sequences (without context)
    cdr_sequences = [
        "HMSMQQVVSAGWERADLVGDAFDV",
        "AASMQQVRSAGWERADLVGDAFEV",
        "ACSMQQVVSAGWSRADLVGDDFDV",
    ]

    # Use unmasked sequences first
    sequences_with_context = [add_context(cdr) for cdr in cdr_sequences]

    # Tokenize sequences
    tokens = tokenize(sequences_with_context, alphabet=alphabet).to(DEVICE)

    # Probe direct model forward output to verify token-logit shape.
    with torch.no_grad():
        output = esm(tokens)

    print("Output Object type :", type(output))

    if isinstance(output, torch.Tensor):
        print("Output shape :", output.shape)
    elif isinstance(output, dict):
        for key, value in output.items():
            shape = value.shape if isinstance(value, torch.Tensor) else "N/A"
            print(f"Key: {key}, Value type: {type(value)}, Value shape: {shape}")
    else:
        print("Unexpected output type from esm(tokens).")

    # Choose one token position in the CDR region:
    # CDR starts at absolute index 104 after adding context.
    cdr_start_idx = 104
    pos_in_cdr = 7
    mask_pos = cdr_start_idx + pos_in_cdr

    # Mask a single position for all sequences and score true-token log-probability there.
    masked_tokens = tokens.clone()
    true_token_ids = tokens[:, mask_pos].clone()
    masked_tokens[:, mask_pos] = mask_token_idx

    with torch.no_grad():
        masked_output = esm(masked_tokens)

    if not isinstance(masked_output, torch.Tensor):
        raise TypeError(
            "Expected tensor logits from esm(masked_tokens), got "
            f"{type(masked_output)} instead."
        )

    log_probs_at_pos = torch.log_softmax(masked_output[:, mask_pos, :], dim=-1)
    true_log_probs = log_probs_at_pos.gather(1, true_token_ids.unsqueeze(1)).squeeze(1)

    print("Mask token idx:", mask_token_idx)
    print("Mask position (absolute token index):", mask_pos)
    print("True token ids at masked position:", true_token_ids)
    print("True log-probs at masked position:", true_log_probs)
    print("All finite:", torch.isfinite(true_log_probs).all().item())

if __name__ == "__main__":
    main()

