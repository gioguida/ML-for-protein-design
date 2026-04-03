from dataclasses import dataclass
from typing import List, Sequence, Optional
from esme.alphabet import Alphabet, tokenize

LEFT_CONTEXT = (
	"EVQLQESGGGLVQPGESLRLSCVGSGSSFGESTLSYYAVSWVRQAPGKGLEWLSIINAGGGDIDYADSVEGRFTISRDNSKETLYLQMTNLRVEDTGVYYCAK"
)
RIGHT_CONTEXT = (
	"WGQGTMVTVSSASTKGPSVFPLAPSSKSTSGGTAALGCLVKDYFPEPVTVSWNSGALTSGVHTFPAVLQSSGLYSLSSVVTVPSSSLGTQTYICNVNHKPSNTKVDKRVEPKSC"
)


@dataclass
class ModelConfig:
	esm_model_path: str = (
        "/cluster/project/krause/flohmann/mgm/oracle_assets/esm2_8m.safetensors"
    )	
	device: str = "cuda"
	use_context: bool = True # If True, expects sequences with context and extracts positions 104:-115; if False, standard slicing 1:-1


@dataclass
class TrainingConfig:
    """Training runtime configuration."""

    batch_size: int = 32
    num_epochs: int = 50
    lr: float = 1e-5
    num_ensembles: int = 1
    subsample: float = 1.0
    patience: int = 20
    device: str = "cuda"
    max_weight: float = float("inf")
    max_train_mutations: Optional[int] = None


def add_context(cdr: str) -> str:
	"""Add fixed heavy-chain context around a CDR sequence."""
	return LEFT_CONTEXT + cdr + RIGHT_CONTEXT


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
