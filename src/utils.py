from esme.alphabet import Alphabet, tokenize

LEFT_CONTEXT = (
	"EVQLQESGGGLVQPGESLRLSCVGSGSSFGESTLSYYAVSWVRQAPGKGLEWLSIINAGGGDIDYADSVEGRFTISRDNSKETLYLQMTNLRVEDTGVYYCAK"
)
RIGHT_CONTEXT = (
	"WGQGTMVTVSSASTKGPSVFPLAPSSKSTSGGTAALGCLVKDYFPEPVTVSWNSGALTSGVHTFPAVLQSSGLYSLSSVVTVPSSSLGTQTYICNVNHKPSNTKVDKRVEPKSC"
)


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
