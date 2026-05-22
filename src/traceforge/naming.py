import random

ADJECTIVES = [
    "brave", "stoic", "amber", "swift", "calm", "bold", "keen",
    "quiet", "sharp", "noble", "clear", "crisp", "firm", "warm",
    "vast", "deep", "light", "dark", "soft", "hard", "bright",
    "cool", "cold", "wise", "true", "pure", "free", "safe",
]

NOUNS = [
    "salmon", "crane", "wolf", "fox", "bear", "hawk", "owl",
    "raven", "tiger", "lion", "whale", "seal", "deer", "elk",
    "eagle", "heron", "finch", "robin", "wren", "swift",
    "cedar", "maple", "birch", "pine", "oak", "ash", "elm",
]


def generate_run_name() -> str:
    return f"{random.choice(ADJECTIVES)}-{random.choice(NOUNS)}"
