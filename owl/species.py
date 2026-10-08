"""The species list the classifier chooses between (see species.txt)."""

from dataclasses import dataclass
from pathlib import Path

CATEGORIES = ("mammal", "bird", "reptile", "amphibian", "pet")


@dataclass(frozen=True)
class Species:
    common: str
    scientific: str
    category: str


def load(path: Path) -> list[Species]:
    species: list[Species] = []
    seen: set[str] = set()
    for number, line in enumerate(path.read_text().splitlines(), 1):
        line = line.split("#", 1)[0].strip()
        if not line:
            continue
        parts = [part.strip() for part in line.split("|")]
        if len(parts) != 3 or not all(parts) or parts[0] not in CATEGORIES:
            raise ValueError(
                f"{path}:{number}: expected 'category | common name | scientific name' "
                f"with a category of {', '.join(CATEGORIES)}"
            )
        category, common, scientific = parts
        if common.lower() in seen:
            raise ValueError(f"{path}:{number}: {common!r} is listed twice")
        seen.add(common.lower())
        species.append(Species(common.lower(), scientific, category))
    if not species:
        raise ValueError(f"{path} lists no species")
    return species
