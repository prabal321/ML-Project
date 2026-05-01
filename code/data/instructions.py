"""
instructions.py - synthetic natural-language instruction generator.

Given a PointNav episode (start position, goal position, scene id),
produce a plausible navigation instruction. Templates are randomized
so the same episode can be phrased multiple ways.

Coordinate convention (Habitat):
    +x  -> right
    +y  -> up (vertical, ignored for navigation)
    +z  -> forward (relative to scene's canonical frame)
We compute a horizontal vector (dx, dz) from start to goal and
classify it into a coarse direction.
"""
from __future__ import annotations
import math
import random
from dataclasses import dataclass
from typing import List, Tuple


# Coarse direction bins (counterclockwise from +x axis)
DIRECTIONS = [
    ("east",       0.0),
    ("northeast",  math.pi / 4),
    ("north",      math.pi / 2),
    ("northwest",  3 * math.pi / 4),
    ("west",       math.pi),
    ("southwest",  -3 * math.pi / 4),
    ("south",      -math.pi / 2),
    ("southeast",  -math.pi / 4),
]

# Friendly scene names
SCENE_NAMES = {
    "apartment_1":       "the apartment",
    "skokloster-castle": "the castle",
    "van-gogh-room":     "the room",
}

# Templates (each takes scene/direction/distance keyword args)
TEMPLATES = [
    "Walk {direction_phrase} for about {distance:.0f} meters in {scene}.",
    "Head {direction} and proceed approximately {distance:.0f} meters.",
    "In {scene}, go {direction} for around {distance:.0f} meters.",
    "Navigate {direction_phrase} until you have moved roughly {distance:.0f} meters.",
    "Move {direction} about {distance:.0f} meters to reach the goal.",
    "Go {direction_phrase} approximately {distance:.0f} meters.",
]

# Direction-phrase variants (optional flavor)
DIRECTION_PHRASES = {
    "east":      ["east",       "to your right",      "right"],
    "northeast": ["northeast",  "forward and right",  "ahead-right"],
    "north":     ["north",      "forward",            "ahead"],
    "northwest": ["northwest",  "forward and left",   "ahead-left"],
    "west":      ["west",       "to your left",       "left"],
    "southwest": ["southwest",  "back and left",      "behind-left"],
    "south":     ["south",      "backward",           "behind you"],
    "southeast": ["southeast",  "back and right",     "behind-right"],
}


@dataclass
class Episode:
    """Minimal episode info needed to generate an instruction."""
    scene_id:        str          # e.g. "apartment_1"
    start_position:  Tuple[float, float, float]
    goal_position:   Tuple[float, float, float]
    episode_id:      str = ""


def coarse_direction(start_xz: Tuple[float, float],
                     goal_xz:  Tuple[float, float]) -> str:
    """Return a coarse 8-way compass direction from start to goal."""
    dx = goal_xz[0] - start_xz[0]
    dz = goal_xz[1] - start_xz[1]
    angle = math.atan2(dz, dx)
    # find closest direction bin
    best = min(DIRECTIONS, key=lambda d: abs(_angle_diff(angle, d[1])))
    return best[0]


def _angle_diff(a: float, b: float) -> float:
    """Smallest signed angle difference, in radians."""
    d = (a - b + math.pi) % (2 * math.pi) - math.pi
    return d


def horizontal_distance(start_pos, goal_pos) -> float:
    dx = goal_pos[0] - start_pos[0]
    dz = goal_pos[2] - start_pos[2]
    return math.hypot(dx, dz)


def generate_instruction(episode: Episode, rng: random.Random | None = None) -> str:
    """Generate one synthetic instruction for an episode."""
    if rng is None:
        rng = random
    sx, _, sz = episode.start_position
    gx, _, gz = episode.goal_position
    direction = coarse_direction((sx, sz), (gx, gz))
    direction_phrase = rng.choice(DIRECTION_PHRASES[direction])
    distance = horizontal_distance(episode.start_position, episode.goal_position)
    scene = SCENE_NAMES.get(episode.scene_id, "the scene")

    template = rng.choice(TEMPLATES)
    return template.format(
        direction=direction,
        direction_phrase=direction_phrase,
        distance=distance,
        scene=scene,
    )


def generate_instructions(episode: Episode, n: int, seed: int = 42) -> List[str]:
    """Generate n diverse instructions for the same episode."""
    rng = random.Random(seed)
    return [generate_instruction(episode, rng) for _ in range(n)]


if __name__ == "__main__":
    # Smoke test on a few real episodes (positions taken from your dataset
    # inspection earlier).
    examples = [
        Episode("skokloster-castle",
                (-3.56, 0.23, 19.48), (-0.62, 0.17, 11.53), "3"),
        Episode("skokloster-castle",
                (3.25, 0.16, 11.27), (-6.97, 0.04, 4.75), "23"),
        Episode("van-gogh-room",
                (1.37, 0.18, 0.11), (3.92, 0.18, 0.25), "3"),
        Episode("van-gogh-room",
                (3.05, 0.18, -1.35), (4.17, 0.18, 0.66), "7"),
    ]

    print("Synthetic instruction generator - smoke test")
    print("=" * 70)
    for ep in examples:
        d = horizontal_distance(ep.start_position, ep.goal_position)
        dir_ = coarse_direction((ep.start_position[0], ep.start_position[2]),
                                (ep.goal_position[0],  ep.goal_position[2]))
        print(f"\nScene: {ep.scene_id}, episode {ep.episode_id}")
        print(f"  start={ep.start_position}, goal={ep.goal_position}")
        print(f"  horizontal distance = {d:.2f} m, direction = {dir_}")
        print("  Sample instructions (3 different phrasings):")
        samples = generate_instructions(ep, n=3, seed=hash(ep.episode_id) & 0xFFFF)
        for i, s in enumerate(samples):
            print(f"    {i+1}. {s}")

    print("\n" + "=" * 70)
    print("instructions.py smoke test complete.")
