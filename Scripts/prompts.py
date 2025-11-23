from importlib import resources
import os
import functools
import random
import inflect

IE = inflect.engine()
ASSETS_PATH = resources.files("Assets")


@functools.cache
def _load_lines(path):
    """
    Load lines from a file. First tries to load from `path` directly, and if that doesn't exist, searches the
    `d3po_pytorch/assets` directory for a file named `path`.
    """
    if not os.path.exists(path):
        newpath = ASSETS_PATH.joinpath(path)
    if not os.path.exists(newpath):
        raise FileNotFoundError(f"Could not find {path} or ddpo_pytorch.assets/{path}")
    path = newpath

    # The strip() method removes any leading, and trailing whitespaces. Leading means at the beginning of the string,
    # trailing means at the end. You can specify which character(s) to remove, if not, any whitespaces will be removed.
    with open(path, "r") as f:
        return [line.strip() for line in f.readlines()]


def from_file(path, low=None, high=None):
    prompts = _load_lines(path)[low:high]
    return random.choice(prompts), {}


def imagenet_all():
    return from_file("imagenet_classes.txt")


def imagenet_animals():
    return from_file("imagenet_classes.txt", 0, 398)


def imagenet_dogs():
    return from_file("imagenet_classes.txt", 151, 269)


def simple_animals():
    return from_file("simple_animals.txt")

def anything_prompt():
    return from_file("anything_prompt.txt")

def unsafe_prompt():
    return from_file("unsafe_prompt.txt")

def balloon_texture():
    return from_file("balloon_texture.txt")

def ambulance_texture():
    return from_file("ambulance_texture.txt")

def cylinder_texture():
    return from_file("cylinder_texture.txt")

def female_head():
    return from_file("female_head.txt")

def male_head():
    return from_file("male_head.txt")

def rabbit_texture():
    return from_file("rabbit_texture.txt")

def napelon_texture():
    return from_file("napelon_texture.txt")

def african_girl_texture():
    return from_file("african_girl_texture.txt")

def cow_texture():
    return from_file("cow_texture.txt")

def dragon_texture():
    return from_file("dragon_texture.txt")