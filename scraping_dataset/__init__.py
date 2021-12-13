"""Automatic collection of dataset using google image search.

Modules
-------
image_scraper
    It collects links to images on a given topic.
create_dataset
    Downloads images corresponding to the list of subjects.

References
----------
flower_types.json
    A text file containing all the required types of flowers and their
    corresponding translations into Russian.
"""

from .image_scraper import *
from .create_dataset import *
