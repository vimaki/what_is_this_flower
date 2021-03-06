#!/usr/bin/env python3

"""Downloads images corresponding to the list of subjects.

For each subject from the list, a folder is created on the local disk,
and a specified number of suitable images are downloaded into it.

Functions
---------
get_image_urls
    Get a list of URL-links to images of flowers of a given type.
download_images
    Downloads images of the required subject.

References
----------
image_scraper.py
    A module containing a scraper class that collects URL-links to
    images of a given subject that are found in a Google search engine.
flower_types.json
    A text file containing all the required types of flowers and their
    corresponding translations into Russian.
"""

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import List

import requests
from fake_useragent import UserAgent

from . import image_scraper

__all__ = ['get_image_urls', 'download_images']

logging.basicConfig(stream=sys.stderr, level=logging.INFO)
logger = logging.getLogger()


def get_image_urls(flower_type: str, n_images: int = 100) -> List[str]:
    """Get a list of URL-links to images of flowers of a given type.

    Parameters
    ----------
    flower_type : str
        The name of the type of flower is substituted into the template
        used to search for images on the Internet.
    n_images : int, optional
        The number of links to suitable images to collect (default is
        100 links).

    Returns
    -------
    list(str)
        A list that is containing strings that represent UTL-links to
        suitable images.
    """

    # Form a search query
    query = '{} flower photo'.format(flower_type)
    image_urls = image_scraper.main(['--query', query,
                                     '--n_images', str(n_images)])
    return image_urls


def download_images(flower_type: str, n_images: int = 100) -> None:
    """Downloads images of the required subject.

    Creates a folder on the local drive to store images of a given
    subject. Downloads into it the images found on the Internet in the
    specified quantity.

    Parameters
    ----------
    flower_type : str
        The name of the type of flower whose images you want to download.
    n_images : int, optional
        The number of images to download. (default is 100 images).

    Returns
    -------
    None
    """

    # Create a folder corresponding to the given subject
    flower_type_folder = f'dataset/{flower_type}'
    if not os.path.exists(flower_type_folder):
        os.makedirs(flower_type_folder)

    # Get a list of links to images
    image_urls = get_image_urls(flower_type, n_images)
    print(f'Number of links for {flower_type}: {len(image_urls)}')  # delete after check correctness

    # Downloading found images
    for i, image_url in enumerate(image_urls):
        # Make 10 attempts to download the image from the link
        for _ in range(10):
            try:
                response = requests.get(
                    image_url, headers={'User-Agent': UserAgent().chrome})

                if response.ok:
                    # Form the file name
                    image_name = f'{flower_type}_{i:0>3}.jpg'
                    file_path = f'{flower_type_folder}/{image_name}'

                    # Write the content obtained by the link to a file
                    with open(file_path, 'wb') as image_file:
                        image_file.write(response.content)

                break

            except requests.exceptions.RequestException:
                logger.warning('Failed to download the image from the link')
                continue


def main(args) -> None:
    parser = argparse.ArgumentParser(description='Downloads a dataset consisting of images')
    parser.add_argument('-n', '--n_images', type=int, default=100,
                        help='The number of images to download')
    args = parser.parse_args(args)

    with open('flower_types.json', 'r') as file_flower_types:
        flower_types = json.load(file_flower_types)
        for flower_type in flower_types:
            # If a folder with this type of flower exists,
            # you don't need to download it again
            if not Path(f'dataset/{flower_type}').exists():
                download_images(flower_type, args.n_images)


if __name__ == '__main__':
    main(sys.argv[1:])
