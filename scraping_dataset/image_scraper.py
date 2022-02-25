#!/usr/bin/env python3

"""It collects links to images on a given topic.

This script finds images in the Google search engine corresponding
to a given topic and collects URL-links to them.

Classes
-------
Scraper
    A web-scraper that collects links to relevant images.

Requirements
------------
ChromeDriver must be installed on your machine.
"""

import argparse
import logging
import sys
import urllib.parse
from typing import List

from retry import retry
from selenium import webdriver
from selenium.common import exceptions as sel_e
from selenium.webdriver.remote.webdriver import WebDriver

__all__ = ['Scraper']

logging.basicConfig(stream=sys.stderr, level=logging.INFO)
logger = logging.getLogger()
retry_logger = None


class Scraper:
    """A web-scraper that collects links to relevant images.

    Using the transmitted query, it searches for relevant images
    in the Google search engine. Collects a specified number of
    URL-links to found images.

    Methods
    -------
    __init__
        Inits Scraper.
    scroll_down_the_page
        Scroll to the bottom of the web page.
    get_thumbnails
        Finds a piece of HTML that is responsible for a thumbnails.
    retry_click
        Performs a click on an element with the possibility of repetition.
    get_image_src
        Gets URL-links out of suitable pieces of HTML.
    get_url_images
        Get a list of URL-links to suitable images.
    """

    CSS_THUMBNAIL = 'img.Q4LuWd'
    CSS_LARGE = 'img.n3VNCb'
    CSS_LOAD_MORE = '.mye4qd'
    selenium_exceptions = (sel_e.ElementClickInterceptedException,
                           sel_e.ElementNotInteractableException,
                           sel_e.StaleElementReferenceException)

    def __init__(self, web_driver: WebDriver, query: str, url_safe: str,
                 url_options: str, n_images: int) -> None:
        """Inits Scraper.

        Parameters
        ----------
        web_driver : WebDriver
            A web-driver instance that performs automated browser control.
        query : str
            The text of the query, which will be written in the Google
            search bar.
        url_safe: str
            Safe search mode.
        url_options: str
            Search options.
        n_images:
            The number of links to suitable images to collect.

        Returns
        -------
        None
        """

        self.search_url = \
            'https://www.google.com/search?safe={safe}&site=&tbm=isch&source=hp&q={q}&oq={q}&gs_l=img&tbs={opt}'.format(
                q=urllib.parse.quote(query),
                safe=url_safe,
                opt=url_options
            )
        self.web_driver = web_driver
        self.n_images = n_images

    def scroll_down_the_page(self) -> None:
        """Scroll to the bottom of the web page."""
        self.web_driver.execute_script(
            'window.scrollTo(0, document.body.scrollHeight);')

    @retry(exceptions=KeyError, tries=6, delay=0.1, backoff=2, logger=retry_logger)
    def get_thumbnails(self, min_amount: int = 0):
        """Finds a piece of HTML that is responsible for a thumbnails."""
        self.web_driver.execute_script(
            "document.querySelector('{}').click();".format(self.CSS_LOAD_MORE))
        thumbnails = self.web_driver.find_elements_by_css_selector(
            self.CSS_THUMBNAIL)
        if len(thumbnails) < min_amount:
            raise KeyError('No more thumbnails')
        return thumbnails

    @retry(exceptions=selenium_exceptions, tries=6, delay=0.1, backoff=2, logger=retry_logger)
    def retry_click(self, element):
        """Performs a click on an element with the possibility of repetition."""
        element.click()

    @retry(exceptions=KeyError, tries=6, delay=0.1, backoff=2, logger=retry_logger)
    def get_image_src(self):
        """Gets URL-links out of suitable pieces of HTML."""
        actual_images = self.web_driver.find_elements_by_css_selector(self.CSS_LARGE)
        sources = []
        for image in actual_images:
            src = image.get_attribute('src')
            if src.startswith('http') and \
                    not src.startswith('https://encrypted-tbn0.gstatic.com/'):
                sources.append(src)
        if not len(sources):
            raise KeyError('There are no large images')
        return sources

    def get_url_images(self) -> List[str]:
        """Get a list of URL-links to suitable images.

        Returns
        -------
        list(str)
            A list that is containing strings that represent UTL-links
            to suitable images.
        """

        self.web_driver.get(self.search_url)
        thumbnails = []
        while len(thumbnails) < self.n_images:
            self.scroll_down_the_page()
            try:
                thumbnails = self.get_thumbnails(min_amount=len(thumbnails))
            except KeyError:
                logger.warning('Cannot load enough thumbnails')
                break
        url_images = []
        for thumbnail in thumbnails:
            try:
                self.retry_click(thumbnail)
            except self.selenium_exceptions:
                logger.warning('Failed to click on the thumbnail')
                continue
            sources = []
            try:
                sources = self.get_image_src()
            except KeyError:
                pass
            if not sources:
                thumbnail_src = thumbnail.get_attribute('src')
                if not thumbnail_src.startswith('data'):
                    logger.warning('src for image not found, using thumbnail')
                    sources = [thumbnail_src]
                else:
                    logger.warning('src for image not found, thumbnail is a data URL')
            for src in sources:
                if src not in url_images:
                    url_images.append(src)
            if len(url_images) > self.n_images:
                break
        return url_images


def main(args):
    parser = argparse.ArgumentParser(description='Downloads images from Google Image Search')
    parser.add_argument('--query', type=str,
                        help='Content for the search string')
    parser.add_argument('--n_images', type=int, default=100,
                        help='The number of images to download')
    parser.add_argument('--safe', type=str, default='off',
                        help='safe search [off|active|images]')
    parser.add_argument('--options', type=str, default='',
                        help='search options, e.g. isz:lt,itp:photo,ift:jpg')
    args = parser.parse_args(args)

    with webdriver.Chrome() as web_driver:
        scraper = Scraper(web_driver, args.query, args.safe,
                          args.options, args.n_images)
        url_images = scraper.get_url_images()

    return url_images


if __name__ == '__main__':
    main(sys.argv[1:])
