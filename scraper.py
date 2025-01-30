import logging
from urllib.parse import urljoin
import os
import json
import scrapy
from scrapy import Selector
from scrapy.crawler import CrawlerProcess
from scrapy.http import Response
from tqdm import tqdm
from typing import Optional
from scrapy_selenium import SeleniumRequest
import random
from time import sleep

CRAWLER_NAME = "shakespeare-spider"
START_URLS = ["https://www.litcharts.com/shakescleare/shakespeare-translations"]
BASE_URL = "https://www.litcharts.com"
DATA_DIR = "./data"

# Spider class to scrape data from the website
class ShakespeareSpider(scrapy.Spider):
    name = CRAWLER_NAME
    start_urls = START_URLS
    custom_settings = {
        'BOT_NAME': 'shakespeare_spider',
        'USER_AGENT': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.36',
        'ROBOTSTXT_OBEY': True,
        'SELENIUM_DRIVER_NAME': 'chrome',
        'SELENIUM_DRIVER_EXECUTABLE_PATH': r'C:\\chromedriver-win64\\chromedriver.exe',
        'SELENIUM_DRIVER_ARGUMENTS': ['--headless'],
        'DOWNLOAD_DELAY': 15,
        'RANDOMIZE_DOWNLOAD_DELAY': True,
        'RETRY_TIMES': 5,
        'RETRY_HTTP_CODES': [202, 503, 504],
        'RETRY_DELAY': 10,
        'CONCURRENT_REQUESTS': 5,
        'CONCURRENT_REQUESTS_PER_DOMAIN': 2,

    }

    def start_requests(self):
        for url in self.start_urls:
            yield SeleniumRequest(
                url = url,
                callback = self.parse,
                wait_time = 60
            )

    # Function to parse the main page
    def parse(self, response: Response, **kwargs):
        base_url = BASE_URL
        next_pages = response.css("a.translation.hoverable::attr(href)").extract()
        print(f"Currently loading Shakespeare's works... No. of Books/Links = {len(next_pages)}")

        if response.status == 202:
            self.logger.warning(f"Encountered 202 status for {response.url}, retrying...")

        # TQDM progress bar
        # Iterates through every book found on the page
        loop = tqdm(
            next_pages,
            position = 0,
            leave = True,
            bar_format = "{desc:<50}{percentage:3.0f}%|{bar:20}{r_bar}",
        )

        # Loop to parse the book urls
        for relative_url in loop:
            url = urljoin(base_url, relative_url) # Combines base URL with book URL
            book_name = relative_url.split("/")[-1]
            yield SeleniumRequest(
                url = url,
                callback = self.parse_book_url,
                cb_kwargs = {"book_name": book_name},
                wait_time = 7
            )

    # Function to parse the book urls
    def parse_book_url(self, response: Response, book_name: Optional[str] = None):
        book_name = book_name or 'Unknown'  # Default to 'Unknown' if no book_name passed
        print(f"Parsing book: {book_name}")
        print(f"Parsing book: {book_name}")
        base_url = BASE_URL
        next_pages = response.css("div.table-of-contents a::attr(href)").extract()
        
        if not next_pages:
            return

        # TQDM progress bar
        # Iterates through every chapter found on the table of contents
        loop = tqdm(
            next_pages,
            position = 1,
            leave = True,
            desc = f"Book: {book_name}",
            bar_format = "{desc:<50}{percentage:3.0f}%|{bar:20}{r_bar}",
        )

        # Loop to parse chapters
        for relative_url in loop:
            url = urljoin(base_url, relative_url) # Combines base URL with relative chapter URL
            yield SeleniumRequest(
                url = url,
                callback = self.parse_chapters,
                cb_kwargs = {
                    "book_name": book_name,
                    "chapter_name": relative_url.split("/")[-1],
                },
                wait_time = 6
            )

    def parse_chapters(self, response: Response, book_name: Optional[str] = None, chapter_name: Optional[str] = None):
        dialogs = {"dialogs": []} # Dictionary that stores extracted text in original and translated format
        text_matches = response.css(".comparison-row").getall()

        # Log the full page HTML for debugging
        page_html = response.text
        with open("page_debug.html", "w") as debug_file:
            debug_file.write(page_html)

        # Log to check if we are getting the expected content
        print(f"Found {len(text_matches)} comparison rows on chapter: {chapter_name}")

        if len(text_matches) == 0:
            print(f"No comparison rows found for chapter: {chapter_name}. Check the page structure or selectors.")

        for i, matches in enumerate(text_matches):
            original_content_html = (
                Selector(text = matches).css(".original-content p.speaker-text").get()
            )
            translated_content_html = (
                Selector(text = matches).css(".translated-content p.speaker-text").get()
            )

            # Log the extracted HTML to debug the content
            print(f"Original content HTML (index {i}): {original_content_html}")
            print(f"Translated content HTML (index {i}): {translated_content_html}")

            try:
                # Extract original and translated dialogues
                original_dialogue = "".join(
                    Selector(text=original_content_html).css("span.line-mapping::text").getall()
                )
                translated_dialogue = "".join(
                    Selector(text=translated_content_html).css("span.line-mapping::text").getall()
                )

            except Exception as e:
                print(f"Error extracting text at index {i}: {e}")
                continue

            # Filter only ASCII characters so only standard English characters remain
            # Store each extracted original and translated dialogue is stored in the dictionary
            dialogs["dialogs"].append(
                {
                    "original": "".join(filter(
                        lambda c: 0 <= ord(c) and ord(c) <= 255,
                        original_dialogue
                    )),

                    "translated": "".join(filter(
                        lambda c: 0 <= ord(c) and ord(c) <= 255,
                        translated_dialogue
                    ))
                }
            )

            print(f"Extracted content: {original_dialogue}, {translated_dialogue}")


        # Log the number of dialogues extracted
        print(f"Extracted {len(dialogs['dialogs'])} dialogues from chapter: {chapter_name}")

        _save_dialogs(dialogs, book_name, chapter_name)

def _save_dialogs(dialogs, book_name, chapter_name):
    output_dir = os.path.join(DATA_DIR, book_name)
    os.makedirs(output_dir, exist_ok = True)
    with open(os.path.join(output_dir, f"{chapter_name}.json"), "w") as f:
        json.dump(dialogs, f, indent = 4)

process = CrawlerProcess()
process.crawl(ShakespeareSpider)
process.start()




