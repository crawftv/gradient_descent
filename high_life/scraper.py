import functools
import sqlite3
from urllib.parse import urljoin

import requests
from bs4 import BeautifulSoup

from agent_splitter import nodes_from_html
from settings import DB_NAME


def web_cache_decorator(func):
    @functools.wraps(func)
    def wrapper(url, parent_url, *args, **kwargs):
        # Initialize or open the database and create the table if it doesn't exist
        conn = sqlite3.connect(DB_NAME)
        cursor = conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS scrape_links (
                url TEXT,
                parent_url TEXT,
                scraped BOOL NOT NULL CHECK (scraped IN (0, 1)),
                page_data TEXT
            )
        """)
        cursor.execute("""
        CREATE UNIQUE INDEX IF NOT EXISTS idx_url_parent_url ON scrape_links (url, parent_url)
        """)
        conn.commit()

        # Check if the URL is already in cache
        cursor.execute('SELECT page_data FROM scrape_links WHERE url = ? AND scraped = 1', (url,))
        cached = cursor.fetchone()

        if cached:
            conn.close()
            print("Returning cached data")
            return cached[0]
        else:
            result = func(url, parent_url, *args, **kwargs)

            # Cache the result
            cursor.execute('INSERT OR REPLACE INTO scrape_links (url,parent_url, scraped, page_data) VALUES (?,?,?, ?)',
                           (url, parent_url, 0, result))
            conn.commit()
            conn.close()
            return result

    return wrapper


@web_cache_decorator
def get_page(url, base_url) -> str:
    url = urljoin(base_url, url)
    response = requests.get(url)
    response.raise_for_status()  # Raises an exception for 4XX or 5XX errors
    return response.text


def scrape_website(url, save_links=False):
    """Fetch and parse the content of a website URL, returning article titles."""
    response = get_page("", url)
    # Parse the HTML content of the webpage with BeautifulSoup
    soup = BeautifulSoup(response, 'html.parser')
    if save_links is True:
        links = [i["href"] for i in soup.find_all('a') if i.get("href") and i["href"].startswith("/")]
        with sqlite3.connect(DB_NAME) as conn:
            for i in links:
                conn.execute(
                    'INSERT OR REPLACE INTO scrape_links (url, parent_url, scraped, page_data) VALUES (?,?, ?, ?)',
                    (i, url, 0, ""))
    # Find elements by their tag and class (modify this according to your needs)
    # This example assumes articles are contained in <article> tags


if __name__ == "__main__":
    # Start by providing a URL to your initial webpage:
    # initial_url = "https://www.chloe-cooks.com/recipes"

    # scrape_website(initial_url, save_links=True)
    with sqlite3.connect(DB_NAME) as conn:
        links = conn.execute("SELECT url,parent_url FROM scrape_links WHERE scraped = 0").fetchall()
        for i in links:
            url = urljoin(i[1], i[0])
            nodes_from_html(url)
            conn.execute()
