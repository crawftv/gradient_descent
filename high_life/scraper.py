import requests
from bs4 import BeautifulSoup


def recursive_scraper(url):
    response = requests.get(url)
    soup = BeautifulSoup(response.content, 'html.parser')
    links = []
    # Find all the links in the page and extract their href attributes
    for link in soup.find_all('a'):
        print(link)
        link.append("")
        if not link.has_attr('href'):  # Check if there is a URL associated with this link
            continue

            # If the link points to another page, call recursive_scraper function on it
            recursive_scraper(link)


if __name__ == "__main__":
    # Start by providing a URL to your initial webpage:
    initial_url = "https://konfektmagazine.com/"
    recursive_scraper(initial_url)
