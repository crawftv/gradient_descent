import sqlite3

from agent_splitter import get_nodes

if __name__ == "__main__":
    conn = sqlite3.connect("scraping.db")
    cursor = conn.cursor()

    # Query for all URLs where complete is False
    cursor.execute('SELECT url FROM urls WHERE complete = ?', (False,))
    incomplete_urls = cursor.fetchall()

    for (url,) in incomplete_urls:
        # Assuming get_nodes(url) is defined elsewhere
        get_nodes(url)

        # Update the URL as complete
        cursor.execute('UPDATE urls SET complete = ? WHERE url = ?', (True, url))
        conn.commit()

    conn.close()
