import requests
from readabilipy import simple_json_from_html_string

monocle_minute = ["https://monocle.com/minute/2024/02/15/", ]

req = requests.get(monocle_minute[0])
article = simple_json_from_html_string(req.text, use_readability=True)
article["content"]

if __name__ == "__main__":
    article = simple_json_from_html_string(req.text, use_readability=True)
