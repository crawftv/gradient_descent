import base64

import requests

# Define the image URL variable
image_url = "https://scontent.cdninstagram.com/v/t39.30808-6/429587290_18425245873036331_9006932432805414794_n.jpg?stp=dst-jpg_e35_p1080x1080_sh0.08&_nc_ht=scontent.cdninstagram.com&_nc_cat=111&_nc_ohc=kNDdnSZJTRgAX9mpPq_&edm=APs17CUAAAAA&ccb=7-5&oh=00_AfDaTro-NfUHSy684WWtuNBeIGMmC8RGo2gdHW5biCUe-A&oe=65F3306A&_nc_sid=10d13b"  # Replace with your actual image URL


# Send a GET request to download the image content


def extract_text_from_photo(image_url):
    response = requests.get(image_url, stream=True)

    # Check for successful download
    if response.status_code == 200:
        # Read the image content in binary mode
        image_data = response.content

        # Encode the image content to base64
        encoded_image = base64.b64encode(image_data).decode("utf-8")

        # Prepare the request body with the encoded image
        request_body = {
            "model": "llava:34b",
            "prompt": "You are a robot with eagle eyes. Extract text from this picture. Do not add any extra info",
            "stream": False,
            "images": [encoded_image]
        }

        # Send the POST request to the API
        response = requests.post("http://localhost:11434/api/generate", json=request_body)

        # Handle the response from the API
        print(response.text)
    else:
        print(f"Error downloading image: {response.status_code}")


if __name__ == "__main__":
    extract_text_from_photo(image_url)
