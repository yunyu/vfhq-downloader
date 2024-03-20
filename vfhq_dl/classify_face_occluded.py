from openai import OpenAI, OpenAIError
from dotenv import load_dotenv
from PIL import Image
import os
from io import BytesIO
import base64
import diskcache
import json
import hashlib
import backoff

load_dotenv()

client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
)
result_cache = diskcache.Cache(".face_occluded_cache")


def string_to_sha256(input_string):
    # Encode the string to bytes
    input_bytes = input_string.encode("utf-8")

    # Create a SHA256 hash object
    sha256_hash = hashlib.sha256()

    # Update the hash object with the bytes-like object (input_bytes)
    sha256_hash.update(input_bytes)

    # Get the hexadecimal representation of the digest
    hex_digest = sha256_hash.hexdigest()

    return hex_digest

@backoff.on_exception(backoff.expo, OpenAIError, max_time=30)
def _infer_from_openai(img_base64_url):
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": "Is there an object directly covering the mouth or jawline? Ignore objects that are merely positioned closely/only cover the jaw or neck - they must actually overlap. Answer YES or NO wrapped in <answer></answer> tags, explaining your reasoning.",
                },
                {
                    "type": "image_url",
                    "image_url": {"url": img_base64_url, "detail": "low"},
                },
            ],
        }
    ]
    # calculate sha256 hash of messages as json string
    message_hash = string_to_sha256(json.dumps(messages))

    # check if the result is in the cache
    if message_hash in result_cache:
        return result_cache[message_hash]

    response = client.chat.completions.create(
        model="gpt-4-vision-preview",
        messages=messages,
        max_tokens=32,
        temperature=0,
    )
    choice = response.choices[0].message.content
    if "<answer>yes" in choice.lower():
        result = True
        result_cache[message_hash] = result
    elif "<answer>no" in choice.lower():
        result = False
        result_cache[message_hash] = result
    else:
        print(f"Unexpected response: {choice}")
        result = False

    return result


def classify_face_occluded(image: Image.Image):
    # Save the image to a BytesIO object as JPEG
    buffered = BytesIO()
    image.save(buffered, format="JPEG")

    # Get the JPEG version of the image as a base64 url
    img_str = base64.b64encode(buffered.getvalue()).decode()

    # Format it as a base64 URL
    img_base64_url = f"data:image/jpeg;base64,{img_str}"
    # print(len(img_base64_url))

    # pass into openai
    try:
        result = _infer_from_openai(img_base64_url)
        return result
    except OpenAIError as e:
        print(f"Error: {e}")
        return False # leave video in if error