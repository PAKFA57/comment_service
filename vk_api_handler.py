import requests
import re
from urllib.parse import urlparse, parse_qs

# Замените на свой VK токен и версию API
VK_TOKEN = "---"
VK_API_VERSION = "5.199"

def extract_ids_from_vk_url(vk_url):
    """
    Извлекает owner_id и post_id из ссылки на пост, фото или видео.
    Примеры:
    - https://vk.com/wall-123456_789
    - https://vk.com/video123456_789
    """
    patterns = [
        r"wall(?P<owner_id>-?\d+)_(?P<post_id>\d+)",
        r"photo(?P<owner_id>-?\d+)_(?P<post_id>\d+)",
        r"video(?P<owner_id>-?\d+)_(?P<post_id>\d+)"
    ]
    
    for pattern in patterns:
        match = re.search(pattern, vk_url)
        if match:
            return int(match.group("owner_id")), int(match.group("post_id"))
    return None, None

def fetch_vk_comments(vk_url, max_comments=100):
    """
    Получает комментарии к посту/видео/фото по ссылке из ВКонтакте.
    """
    owner_id, post_id = extract_ids_from_vk_url(vk_url)
    if owner_id is None or post_id is None:
        raise ValueError("Не удалось извлечь owner_id и post_id из ссылки")

    comments = []
    count = 100
    offset = 0

    while len(comments) < max_comments:
        response = requests.get("https://api.vk.com/method/wall.getComments", params={
            "owner_id": owner_id,
            "post_id": post_id,
            "access_token": VK_TOKEN,
            "v": VK_API_VERSION,
            "count": min(count, max_comments - len(comments)),
            "offset": offset,
            "extended": 0,
            "sort": "asc",
            "preview_length": 0
        })
        data = response.json()

        if "error" in data:
            raise Exception(f"VK API error: {data['error'].get('error_msg')}")

        items = data.get("response", {}).get("items", [])
        if not items:
            break

        for item in items:
            text = item.get("text", "").strip()
            if text:
                comments.append(text)

        offset += len(items)

    return comments
