"""
Forum Scraper for TDS Course Discussions

This script scrapes the Tools in Data Science course forum and saves
the discussions as JSON files for later processing.
"""

import os
import json
import requests
import time
from datetime import datetime
import re
from typing import List, Dict
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Constants for forum scraping
BASE_URL = "https://discourse.onlinedegree.iitm.ac.in"
CATEGORY_SLUG = "courses/tds-kb"
CATEGORY_ID = 34
START_DATE = datetime(2025, 1, 1)
END_DATE = datetime(2025, 4, 14, 23, 59, 59)

# Authentication cookie for accessing the forum
COOKIES = {
    "_t": os.getenv("_t")
}

if not COOKIES["_t"]:
    raise ValueError("Environment variable '_t' not set. Please add it to your .env file.")

# Headers for making requests
HEADERS = {
    "User-Agent": "Mozilla/5.0"
}

def clean_text(text: str) -> str:
    """
    Clean and format text content from forum posts.
    
    Args:
        text: Raw text content from HTML
        
    Returns:
        Cleaned and formatted text string
    """
    if not text:
        return ""
    # Remove HTML tags
    text = re.sub(r'<[^>]+>', '', text)
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text)
    # Remove special characters
    text = re.sub(r'[^\w\s\-.,;:!?()\[\]{}]', '', text)
    return text.strip()

def get_all_topics():
    """
    Fetch all topics from the forum category using the Discourse API.
    
    Returns:
        List of topic dictionaries
    """
    all_topics = []
    page = 0
    print("Starting to fetch all topics from category...")
    while True:
        url = f"{BASE_URL}/c/{CATEGORY_SLUG}/{CATEGORY_ID}.json?page={page}"
        resp = requests.get(url, headers=HEADERS, cookies=COOKIES)
        if resp.status_code != 200:
            print(f"Failed to fetch page {page}: status code {resp.status_code}")
            break
        data = resp.json()
        topics = data.get("topic_list", {}).get("topics", [])
        if not topics:
            print(f"No more topics found at page {page}. Stopping pagination.")
            break

        print(f"Fetched {len(topics)} topics from page {page}")
        all_topics.extend(topics)
        page += 1
        time.sleep(1)  # polite delay to avoid hammering the server

    print(f"Total topics fetched: {len(all_topics)}")
    return all_topics

def filter_topics_by_date(topics, start_date, end_date):
    """
    Filter topics based on their creation date.
    
    Args:
        topics: List of topic dictionaries
        start_date: Start date for filtering
        end_date: End date for filtering
        
    Returns:
        List of filtered topic dictionaries
    """
    filtered = []
    for topic in topics:
        created_at_str = topic.get("created_at")
        if not created_at_str:
            continue
        try:
            created_at = datetime.strptime(created_at_str, "%Y-%m-%dT%H:%M:%S.%fZ")
        except ValueError:
            # Sometimes the microseconds part might be missing
            created_at = datetime.strptime(created_at_str, "%Y-%m-%dT%H:%M:%SZ")
        if start_date <= created_at <= end_date:
            filtered.append(topic)
    print(f"Filtered {len(filtered)} topics between {start_date.date()} and {end_date.date()}")
    return filtered

def get_posts_for_topic(topic_id: str) -> List[Dict]:
    """
    Get all posts for a topic using the Discourse API.
    
    Args:
        topic_id: ID of the topic
        
    Returns:
        List of post dictionaries
    """
    url = f"{BASE_URL}/t/{topic_id}.json"
    resp = requests.get(url, headers=HEADERS, cookies=COOKIES)
    if resp.status_code != 200:
        print(f"Failed to fetch posts for topic {topic_id}: status code {resp.status_code}")
        return []
    
    data = resp.json()
    posts = data.get("post_stream", {}).get("posts", [])
    
    # Process each post
    processed_posts = []
    for post in posts:
        processed_post = {
            'id': str(post.get('id')),
            'author': post.get('username'),
            'content': post.get('cooked', ''),  # Keep original HTML content
            'created_at': post.get('created_at'),
            'url': f"{BASE_URL}/t/{topic_id}/{post.get('post_number')}"
        }
        processed_posts.append(processed_post)
    
    return processed_posts

def scrape_forum(output_dir: str) -> None:
    """
    Scrape the TDS course forum and save content as JSON files.
    
    Args:
        output_dir: Directory to save JSON files
    """
    try:
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Get all topics
        print("Fetching topics...")
        all_topics = get_all_topics()
        
        # Filter topics by date
        print("Filtering topics by date...")
        filtered_topics = filter_topics_by_date(all_topics, START_DATE, END_DATE)
        
        # Save filtered topics
        topics_file = os.path.join(output_dir, 'tds_forum_topics_filtered.json')
        with open(topics_file, 'w', encoding='utf-8') as f:
            json.dump(filtered_topics, f, indent=2)
        print(f"Saved {len(filtered_topics)} topics to {topics_file}")
        
        # Get posts for each filtered topic
        print("Fetching posts...")
        all_posts = []
        for topic in filtered_topics:
            print(f"Processing topic: {topic.get('title', 'Unknown')}")
            posts = get_posts_for_topic(topic['id'])
            
            # Add topic info to each post
            for post in posts:
                post['topic_id'] = topic['id']
                post['topic_title'] = topic.get('title', 'Unknown')
            
            all_posts.extend(posts)
            time.sleep(1)  # polite delay between topics
        
        # Save posts
        posts_file = os.path.join(output_dir, 'tds_forum_posts_filtered.json')
        with open(posts_file, 'w', encoding='utf-8') as f:
            json.dump(all_posts, f, indent=2)
        print(f"Saved {len(all_posts)} posts to {posts_file}")
        
        print("Forum scraping complete!")
        
    except Exception as e:
        print(f"Error scraping forum: {str(e)}")

if __name__ == "__main__":
    # Configuration
    OUTPUT_DIR = "."
    
    # Start scraping
    print("Starting forum scraping...")
    scrape_forum(OUTPUT_DIR)
