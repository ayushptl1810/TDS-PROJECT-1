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
from typing import List, Dict, Optional
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
    # Remove special characters but keep important punctuation
    text = re.sub(r'[^\w\s\-.,;:!?()\[\]{}]', '', text)
    return text.strip()

def get_all_topics() -> List[Dict]:
    """
    Get all topics from the forum using the Discourse API.
    Handles pagination to get all topics.
    
    Returns:
        List of topic dictionaries including the topic slug
    """
    all_topics = []
    page = 0
    
    print("Starting to fetch all topics from category...")
    while True:
        try:
            # Use the correct category URL with slug and ID
            url = f"{BASE_URL}/c/{CATEGORY_SLUG}/{CATEGORY_ID}.json?page={page}"
            resp = requests.get(url, headers=HEADERS, cookies=COOKIES)
            resp.raise_for_status()
            
            data = resp.json()
            topics = data.get("topic_list", {}).get("topics", [])
            
            if not topics:
                print(f"No more topics found at page {page}. Stopping pagination.")
                break
            
            # Process each topic to include the slug
            for topic in topics:
                # Get the topic slug from the URL
                topic_url = topic.get('url', '')
                if topic_url:
                    # Extract slug from URL (format: /t/slug/id)
                    parts = topic_url.split('/')
                    if len(parts) >= 3:
                        topic['slug'] = parts[2]  # The slug is the third part
                    else:
                        # Fallback to a sanitized version of the title
                        topic['slug'] = topic.get('title', '').lower().replace(' ', '-')
                else:
                    # Fallback to a sanitized version of the title
                    topic['slug'] = topic.get('title', '').lower().replace(' ', '-')
            
            all_topics.extend(topics)
            print(f"Fetched {len(topics)} topics from page {page}")
            
            page += 1
            time.sleep(1)  # Polite delay between requests
            
        except requests.exceptions.RequestException as e:
            print(f"Error fetching topics at page {page}: {str(e)}")
            if isinstance(e, requests.exceptions.HTTPError) and e.response.status_code == 429:
                retry_after = int(e.response.headers.get('Retry-After', 60))
                print(f"Rate limited. Waiting {retry_after} seconds...")
                time.sleep(retry_after)
                continue
            break
    
    print(f"Total topics fetched: {len(all_topics)}")
    return all_topics

def filter_topics_by_date(topics: List[Dict], start_date: datetime, end_date: datetime) -> List[Dict]:
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

def get_posts_for_topic(topic_id: str, topic_slug: str, max_retries: int = 3) -> List[Dict]:
    """Get all posts for a topic using pagination."""
    all_posts = []
    seen_post_ids = set()
    page = 0
    
    # First get topic details to get total post count
    try:
        url = f"{BASE_URL}/t/{topic_slug}/{topic_id}.json"
        resp = requests.get(url, headers=HEADERS, cookies=COOKIES)
        resp.raise_for_status()
        
        data = resp.json()
        topic_data = data.get("post_stream", {}).get("posts", [{}])[0]  # First post contains topic info
        total_posts = topic_data.get("posts_count", 0)
        print(f"Total posts in topic {topic_id}: {total_posts}")
        
        if not total_posts:
            print(f"No posts found for topic {topic_id}")
            return []
            
    except requests.exceptions.RequestException as e:
        print(f"Error fetching topic details for {topic_id}: {str(e)}")
        return []
    
    # Now fetch all posts using the correct pagination endpoint
    while True:
        try:
            # Use the exact URL structure that works for pagination
            url = f"{BASE_URL}/t/{topic_slug}/{topic_id}.json"
            params = {"page": page}
            print(f"Fetching page {page} from: {url}?page={page}")  # Debug print
            
            resp = requests.get(url, params=params, headers=HEADERS, cookies=COOKIES)
            resp.raise_for_status()
            
            data = resp.json()
            post_stream = data.get("post_stream", {})
            posts = post_stream.get("posts", [])
            stream = post_stream.get("stream", [])  # Get the stream of post IDs
            
            if not posts:
                print(f"No more posts found for topic {topic_id} at page {page}")
                break
            
            # Track new posts and check for duplicates
            new_posts = []
            for post in posts:
                post_id = str(post.get('id'))
                if post_id not in seen_post_ids:
                    seen_post_ids.add(post_id)
                    # Only keep the fields we need for the vector store
                    processed_post = {
                        'topic_id': topic_id,
                        'post_number': post.get('post_number'),
                        'title': topic_data.get('title', ''),  # Get title from topic data
                        'content': post.get('cooked', ''),  # Keep original HTML content
                        'url': f"{BASE_URL}/t/{topic_slug}/{topic_id}/{post.get('post_number')}",  # Use correct URL structure
                        'author': post.get('username', ''),
                        'created_at': post.get('created_at', '')
                    }
                    new_posts.append(processed_post)
            
            if new_posts:
                print(f"Fetched {len(new_posts)} new posts from page {page} of topic {topic_id}")
                all_posts.extend(new_posts)
            else:
                print(f"No new posts found on page {page}")
            
            # Check if we've seen all posts from the stream
            if stream and all(post_id in seen_post_ids for post_id in stream):
                print(f"Reached end of stream for topic {topic_id}")
                break
                
            # If we haven't reached the total count yet, continue pagination
            if len(seen_post_ids) < total_posts:
                page += 1
            else:
                print(f"Reached total post count ({total_posts}) for topic {topic_id}")
                break
                
        except requests.exceptions.RequestException as e:
            print(f"Error fetching posts for topic {topic_id} at page {page}: {str(e)}")
            if max_retries > 0:
                print(f"Retrying... ({max_retries} attempts left)")
                max_retries -= 1
                continue
            break
    
    print(f"Processed {len(all_posts)} posts for topic {topic_id}")
    return all_posts

def scrape_forum(output_dir: str) -> None:
    """
    Scrape the forum and save all posts as a single JSON file.
    Only includes fields needed for the vector store.
    """
    print("Starting forum scraping...")
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Get all topics
    print("Fetching topics...")
    topics = get_all_topics()
    
    # Filter topics by date
    print("Filtering topics by date...")
    start_date = datetime(2025, 1, 1)
    end_date = datetime(2025, 4, 14)
    filtered_topics = filter_topics_by_date(topics, start_date, end_date)
    
    # Fetch all posts
    print("Fetching ALL posts...")
    all_posts = []
    total_topics = len(filtered_topics)
    
    for i, topic in enumerate(filtered_topics, 1):
        topic_id = str(topic['id'])
        topic_slug = topic.get('slug', '')  # Get the topic slug from the topic data
        print(f"\nProcessing topic {i}/{total_topics}: {topic.get('title', 'Unknown Title')}")
        
        # Get all posts for this topic
        posts = get_posts_for_topic(topic_id, topic_slug)
        all_posts.extend(posts)
        
        print(f"Processed {len(posts)} posts for topic {i}")
        print(f"Total posts collected so far: {len(all_posts)}")
    
    # Save all posts to a single JSON file
    output_file = os.path.join(output_dir, "tds_forum_posts.json")
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump({
            "metadata": {
                "date_range": {
                    "start": start_date.isoformat(),
                    "end": end_date.isoformat()
                },
                "completion_time": datetime.now().isoformat(),
                "total_posts": len(all_posts),
                "total_topics": total_topics
            },
            "posts": all_posts
        }, f, indent=2, ensure_ascii=False)
    
    print(f"\nScraping complete!")
    print(f"Total posts collected: {len(all_posts)}")
    print(f"Output saved to: {output_file}")
    
    # Save summary separately for monitoring
    summary_file = os.path.join(output_dir, "tds_forum_scrape_summary.json")
    with open(summary_file, 'w', encoding='utf-8') as f:
        json.dump({
            "date_range": {
                "start": start_date.isoformat(),
                "end": end_date.isoformat()
            },
            "completion_time": datetime.now().isoformat(),
            "total_posts": len(all_posts),
            "total_topics": total_topics,
            "topics_processed": [{
                "id": topic['id'],
                "title": topic.get('title', 'Unknown Title'),
                "post_count": len([p for p in all_posts if str(p['topic_id']) == str(topic['id'])])
            } for topic in filtered_topics]
        }, f, indent=2, ensure_ascii=False)

if __name__ == "__main__":
    # Configuration
    OUTPUT_DIR = "."
    
    # Start scraping
    print("Starting forum scraping...")
    scrape_forum(OUTPUT_DIR)
