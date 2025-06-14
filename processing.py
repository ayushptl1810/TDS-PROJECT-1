import json
import re

def clean_text(text):
    if not text:
        return ""
    # Lowercase
    text = text.lower()
    # Remove HTML tags if any (simple regex)
    text = re.sub(r'<[^>]+>', ' ', text)
    # Remove punctuation except spaces and alphanumerics
    text = re.sub(r'[^a-z0-9\s]', ' ', text)
    # Normalize whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def preprocess(topics_file, posts_file, output_file):
    with open(topics_file, 'r', encoding='utf-8') as f:
        topics = json.load(f)
    with open(posts_file, 'r', encoding='utf-8') as f:
        posts = json.load(f)
    
    processed_docs = []
    for topic in topics:
        topic_id = str(topic['id'])
        
        # Get all posts for this topic
        topic_posts = [post for post in posts if str(post.get('topic_id')) == topic_id]
        
        for post in topic_posts:
            post_number = post.get('post_number', '')
            if post_number != 1:  # Skip first post
                post_url = post.get('url', '')
                post_text = clean_text(post.get('content', ''))  # Use content field which was cooked from scraper
                post_id = post.get('id')

                processed_docs.append({
                    'id': post_id,
                    'topic_id': topic_id,
                    'search_text': post_text,
                    'url': post_url
                })
            else:
                print(f"Skipping first post for topic {topic_id}")
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(processed_docs, f, ensure_ascii=False, indent=2)
    print(f"Saved preprocessed data for {len(processed_docs)} posts to {output_file}")

if __name__ == "__main__":
    preprocess("tds_forum_topics_filtered.json", "tds_forum_posts_filtered.json", "processed_docs.json")
