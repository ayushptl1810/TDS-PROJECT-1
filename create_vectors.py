"""
Vector Store Creation Script

This script creates vector embeddings for course content and forum posts,
storing them in FAISS indices for efficient similarity search.
"""

import os
import json
import pickle
from fastembed import TextEmbedding, ImageEmbedding
import requests
import faiss
import numpy as np
from tqdm import tqdm
from bs4 import BeautifulSoup
from PIL import Image
from io import BytesIO
import re
from datetime import datetime

# Constants for embedding dimensions
TEXT_DIMENSION = 384  # all-MiniLM-L6-v2 dimension
IMAGE_DIMENSION = 512  # CLIP dimension

def extract_image_urls(html_content):
    soup = BeautifulSoup(html_content, "html.parser")
    return [img['src'] for img in soup.find_all('img') if 'src' in img.attrs]

def download_and_preprocess_image(url):
    try:
        response = requests.get(url, timeout=5)
        response.raise_for_status()
        img = Image.open(BytesIO(response.content)).convert('RGB')
        # Resize to CLIP's expected size
        img = img.resize((224, 224))
        return img
    except Exception as e:
        print(f"Failed to download image: {url} | Error: {e}")
        return None

def clean_text(text: str) -> str:
    """Clean and normalize text for better matching."""
    if not text:
        return ""
    # Remove HTML tags but preserve important formatting
    text = re.sub(r'<div[^>]*>', ' ', text)
    text = re.sub(r'</div>', ' ', text)
    text = re.sub(r'<span[^>]*>', ' ', text)
    text = re.sub(r'</span>', ' ', text)
    text = re.sub(r'<a[^>]*>', ' ', text)
    text = re.sub(r'</a>', ' ', text)
    text = re.sub(r'<p>', ' ', text)
    text = re.sub(r'</p>', ' ', text)
    text = re.sub(r'<br>', ' ', text)
    text = re.sub(r'<strong>', ' ', text)
    text = re.sub(r'</strong>', ' ', text)
    text = re.sub(r'<em>', ' ', text)
    text = re.sub(r'</em>', ' ', text)
    # Keep mentions but clean them
    text = re.sub(r'@(\w+)', r'@\1', text)
    # Normalize whitespace
    text = re.sub(r'\s+', ' ', text)
    # Remove special characters but keep important punctuation
    text = re.sub(r'[^\w\s.,!?@-]', ' ', text)
    return text.strip()

def extract_user_role(html_content: str) -> str:
    """Extract user role from HTML content."""
    role = "Student"  # Default role
    if "user-title--community-ta" in html_content:
        role = "Community-TA"
    elif "user-title--instructor" in html_content:
        role = "Instructor"
    elif "user-title--teaching-assistant" in html_content:
        role = "Teaching-Assistant"
    return role

def extract_post_metadata(html_content: str, post_data: dict) -> dict:
    """Extract additional metadata from post HTML."""
    metadata = post_data.copy()
    
    # Extract user role
    metadata["role"] = extract_user_role(html_content)
    
    # Extract post date
    date_match = re.search(r'data-time="(\d+)"', html_content)
    if date_match:
        timestamp = int(date_match.group(1))
        metadata["post_date"] = datetime.fromtimestamp(timestamp/1000).isoformat()
    
    # Extract topic category
    category_match = re.search(r'category-(\w+)', html_content)
    if category_match:
        metadata["category"] = category_match.group(1)
    
    # Extract mentions
    mentions = re.findall(r'<a class="mention" href="/u/([^"]+)">', html_content)
    if mentions:
        metadata["mentions"] = mentions
    
    # Extract post number from URL
    url_match = re.search(r'/(\d+)(?:/(\d+))?$', metadata.get("url", ""))
    if url_match:
        metadata["topic_id"] = url_match.group(1)
        if url_match.group(2):
            metadata["post_number"] = int(url_match.group(2))
    
    return metadata

def create_md_vectors():
    """
    Create vector embeddings for markdown files in the tds_pages_md directory.
    Saves both the FAISS index and metadata for later use.
    """
    print("Loading markdown files...")
    
    # Initialize the sentence transformer model
    model = TextEmbedding('sentence-transformers/all-MiniLM-L6-v2')
    
    # Get list of markdown files
    md_dir = "tds_pages_md"
    md_files = [f for f in os.listdir(md_dir) if f.endswith(".md")]
    
    # Prepare data structures
    texts = []
    metadata = []
    
    # Process each markdown file
    for filename in tqdm(md_files, desc="Processing markdown files"):
        file_path = os.path.join(md_dir, filename)
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()
                
                # Extract title and URL from frontmatter if present
                title = filename.replace(".md", "").replace("_", " ")
                original_url = None
                
                if content.startswith("---"):
                    try:
                        frontmatter_end = content.find("---", 3)
                        if frontmatter_end != -1:
                            frontmatter = content[3:frontmatter_end]
                            for line in frontmatter.split("\n"):
                                if line.startswith("title:"):
                                    title = line[6:].strip().strip('"\'')
                                elif line.startswith("original_url:"):
                                    original_url = line[13:].strip().strip('"\'')
                    except Exception as e:
                        print(f"Error parsing frontmatter in {filename}: {str(e)}")
                
                # Clean content
                content = clean_text(content)
                
                # Extract keywords from title and content
                title_keywords = set(title.lower().split())
                content_keywords = set(re.findall(r'\b\w+\b', content.lower()))
                keywords = title_keywords.union(content_keywords)
                
                # Create structured search text
                search_text = f"""Title: {title}
Content: {content}
Keywords: {', '.join(keywords)}"""
                
                # Only construct URL if not found in frontmatter
                if not original_url:
                    # Remove any .md extension and convert to URL format
                    url_path = filename.replace(".md", "").lower()
                    # Convert filename to URL format
                    url_path = url_path.replace("__", "-").replace("_", "-")
                    original_url = f"https://tds.s-anand.net/#/{url_path}"
                
                texts.append(search_text)
                metadata.append({
                    "filename": filename,
                    "file_path": file_path,
                    "title": title,
                    "content": content,
                    "url": original_url,
                    "search_text": search_text,
                    "keywords": list(keywords)
                })
        except Exception as e:
            print(f"Error processing {filename}: {str(e)}")
    
    print(f"Creating embeddings for {len(texts)} markdown files...")
    
    # Create embeddings
    embeddings = np.array(list(model.embed(texts)), dtype='float32')
    
    # Normalize embeddings for cosine similarity
    faiss.normalize_L2(embeddings)
    
    # Create FAISS index with cosine similarity (IP = Inner Product)
    index = faiss.IndexFlatIP(TEXT_DIMENSION)
    index.add(embeddings)
    
    # Save index and metadata
    print("Saving markdown index and metadata...")
    faiss.write_index(index, "vector_store/md_index.faiss")
    with open("vector_store/md_metadata.pkl", "wb") as f:
        pickle.dump(metadata, f)

def create_json_vectors():
    """
    Create vector embeddings for forum posts from JSON files.
    Uses text embeddings for search and optionally combines with image embeddings.
    """
    print("Loading forum posts...")

    # Initialize both models
    text_model = TextEmbedding('sentence-transformers/all-MiniLM-L6-v2')
    image_model = ImageEmbedding('Qdrant/clip-ViT-B-32-vision')

    # Load and validate JSON data
    try:
        with open("tds_forum_posts.json", "r", encoding="utf-8") as f:
            data = json.load(f)
            
        # Validate JSON structure
        if not isinstance(data, dict) or "posts" not in data or "metadata" not in data:
            raise ValueError("Invalid JSON format: missing 'posts' or 'metadata' fields")
            
        posts = data["posts"]
        metadata = data["metadata"]
        
        # Validate posts array
        if not isinstance(posts, list):
            raise ValueError("Invalid JSON format: 'posts' must be an array")
            
        print(f"Loaded {len(posts)} posts from JSON")
        print(f"Date range: {metadata.get('date_range', {}).get('start')} to {metadata.get('date_range', {}).get('end')}")
        
    except json.JSONDecodeError as e:
        print(f"Error decoding JSON file: {str(e)}")
        return
    except Exception as e:
        print(f"Error loading JSON file: {str(e)}")
        return

    embeddings = []
    metadata_list = []
    processed_count = 0
    error_count = 0

    for post in tqdm(posts, desc="Processing forum posts"):
        try:
            # Validate required fields
            required_fields = ['topic_id', 'post_number', 'title', 'content', 'url', 'author', 'created_at']
            missing_fields = [field for field in required_fields if field not in post]
            if missing_fields:
                print(f"Warning: Post missing required fields: {missing_fields}")
                error_count += 1
                continue

            # Extract and clean content
            html_content = post.get("content", "")
            content = clean_text(html_content)
            title = clean_text(post.get("title", ""))
            
            # Extract additional metadata
            post_metadata = {
                "title": title,
                "content": content,
                "url": post.get("url", ""),
                "author": post.get("author", ""),
                "topic_title": post.get("title", ""),  # Using title as topic_title since it's the same
                "topic_id": post.get("topic_id", ""),
                "post_number": post.get("post_number", 0),
                "created_at": post.get("created_at", ""),
                "role": "user"  # Default role
            }
            
            # Create structured search text
            search_text = f"""Title: {title}
Content: {content}
Author: {post_metadata['author']} ({post_metadata['role']})
Date: {post_metadata['created_at']}"""
            
            # Get text embedding
            text_emb = np.array(list(text_model.embed([search_text]))[0], dtype=np.float32)
            faiss.normalize_L2(text_emb.reshape(1, -1))
            text_emb = text_emb.reshape(-1)
            
            # For now, we'll just use text embeddings
            final_emb = text_emb
            
            embeddings.append(final_emb)
            metadata_list.append(post_metadata)
            processed_count += 1
            
        except Exception as e:
            print(f"Error processing post {post.get('topic_id', 'unknown')}/{post.get('post_number', 'unknown')}: {str(e)}")
            error_count += 1
            continue

    # Create FAISS index
    if embeddings:
        print(f"\nCreating FAISS index with {len(embeddings)} embeddings...")
        embeddings_array = np.array(embeddings, dtype='float32')
        faiss.normalize_L2(embeddings_array)  # Normalize for cosine similarity
        index = faiss.IndexFlatIP(embeddings_array.shape[1])  # Use Inner Product for cosine similarity
        index.add(embeddings_array)
        
        # Save index and metadata
        os.makedirs("vector_store", exist_ok=True)
        print("Saving index and metadata...")
        faiss.write_index(index, "vector_store/json_index.faiss")
        with open("vector_store/json_metadata.pkl", "wb") as f:
            pickle.dump(metadata_list, f)
        
        print(f"\nVector store creation complete!")
        print(f"Successfully processed: {processed_count} posts")
        print(f"Errors encountered: {error_count} posts")
        print(f"Total embeddings created: {len(embeddings)}")
    else:
        print("No embeddings created - no valid posts found")

if __name__ == "__main__":
    # Create vector store directory if it doesn't exist
    os.makedirs("vector_store", exist_ok=True)
    
    # Create vectors for both markdown and forum content
    create_json_vectors()
    create_md_vectors()
    
    print("Vector store creation complete!") 