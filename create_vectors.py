"""
Vector Store Creation Script

This script creates vector embeddings for course content and forum posts,
storing them in Pinecone vector database for efficient similarity search.
"""

import os
import json
import pickle
from pinecone import Pinecone
import requests
import numpy as np
from tqdm import tqdm
from bs4 import BeautifulSoup
from PIL import Image
from io import BytesIO
import re
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()

# Pinecone configuration
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
INDEX_NAME = "llama-text-embed-v2-index"

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

def delete_pinecone_index():
    """
    Delete the Pinecone index if it exists.
    """
    print("Deleting existing Pinecone index...")
    
    pc = Pinecone(api_key=PINECONE_API_KEY)
    
    try:
        pc.delete_index(INDEX_NAME)
        print(f"Index '{INDEX_NAME}' deleted successfully")
        
        # Wait for deletion to complete
        import time
        while True:
            try:
                pc.Index(INDEX_NAME)
                print("Waiting for deletion to complete...")
                time.sleep(5)
            except Exception:
                print("Index deletion completed")
                break
    except Exception as e:
        if "not found" in str(e).lower() or "404" in str(e):
            print(f"Index '{INDEX_NAME}' doesn't exist")
        else:
            print(f"Error deleting index: {str(e)}")

def create_pinecone_index():
    """
    Create the Pinecone index if it doesn't exist.
    Note: The index must be created manually through the Pinecone console with integrated inference enabled.
    """
    print("Checking Pinecone index...")
    
    pc = Pinecone(api_key=PINECONE_API_KEY)
    
    # Check if index exists
    try:
        index = pc.Index(INDEX_NAME)
        print(f"Index '{INDEX_NAME}' already exists")
        
        # Test if integrated inference is working
        try:
            test_record = {"id": "test", "text": "test"}
            index.upsert_records(namespace="test", records=[test_record])
            print("Integrated inference is working correctly")
            return index
        except Exception as e:
            if "integrated inference is not configured" in str(e).lower():
                print("ERROR: Index exists but doesn't have integrated inference configured.")
                print("Please create the index manually through the Pinecone console:")
                print("1. Go to https://app.pinecone.io/")
                print("2. Create a new index named 'llama-text-embed-v2-index'")
                print("3. Set dimension to 1024")
                print("4. Set metric to 'cosine'")
                print("5. Enable 'Integrated inference' and select 'llama-text-embed-v2' model")
                print("6. Choose serverless with AWS us-east-1")
                raise Exception("Index needs to be created manually with integrated inference")
            else:
                raise e
        
    except Exception as e:
        if "not found" in str(e).lower() or "404" in str(e):
            print("ERROR: Index 'llama-text-embed-v2-index' not found.")
            print("Please create the index manually through the Pinecone console:")
            print("1. Go to https://app.pinecone.io/")
            print("2. Create a new index named 'llama-text-embed-v2-index'")
            print("3. Set dimension to 1024")
            print("4. Set metric to 'cosine'")
            print("5. Enable 'Integrated inference' and select 'llama-text-embed-v2' model")
            print("6. Choose serverless with AWS us-east-1")
            raise Exception("Index needs to be created manually with integrated inference")
        else:
            raise e

def create_md_vectors(index):
    """
    Create vector embeddings for markdown files in the tds_pages_md directory.
    Stores them in Pinecone vector database for efficient similarity search.
    """
    print("Loading markdown files...")
    
    # Get list of markdown files
    md_dir = "tds_pages_md"
    md_files = [f for f in os.listdir(md_dir) if f.endswith(".md")]
    
    # Prepare data for Pinecone upsert
    records = []
    metadata_list = []
    
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
                
                # Create structured search text (truncated to reduce size)
                search_text = f"""Title: {title[:100]}
Content: {content[:1000]}
Keywords: {', '.join(list(keywords)[:5])}"""
                
                # Only construct URL if not found in frontmatter
                if not original_url:
                    # Remove any .md extension and convert to URL format
                    url_path = filename.replace(".md", "").lower()
                    # Convert filename to URL format
                    url_path = url_path.replace("__", "-").replace("_", "-")
                    original_url = f"https://tds.s-anand.net/#/{url_path}"
                
                # Create unique ID for the record
                record_id = f"md_{filename.replace('.md', '')}"
                
                # Prepare record for Pinecone
                record = {
                    "id": record_id,
                    "text": search_text
                }
                
                records.append(record)
                
                # Store minimal metadata to stay under 40KB limit
                metadata_list.append({
                    "id": record_id,
                    "filename": filename,
                    "title": title[:100],  # Truncate title further
                    "url": original_url,
                    "type": "markdown"
                })
                
        except Exception as e:
            print(f"Error processing {filename}: {str(e)}")
    
    print(f"Creating embeddings for {len(records)} markdown files...")
    
    # Upsert records to Pinecone in batches
    batch_size = 20  # Reduce batch size further for better reliability
    successful_uploads = 0
    failed_uploads = 0
    
    for i in range(0, len(records), batch_size):
        batch = records[i:i + batch_size]
        max_retries = 3
        retry_count = 0
        
        while retry_count < max_retries:
            try:
                index.upsert_records(
                    namespace="markdown-content",
                    records=batch
                )
                successful_uploads += len(batch)
                print(f"Upserted batch {i//batch_size + 1}/{(len(records) + batch_size - 1)//batch_size} ({len(batch)} records)")
                break
            except Exception as e:
                retry_count += 1
                print(f"Error upserting batch {i//batch_size + 1} (attempt {retry_count}): {str(e)}")
                if retry_count >= max_retries:
                    # Try uploading records individually
                    print(f"Trying individual upload for batch {i//batch_size + 1}")
                    individual_success = 0
                    for j, record in enumerate(batch):
                        try:
                            index.upsert_records(
                                namespace="markdown-content",
                                records=[record]
                            )
                            individual_success += 1
                            successful_uploads += 1
                        except Exception as individual_error:
                            print(f"Failed to upload individual record {i+j+1}: {str(individual_error)}")
                            failed_uploads += 1
                    print(f"Individual upload result: {individual_success}/{len(batch)} successful")
                else:
                    import time
                    time.sleep(2)  # Wait before retry
    
    print(f"Upload summary: {successful_uploads} successful, {failed_uploads} failed out of {len(records)} total")
    
    # Save metadata locally for reference
    print("Saving markdown metadata...")
    os.makedirs("vector_store", exist_ok=True)
    with open("vector_store/md_metadata.pkl", "wb") as f:
        pickle.dump(metadata_list, f)
    
    print(f"Successfully processed {len(records)} markdown files")

def create_json_vectors(index):
    """
    Create vector embeddings for forum posts from JSON files.
    Stores them in Pinecone vector database for efficient similarity search.
    """
    print("Loading forum posts...")

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

    records = []
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
                "topic_id": post.get("topic_id", ""),
                "post_number": post.get("post_number", 0),
                "role": "user"  # Default role
            }
            
            # Create structured search text
            search_text = f"""Title: {title}
Content: {content}
Author: {post_metadata['author']} ({post_metadata['role']})"""
            
            # Create unique ID for the record
            record_id = f"post_{post_metadata['topic_id']}_{post_metadata['post_number']}"
            
            # Prepare record for Pinecone
            record = {
                "id": record_id,
                "text": search_text
            }
            
            records.append(record)
            
            # Store metadata separately
            metadata_list.append({
                "id": record_id,
                **post_metadata,
                "type": "forum_post"
            })
            
            processed_count += 1
            
        except Exception as e:
            print(f"Error processing post {post.get('topic_id', 'unknown')}/{post.get('post_number', 'unknown')}: {str(e)}")
            error_count += 1
            continue

    # Upsert records to Pinecone
    if records:
        print(f"\nUpserting {len(records)} records to Pinecone...")
        
        # Upsert records in batches
        batch_size = 90
        for i in range(0, len(records), batch_size):
            batch = records[i:i + batch_size]
            try:
                index.upsert_records(
                    namespace="forum-posts",
                    records=batch
                )
                print(f"Upserted batch {i//batch_size + 1}/{(len(records) + batch_size - 1)//batch_size}")
            except Exception as e:
                print(f"Error upserting batch {i//batch_size + 1}: {str(e)}")
        
        # Save metadata locally for reference
        os.makedirs("vector_store", exist_ok=True)
        print("Saving forum posts metadata...")
        with open("vector_store/json_metadata.pkl", "wb") as f:
            pickle.dump(metadata_list, f)
        
        print(f"\nVector store creation complete!")
        print(f"Successfully processed: {processed_count} posts")
        print(f"Errors encountered: {error_count} posts")
        print(f"Total records created: {len(records)}")
    else:
        print("No records created - no valid posts found")

def check_and_fix_markdown_vectors(index):
    """
    Check what markdown files are in the Pinecone index and re-upload missing ones.
    """
    print("Checking markdown vectors in Pinecone index...")
    
    # Get index stats
    stats = index.describe_index_stats()
    if 'namespaces' in stats and 'markdown-content' in stats['namespaces']:
        current_count = stats['namespaces']['markdown-content']['vector_count']
        print(f"Current markdown vectors in index: {current_count}")
    else:
        current_count = 0
        print("No markdown-content namespace found")
    
    # Check how many markdown files we should have
    md_dir = "tds_pages_md"
    md_files = [f for f in os.listdir(md_dir) if f.endswith(".md")]
    expected_count = len(md_files)
    print(f"Expected markdown files: {expected_count}")
    
    if current_count < expected_count:
        print(f"Missing {expected_count - current_count} markdown files. Re-uploading...")
        create_md_vectors(index)
    else:
        print("All markdown files are uploaded successfully!")

if __name__ == "__main__":
    # Create vector store directory if it doesn't exist (for metadata storage)
    os.makedirs("vector_store", exist_ok=True)
    
    # Create or get the Pinecone index
    index = create_pinecone_index()
    
    # Check and fix markdown vectors
    check_and_fix_markdown_vectors(index)
    
    # Create vectors for forum posts
    create_json_vectors(index)
    
    print("Pinecone vector store creation complete!") 