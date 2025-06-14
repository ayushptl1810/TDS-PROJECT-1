"""
Vector Store Creation Script

This script creates vector embeddings for course content and forum posts,
storing them in FAISS indices for efficient similarity search.
"""

import os
import json
import pickle
from fastembed import TextEmbedding
import requests
import faiss
import numpy as np
from tqdm import tqdm

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
                texts.append(content)
                metadata.append({
                    "filename": filename,
                    "file_path": file_path,
                    "original_url": f"https://tds.s-anand.net/#/{filename.replace('.md', '')}"
                })
        except Exception as e:
            print(f"Error processing {filename}: {str(e)}")
    
    print(f"Creating embeddings for {len(texts)} markdown files...")
    
    # Create embeddings
    embeddings = np.array(list(model.embed(texts)), dtype='float32')
    
    # Create FAISS index
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(np.array(embeddings).astype('float32'))
    
    # Save index and metadata
    print("Saving markdown index and metadata...")
    faiss.write_index(index, "vector_store/md_index.faiss")
    with open("vector_store/md_metadata.pkl", "wb") as f:
        pickle.dump(metadata, f)

def create_json_vectors():
    """
    Create vector embeddings for forum posts from JSON files.
    Saves both the FAISS index and metadata for later use.
    """
    print("Loading forum posts...")
    
    # Initialize the sentence transformer model
    text_model = TextEmbedding('sentence-transformers/all-MiniLM-L6-v2')

    
    # Load forum data
    with open("tds_forum_posts_filtered.json", "r", encoding="utf-8") as f:
        posts = json.load(f)
    
    # Prepare data structures
    texts = []
    metadata = []
    
    # Process each post
    for post in tqdm(posts, desc="Processing forum posts"):
        try:
            # Combine topic title and content for better context
            search_text = f"{post['topic_title']}\n{post['content']}"
            texts.append(search_text)
            metadata.append({
                "id": post["id"],
                "url": post["url"],
                "search_text": search_text
            })

        except Exception as e:
            print(f"Error processing post {post.get('id', 'unknown')}: {str(e)}")
    
    print(f"Creating embeddings for {len(texts)} forum posts...")
    
    # Create embeddings
    text_embeddings = np.array(list(text_model.embed(texts)), dtype='float32')

    print(f"Saving {len(text_embeddings)} total embeddings...")

    # Save FAISS index
    dimension = text_embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(text_embeddings)
    
    # Save index and metadata
    print("Saving forum index and metadata...")
    faiss.write_index(index, "vector_store/json_index.faiss")
    with open("vector_store/json_metadata.pkl", "wb") as f:
        pickle.dump(metadata, f)

if __name__ == "__main__":
    # Create vector store directory if it doesn't exist
    os.makedirs("vector_store", exist_ok=True)
    
    # Create vectors for both markdown and forum content
    create_md_vectors()
    create_json_vectors()
    
    print("Vector store creation complete!") 