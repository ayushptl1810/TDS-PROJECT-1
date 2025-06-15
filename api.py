"""
TDS Virtual TA API Server

This module implements a FastAPI server that provides a virtual teaching assistant
for the Tools in Data Science course. It uses Gemini AI for generating answers
and maintains a vector store of course content and forum posts for context.
"""

import os
import json
from typing import List, Dict, Optional, Union, Any
from fastapi import FastAPI, Request, BackgroundTasks, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastembed import TextEmbedding, ImageEmbedding
import faiss
import numpy as np
import pickle
import google.generativeai as genai
from dotenv import load_dotenv
import re
import shutil
import time
from PIL import Image
from io import BytesIO
from functools import lru_cache
import asyncio
from concurrent.futures import ThreadPoolExecutor

# Load environment variables
load_dotenv()

# Constants
BASE_URL = "https://discourse.onlinedegree.iitm.ac.in"
TEXT_DIMENSION = 384  # all-MiniLM-L6-v2 dimension
IMAGE_DIMENSION = 512  # CLIP dimension
CACHE_SIZE = 1000  # Number of search results to cache
SEARCH_EXECUTOR = ThreadPoolExecutor(max_workers=4)  # For parallel search operations

# Initialize FastAPI app
app = FastAPI(title="TDS Virtual TA API")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

search_engine = None

# ✅ NEW: FastAPI startup hook to preload everything
@app.on_event("startup")
def load_on_startup():
    global search_engine
    print("=== Starting server initialization ===")
    start = time.time()
    search_engine = SearchEngine()
    end = time.time()
    print(f"=== Initialization complete in {end - start:.2f} seconds ===")


class SearchEngine:
    """
    Search engine that combines vector search with Gemini AI for answering questions.
    Maintains indices of course content and forum posts for context-aware responses.
    """
    
    def __init__(self):
        """Initialize the search engine by loading models and indices."""
        print("Initializing SearchEngine...")
        
        # Load the sentence transformer model for vector embeddings
        print("Loading sentence transformer model...")
        self.text_model = TextEmbedding('sentence-transformers/all-MiniLM-L6-v2')
        
        # Load CLIP model for image embeddings
        print("Loading CLIP model...")
        self.image_model = ImageEmbedding('Qdrant/clip-ViT-B-32-vision')
        
        # Load FAISS indices for fast similarity search
        print("Loading FAISS indices...")
        self.md_index = faiss.read_index("vector_store/md_index.faiss")
        self.json_index = faiss.read_index("vector_store/json_index.faiss")
        
        # Load metadata for mapping indices to content
        print("Loading metadata...")
        with open("vector_store/md_metadata.pkl", "rb") as f:
            self.md_metadata = pickle.load(f)
        with open("vector_store/json_metadata.pkl", "rb") as f:
            self.json_metadata = pickle.load(f)
        
        # Load markdown content for course materials
        print("Loading markdown content...")
        self.md_content = {}
        for meta in self.md_metadata:
            try:
                with open(meta["file_path"], "r", encoding="utf-8") as f:
                    self.md_content[meta["filename"]] = f.read()
            except Exception as e:
                print(f"Error loading file {meta['file_path']}: {str(e)}")
        
        # Initialize Gemini AI model
        print("Initializing Gemini model...")
        GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
        if not GOOGLE_API_KEY:
            raise ValueError("GOOGLE_API_KEY environment variable not set")
        genai.configure(api_key=GOOGLE_API_KEY)
        self.gemini = genai.GenerativeModel('gemini-1.5-flash')
        
        print("SearchEngine initialization complete")

    def preprocess_image(self, image: Union[Image.Image, bytes]) -> Image.Image:
        """Preprocess image for CLIP model."""
        if isinstance(image, bytes):
            image = Image.open(BytesIO(image)).convert('RGB')
        return image.resize((224, 224))

    @lru_cache(maxsize=CACHE_SIZE)
    def get_query_embedding(self, query: Optional[str] = None, image_hash: Optional[str] = None) -> np.ndarray:
        """Get embedding for query (text and/or image) with caching."""
        try:
            if query and image_hash:
                # Get both embeddings
                text_emb = np.array(list(self.text_model.embed([query]))[0], dtype='float32')
                img_emb = np.array(list(self.image_model.embed([image_hash]))[0], dtype='float32')
                
                # Normalize embeddings
                faiss.normalize_L2(text_emb.reshape(1, -1))
                faiss.normalize_L2(img_emb.reshape(1, -1))
                text_emb = text_emb.reshape(-1)
                img_emb = img_emb.reshape(-1)
                
                return {
                    'text_emb': text_emb,
                    'image_emb': img_emb,
                    'has_text': True,
                    'has_image': True
                }
            elif query:
                # Text-only query
                text_emb = np.array(list(self.text_model.embed([query]))[0], dtype='float32')
                faiss.normalize_L2(text_emb.reshape(1, -1))
                text_emb = text_emb.reshape(-1)
                return {
                    'text_emb': text_emb,
                    'image_emb': None,
                    'has_text': True,
                    'has_image': False
                }
            elif image_hash:
                # Image-only query
                img_emb = np.array(list(self.image_model.embed([image_hash]))[0], dtype='float32')
                faiss.normalize_L2(img_emb.reshape(1, -1))
                img_emb = img_emb.reshape(-1)
                return {
                    'text_emb': None,
                    'image_emb': img_emb,
                    'has_text': False,
                    'has_image': True
                }
            else:
                raise ValueError("Either query text or image must be provided")
        except Exception as e:
            print(f"Error in get_query_embedding: {str(e)}")
            raise

    def clean_text(self, text: str) -> str:
        """Clean and normalize text for better matching."""
        if not text:
            return ""
        # Remove HTML tags
        text = re.sub(r'<[^>]+>', ' ', text)
        # Normalize whitespace
        text = re.sub(r'\s+', ' ', text)
        # Remove special characters but keep important punctuation
        text = re.sub(r'[^\w\s.,!?-]', ' ', text)
        return text.strip()

    def _calculate_semantic_similarity(self, query: str, text: str) -> float:
        """Calculate semantic similarity between query and text using embeddings."""
        if not text:
            return 0.0
        
        try:
            # Get embeddings and convert to numpy arrays
            query_emb = np.array(list(self.text_model.embed([query]))[0], dtype=np.float32)
            text_emb = np.array(list(self.text_model.embed([text]))[0], dtype=np.float32)
            
            # Normalize embeddings
            query_emb = query_emb / np.linalg.norm(query_emb)
            text_emb = text_emb / np.linalg.norm(text_emb)
            
            # Calculate cosine similarity
            return float(np.dot(query_emb, text_emb))
        except Exception as e:
            print(f"Error in semantic similarity calculation: {str(e)}")
            return 0.0

    def _normalize_url(self, url: str, post_number: Optional[int] = None) -> str:
        """Normalize URL to use exact format: BASE_URL/t/title-slug/topic_id/post_number"""
        if not url:
            return ""
        
        try:
            # First, clean up any URL encoding and special characters
            url = url.replace('[', '').replace(']', '')  # Remove square brackets
            url = re.sub(r'---+', '-', url)  # Replace multiple dashes with single dash
            url = re.sub(r'[^\w\-/.:]', '', url)  # Keep only alphanumeric, dashes, slashes, dots, and colons
            
            # Extract components from URL
            url_match = re.search(r'/t/(?:([^/]+)/)?(\d+)(?:/(\d+))?$', url)
            if url_match:
                topic_slug = url_match.group(1)
                topic_id = url_match.group(2)
                post_num = post_number or url_match.group(3)
                
                # Clean up the topic slug
                if topic_slug:
                    # Remove any remaining special characters and normalize dashes
                    topic_slug = re.sub(r'[^\w\-]', '-', topic_slug)
                    topic_slug = re.sub(r'-+', '-', topic_slug)  # Replace multiple dashes with single dash
                    topic_slug = topic_slug.strip('-')  # Remove leading/trailing dashes
                
                # Construct URL in exact format
                if topic_slug:
                    if post_num:
                        return f"{BASE_URL}/t/{topic_slug}/{topic_id}/{post_num}"
                    return f"{BASE_URL}/t/{topic_slug}/{topic_id}"
                return f"{BASE_URL}/t/{topic_id}"
            
            return url
            
        except Exception as e:
            print(f"Error normalizing URL {url}: {str(e)}")
            return url

    def _urls_match(self, url1: str, url2: str) -> bool:
        """Check if two URLs match regardless of format."""
        if not url1 or not url2:
            return False
        
        # Extract topic ID and post number from both URLs
        def extract_ids(url):
            match = re.search(r'/t/(?:[^/]+/)?(\d+)(?:/(\d+))?$', url)
            if match:
                return (match.group(1), match.group(2))
            return (None, None)
        
        id1, post1 = extract_ids(url1)
        id2, post2 = extract_ids(url2)
            
        # URLs match if they have the same topic ID and post number
        return id1 == id2 and (post1 == post2 or (not post1 and not post2))

    async def search(self, query: str, image_data: Optional[bytes] = None, top_k: int = 5, min_score: float = 0.15) -> List[Dict]:
        """Search for relevant content using both text and image data."""
        try:
            # Get query embedding
            image_hash = None
            if image_data:
                # Create a simple hash of image data for caching
                image_hash = str(hash(image_data))
            
            query_embedding = await asyncio.get_event_loop().run_in_executor(
                SEARCH_EXECUTOR,
                self.get_query_embedding,
                query,
                image_hash
            )
            
            # Extract key terms from query
            query_terms = set(query.lower().split())
            important_terms = {term for term in query_terms if len(term) > 3}

            # Search in both indices concurrently
            async def search_index(index, metadata, content_dict=None, is_forum=False):
                scores, indices = await asyncio.get_event_loop().run_in_executor(
                    SEARCH_EXECUTOR,
                    index.search,
                    np.array([query_embedding['text_emb']], dtype=np.float32),
                    min(top_k * 5, index.ntotal)
                )
                
                results = []
                for score, idx in zip(scores[0], indices[0]):
                    if idx == -1 or score < 0.1:
                        continue
                    
                    meta = metadata[idx]
                    title = meta.get("title", "").lower()
                    content = content_dict.get(meta["filename"], "") if content_dict else meta.get("content", "").lower()
                    url = meta.get("url", "")
                    
                    if is_forum:
                        url = self._normalize_url(url, meta.get("post_number"))
                    
                    # Calculate scores
                    title_sim = self._calculate_semantic_similarity(query, title)
                    content_sim = self._calculate_semantic_similarity(query, content)
                    phrase_score = self._calculate_phrase_match_score(query, title, content)
                    context_score = self._calculate_context_score(query, title, content)
                    
                    # Calculate term overlap with higher weight for forum posts
                    title_terms = set(title.split())
                    content_terms = set(content.split())
                    term_overlap = len(important_terms.intersection(title_terms)) / len(important_terms) * (0.4 if not is_forum else 0.5) + \
                                 len(important_terms.intersection(content_terms)) / len(important_terms) * (0.8 if not is_forum else 0.9)
                    
                    # Get context for forum posts
                    context = []
                    if is_forum and url and meta.get("post_number"):
                        post_number = meta.get("post_number")
                        topic_id = meta.get("topic_id")
                        
                        # Get parent post if exists
                        if post_number > 1:
                            parent_idx = next((i for i, m in enumerate(metadata) if m.get("post_number") == post_number-1 and m.get("topic_id") == topic_id), None)
                            if parent_idx is not None:
                                parent_meta = metadata[parent_idx]
                                parent_content = parent_meta.get("content", "").lower()
                                if self._calculate_semantic_similarity(query, parent_content) > 0.05:
                                    parent_url = self._normalize_url(parent_meta.get("url", ""), post_number-1)
                                    context.append({
                                        "content": parent_content,
                                        "author": parent_meta.get("author", ""),
                                        "url": parent_url,
                                        "is_parent": True
                                    })
                        
                        # Get next post if exists
                        next_idx = next((i for i, m in enumerate(metadata) if m.get("post_number") == post_number+1 and m.get("topic_id") == topic_id), None)
                        if next_idx is not None:
                            next_meta = metadata[next_idx]
                            next_content = next_meta.get("content", "").lower()
                            if self._calculate_semantic_similarity(query, next_content) > 0.05:
                                next_url = self._normalize_url(next_meta.get("url", ""), post_number+1)
                                context.append({
                                    "content": next_content,
                                    "author": next_meta.get("author", ""),
                                    "url": next_url,
                                    "is_parent": False
                                })
                    
                    # Add context similarity
                    context_sim = 0.0
                    if context:
                        context_text = " ".join([ctx["content"] for ctx in context])
                        context_sim = self._calculate_semantic_similarity(query, context_text)
                    
                    # Combine scores with higher weight for forum posts
                    final_score = (
                        score * 0.1 +
                        title_sim * (0.2 if not is_forum else 0.25) +
                        content_sim * (0.4 if not is_forum else 0.45) +
                        phrase_score * (0.2 if not is_forum else 0.25) +
                        context_score * 0.05 +
                        context_sim * (0.0 if not is_forum else 0.1) +
                        term_overlap * (0.15 if not is_forum else 0.2)
                    )
                    
                    if final_score < min_score:
                        continue
                    
                    result = {
                        "score": final_score,
                        "title": meta.get("title", ""),
                        "url": url,
                        "content": content[:200] + "..." if len(content) > 200 else content,
                        "source": "markdown" if content_dict else "forum_post",
                        "author": meta.get("author", "").lower()
                    }
                    
                    if context:
                        result["context"] = context
                    
                    results.append(result)
                
                return results

            # Run searches concurrently
            md_task = search_index(self.md_index, self.md_metadata, self.md_content, False)
            json_task = search_index(self.json_index, self.json_metadata, None, True)
            
            md_results, json_results = await asyncio.gather(md_task, json_task)
            
            # Combine and sort results
            all_results = []
            all_results.extend(md_results)
            all_results.extend(json_results)
            all_results.sort(key=lambda x: x["score"], reverse=True)
            
            return all_results[:top_k]
            
        except Exception as e:
            import traceback
            print(f"\n=== ERROR IN SEARCH ===")
            print(f"Error type: {type(e).__name__}")
            print(f"Error message: {str(e)}")
            print("\nFull traceback:")
            print(traceback.format_exc())
            return []

    def _calculate_phrase_match_score(self, query: str, title: str, content: str) -> float:
        """Calculate score based on exact phrase matches."""
        # Extract important phrases from query (2+ words)
        query_words = query.split()
        phrases = []
        for i in range(len(query_words) - 1):
            phrases.append(" ".join(query_words[i:i+2]))
        
        # Check for exact phrase matches
        score = 0.0
        for phrase in phrases:
            if phrase in title:
                score += 0.5  # Higher weight for title matches
            if phrase in content:
                score += 0.3  # Lower weight for content matches
        
        return min(1.0, score)  # Cap at 1.0

    def _calculate_context_score(self, query: str, title: str, content: str) -> float:
        """Calculate how well the content matches the query context."""
        # Extract key terms and their context
        query_terms = set(query.split())
        title_terms = set(title.split())
        content_terms = set(content.split())
        
        # Calculate term overlap
        term_overlap = len(query_terms.intersection(title_terms)) / len(query_terms)
        
        # Check for term proximity in content
        proximity_score = 0.0
        if content:
            words = content.split()
            for i, word in enumerate(words):
                if word in query_terms:
                    # Check surrounding context (3 words before and after)
                    context = words[max(0, i-3):min(len(words), i+4)]
                    context_terms = set(context)
                    proximity_score += len(query_terms.intersection(context_terms)) / len(query_terms)
        
        return (term_overlap * 0.6 + min(1.0, proximity_score) * 0.4)

    def _has_exact_match(self, query: str, text: str) -> bool:
        """Check if query has an exact match in text."""
        query_words = query.split()
        text_words = text.split()
        
        # Check for exact phrase matches (2+ words)
        for i in range(len(text_words) - 1):
            phrase = " ".join(text_words[i:i+2])
            if phrase in query:
                return True
        
        return False

    async def generate_answer(self, question: str, search_results: List[Dict]) -> Dict:
        """
        Generate an answer using Gemini AI based on search results.
        
        Args:
            question: The question to answer
            search_results: List of relevant search results for context
            
        Returns:
            Dictionary containing answer and links in the exact format:
            {
                "answer": "string",
                "links": [
                    {
                        "url": "string",
                        "text": "string"
                    }
                ]
            }
        """
        try:
            print(f"\n=== DEBUG: Generate Answer ===")
            print(f"Question: {question}")
            print(f"Number of search results: {len(search_results)}")
            
            if not search_results:
                print("No search results found")
                return {
                    "answer": "I don't know the answer as I couldn't find any relevant information in the course materials.",
                    "links": []
                }

            # Format sources with explicit URLs and relevance scores
            sources_text = "\n\n".join([
                f"=== Source {i+1} (Relevance Score: {r['score']:.2f}) ===\n"
                f"Title: {r.get('title', 'No title')}\n"
                f"URL: {r['url']}\n"
                f"Author: {r.get('author', 'Unknown')} ({r.get('role', 'Student')})\n"
                f"Content:\n{r['content'][:500]}..."
                + (f"\n\nContext:\n" + "\n".join([
                    f"{'Parent post' if ctx.get('is_parent') else 'Reply'} by {ctx['author']}:\n{ctx['content'][:200]}..."
                    for ctx in r.get('context', [])
                ]) if r.get('context') else "")
                for i, r in enumerate(search_results[:5])
            ])

            # Updated prompt to emphasize handling contradictory statements and exact response format
            prompt = f"""You are a Teaching Assistant for the Tools in Data Science course at IIT Madras. Your task is to answer student questions using the provided course materials.

Question: {question}

Here are the relevant course materials or forum posts (sorted by relevance):

{sources_text}

CRITICAL INSTRUCTIONS:
1. You MUST use the provided course materials to answer the question
2. Pay special attention to responses from:
   - Course Instructors (@rajan.iitm, @dibujohn)
   - Teaching Assistants (@21f3001136, @sandeepstele)
3. For specific topics, prioritize:
   - Assignment/Project queries: TA responses
   - Quiz queries: @rajan.iitm's responses
   - ROE/Exam queries: @dibujohn's responses
4. Consider the FULL context of forum posts, including parent posts and replies
5. If a post is part of a conversation, consider the entire thread for context
6. Consider ALL provided sources, not just the highest scoring one
7. If multiple sources have relevant information, combine them in your answer
8. You MUST include the exact URLs from ALL relevant sources in your answer
9. For handling contradictory statements:
   - ALWAYS prioritize official course materials over forum posts
   - If forum posts contradict each other, prioritize in this order:
     1. Course Instructor responses (@rajan.iitm, @dibujohn)
     2. Teaching Assistant responses (@21f3001136, @sandeepstele)
     3. Student responses (only if no instructor/TA response exists)
   - If an instructor/TA later corrects or updates information, use their latest response
   - If there are multiple instructor/TA responses, use the most recent one
   - If contradictions exist between instructors/TAs, prioritize the course instructor's response
10. For questions about specific requirements or scores:
    - Extract and state the EXACT numbers/values mentioned
    - If there are multiple values, explain which one applies based on instructor/TA priority
    - If a value is not explicitly stated, say so clearly
    - When describing score calculations, use the EXACT format mentioned in the sources
    - Do not make assumptions about score formats not explicitly stated
11. For questions about tools or software:
    - State clearly whether something is allowed or not allowed
    - If alternatives are mentioned, list ALL options with their specific conditions
    - If there are restrictions, state them explicitly
    - When discussing alternatives, explain the exact implications of each choice
    - If students suggest alternatives not mentioned by instructors/TAs, note this explicitly
12. For questions about dates or deadlines:
    - State the EXACT date if available
    - If a date is not available, explain why (e.g., not yet announced)
    - If there are multiple dates, clarify which is which
    - If dates are updated, use the most recent instructor/TA announcement
13. Only say "I don't know" if NONE of the sources have relevant information
14. Format your response EXACTLY as follows:

Answer: [your answer using ALL relevant sources, being explicit about specific values, requirements, or restrictions]

Sources:
1. URL: [exact_url_1], Text: [brief quote or description with specific values]
2. URL: [exact_url_2], Text: [brief quote or description with specific values]
[Include ALL relevant sources, not just the highest scoring one]"""

            print("\nSending prompt to Gemini...")

            # Generate with very low temperature for consistency
            response = await asyncio.get_event_loop().run_in_executor(
                SEARCH_EXECUTOR,
                lambda: self.gemini.generate_content(
                    prompt,
                    generation_config={
                        "temperature": 0.1,
                        "top_p": 0.1,
                        "top_k": 1,
                        "max_output_tokens": 1000
                    }
                )
            )

            if response and hasattr(response, 'text'):
                answer_text = response.text.strip()
                print(f"\nGemini Response:\n{answer_text}")
                
                # Extract answer and sources
                answer_match = re.search(r'Answer:\s*(.*?)(?=\nSources:|$)', answer_text, re.DOTALL)
                sources_match = re.search(r'Sources:\s*(.*?)$', answer_text, re.DOTALL)
                
                if answer_match and sources_match:
                    answer = answer_match.group(1).strip()
                    sources_text = sources_match.group(1).strip()
                    
                    # Clean up the answer text
                    answer = re.sub(r'---.*?---', '', answer, flags=re.DOTALL)
                    answer = re.sub(r'\s+', ' ', answer).strip()
                    
                    # Extract links from sources
                    links = []
                    for line in sources_text.split('\n'):
                        if line.strip():
                            url_match = re.search(r'URL:\s*(https?://[^\s,]+)', line)
                            text_match = re.search(r'Text:\s*"([^"]+)"', line)
                            if url_match:
                                url = url_match.group(1)
                                text = text_match.group(1) if text_match else "Source"
                                # Only add if we haven't seen this URL
                                if not any(link["url"] == url for link in links):
                                    links.append({
                                        "url": url,
                                        "text": text
                                    })
                    
                    # If no links were found in the response, use search results
                    if not links:
                        for result in search_results[:3]:
                            if result.get("url"):
                                url = result["url"]
                                # For Docker content, use the exact URL
                                if "docker" in url.lower():
                                    url = "https://tds.s-anand.net/#/docker"
                                if not any(link["url"] == url for link in links):
                                    links.append({
                                        "url": url,
                                        "text": result.get("title", "Source")
                                    })
                    
                    return {
                        "answer": answer,
                        "links": links[:3]  # Limit to top 3 links
                    }
                
                # If we couldn't parse the response properly, construct a basic answer
                return {
                    "answer": answer_text,
                    "links": [
                        {
                            "url": result["url"],
                            "text": result.get("title", "Source")
                        }
                        for result in search_results[:3]
                        if result.get("url")
                    ]
                }

            # If no response from Gemini, construct a basic answer from search results
            if search_results:
                relevant_sources = [r for r in search_results if r["score"] > 0.3]
                if relevant_sources:
                    answer_parts = []
                    for source in relevant_sources[:3]:
                        content = re.sub(r'---.*?---', '', source["content"], flags=re.DOTALL)
                        content = re.sub(r'\s+', ' ', content).strip()
                        answer_parts.append(f"Source {source['url']} states: {content[:200]}...")
                    
                    return {
                        "answer": "Based on the course materials:\n\n" + "\n\n".join(answer_parts),
                        "links": [
                            {
                                "url": result["url"],
                                "text": result.get("title", "Source")
                            }
                            for result in relevant_sources[:3]
                            if result.get("url")
                        ]
                    }

            return {
                "answer": "I don't know the answer as I couldn't find any relevant information in the course materials.",
                "links": []
            }

        except Exception as e:
            print(f"Error generating answer: {str(e)}")
            return {
                "answer": "I don't know the answer as I encountered an error while processing your request.",
                "links": []
            }

@app.get("/disk-usage")
def get_disk_usage():
    total, used, free = shutil.disk_usage("/")
    return JSONResponse({
        "total_gb": round(total / (1024**3), 2),
        "used_gb": round(used / (1024**3), 2),
        "free_gb": round(free / (1024**3), 2),
    })

@app.post("/api/")
async def answer_question(
    request: Request,
    background_tasks: BackgroundTasks,
    image: Optional[UploadFile] = File(None)
):
    """
    Handle incoming questions and return answers with sources.
    Supports both text and image queries.
    """
    try:
        # Get request body
        body = await request.body()
        body_str = body.decode('utf-8')
        
        # Extract question - handle both JSON and string formats
        question = None
        
        # Try to parse as JSON first
        try:
            json_body = json.loads(body_str)
            if isinstance(json_body, dict) and "question" in json_body:
                question = json_body["question"]
            elif isinstance(json_body, str):
                question = json_body
        except json.JSONDecodeError:
            # If not JSON, try to extract from string
            if "{{prompt}}" in body_str:
                # This is a promptfoo template string
                question = body_str.replace("{{prompt}}", "").strip()
            else:
                # Try to extract question using patterns
                patterns = [
                    r'"question"\s*:\s*"([^"]+)"',  # JSON format
                    r'question=([^&]+)',  # URL encoded
                    r'question:\s*([^\n]+)',  # Plain text
                ]
                for pattern in patterns:
                    matches = re.findall(pattern, body_str)
                    if matches:
                        question = matches[0].strip()
                        break
        
        # Process image if provided
        image_data = None
        if image:
            image_data = await image.read()
        
        if not question and not image_data:
            return JSONResponse(
                status_code=400,
                content={
                    "answer": "Please provide either a question or an image.",
                    "links": []
                }
            )
        
        # Process the request
        try:
            # Search and generate answer concurrently
            search_task = search_engine.search(question, image_data, top_k=3, min_score=0.2)
            search_results = await search_task
            
            if not search_results:
                return JSONResponse(content={
                    "answer": "I don't know the answer as I couldn't find any relevant information in the course materials.",
                    "links": []
                })
            
            response = await search_engine.generate_answer(question or "What is in this image?", search_results)
            
            # Format response
            formatted_response = {
                "answer": response["answer"].strip(),
                "links": [
                    {
                        "url": str(link.get("url", "")).strip(),
                        "text": str(link.get("text", "Source")).strip()
                    }
                    for link in response["links"]
                    if link.get("url")
                ]
            }
            
            return JSONResponse(content=formatted_response)
            
        except Exception as e:
            raise
    
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={
                "answer": f"I encountered an error while processing your request: {str(e)}",
                "links": []
            }
        )

if __name__ == "__main__":
    import uvicorn
    print("Starting TDS Virtual TA API server...")
    uvicorn.run(
        "api:app",
        host="0.0.0.0",
        port=8000,
        log_level="info",  # We handle our own logging
        workers=1,  # Use single worker for easier logging
        loop="uvloop",
        limit_concurrency=100
    )
