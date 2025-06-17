"""
TDS Virtual TA API Server

This module implements a FastAPI server that provides a virtual teaching assistant
for the Tools in Data Science course. It uses Gemini AI for generating answers
and maintains a vector store of course content and forum posts for context.
"""

import os
import json
from typing import List, Dict, Optional
from fastapi import FastAPI, Request, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pinecone import Pinecone
import pickle
import google.generativeai as genai
from dotenv import load_dotenv
import re
import shutil
import time
from PIL import Image
from io import BytesIO
import asyncio
import base64

# Load environment variables
load_dotenv()

# Constants
BASE_URL = "https://discourse.onlinedegree.iitm.ac.in"
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
INDEX_NAME = "llama-text-embed-v2-index"
REQUEST_SEMAPHORE = asyncio.Semaphore(1)  # Limit based on environment

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
        
        # Initialize Pinecone client
        print("Initializing Pinecone client...")
        self.pc = Pinecone(api_key=PINECONE_API_KEY)
        self.index = self.pc.Index(INDEX_NAME)
        
        # Debug: Check index stats
        try:
            stats = self.index.describe_index_stats()
            print(f"Index stats: {stats}")
            if 'namespaces' in stats:
                print(f"Available namespaces: {list(stats['namespaces'].keys())}")
        except Exception as e:
            print(f"Error getting index stats: {e}")
        
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
                # Create file path from filename
                file_path = os.path.join("tds_pages_md", meta["filename"])
                with open(file_path, "r", encoding="utf-8") as f:
                    self.md_content[meta["filename"]] = f.read()
            except Exception as e:
                print(f"Error loading file {file_path}: {str(e)}")
        
        # Initialize Gemini AI model
        print("Initializing Gemini model...")
        GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
        if not GOOGLE_API_KEY:
            raise ValueError("GOOGLE_API_KEY environment variable not set")
        genai.configure(api_key=GOOGLE_API_KEY)
        self.gemini = genai.GenerativeModel('gemini-1.5-flash')
        
        print("SearchEngine initialization complete")

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

    async def process_image_with_gemini(self, image_data: bytes) -> str:
        """Use Gemini to generate a textual description of the image."""
        try:
            # Convert bytes to PIL Image
            image = Image.open(BytesIO(image_data))
            
            # Create prompt for image description
            prompt = """
            Please describe this image in detail, focusing on:
            - What you see in the image
            - Any text, diagrams, or visual elements
            - The context or subject matter
            - Any relevant technical or educational content
            
            Provide a clear, descriptive text that could be used to search for related information.
            Keep the description concise but comprehensive.
            """
            
            # Generate description using Gemini
            response = self.gemini.generate_content([prompt, image])
            
            if response and hasattr(response, 'text'):
                description = response.text.strip()
                return description
            else:
                return "Unable to process image"
                
        except Exception as e:
            return "Error processing image"

    async def search(self, query: str, image_data: Optional[bytes] = None, top_k: int = 5, min_score: float = 0.15) -> List[Dict]:
        """Search for relevant content using text and/or image data."""
        try:
            # If image is provided, get description from Gemini
            if image_data:
                image_description = await self.process_image_with_gemini(image_data)
                
                # Combine question with image description
                if query:
                    combined_query = f"{query} Image shows: {image_description}"
                else:
                    combined_query = f"Image shows: {image_description}"
            else:
                combined_query = query
            
            # Extract key terms from query
            query_terms = set(combined_query.lower().split())
            important_terms = {term for term in query_terms if len(term) > 3}

            # Search in both namespaces
            async def search_namespace(namespace: str, metadata_list: List[Dict], content_dict=None, is_forum=False):
                try:
                    # Query Pinecone with the combined query using the correct API format
                    print(f"Querying Pinecone with: {combined_query[:100]}...")
                    print(f"Namespace: {namespace}")
                    
                    # Use the correct Pinecone API format for semantic search
                    results = self.index.search(
                        namespace=namespace,
                        query={
                            "top_k": min(top_k * 3, 100),
                            "inputs": {
                                'text': combined_query
                            }
                        }
                    )
                    
                    processed_results = []
                    # Handle the new results structure
                    if 'result' in results and 'hits' in results['result']:
                        matches = results['result']['hits']
                    else:
                        matches = []
                    
                    for match in matches:
                        # Handle the new match structure
                        score = match.get('_score', 0)
                        match_id = match.get('_id', '')
                        
                        if score < 0.1:  # Filter low scores
                            continue
                        
                        # Find corresponding metadata
                        meta = next((m for m in metadata_list if m["id"] == match_id), None)
                        if not meta:
                            continue
                        
                        title = meta.get("title", "").lower()
                        content = content_dict.get(meta["filename"], "") if content_dict else meta.get("content", "").lower()
                        url = meta.get("url", "")
                        
                        if is_forum:
                            url = self._normalize_url(url, meta.get("post_number"))
                        
                        # Calculate additional scores
                        phrase_score = self._calculate_phrase_match_score(combined_query, title, content)
                        context_score = self._calculate_context_score(combined_query, title, content)
                        
                        # Calculate term overlap
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
                                parent_meta = next((m for m in metadata_list if m.get("post_number") == post_number-1 and m.get("topic_id") == topic_id), None)
                                if parent_meta:
                                    parent_content = parent_meta.get("content", "").lower()
                                    parent_url = self._normalize_url(parent_meta.get("url", ""), post_number-1)
                                    context.append({
                                        "content": parent_content,
                                        "author": parent_meta.get("author", ""),
                                        "url": parent_url,
                                        "is_parent": True
                                    })
                            
                            # Get next post if exists
                            next_meta = next((m for m in metadata_list if m.get("post_number") == post_number+1 and m.get("topic_id") == topic_id), None)
                            if next_meta:
                                next_content = next_meta.get("content", "").lower()
                                next_url = self._normalize_url(next_meta.get("url", ""), post_number+1)
                                context.append({
                                    "content": next_content,
                                    "author": next_meta.get("author", ""),
                                    "url": next_url,
                                    "is_parent": False
                                })
                        
                        # Combine scores with higher weight for forum posts
                        final_score = (
                            score * 0.6 +  # Pinecone similarity score
                            phrase_score * (0.2 if not is_forum else 0.25) +
                            context_score * 0.05 +
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
                        
                        processed_results.append(result)
                    
                    return processed_results
                    
                except Exception as e:
                    print(f"Error searching namespace {namespace}: {str(e)}")
                    return []

            # Search both namespaces
            md_results = await search_namespace("markdown-content", self.md_metadata, self.md_content, False)
            json_results = await search_namespace("forum-posts", self.json_metadata, None, True)
            
            # Debug: Print search results
            print(f"Markdown results: {len(md_results)}")
            print(f"Forum results: {len(json_results)}")
            if md_results:
                print(f"Top markdown result: {md_results[0].get('title', 'No title')} - Score: {md_results[0].get('score', 0)}")
            if json_results:
                print(f"Top forum result: {json_results[0].get('title', 'No title')} - Score: {json_results[0].get('score', 0)}")
            
            # Combine and sort results
            all_results = []
            all_results.extend(md_results)
            all_results.extend(json_results)
            all_results.sort(key=lambda x: x["score"], reverse=True)
            
            print(f"Total combined results: {len(all_results)}")
            if all_results:
                print(f"Best overall result: {all_results[0].get('title', 'No title')} - Score: {all_results[0].get('score', 0)}")
            
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
            if not search_results:
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
   - Course Instructors (@iamprasna, @carlton)
   - Teaching Assistants (@Jivraj @HritikRoshan_HRM)
3. For specific topics, prioritize:
   - Assignment/Project queries: TA responses
   - Quiz queries: @iamprasna's responses
   - ROE/Exam queries: @carlton's responses
4. Consider the FULL context of forum posts, including parent posts and replies
5. If a post is part of a conversation, consider the entire thread for context
6. Consider ALL provided sources, not just the highest scoring one
7. If multiple sources have relevant information, combine them in your answer
8. You MUST include the exact URLs from ALL relevant sources in your answer
9. For handling contradictory statements:
   - ALWAYS prioritize official course materials over forum posts
   - If forum posts contradict each other, prioritize in this order:
     1. Course Instructor responses (@iamprasna, @carlton)
     2. Teaching Assistant responses (@Jivraj @HritikRoshan_HRM)
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

            # Generate with very low temperature for consistency
            response = self.gemini.generate_content(
            prompt,
            generation_config={
                "temperature": 0.1,
                "top_p": 0.1,
                "top_k": 1,
                    "max_output_tokens": 1000
            }
            )

            if response and hasattr(response, 'text'):
                answer_text = response.text.strip()
                
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
            return {
                "answer": "I don't know the answer as I encountered an error while processing your request.",
                "links": []
            }

@app.get("/")
async def root():
    return {"message": "TDS Virtual TA API is running", "endpoints": ["/api", "/disk-usage"]}

@app.get("/disk-usage")
def get_disk_usage():
    total, used, free = shutil.disk_usage("/")
    return JSONResponse({
        "total_gb": round(total / (1024**3), 2),
        "used_gb": round(used / (1024**3), 2),
        "free_gb": round(free / (1024**3), 2),
    })

@app.post("/api")
async def answer_question(
    request: Request,
    image: Optional[UploadFile] = File(None)
):
    """
    Handle incoming questions and return answers with sources.
    Supports both text and image queries.
    """
    try:
        # Get form data first
        form_data = await request.form()
        
        # Extract question from form data or JSON body
        question = None
        image_path = None
        
        # Try form data first
        if "question" in form_data:
            question = form_data["question"]
        else:
            # If not in form data, try JSON body
            try:
                body = await request.json()
                if isinstance(body, dict):
                    if "question" in body:
                        question = body["question"]
                    if "image" in body:
                        image_path = body["image"]
                elif isinstance(body, str):
                    question = body
            except json.JSONDecodeError:
                # If not JSON, try to extract from raw body
                body = (await request.body()).decode('utf-8')
                if "{{prompt}}" in body:
                # This is a promptfoo template string
                    question = body.replace("{{prompt}}", "").strip()
                else:
                    # Try to extract question using patterns
                    patterns = [
                        r'"question"\s*:\s*"([^"]+)"',  # JSON format
                        r'question=([^&]+)',  # URL encoded
                        r'question:\s*([^\n]+)',  # Plain text
                    ]
                    for pattern in patterns:
                        matches = re.findall(pattern, body)
                        if matches:
                            question = matches[0].strip()
                            break
        
        # Process image if provided
        image_data = None
        if image:
            # Image uploaded via multipart form
            image_data = await image.read()
        elif image_path:
            # Image path provided in JSON body
            try:
                # Check if it's a base64-encoded image
                if image_path.startswith("data:image/") or len(image_path) > 1000:
                    # This looks like a base64-encoded image
                    
                    # Remove data URL prefix if present
                    if image_path.startswith("data:image/"):
                        # Extract base64 part after comma
                        image_path = image_path.split(",", 1)[1]
                    
                    # Decode base64 to bytes
                    image_data = base64.b64decode(image_path)
                else:
                    # Remove file:// prefix if present
                    if image_path.startswith("file://"):
                        image_path = image_path[7:]
                    
                    with open(image_path, 'rb') as f:
                        image_data = f.read()
            except Exception as e:
                image_data = None
        
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
            # Limit concurrent requests to prevent resource exhaustion
            async with REQUEST_SEMAPHORE:
                start_time = time.time()
                
                # Search and generate answer concurrently
                search_task = search_engine.search(question, image_data, top_k=3, min_score=0.15)
                search_results = await search_task
                search_time = time.time() - start_time
                
                if not search_results:
                    return JSONResponse(content={
                        "answer": "I don't know the answer as I couldn't find any relevant information in the course materials.",
                        "links": []
                    })
                
                answer_start = time.time()
                response = await search_engine.generate_answer(question or "What is in this image?", search_results)
                answer_time = time.time() - answer_start
                total_time = time.time() - start_time
                
                print(f"Request timing - Search: {search_time:.2f}s, Answer: {answer_time:.2f}s, Total: {total_time:.2f}s")
                
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

@app.post("/api/")
async def answer_question_with_slash(
    request: Request,
    image: Optional[UploadFile] = File(None)
):
    """Handle requests to /api/ (with trailing slash) to prevent redirects."""
    return await answer_question(request, image)

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
