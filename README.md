# TDS Virtual Teaching Assistant

A virtual teaching assistant for the Tools in Data Science (TDS) course at IIT Madras, powered by Pinecone vector search and Google's Gemini AI.

## Overview

This project implements an AI-powered teaching assistant that can answer student questions by searching through course materials and forum discussions. It uses semantic search with Pinecone's vector database and Gemini AI for natural language understanding and response generation.

## Features

- **Semantic Search**: Uses Pinecone's `llama-text-embed-v2-index` for high-quality semantic search over course content
- **Multi-Source Knowledge**: Combines information from:
  - Course website materials
  - Forum discussions
  - Student-TA interactions
- **Smart Answer Generation**: Uses Gemini AI to generate contextual, accurate responses
- **Source Attribution**: Provides links to original sources for verification
- **Image Support**: Can process images in questions using Gemini's vision capabilities

## Search Architecture

The system employs a sophisticated multi-type search strategy that combines vector similarity with dynamic keyword importance:

1. **Vector Similarity** (60% weight):

   - Uses Pinecone's `llama-text-embed-v2` model for semantic similarity
   - Searches across two namespaces: course materials and forum posts
   - Initial filtering of results with scores < 0.1

2. **Dynamic Keyword Importance** (25-30% weight):

   - Uses TF-IDF (Term Frequency-Inverse Document Frequency) to identify important terms
   - Automatically learns word importance from the document collection
   - Boosts scores for:
     - Technical terms (identified by IDF scores)
     - Words containing uppercase (e.g., API, GA4)
     - Words containing numbers or special characters
   - Caches word importance calculations for efficiency

3. **Term Overlap** (10-15% weight):

   - Measures intersection between query terms and content
   - Different weights for title (0.4-0.5) and content (0.8-0.9)
   - Higher weights for forum posts vs. course materials

4. **Forum Context Enhancement**:
   - Includes parent and child posts for forum results
   - Maintains conversation context
   - Considers author roles (instructor/TA/student)

The system dynamically learns what terms are important based on their usage patterns in the course materials and forum posts, rather than relying on predefined rules or stop words. This allows it to:

- Automatically identify technical and course-specific terminology
- Adapt to new terminology as it appears in the course
- Reduce the impact of common, non-informative words
- Cache frequent calculations for better performance

The final score is a weighted combination of these factors, with results below the minimum threshold (default: 0.15) being filtered out. Results are then sorted by final score and returned with their original sources.

## Project Structure

```
├── api.py                 # Main FastAPI server implementation
├── create_vectors.py      # Script to create and update vector embeddings
├── scraper_website.py     # Scraper for course website content
├── scraper_forum.py       # Scraper for forum discussions
├── requirements.txt       # Python dependencies
├── vector_store/         # Directory for vector store metadata
├── tds_pages_md/        # Scraped markdown content
```

## Key Components

### 1. Data Collection (`scraper_*.py`)

- `scraper_website.py`: Scrapes the TDS course website using Playwright
- `scraper_forum.py`: Scrapes the course forum using the Discourse API
- Saves content in structured formats (Markdown/JSON) for processing

### 2. Vector Creation (`create_vectors.py`)

- Processes scraped content
- Creates embeddings using Pinecone's integrated inference
- Handles metadata size limits and batch processing
- Implements retry logic for failed uploads

### 3. API Server (`api.py`)

- FastAPI server with the following endpoints:
  - `/api`: Main endpoint for question answering
  - `/disk-usage`: System monitoring endpoint
- Features:
  - Concurrent request handling
  - Image processing support
  - Smart context aggregation
  - Source prioritization based on author roles

## Setup

1. Clone the repository
2. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

3. Set up environment variables in `.env`:

   ```
   PINECONE_API_KEY=your_key_here
   GOOGLE_API_KEY=your_key_here
   _t=your_discourse_token  # For forum access
   ```

4. Create the Pinecone index:

   - Index name: `llama-text-embed-v2-index`
   - Model: `llama-text-embed-v2`
   - Metric: `cosine`

5. Run the scrapers:

   ```bash
   python scraper_website.py
   python scraper_forum.py
   ```

6. Create vectors:

   ```bash
   python create_vectors.py
   ```

7. Start the API server:
   ```bash
   uvicorn api:app --host 0.0.0.0 --port 8000
   ```

## API Usage

Send questions to the `/api` endpoint:

```bash
curl -X POST http://localhost:8000/api \
  -H "Content-Type: application/json" \
  -d '{"question": "What is the deadline for GA4?"}'
```

For image questions, use multipart/form-data:

```bash
curl -X POST http://localhost:8000/api \
  -F "question=What's wrong with this code?" \
  -F "image=@screenshot.png"
```

## Response Format

```json
{
  "answer": "Based on the course materials...",
  "links": [
    {
      "url": "https://discourse.onlinedegree.iitm.ac.in/t/...",
      "text": "Source Title"
    }
  ]
}
```

## Deployment

The project is deployment on Render so it may take few seconds to start up.

## License

See the [LICENSE](LICENSE) file for details.
