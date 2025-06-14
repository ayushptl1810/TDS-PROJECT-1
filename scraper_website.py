"""
Website Scraper for TDS Course Content

This script scrapes the Tools in Data Science course website and saves
the content as markdown files for later processing.
"""

import os
import json
import re
from datetime import datetime
from urllib.parse import urljoin
from markdownify import markdownify as md
from playwright.sync_api import sync_playwright
import requests
from bs4 import BeautifulSoup
import time

BASE_URL = "https://tds.s-anand.net/#/2025-01/"
BASE_ORIGIN = "https://tds.s-anand.net"
OUTPUT_DIR = "tds_pages_md"
METADATA_FILE = "metadata.json"

visited = set()
metadata = []

def sanitize_filename(title):
    return re.sub(r'[\\/*?:"<>|]', "_", title).strip().replace(" ", "_")

def extract_all_internal_links(page):
    links = page.eval_on_selector_all("a[href]", "els => els.map(el => el.href)")
    return list(set(
        link for link in links
        if BASE_ORIGIN in link and '/#/' in link
    ))

def wait_for_article_and_get_html(page):
    page.wait_for_selector("article.markdown-section#main", timeout=10000)
    return page.inner_html("article.markdown-section#main")

def crawl_page(page, url):
    if url in visited:
        return
    visited.add(url)

    print(f"📄 Visiting: {url}")
    try:
        page.goto(url, wait_until="domcontentloaded")
        page.wait_for_timeout(1000)
        html = wait_for_article_and_get_html(page)
    except Exception as e:
        print(f"❌ Error loading page: {url}\n{e}")
        return

    # Extract title and save markdown
    title = page.title().split(" - ")[0].strip() or f"page_{len(visited)}"
    filename = sanitize_filename(title)
    filepath = os.path.join(OUTPUT_DIR, f"{filename}.md")

    markdown = md(html)
    with open(filepath, "w", encoding="utf-8") as f:
        f.write(f"---\n")
        f.write(f"title: \"{title}\"\n")
        f.write(f"original_url: \"{url}\"\n")
        f.write(f"downloaded_at: \"{datetime.now().isoformat()}\"\n")
        f.write(f"---\n\n")
        f.write(markdown)

    metadata.append({
        "title": title,
        "filename": f"{filename}.md",
        "original_url": url,
        "downloaded_at": datetime.now().isoformat()
    })

    # Recursively crawl all links found on the page
    links = extract_all_internal_links(page)
    for link in links:
        if link not in visited:
            crawl_page(page, link)

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    global visited, metadata

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        context = browser.new_context()
        page = context.new_page()
        crawl_page(page, BASE_URL)

        with open(METADATA_FILE, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)

        print(f"\n✅ Completed. {len(metadata)} pages saved.")
        browser.close()

def clean_text(text: str) -> str:
    """
    Clean and format text content from HTML.
    
    Args:
        text: Raw text content from HTML
        
    Returns:
        Cleaned and formatted text string
    """
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text)
    # Remove special characters
    text = re.sub(r'[^\w\s\-.,;:!?()\[\]{}]', '', text)
    return text.strip()

def extract_content(url: str) -> tuple[str, str]:
    """
    Extract content from a webpage.
    
    Args:
        url: URL of the webpage to scrape
        
    Returns:
        Tuple of (title, content) where content is in markdown format
    """
    try:
        # Add delay to be respectful to the server
        time.sleep(1)
        
        # Fetch the webpage
        response = requests.get(url)
        response.raise_for_status()
        
        # Parse HTML
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Extract title
        title = soup.find('h1').text.strip() if soup.find('h1') else "Untitled"
        
        # Extract main content
        content = []
        main_content = soup.find('div', class_='content')
        
        if main_content:
            # Process each element in the content
            for element in main_content.find_all(['h1', 'h2', 'h3', 'p', 'ul', 'ol', 'pre', 'code']):
                if element.name in ['h1', 'h2', 'h3']:
                    # Add markdown headers
                    level = int(element.name[1])
                    content.append(f"{'#' * level} {element.text.strip()}\n")
                elif element.name == 'p':
                    # Add paragraphs
                    content.append(f"{clean_text(element.text)}\n")
                elif element.name in ['ul', 'ol']:
                    # Add lists
                    for item in element.find_all('li'):
                        content.append(f"- {clean_text(item.text)}\n")
                elif element.name in ['pre', 'code']:
                    # Add code blocks
                    content.append(f"```\n{element.text.strip()}\n```\n")
        
        return title, '\n'.join(content)
    
    except Exception as e:
        print(f"Error extracting content from {url}: {str(e)}")
        return "Error", f"Failed to extract content: {str(e)}"

def scrape_website(base_url: str, output_dir: str) -> None:
    """
    Scrape the TDS course website and save content as markdown files.
    
    Args:
        base_url: Base URL of the course website
        output_dir: Directory to save markdown files
    """
    try:
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Fetch the main page
        response = requests.get(base_url)
        response.raise_for_status()
        
        # Parse HTML
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Find all links to course content
        links = []
        for link in soup.find_all('a'):
            href = link.get('href')
            if href and not href.startswith(('http', '#')):
                full_url = urljoin(base_url, href)
                if full_url.startswith(base_url):
                    links.append(full_url)
        
        # Remove duplicates while preserving order
        links = list(dict.fromkeys(links))
        
        # Process each link
        for url in links:
            try:
                # Extract content
                title, content = extract_content(url)
                
                # Create filename from URL
                filename = url.split('/')[-1]
                if not filename.endswith('.md'):
                    filename = f"{filename}.md"
                
                # Save content
                filepath = os.path.join(output_dir, filename)
                with open(filepath, 'w', encoding='utf-8') as f:
                    f.write(f"# {title}\n\n")
                    f.write(content)
                
                print(f"Saved {filename}")
                
            except Exception as e:
                print(f"Error processing {url}: {str(e)}")
                continue
        
        print("Scraping complete!")
        
    except Exception as e:
        print(f"Error scraping website: {str(e)}")

if __name__ == "__main__":
    main()