import requests
import time
from typing import List, Dict, Any, Optional
from googlesearch import search
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

from models.schemas import DocumentChunk
from config.settings import settings
from utils.logger import get_logger

logger = get_logger(__name__)

class WebScraper:
    """Web scraper for educational content from Google search"""

    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        })
        self.driver = None

    def search_educational_content(self, topic: str, subtopic: str = "",
                                 max_results: int = None) -> List[DocumentChunk]:
        """Search for educational content related to the topic"""
        try:
            max_results = max_results or settings.MAX_GOOGLE_RESULTS
            logger.info(f"Searching for educational content: {topic} {subtopic}")

            # Build search query
            search_query = self._build_search_query(topic, subtopic)
            logger.info(f"Search query: {search_query}")

            # Get URLs from Google search
            urls = self._google_search(search_query, max_results)

            if not urls:
                logger.warning("No URLs found in search")
                return []

            # Scrape content from URLs
            chunks = []
            for i, url in enumerate(urls):
                try:
                    logger.info(f"Scraping URL {i+1}/{len(urls)}: {url[:100]}...")

                    content = self._scrape_url_content(url)
                    if content:
                        chunk = self._create_chunk_from_content(content, url, topic, i)
                        chunks.append(chunk)

                    # Respectful delay
                    time.sleep(settings.WEB_SCRAPING_DELAY)

                except Exception as e:
                    logger.error(f"Error scraping URL {url}: {e}")
                    continue

            logger.info(f"Successfully scraped {len(chunks)} chunks from web")
            return chunks

        except Exception as e:
            logger.error(f"Error in web scraping: {e}")
            return []

    def _build_search_query(self, topic: str, subtopic: str) -> str:
        """Build optimized search query for educational content"""
        base_query = f"{topic}"
        if subtopic:
            base_query += f" {subtopic}"

        # Add educational context keywords
        educational_terms = [
            "pembelajaran", "modul ajar", "materi pelajaran", "kurikulum",
            "pendidikan", "siswa", "guru", "sekolah"
        ]

        # Indonesian educational sites
        site_filters = [
            "site:kemdikbud.go.id", "site:guruberbagi.kemdikbud.go.id",
            "site:ruangguru.com", "site:zenius.net", "site:quipper.com"
        ]

        # Combine query with educational terms
        query = f"{base_query} {' OR '.join(educational_terms[:3])}"

        # Add some site filters (not all to avoid being too restrictive)
        if len(site_filters) > 0:
            query += f" OR {site_filters[0]}"

        return query

    def _google_search(self, query: str, max_results: int) -> List[str]:
        """Perform Google search and return URLs"""
        try:
            urls = []

            # Use googlesearch library
            for url in search(query, num_results=max_results, lang='id', safe='active'):
                urls.append(url)
                if len(urls) >= max_results:
                    break

            # Filter out non-educational or problematic URLs
            filtered_urls = []
            for url in urls:
                if self._is_valid_educational_url(url):
                    filtered_urls.append(url)

            logger.info(f"Found {len(filtered_urls)} valid URLs from {len(urls)} total")
            return filtered_urls

        except Exception as e:
            logger.error(f"Error in Google search: {e}")
            return []

    def _is_valid_educational_url(self, url: str) -> bool:
        """Check if URL is suitable for educational content scraping"""
        # Educational domains
        educational_domains = [
            'kemdikbud.go.id', 'guruberbagi.kemdikbud.go.id', 'ruangguru.com',
            'zenius.net', 'quipper.com', 'brainly.co.id', 'kompas.com',
            'wikipedia.org', 'id.wikipedia.org', 'edu'
        ]

        # Blocked domains/patterns
        blocked_patterns = [
            'youtube.com', 'facebook.com', 'instagram.com', 'twitter.com',
            'tiktok.com', 'pinterest.com', 'pdf', 'login', 'register'
        ]

        url_lower = url.lower()

        # Check if it's an educational domain
        is_educational = any(domain in url_lower for domain in educational_domains)

        # Check if it contains blocked patterns
        is_blocked = any(pattern in url_lower for pattern in blocked_patterns)

        # Allow educational domains or general web content that's not blocked
        return is_educational or not is_blocked

    def _scrape_url_content(self, url: str) -> Optional[str]:
        """Scrape content from a single URL"""
        try:
            # Try simple requests first
            response = self.session.get(url, timeout=10)
            response.raise_for_status()

            # Parse with BeautifulSoup
            soup = BeautifulSoup(response.content, 'html.parser')

            # Remove script and style elements
            for script in soup(["script", "style", "nav", "footer", "header"]):
                script.decompose()

            # Extract main content
            content = self._extract_main_content(soup)

            if content and len(content.strip()) > 100:  # Minimum content length
                return content.strip()

            # If simple requests didn't work well, try Selenium (if needed)
            return self._scrape_with_selenium(url)

        except Exception as e:
            logger.error(f"Error scraping URL {url}: {e}")
            return None

    def _extract_main_content(self, soup: BeautifulSoup) -> str:
        """Extract main content from HTML soup"""
        content_parts = []

        # Try different content selectors
        content_selectors = [
            'main', 'article', '.content', '.main-content',
            '.post-content', '.entry-content', '.article-content'
        ]

        # Find main content area
        main_content = None
        for selector in content_selectors:
            main_content = soup.select_one(selector)
            if main_content:
                break

        # If no specific content area found, use body
        if not main_content:
            main_content = soup.find('body')

        if main_content:
            # Extract paragraphs and headings
            for element in main_content.find_all(['p', 'h1', 'h2', 'h3', 'h4', 'li']):
                text = element.get_text().strip()
                if text and len(text) > 20:  # Filter out very short text
                    content_parts.append(text)

        return '\n\n'.join(content_parts)

    def _scrape_with_selenium(self, url: str) -> Optional[str]:
        """Scrape URL using Selenium for JavaScript-heavy sites"""
        try:
            if not self.driver:
                self._init_selenium_driver()

            if not self.driver:
                return None

            self.driver.get(url)

            # Wait for content to load
            WebDriverWait(self.driver, 10).until(
                EC.presence_of_element_located((By.TAG_NAME, "body"))
            )

            # Extract text content
            body = self.driver.find_element(By.TAG_NAME, "body")
            text_content = body.text

            if len(text_content.strip()) > 100:
                return text_content.strip()

            return None

        except Exception as e:
            logger.error(f"Selenium scraping error for {url}: {e}")
            return None

    def _init_selenium_driver(self):
        """Initialize Selenium WebDriver"""
        try:
            chrome_options = Options()
            chrome_options.add_argument("--headless")  # Run headless
            chrome_options.add_argument("--no-sandbox")
            chrome_options.add_argument("--disable-dev-shm-usage")
            chrome_options.add_argument("--disable-gpu")
            chrome_options.add_argument("--window-size=1920,1080")
            chrome_options.add_argument("--user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36")

            self.driver = webdriver.Chrome(options=chrome_options)
            logger.info("Selenium WebDriver initialized")

        except Exception as e:
            logger.error(f"Failed to initialize Selenium: {e}")
            logger.info("Continuing without Selenium support")
            self.driver = None

    def _create_chunk_from_content(self, content: str, url: str, topic: str, index: int) -> DocumentChunk:
        """Create DocumentChunk from scraped content"""
        # Clean and truncate content
        clean_content = self._clean_scraped_content(content)

        # Create chunk ID
        chunk_id = f"web_{topic.replace(' ', '_')}_{index:03d}"

        chunk = DocumentChunk(
            chunk_id=chunk_id,
            content=clean_content,
            source_file=url,
            metadata={
                "source": "web_scraping",
                "url": url,
                "topic": topic,
                "scrape_index": index,
                "content_length": len(clean_content),
                "domain": self._extract_domain(url)
            }
        )

        return chunk

    def _clean_scraped_content(self, content: str) -> str:
        """Clean scraped content"""
        # Remove extra whitespace
        lines = [line.strip() for line in content.split('\n') if line.strip()]

        # Remove very short lines (likely navigation or ads)
        meaningful_lines = [line for line in lines if len(line) > 15]

        # Join with double newlines for readability
        cleaned_content = '\n\n'.join(meaningful_lines)

        # Truncate if too long (to avoid token limits)
        if len(cleaned_content) > 2000:
            cleaned_content = cleaned_content[:2000] + "..."

        return cleaned_content

    def _extract_domain(self, url: str) -> str:
        """Extract domain from URL"""
        try:
            from urllib.parse import urlparse
            parsed = urlparse(url)
            return parsed.netloc
        except:
            return "unknown"

    def close(self):
        """Close resources"""
        if self.driver:
            try:
                self.driver.quit()
            except:
                pass

        if self.session:
            self.session.close()

class ContentEnhancer:
    """Enhance local content with web-scraped information"""

    def __init__(self):
        self.web_scraper = WebScraper()

    def enhance_chunks(self, local_chunks: List[DocumentChunk],
                      topic: str, subtopic: str = "") -> List[DocumentChunk]:
        """Enhance local chunks with additional web content"""
        try:
            logger.info(f"Enhancing content for topic: {topic}")

            # Determine if we need additional content
            if self._sufficient_local_content(local_chunks, topic):
                logger.info("Sufficient local content available, skipping web search")
                return local_chunks

            # Search for additional content
            web_chunks = self.web_scraper.search_educational_content(
                topic, subtopic, max_results=5
            )

            # Combine local and web chunks
            enhanced_chunks = local_chunks + web_chunks

            logger.info(f"Enhanced with {len(web_chunks)} web chunks. Total: {len(enhanced_chunks)}")
            return enhanced_chunks

        except Exception as e:
            logger.error(f"Error enhancing content: {e}")
            return local_chunks  # Return original chunks on error
        finally:
            self.web_scraper.close()

    def _sufficient_local_content(self, chunks: List[DocumentChunk], topic: str) -> bool:
        """Check if local content is sufficient"""
        if len(chunks) < 3:  # Need at least 3 chunks
            return False

        # Check topic relevance
        topic_lower = topic.lower()
        relevant_chunks = 0

        for chunk in chunks:
            chunk_lower = chunk.content.lower()
            if any(word in chunk_lower for word in topic_lower.split()):
                relevant_chunks += 1

        # Need at least 50% relevance
        relevance_ratio = relevant_chunks / len(chunks)
        return relevance_ratio >= 0.5 and len(chunks) >= 5
