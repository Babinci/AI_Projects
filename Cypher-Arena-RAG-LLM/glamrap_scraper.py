import requests
from bs4 import BeautifulSoup
import json
from datetime import datetime
import time
from urllib.parse import urljoin
from typing import Dict, Optional, List, Set
import os
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from tqdm import tqdm
class GlamRapScraper:
    def __init__(self, base_url: str = "https://glamrap.pl"):
        self.base_url = base_url
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        })

    def get_article_content(self, url: str) -> Optional[Dict]:
        """
        Scrape a single article page and extract relevant content.
        
        Args:
            url: The URL of the article to scrape
            
        Returns:
            Dictionary containing the article's content or None if extraction fails
        """
        try:
            time.sleep(2)
            
            response = self.session.get(url)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.text, 'html.parser')
            
            article_data = {
                'url': url,
                'scrape_timestamp': datetime.now().isoformat()
            }
            
            # Find main article content within the first content-main div
            article = soup.find('article')
            if not article:
                print(f"Could not find article content for {url}")
                return None
                
            # Extract title
            title = article.find('h1')
            article_data['title'] = title.text.strip() if title else None
            
            # Extract date
            date = article.find('time')
            article_data['published_date'] = date.get('datetime') if date else None
            
            # Extract main content from the first content section
            content_main = article.find('div', {'id': 'mvp-content-main'})
            if content_main:
                # Get all paragraphs up to the first ad or social media section
                content_parts = []
                for p in content_main.find_all('p'):
                    # Stop if we hit the paragraph about theLAKE
                    if 'theLAKE' in p.text:
                        content_parts.append(p.text.strip())
                        break
                    content_parts.append(p.text.strip())
                
                article_data['content'] = '\n'.join(filter(None, content_parts))
            
            # Extract tags
            tags = article.find_all('a', {'rel': 'tag'})
            article_data['tags'] = [tag.text.strip() for tag in tags if tag.text]
            
            return article_data
            
        except requests.RequestException as e:
            print(f"Error fetching {url}: {str(e)}")
            return None
        except Exception as e:
            print(f"Error processing {url}: {str(e)}")
            return None
        

# ###example usage 
# if __name__ == "__main__":
#     scraper = GlamRapScraper()
#     # test_url = "https://glamrap.pl/bonus-rpk-nagral-piosenke-jak-zadbac-o-forme-w-wiezieniu/"
#     article = scraper.get_article_content(test_url)

def scrape_and_save_articles(input_file: str, output_file: str):
    """
    Scrape articles from URLs in input JSON file and save results continuously to output JSON file.
    
    Args:
        input_file: Path to JSON file containing article URLs
        output_file: Path where to save the scraped articles
    """
    # Initialize scraper
    scraper = GlamRapScraper()
    
    # Load URLs
    with open(input_file, 'r', encoding='utf-8') as f:
        urls = json.load(f)
    
    # Create or load existing results
    if os.path.exists(output_file):
        with open(output_file, 'r', encoding='utf-8') as f:
            results = json.load(f)
    else:
        results = []
    
    # Setup progress bar
    pbar = tqdm(urls, desc="Scraping articles")
    
    try:
        for url in pbar:
            # Update progress bar description
            pbar.set_description(f"Scraping: {url[:50]}...")
            
            # Scrape article
            article = scraper.get_article_content(url)
            
            # Append to results
            results.append(article)
            
            # Save after each article (to prevent data loss)
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(results, f, ensure_ascii=False, indent=2)
            
    except KeyboardInterrupt:
        print("\nScraping interrupted by user. Saving progress...")
    finally:
        # Final save
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        
        # Print summary
        total_articles = len(results)
        successful_articles = len([a for a in results if a is not None])
        print(f"\nScraping completed:")
        print(f"Total articles processed: {total_articles}")
        print(f"Successfully scraped: {successful_articles}")
        print(f"Failed: {total_articles - successful_articles}")
        print(f"Results saved to: {output_file}")

if __name__ == "__main__":
    input_file = '/home/wojtek/AI_Projects/Jupyters/glamrap_articles_urls.json'
    output_file = f'glamrap_articles_content_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
    
    scrape_and_save_articles(input_file, output_file)


class GlamRapUrlScraper:
    def __init__(self, base_url: str = "https://glamrap.pl"):
        self.base_url = base_url
        self.session = requests.Session()
        
        # Configure retries
        retries = Retry(
            total=5,
            backoff_factor=1,
            status_forcelist=[429, 500, 502, 503, 504]
        )
        adapter = HTTPAdapter(max_retries=retries)
        self.session.mount('http://', adapter)
        self.session.mount('https://', adapter)
        
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1',
        })
        
        # Subpages to include (these contain article links we want)
        self.valid_subpages = {
            '/singiel/',
            '/teledysk/',
            '/wywiad/',
            '/felieton/',
            '/ranking/',
            '/premiery-plyt/'
        }
        
        # Pages to exclude
        self.excluded_paths = {
            '/reprezentuj-siebie-jd/',
            '/logowanie/',
            '/kontakt/',
            '/o-nas/',
            '/patronat/',
            '/reklama/',
            '/kontakt2/',
            '/regulamin/',
            '/polityka-cookies/',
            '/polityka-prywatnosci/'
        }
        
        # Set to store unique article URLs
        self.article_urls: Set[str] = set()
        
        # File to save progress
        self.save_file = '/home/wojtek/AI_Projects/Jupyters/glamrap_articles_urls.json'
        
        # Load existing URLs if file exists
        self.load_existing_urls()

    def load_existing_urls(self):
        """Load existing URLs from save file if it exists"""
        try:
            if os.path.exists(self.save_file):
                with open(self.save_file, 'r', encoding='utf-8') as f:
                    self.article_urls = set(json.load(f))
                print(f"Loaded {len(self.article_urls)} existing URLs")
        except Exception as e:
            print(f"Error loading existing URLs: {e}")

    def save_urls(self):
        """Save current URLs to file"""
        try:
            with open(self.save_file, 'w', encoding='utf-8') as f:
                json.dump(list(self.article_urls), f, ensure_ascii=False, indent=2)
            print(f"Saved {len(self.article_urls)} URLs to {self.save_file}")
        except Exception as e:
            print(f"Error saving URLs: {e}")

    def is_article_url(self, url: str) -> bool:
        """Check if URL is an article and not a subpage or excluded path"""
        path = url.replace(self.base_url, '')
        
        # Exclude specific paths
        if path in self.excluded_paths:
            return False
            
        # Exclude subpages
        if path in self.valid_subpages:
            return False
            
        # Exclude pagination pages
        if path.startswith('/page/'):
            return False
            
        # Exclude external links
        if not url.startswith(self.base_url):
            return False
            
        # Must have more than just the base URL
        if url == self.base_url or url == f"{self.base_url}/":
            return False
            
        return True

    def extract_urls_from_page(self, url: str) -> Set[str]:
        """Extract all valid article URLs from a single page"""
        new_urls = set()
        try:
            print(f"\nFetching {url}")
            response = self.session.get(url, timeout=10)
            response.raise_for_status()
            
            # Save the raw HTML for debugging
            with open('last_page.html', 'w', encoding='utf-8') as f:
                f.write(response.text)
            
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Debug: Print page structure
            print("\nPage structure analysis:")
            main_content = soup.find('div', {'id': 'mvp-main-body-wrap'})
            if main_content:
                print("Found main content wrapper")
            else:
                print("WARNING: Main content wrapper not found!")
            
            # Try different approaches to find articles
            articles_found = 0
            
            # Look for article links in various ways
            article_selectors = [
                ('a', {'rel': 'bookmark'}),  # Common pattern for article links
                ('h2', {'class': 'mvp-blog-story-title'}),  # Article titles
                ('div', {'class': 'mvp-blog-story-wrap'}),  # Article wrappers
            ]
            
            for tag, attrs in article_selectors:
                elements = soup.find_all(tag, attrs)
                print(f"\nFound {len(elements)} elements with {tag} {attrs}")
                
                for element in elements:
                    if tag == 'a':
                        href = element.get('href')
                    else:
                        link_elem = element.find('a')
                        href = link_elem.get('href') if link_elem else None
                    
                    if href:
                        link = urljoin(self.base_url, href)
                        if self.is_article_url(link):
                            new_urls.add(link)
                            articles_found += 1
            
            print(f"\nTotal articles found on page: {articles_found}")
            
            # If we found no articles, this might indicate a problem
            if articles_found == 0:
                print("WARNING: No articles found on page!")
                print("Page title:", soup.title.string if soup.title else "No title found")
                
        except requests.exceptions.RequestException as e:
            print(f"\nRequest error processing {url}: {str(e)}")
            print(f"Response status code: {getattr(e.response, 'status_code', 'N/A')}")
            print(f"Response headers: {getattr(e.response, 'headers', {})}")
        except Exception as e:
            print(f"\nGeneral error processing {url}: {str(e)}")
            
        return new_urls

    def scrape_urls(self, start_page: int = 1, end_page: int = 4297, save_interval: int = 10):
        """Scrape article URLs from all pages in range"""
        try:
            for page in range(start_page, end_page + 1):
                # Construct page URL
                if page == 1:
                    page_url = self.base_url
                else:
                    page_url = f"{self.base_url}/page/{page}/"
                
                # Get URLs from main page
                new_urls = self.extract_urls_from_page(page_url)
                old_count = len(self.article_urls)
                self.article_urls.update(new_urls)
                new_count = len(self.article_urls)
                
                print(f"\nPage {page}: Found {len(new_urls)} URLs, {new_count - old_count} new")
                
                # Save progress periodically
                if page % save_interval == 0:
                    self.save_urls()
                    
                # Longer delay between requests
                time.sleep(5)  # Increased delay
                
        except KeyboardInterrupt:
            print("\nScraping interrupted by user")
            self.save_urls()
        except Exception as e:
            print(f"\nError during scraping: {e}")
            self.save_urls()
            
        # Final save
        self.save_urls()

