import os
import csv
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
from langdetect import detect

class WebCrawler:
    def __init__(self, seed_url, domain, max_pages=50):
        self.seed_url = seed_url
        self.domain = domain
        self.visited = set()
        self.report = []
        self.max_pages = max_pages

        # Create a separate repository folder for this domain
        self.repo_path = os.path.join("repository", self.domain.replace(".", "_"))
        os.makedirs(self.repo_path, exist_ok=True)

    def valid_url(self, url):
        parsed_url = urlparse(url)
        return parsed_url.scheme in ["http", "https"] and self.domain in parsed_url.netloc

    def crawl(self):
        print(f"\n🌍 Starting crawl for domain: {self.domain}")
        to_crawl = [self.seed_url]
        
        while to_crawl and len(self.visited) < self.max_pages:
            url = to_crawl.pop(0)
            if url in self.visited or not self.valid_url(url):
                continue
            
            try:
                print(f"Crawling: {url}")
                response = requests.get(url, timeout=10, headers={"User-Agent": "Mozilla/5.0"})
                response.raise_for_status()
                response.encoding = "utf-8" # Assume content of page is in UTF-8
                soup = BeautifulSoup(response.text, "html.parser")

                # Detect page language
                try:
                    lang = detect(soup.get_text())
                except:
                    lang = "unknown"

                # Save full HTML content in the domain-specific folder
                filename = os.path.join(self.repo_path, f"page_{len(self.visited)}.html")
                with open(filename, "w", encoding="utf-8") as file:
                    file.write(response.text)
                
                print(f"Saved: {filename}")

                # Find all outlinks
                links = set(urljoin(url, a['href']) for a in soup.find_all("a", href=True))
                links = {link for link in links if self.valid_url(link)}

                # Store the URL and number of outlinks
                self.report.append((url, len(links), lang, filename))

                # Mark the page as visited
                self.visited.add(url)
                
                # Add new links to the queue
                to_crawl.extend(links - self.visited)

            except requests.RequestException as e:
                print(f"Failed to crawl {url}: {e}")
                continue

        self.save_report()
        print(f"🚀 Finished crawling {self.domain}. Pages visited: {len(self.visited)}")

    def save_report(self):
        # Save the crawling report to a single report.csv file
        with open("report.csv", "a", newline="", encoding="utf-8") as file:
            writer = csv.writer(file)
            writer.writerows(self.report)

if __name__ == "__main__":
    seed_urls = [
        "https://www.deutschland.de/de",  # German news
        "https://www.lefigaro.fr",  # French news
        "https://www.telegraaf.nl/",  # Dutch news
        "https://www.cnn.com/" # English News
    ]

    domain_restrictions = ["deutschland.de", "lefigaro.fr", "telegraaf.nl", "cnn.com"]

    # Create a report file with headers
    with open("report.csv", "w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["URL", "Outlinks", "Language", "Filename"])

    # Start separate crawlers for each domain
    for url, domain in zip(seed_urls, domain_restrictions):
        crawler = WebCrawler(url, domain, max_pages=50)
        crawler.crawl()
