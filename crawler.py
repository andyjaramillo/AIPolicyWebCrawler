import logging
import re
import sys
import os
from serpapi import GoogleSearch
from dotenv import load_dotenv
from bs4 import BeautifulSoup
from queue import Queue
from queue import PriorityQueue 
from difflib import SequenceMatcher
from urllib import parse, request
from urllib.parse import urlparse, urljoin

logging.basicConfig(level=logging.DEBUG, filename='output.log', filemode='w')
with open("queries.txt", "r") as file:
    queries = file.readlines()
queries = [query.strip() for query in queries]
visitlog = logging.getLogger('visited')
extractlog = logging.getLogger('extracted')
MAX_CRAWL = 10
BASEDIR = os.path.abspath(os.path.dirname(__file__))
load_dotenv(os.path.join(BASEDIR, '.env'))
api_key = os.getenv('SERPAPI_KEY')
 


def parse_links(root, html):
    soup = BeautifulSoup(html, 'html.parser')
    root_url = urlparse(root)
    root_href = root_url.path
    for link in soup.find_all('a'):
        href = link.get('href')
        if href and not is_self_referencing(root_href, href):
            text = link.string
            if not text:
                text = ''
            text = re.sub('\s+', ' ', text).strip()
            yield (parse.urljoin(root, link.get('href')), text)


def similarityComparison(str1, str2):
    return SequenceMatcher(None, str1, str2).ratio()

def is_self_referencing(root_href, child_href):
    try:
        
        return root_href in child_href
    except Exception as e:
        print(e, child_href)
        return False


def parse_links_sorted(root, html):
    # TODO: implement
    #priority function: similarity to root
    #We give a url on the start of the program and ask to crawl, it only makes sense to want to crawl
    #everything that is closest to the url given. This is done just by comparison to the root

    links = get_links(root)
    queue = PriorityQueue()
    
    for link_tuple in links:
        score = similarityComparison(link_tuple[0], root)
        queue.put((-score, link_tuple))
    return list(queue.queue)



def get_domain(url):
    
    pattern = r"(?:https?://)?(?:www\.)?([^/]+)"
    match = re.match(pattern, url)
    if match:
        return match.group(1)
    else:
        return None


def get_links(url):
    res = request.urlopen(url)
    return list(parse_links(url, res.read()))


def get_nonlocal_links(url):
    '''Get a list of links on the page specificed by the url,
    but only keep non-local links and non self-references.
    Return a list of (link, title) pairs, just like get_links()'''

    # TODO: implement
    current_url_domain = get_domain(url)
    links = get_links(url)
    filtered = []
    for link_tuple in links:
        if current_url_domain != get_domain(link_tuple[0]):
            filtered.append(link_tuple)
    
    return filtered


def crawl(root, wanted_content):
    '''Crawl the url specified by `root`.
    `wanted_content` is a list of content types to crawl
    `within_domain` specifies whether the crawler should limit itself to the domain of `root`
    '''
    # TODO: implement
    queue = Queue()
    queue.put(root)

    visited = []
    extracted = []
    iterator = 0
    while not queue.empty() and iterator < MAX_CRAWL:
        url = queue.get()
        iterator += 1
        if url not in visited: 
            try:
                req = request.urlopen(url)
                try:
                    html = req.read()
                    visited.append(url)
                    visitlog.debug(url)
                    for ex in extract_pdf_links(url, html):
                        extracted.append(ex)
                        extractlog.debug(ex)
                    for score, link in parse_links_sorted(url, html):
                        try:
                            child_link = request.urlopen(link[0])
                            if child_link.headers.get('Content-Type').split(';')[0].strip() in wanted_content or len(wanted_content) == 0: 
                                if link[0] not in visited:
                                    queue.put(link[0])
                        except Exception as e:
                            print(e, link[0])
                        
                except Exception as e:
                    print(e, url)
            except Exception as e:
                print(e, url)
        
        
           
    return visited, extracted


def extract_pdf_links(address, html):
    '''Extract contact information from html, returning a list of (url, category, content) pairs,
    where category is one of PHONE, ADDRESS, EMAIL'''

    
    soup = BeautifulSoup(html, 'html.parser')
    pdf_links = []
    for link in soup.find_all('a', href=True):
        href = link['href']
        if href.endswith('.pdf'):
            absolute_url = urljoin(address, href)
            pdf_links.append(absolute_url)

    return pdf_links

def writelines(filename, data):
    with open(filename, 'w') as fout:
        for d in data:
            print(d, file=fout)


def run():
    ###TODO Change this to list of queries from queries.txt
    params = {
        "engine" : "google",
        "q": "Artificial intelligence policy reports",
        "api_key": api_key
    }
    search = GoogleSearch(params)
    results = search.get_dict()
    organic_results = results["organic_results"]
    ### now get the list of links
    links = []
    for result in organic_results:
        links.append(result["link"])
    with open("seed_links.txt", "r") as links_file:
        seed_links = links_file.readlines()
    seed_links = [link.strip() for link in seed_links]
    links += seed_links
    ###TODO: Change to for loop iterating over list of links
    visited, extracted = crawl(links[0], ["text/html"])

    # site = sys.argv[1]
    # headers_list = []
    # links = get_links(site)
    # writelines('links.txt', links)
    # nonlocal_links = get_nonlocal_links(site)
    # writelines('nonlocal.txt', nonlocal_links)
    # res = request.urlopen(site)
    # if len(sys.argv) > 3:
    #     # then a parameter was passed in
    #     header = sys.argv[4]
    #     header.split(',')
    #     headers_list = header

    
    # writelines('visited.txt', visited)
    # writelines('extracted.txt', extracted)
    # return visited,extracted

if __name__ == "__main__":
    run()

