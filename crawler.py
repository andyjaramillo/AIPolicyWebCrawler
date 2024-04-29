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
    #priority function: similarity to root
    #We give a url on the start of the program and ask to crawl, it only makes sense to want to crawl
    #everything that is closest to the url given. This is done just by comparison to the root
    print('got here to get links')
    links = get_links(root) # list of tuples (link, html of the link)
    queue = PriorityQueue()
    
    for link_tuple in links:
        score = similarityComparison(link_tuple[0], root) # link_tuple[0] is the link, link_tuple[1] is the text
        queue.put((-score, link_tuple[0]))
    return list(queue.queue) # return a list of tuples (score, link)



def get_domain(url):
    
    pattern = r"(?:https?://)?(?:www\.)?([^/]+)"
    match = re.match(pattern, url)
    if match:
        return match.group(1)
    else:
        return None


def get_links(url):
    print('inside get_links')
    res = request.urlopen(url).decode('utf-8')
    print('got html')
    return list(parse_links(url, res.read())) # return a list of tuples (link, html of the url)


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


def crawl(roots, wanted_content):
    '''Crawl the url specified by `root`.
    `wanted_content` is a list of content types to crawl
    `within_domain` specifies whether the crawler should limit itself to the domain of `root`
    '''
    queue = Queue()
    for root in roots:
        queue.put(root) # add all the seed links to the queue

    visited = []
    extracted = []
    iterator = 0
    
    while not queue.empty() and iterator < MAX_CRAWL:
        url = queue.get()
        iterator += 1
        
        if url not in visited: 
           
            try:
                print('not visited yet', url)
                req = request.urlopen(url)
                try:
                    html = req.read()
                    visited.append(url)
                    visitlog.debug(url)
                    for ex in extract_pdf_links(url, html): # extract pdf links from visited url
                        extracted.append(ex)
                        extractlog.debug(ex)
                    print('parsing links in sorted order')
                    for score, links in parse_links_sorted(url, html): # extract links from vissited url
                        print('starts here')
                        print('links', links)
                        try:
                            child_link = request.urlopen(links[0])
                            if child_link.headers.get('Content-Type').split(';')[0].strip() in wanted_content or len(wanted_content) == 0: 
                                if links[0] not in visited:
                                    queue.put(links[0])
                        except Exception as e:
                            print(e, links[0])
                        
                except Exception as e:
                    print(e, url)
            except Exception as e:
                print(e, url)
        
        
           
    return visited, extracted


def extract_pdf_links(address, html):
    '''Extract pdfs from the html of a linked page'''

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

def read_queries_from_file(filename):
        with open(filename, 'r') as file:
            queries = file.readlines()
        return [query.strip() for query in queries]

def run():
    ###TODO Change this to list of queries from queries.txt
    

    params = {
        "hl": "en",
        "gl": "us",
        "engine": "google_scholar",
        "q": read_queries_from_file('queries.txt'),
        "api_key": api_key
    }
    
    # print(params)

    search = GoogleSearch(params)
    results = search.get_dict()
    # print(results)
    organic_results = results["organic_results"]
    ### now get the list of links from google
    # print(organic_results)
    links = []
    pdfs = []
    for result in organic_results:
        link = result["link"]
        if not link.endswith('.pdf'):
            links.append(link) # extract links in the google search
        else:
            pdfs.append(link) # extract pdfs if it is the seed link
    with open("seed_links.txt", "r") as links_file:
        seed_links = links_file.readlines() # put each line of the file into list of strings elements 
    seed_links = [link.strip() for link in seed_links] # extract the links from seed_links.txt
    links += seed_links
    # print(links)
    
    
    # go through seed links and links extracted from google search
    visited, extracted = crawl(links, ["text/html"]) # crawl the google + seed links for pdfs
    extracted += pdfs # add pdfs from google search to the extracted list
    writelines('links.txt', links) # see what links extracted from seed links + serpapi
    # nonlocal_links = get_nonlocal_links(site)
    # writelines('nonlocal.txt', nonlocal_links)
    # res = request.urlopen(site)
    # if len(sys.argv) > 3:
    #     # then a parameter was passed in
    #     header = sys.argv[4]
    #     header.split(',')
    #     headers_list = header

    
    writelines('visited.txt', visited)
    writelines('extracted.txt', extracted)
    return visited,extracted

if __name__ == "__main__":
    run()

