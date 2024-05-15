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
MAX_CRAWL = 5
MAX_PAGE_LINKS = 5
BASEDIR = os.path.abspath(os.path.dirname(__file__))
load_dotenv(os.path.join(BASEDIR, '.env'))
api_key = os.getenv('SERPAPI_KEY')
 


def parse_links(url, html):
    '''
    Parse the links in the html of a page.
    Return a list of (link, text) tuples which are the links found on the url.
    '''
    soup = BeautifulSoup(html, 'html.parser')
    # parse the root url into its components
    root_url = urlparse(url)
    root_href = root_url.path
    links = soup.find_all('a')
    num_links = 0
    for link in links:
        if num_links >= MAX_PAGE_LINKS:
            break
        # gets the href attribute of the <a> tag, which is the URL that the link points to.
        href = link.get('href')
        # remove self-referencing links
        if href and not is_self_referencing(root_href, href):
            # TODO we dont use the anchor text, we shoould remove it
            anchor_text = link.string
            # print('text', text)
            if not anchor_text:
                anchor_text = ''
            # this is to remove any extra whitespace in the anchor text
            anchor_text = re.sub('\s+', ' ', anchor_text).strip()
            num_links += 1
            # urljoin() is used to resolve the relative URL to an absolute URL with the paths combined
            yield (parse.urljoin(url, link.get('href')), anchor_text)


def similarityComparison(str1, str2):
    return SequenceMatcher(None, str1, str2).ratio()

def is_self_referencing(root_href, child_href):
    try:
        
        return root_href in child_href
    except Exception as e:
        print(e, child_href)
        return False


def parse_links_sorted(url, html):
    '''
    Looks for links in the html of a page and sorts them by similarity to the root
    Returns a list of (score, link) tuples which are the links found on the url and their score
    '''
    #priority function: similarity to root
    #We give a url on the start of the program and ask to crawl, it only makes sense to want to crawl
    #everything that is closest to the url given. This is done just by comparison to the root
    print('getting links from url', url)
    links = get_links(url) # list of tuples (link, anchor text)
    # print('links within url', links)
    queue = PriorityQueue()
    
    for link_tuple in links:
        # sort links by descending similarity score to the root url
        score = similarityComparison(link_tuple[0], url) # link_tuple[0] is the link, link_tuple[1] is the text
        queue.put((-score, link_tuple[0]))
    print('sorted the links')
    return list(queue.queue) # return a list of tuples (score, link)



def get_domain(url):
    
    pattern = r"(?:https?://)?(?:www\.)?([^/]+)"
    match = re.match(pattern, url)
    if match:
        return match.group(1)
    else:
        return None


def get_links(url):
    '''
    Get a list of links on the webpage specified by the url.
    Return a list of tuples (link, anchor text) from the page specified by the url
    '''
    res = request.urlopen(url)
    print('got html')
    html_content = res.read().decode('utf-8')
    return list(parse_links(url, html_content)) # return a list of tuples (link, anchortext of the link)


def get_nonlocal_links(url):
    '''Get a list of links on the page specificed by the url,
    but only keep non-local links and non self-references.
    Return a list of (link, title) pairs, just like get_links()'''

    current_url_domain = get_domain(url)
    links = get_links(url)
    filtered = []
    for link_tuple in links:
        if current_url_domain != get_domain(link_tuple[0]):
            filtered.append(link_tuple)
    
    return filtered


def crawl(seed_links, wanted_content):
    '''Crawl the url specified by `root`.
    `wanted_content` is a list of content types to crawl
    `within_domain` specifies whether the crawler should limit itself to the domain of `root`
    '''
    queue = Queue()
    for link in seed_links:
        queue.put(link) # add all the seed links to the queue

    visited = []
    extracted = []
    iterator = 0
    
    while not queue.empty() and iterator < MAX_CRAWL:
        url = queue.get()
        print('url from seedlinks', url)
        
        if url not in visited: 
           
            try:
                print('not visited yet', url)
                req = request.urlopen(url)
                print('opened url')
                try:
                    html = req.read()
                    visited.append(url)
                    visitlog.debug(url)
                    print('extracting pdf links')
                    for ex in extract_pdf_links(url, html): # extract pdf links from visited url
                        extracted.append(ex)
                        extractlog.debug(ex)
                    print('parsing links in sorted order')
                    for score, link in parse_links_sorted(url, html): # extract links from visited url
                        # print('links', links)
                        try:
                            print('trying to open link', link)
                            child_link = request.urlopen(link)
                            # if link matches the wanted content type, add it to the queue to search
                            if child_link.headers.get('Content-Type').split(';')[0].strip() in wanted_content or len(wanted_content) == 0: 
                                if link not in visited:
                                    # TODO should we limit the number of links we add to the queue that were found on the page?
                                    queue.put(link)
                        except Exception as e:
                            print(e, link)
                    # opening a valid seed link counts as a crawl
                    iterator += 1
                        
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
    # returns JSON output
    search = GoogleSearch(params)
    # get JSON output as dictionary
    results = search.get_dict()
    # parses the JSON output to get the organic results section
    organic_results = results["organic_results"]
    ### now get the list of links from google
    links = []
    pdfs = []
    # check for pdfs in the google search
    for result in organic_results:
        link = result["link"]
        # TODO immprove methods to identify pdfs
        if not link.endswith('.pdf'):
            links.append(link) # extract links in the google search    
        else:
            pdfs.append(link) # extract pdfs if it is the seed link
    writelines('serpapi_links.txt', links) # write the links from the google search to a file
    with open("seed_links.txt", "r") as links_file:
        seed_links = links_file.readlines() # put each line of the file into list of strings elements 
    seed_links = [link.strip() for link in seed_links] # extract the links from seed_links.txt
    links += seed_links
    
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

