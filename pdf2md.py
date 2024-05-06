import PyPDF2
from markdownify import markdownify as md
from urllib import parse, request
from io import BytesIO
from markdown_it import MarkdownIt

def pdf_to_markdown(pdf_file):
    reader = PyPDF2.PdfReader(pdf_file)
    text = ''
    for page_num in range(len(reader.pages)):
        text += reader.pages[page_num].extract_text()
    return md(text)

def extract_headers(markdown_content):
    # Initialize Markdown parser
    md = MarkdownIt()
    # Parse Markdown content
    tokens = md.parse(markdown_content)
    headers = []
    # Loop through parsed tokens
    for token in tokens:
        # Check if token represents a heading
        if token.type == 'heading_open':
            # Extract header text from following text token
            header_text = next(t for t in tokens if t.type == 'inline' and t.tag == 'text').content
            headers.append(header_text)
    return headers

def extract_paragraphs(markdown_content):
    md = MarkdownIt()
    tokens = md.parse(markdown_content)
    paragraphs = []
    token_iter = iter(tokens)  # Convert tokens list to an iterator
    for token in token_iter:
        if token.type == 'paragraph_open':
            # Find the text content within the paragraph
            paragraph_tokens = []
            while True:
                try:
                    next_token = next(token_iter)
                    if next_token.type == 'paragraph_close':
                        break
                    paragraph_tokens.append(next_token)
                except StopIteration:
                    break
            # Extract text content from inline tokens
            paragraph_text = ''.join(t.content for t in paragraph_tokens if t.type == 'inline' and t.tag == 'text')
            paragraphs.append(paragraph_text)
    return paragraphs

def extract_lists(markdown_content):
    md = MarkdownIt()
    tokens = md.parse(markdown_content)
    lists = []
    for token in tokens:
        if token.type == 'list_open':
            list_items = []
            while True:
                next_token = next(tokens)
                if next_token.type == 'list_close':
                    break
                if next_token.type == 'list_item_open':
                    list_item_text = next(t for t in tokens if t.type == 'inline' and t.tag == 'text').content
                    list_items.append(list_item_text)
            lists.append(list_items)
    return lists

def extract_bookmarks(pdf_file):
    reader = PyPDF2.PdfReader(pdf_file)
    bookmarks = []
    for page_num in range(len(reader.pages)):
        page = reader.pages[page_num].extract_text()
        if '/Outlines' in page:
            bookmarks += reader.getDestinationPageNumbers()
    return bookmarks


with open("sample_pdf_test.txt", "r") as extracted_file:
        # logs is a list of pdf links
        logs = extracted_file.readlines()
extracted = logs
label_counter = 101
pdf_links=[]
# put all pdf links in a list
for line in extracted:
    current_line_array = line.strip()
    pdf_links.append(current_line_array)
    
# print("first link:",pdf_links[0])
# Usage example
response = request.urlopen(pdf_links[0])
# .read() returns binary data of a file
pdf_file = BytesIO(response.read()) 
pdf_reader = PyPDF2.PdfReader(pdf_file) 
bookmarks = extract_bookmarks(pdf_file)
print("extracted bookmarks from pdf file", bookmarks)
markdown_content = pdf_to_markdown(pdf_file)
# print(markdown_content)

# Save markdown_content to a file
with open('pdf2markdown.md', 'w') as f:
    f.write(markdown_content)
    
# Extract headers from Markdown content
headers = extract_headers(markdown_content)
paragraphs = extract_paragraphs(markdown_content)
lists = extract_lists(markdown_content)
print("Headers:", headers)
print("Paragraphs:", paragraphs)
print("Lists:", lists)
# Print extracted headers
# for header in headers:
#     print(header)

# for paragraph in paragraphs:
#     print(paragraph)
    
# for list_item in lists:
#     print(list_item)