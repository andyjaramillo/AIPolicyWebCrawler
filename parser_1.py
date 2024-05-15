import PyPDF2
from urllib import parse, request
from io import BytesIO
from datetime import datetime
from model import Document
import nltk
# import fitz
from nltk.tokenize import sent_tokenize
from model import model
from crawler import run



def time_parse(time_string):
    # Sample PDF date string
    pdf_date_string = time_string

    # Remove the 'D:' prefix
    pdf_date_string = pdf_date_string[2:]
    pdf_date_string = pdf_date_string.replace("'", "")

    # Parse the PDF date string
    creation_date = datetime.strptime(pdf_date_string, '%Y%m%d%H%M%S%z')

    # Format the parsed date to display only year, month, and day
    formatted_creation_date = creation_date.strftime('%Y-%m-%d')

    # Print the formatted creation date
    return formatted_creation_date


def retrieve_PDF_text(pdf_urls, label_counter):
    documents = []
    for pdf_url in pdf_urls:
        one_pdf_of_documents = []
        try:
            print(pdf_url)
            response = request.urlopen(pdf_url)
            pdf_file = BytesIO(response.read()) 
            pdf_reader = PyPDF2.PdfReader(pdf_file) 
            metadata = pdf_reader.metadata
            print("Pdf pages", len(pdf_reader.pages))
            print("read pdf file")
        # metadata['/CreationDate']
        # metadata['/Author']
        # metadata['/ModDate']
        # metadata['/Title']
            # create a new document object for each page in the pdf
            for page_num in range(len(pdf_reader.pages)):
                page_text = pdf_reader.pages[page_num].extract_text().strip().replace('\n', '').replace('\xa0', ' ') # Extract html from pdf
                newDocument = Document(doc_id=label_counter, text=sent_tokenize(page_text), created=time_parse(metadata['/CreationDate']),modified=time_parse(metadata['/CreationDate']), title=sent_tokenize(metadata.title), author=metadata.author, url=pdf_url)
                one_pdf_of_documents.append(newDocument)
                label_counter += 1
            print("extracted metadata from pdf file")
            documents.append(one_pdf_of_documents)
        except Exception as e:
            print("Error to retrive PDF text from ", e, pdf_url)
    
    return documents


def parse_output():
    with open("extracted.txt", "r") as extracted_file:
    # with open("sample_pdf_test.txt", "r") as extracted_file:
        logs = extracted_file.readlines()
    extracted = logs
    label_counter = 101
    pdf_links=[]
    for line in extracted:
        current_line_array = line.strip().split(':')
        pdf_links.append(current_line_array[0] + ':' + current_line_array[1])
        # print(pdf_links)
        # print("got here 3")
    documents = retrieve_PDF_text(pdf_links, label_counter=label_counter) if len(pdf_links) > 0 else []
    return documents
    # for link in links:
    #     retrieve_PDF_text(link)

if __name__ == "__main__":
    parse_output()