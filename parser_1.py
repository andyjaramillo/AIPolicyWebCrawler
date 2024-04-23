import PyPDF2
from urllib import parse, request
from io import BytesIO
from datetime import datetime
from model import Document
import nltk

from nltk.tokenize import sent_tokenize



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
        try:
            response = request.urlopen(pdf_url)
            pdf_file = BytesIO(response.read())
            pdf_reader = PyPDF2.PdfReader(pdf_file)
            metadata = pdf_reader.metadata
        # metadata['/CreationDate']
        # metadata['/Author']
        # metadata['/ModDate']
        # metadata['/Title']
            for page_num in range(len(pdf_reader.pages)):
                page_text = pdf_reader.pages[page_num].extract_text()
                newDocument = Document(doc_id=label_counter, text=sent_tokenize(page_text), created=time_parse(metadata['/CreationDate']),modified=time_parse(metadata['/CreationDate']), title=metadata['/Title'], author=metadata['/Author'], url=pdf_url)
                documents.append(newDocument)
            #     label_counter += 1
        except Exception as e:
            print("Error to retrive PDF text from ", e, pdf_url)
    
    return documents


def parse_output():
    with open("output.log", "r") as output_file:
        logs = output_file.readlines()
    output=logs
    label_counter = 101
    links=[]
    for line in output:
        current_line_array = line.strip().split(':')
        if current_line_array[1] == "extracted":
            ## we have a pdf
            links.append(current_line_array[2] + ':' + current_line_array[3])
    # print(links)
    print("got here 3")
    documents = retrieve_PDF_text(links, label_counter=label_counter) if len(links) > 0 else []
    return documents
    # for link in links:
    #     retrieve_PDF_text(link)
