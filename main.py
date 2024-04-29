from crawler import run
from parser_1 import parse_output
from model import model
from crawler import run



def main():
    # call crawl function
    print("Crawling started")
    run()
    print("Crawling complete")
    document_array = parse_output()
    print("Parsing complete")
    print("document_array: ", document_array[0])
    model([document_array])



if __name__ == "__main__":
    main()