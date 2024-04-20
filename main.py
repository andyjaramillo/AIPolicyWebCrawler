from crawler import run
from parser_1 import parse_output
from model import model


def main():
    # call crawl function
    run()
    print("Crawling complete")
    document_array = parse_output()
    print("Parsing complete")
    model([document_array])


if __name__ == "__main__":
    main()