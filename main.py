from crawler import run
from parser_1 import parse_output
from model import model, Document
from crawler import run


def main():
   run()
   document_array = parse_output()
   print("Parsing complete")
   result = model(document_array)
   return result


if __name__ == "__main__":
    main()