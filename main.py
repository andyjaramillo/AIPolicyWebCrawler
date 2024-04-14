from parser_1 import parse_output
from model import model


def main():
    document_array = parse_output()
    model([document_array])


if __name__ == "__main__":
    main()