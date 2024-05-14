from crawler import run
from parser_1 import parse_output
from model import model, Document
from crawler import run
import ast

def save_document_array(document_array):
    # Save document_array to a file
    with open("document_array.txt", "w") as f:
        for document in document_array:
            f.write(str(document) + "\n")

def load_document_array():
    document_array = []
    with open("document_array.txt", "r") as f: 
        lines = f.readlines()  # Read all lines from the file
        index = 0
        while index < len(lines):
            split_line = lines[index].split()
          #  print(split_line)
            if split_line[0] == "Document":
                ## we are at a new document. We need to add it to document array
                doc_id = ""
                text = []
                labels = []
                date_created = ""
                date_modified = ""
                title = ""
                author = ""
                url = ""
                ##lines
                doc_id = split_line[2]
                index += 1
                while index < len(lines) and not lines[index].startswith("Labels:"):
                    if "Text:" in lines[index].strip():
                        text.append(lines[index].strip().replace("Text:", ""))
                    else:    
                        text.append(lines[index].strip())
                    index += 1
                text = " ".join(text)
                text = ast.literal_eval(text)
                split_line = lines[index]
                labels = split_line.split(":")[1].strip()
                index += 1
                split_line = lines[index]
                date_created = split_line.split(":")[1].strip()
                index += 1
                split_line = lines[index]
                date_modified = split_line.split(":")[1].strip()
                index += 1
                split_line = lines[index]
                title = split_line.split(":")[1].strip()
                index += 1
                split_line = lines[index]
                author = split_line.split(":")[1].strip()
                index += 1
                split_line = lines[index]
                url = split_line.split("URL:")[1].strip()
                document_array.append(Document(doc_id=doc_id, text=text, created=date_created, modified=date_modified, title=title, author=author, url=url))
            index += 1
    return document_array


def main():
    # call crawl function
   run()
#    print("Crawling complete")
   document_array = parse_output()
   print("Parsing complete")
   #save_document_array(document_array)
   # document_array = load_document_array()
   #print(document_array)
   result = model(document_array)
   return result


if __name__ == "__main__":
    main()