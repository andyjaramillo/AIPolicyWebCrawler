from typing import Dict, List, NamedTuple
from nltk.stem import SnowballStemmer
from collections import Counter, defaultdict
import math
import string
from numpy.linalg import norm

stemmer = SnowballStemmer('english')
WITHIN_PAGE_PROBABILITY_MINIMUM = 0.75
PDF_LABEL_PROBABILITY = 3

class Document:
    def __init__(self, doc_id: int, text: str, created: str, modified :str , title: str, author: str, url: str):
        self.doc_id = doc_id
        self.text = text
        self.labels = []
        self.created = created
        self.modified = modified
        self.title = title
        self.author = author
        self.url = url
    
    def sections(self):
        return [self.author, self.title, self.text]

    def __repr__(self):
        return (f"doc_id: {self.doc_id}\n" +
                f"  text: {self.text}\n" +
                f"  label: {self.labels}\n" +
                f"  date: {self.created}\n" +
                f"  title: {self.title}\n" +
                f"  author: {self.author}")
    def __lt__(self, other):
        ### TODO ADD SORTED, JUST COMPARING THE SCORES
        average_scores = self.text
        return self.number < other.number
    


def read_labels_from_file(file_path: str) -> List[str]:
    with open(file_path, 'r') as file:
        labels = file.readlines()
        labels = [label.strip() for label in labels]  # Remove leading/trailing whitespace and newline characters
    return labels

def read_stopwords(file):
    with open(file) as f:
        return set([x.strip() for x in f.readlines()])

stopwords = read_stopwords('common_words')

def stem_doc(doc: Document):
    return Document(doc.doc_id, *[[stemmer.stem(word) for word in sec] for sec in doc.sections()])

def stem_docs(docs: List[Document]):
    return [stem_doc(doc) for doc in docs]

def remove_stopwords_doc(doc: Document):
    return Document(doc_id=doc.doc_id, created=doc.created, modified=doc.modified, url=doc.url ,
                    author=[word for word in doc.author.split() if word not in stopwords and word not in string.punctuation], 
                    title=[word for word in doc.title.split() if word not in stopwords and word not in string.punctuation], 
                    text = [word for sentence in doc.text for word in sentence.split() if word not in stopwords and word not in string.punctuation])

def remove_stopwords(docs: List[Document]):
    return [remove_stopwords_doc(doc) for doc in docs]



def process_docs_array(docs_array):

    processed_docs = remove_stopwords(docs_array)
    #processed_docs = stem_docs(processed_docs)
    return processed_docs

def process_queries_array(query_array):
    doc_id = 1
    docs = []
    for query in query_array:
        docs.append(Document(doc_id=doc_id, text=query.split(), created="", modified="", title="", author="", url=""))
    return docs

### Term-Document Matrix


def compute_doc_freqs(docs: List[Document]):
    '''
    Computes document frequency, i.e. how many documents contain a specific word
    '''
    freq = Counter()
    for doc in docs:
        words = set()
        for sec in doc.sections():
            for word in sec:
                words.add(word)
        for word in words:
            freq[word] += 1
    return freq


def compute_expo_tfidf(doc, doc_freqs, n, terms):
    # we need doc frequency and term frequency
    decay_dict = defaultdict(float)

    for sentence in doc.text:
        distances = [abs(sentence.index(word) - sentence.index(term)) for word in sentence for term in terms if term in sentence]
        for word in sentence:
            if word in terms:
                # Compute the decay for key terms based on distances
                word_decay = sum(1 / (distance + 1) for distance in distances if distance > 0)  # Avoid division by zero
                decay_dict[word] += word_decay
            else:
                # Assign a lower weight to non-key terms
                decay_dict[word] += 0.1  # Adjust this weight as needed
    return dict(decay_dict)




### Term Similarity

### Vector Similarity

def dictdot(x: Dict[str, float], y: Dict[str, float]):
    '''
    Computes the dot product of vectors x and y, represented as sparse dictionaries.
    '''
    keys = list(x.keys()) if len(x) < len(y) else list(y.keys())
    return sum(x.get(key, 0) * y.get(key, 0) for key in keys)

def cosine_sim(x, y):
    '''
    Computes the cosine similarity between two sparse term vectors represented as dictionaries.
    '''
    num = dictdot(x, y)
    if num == 0:
        return 0
    return num / (norm(list(x.values())) * norm(list(y.values())))


def model(pdf_docs_array):
    """
        pdf_docs_array: a list of formatted pdfs
        Each pdf, rather than the raw doc, is an array of individual
        Document objects, and in each are the text and metadata
    """
    labels = read_labels_from_file('label.txt')
    terms = read_labels_from_file('key_terms')
    processed_labels = process_queries_array(labels)
    pdfs_with_labels = []
    for docs_array in pdf_docs_array:
        processed_docs = process_docs_array(docs_array)

        doc_freqs = compute_doc_freqs(processed_docs)
        doc_vectors = [compute_expo_tfidf(doc, doc_freqs, len(processed_docs), terms) for doc in processed_docs]

        metrics = []

        for doc_vector in doc_vectors:
            """
            Iterate over each doc, and for each doc we go through all labels and calculate probabilities for those
            and append to the labels array
            """

            doc_metric = []
            for label in processed_labels:
                label_vec = compute_expo_tfidf(label, doc_freqs, len(processed_docs), terms=terms)
                results = search(doc_vector, label_vec)
                doc_metric.append((label, results))
            metrics.append(doc_metric)
        
            """
            Take top probabilities above 0.75
            """
            for element in doc_metric:
                if element[0] < WITHIN_PAGE_PROBABILITY_MINIMUM:
                    doc_metric.remove(element)
            
            metrics.append(doc_metric)
        """
        Perform label congregation
        """
        label_to_page = defaultdict(int)
        for metric in metrics:
            for label_tuple in metric:
                label_to_page[label_tuple[0]] += 1

        for label, count in label_to_page.items():
            if count < PDF_LABEL_PROBABILITY:
                del label_to_page[label]
        pdf_docs_array.append(({
            "title": docs_array[0].title,
            "created": docs_array[0].created,
            "modified":docs_array[0].modified,
            "url": docs_array[0].url
        },label_to_page))
        
    return pdfs_with_labels    
        

        





def search(doc_vector, label_vec):
    score = cosine_sim(label_vec, doc_vector)
    return score
