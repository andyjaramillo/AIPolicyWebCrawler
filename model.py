from typing import Dict, List, NamedTuple
from nltk.stem import SnowballStemmer
from collections import Counter, defaultdict
from sklearn.feature_extraction.text import TfidfVectorizer
import skfuzzy as fuzz
from sklearn.preprocessing import LabelEncoder
import math
import string
from numpy.linalg import norm
import numpy as np

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
        return [self.title, self.text]

    def __repr__(self):
        return (f"doc_id: {self.doc_id}\n" +
                f"  text: {self.text}\n" +
                f"  created: {self.created}\n" +
                f"  modified: {self.modified}\n" +
                f"  title: {self.title}\n" +
                f"  author: {self.author}\n" + 
                f"  url: {self.url}\n"
                )
    
    def __str__(self):
        return (f"Document ID: {self.doc_id}\n" +
                f"Text: {self.text}\n" +
                f"Labels: {self.labels}\n" +
                f"Date Created: {self.created}\n" +
                f"Last Modified: {self.modified}\n" +
                f"Title: {self.title}\n" +
                f"Author: {self.author}\n" +
                f"URL: {self.url}")

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
        docs.append(Document(doc_id=doc_id, text=query, created="", modified="", title="", author="", url=""))
    return docs

### Term-Document Matrix

# TODO Isnt this receiving one of the docs from the pdf?
def compute_doc_freqs(docs: List[Document]):
    '''
    Computes document frequency, i.e. how many documents contain a specific word
    '''
    freq = Counter()
    for doc in docs:
        words = set()
        for word in (doc.title or "").split():
            # include words in the title
            words.add(word)
        for element in doc.text:
            # add words in the text
            for word in element.split():
                words.add(word)
        for word in words:
            freq[word] += 1
    return freq


def compute_expo_tfidf(doc, doc_freqs, n, terms):
    # we need doc frequency and term frequency
    decay_dict = defaultdict(float)

    tf_matrix = []
    words_set = set()
    for sentence in doc.text:
        sentence_array = sentence.split()
        tf_row = {}
        for term in sentence_array:
            words_set.add(term)
            tf_row[term] = sentence.count(term) / len(sentence)
        tf_matrix.append(tf_row)

    for i,sentence in enumerate(tf_matrix):
        for word,weight in sentence.items():
            if word in doc_freqs:
                tf_matrix[i][word] = (weight *  math.log2((n/doc_freqs[word])))

    return (doc, tf_matrix, words_set)
    
    
        
    #     distances = [abs(sentence_array.index(word) - sentence_array.index(term)) for word in sentence_array]
    #     for word in sentence:
    #         if word in terms:
    #             # Compute the decay for key terms based on distances
    #             word_decay = sum(1 / (distance + 1) for distance in distances if distance > 0)  # Avoid division by zero
    #             decay_dict[word] += word_decay
    #         else:
    #             # Assign a lower weight to non-key terms
    #             decay_dict[word] += 0.1  # Adjust this weight as needed
    # return dict(decay_dict)


def compute_relevant_labels(label_all_array):
    
    labels = Counter()

    for relevant_labels in label_all_array:
        for label in relevant_labels:
            labels[label] += 1

    
    filtered_labels = [label for label, count in labels.items() if count > 3 ]
    return filtered_labels


def compute_label_congregation(metrics):
    """
    Data structure:
    [
    (doc, [labels]),
    (doc, [labels])
    ]
    """
    labels = Counter()
    for tuplet in metrics:
        label_list = tuplet[1]
        for label in label_list:
            labels[label] += 1
    
    filtered_labels = [label for label, count in labels.items() if count > 3 ]
    return filtered_labels

# def convert_doc_vector_for_heat_map(matrix_tuplet):
#     """
#     Data structure of a doc vector
#     {"erqwe":0.04, ...}
#     where its a dictionary of for each sentence and their associated weight from expo_tfidf.
#     In this case we pass in matrix_tuplet, which will be
#     [
#     (
#     Doc,
#     [labels],
#     [{doc vectors}]
#     )
#     ]
#     """
#     data_set = []
#     for tuplet in matrix_tuplet:
#        doc_itself = tuplet[0]
#        labels = tuplet[1]
#        doc_vectors = tuplet[2]
#        words_dict = dict()
#        for doc_vector in doc_vectors:
#            ## this just is appending dictionaries together so we get one big dictionary
#            words_dict.update(doc_vector)
#        data_set.append({
#            'doc': doc_itself,
#            'labels': labels,
#            doc
#        })


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
    label_encoder = LabelEncoder()
    designated_labels_encoded = label_encoder.fit_transform(labels)
    # map the encoded integer to the label
    encoded_int_to_label_map = {encoded_int: label for label,encoded_int in zip(labels, designated_labels_encoded)}
    # print("test",designated_labels_encoded)
    # print("encoded",encoded_int_to_label_map)
    processed_labels = process_queries_array(labels)
    pdfs_with_labels = []
    data_metrics = []
    for doc in pdf_docs_array:
        processed_doc = doc
        doc_freqs = compute_doc_freqs(processed_doc)
        doc_vectors = [compute_expo_tfidf(doc, doc_freqs, len(processed_doc), labels) for doc in processed_doc]
        num_clusters = len(labels)

        """
        Now we will perform Fuzzy C Means clustering (FCM) for each page so each page has the labels
        """

        metrics=[]
        heat_map_data = dict()
        for tuplet in doc_vectors:
            doc_itself = tuplet[0]
            doc_vector = tuplet[1]
            terms = tuplet[2]
            #tf_array = np.array([[tf_row[term] for term in terms if term in tf_row] for tf_row in doc_vector])
            tf_array = np.zeros((len(doc_vector), len(terms)))

            # Iterate over each document vector and update tf_array
            for i, tf_row in enumerate(doc_vector):
                for j, term in enumerate(terms):
                    if term in tf_row:
                        tf_array[i, j] = tf_row[term]  
                    # else:
                    #     tf_array[i, j] = 0  
         #   initial_centroids = np.array([doc_vector.toarray()[np.where(designated_labels_encoded == label)[0]].mean(axis=0) for label in range(num_clusters)])
         
            cntr, u, _, _, _, _, _ = fuzz.cluster.cmeans(tf_array.T, num_clusters, 2, error=0.005, maxiter=1000)
            cluster_to_encoded_index_map = {cluster_index: encoded_int for encoded_int, cluster_index in zip(designated_labels_encoded, np.argmax(u, axis=0))}
            threshold = 0.051 # Set the threshold for membership values
            
            relevant_labels_all = []

            for i in range(len(doc_vector)):  # Iterate over each data point      
                cluster_membership = u[:, i]  # Get the membership values of the data point for all clusters
                relevant_clusters = np.where(cluster_membership > threshold)[0] # Find clusters where membership values are above the threshold
                if len(relevant_clusters) > 0:
                    relevant_encoded_int = [cluster_to_encoded_index_map.get(cluster, 0) for cluster in relevant_clusters]
                    relevant_labels = [encoded_int_to_label_map[encoded_int] for encoded_int in relevant_encoded_int]
                    relevant_labels_all.append(relevant_labels)
            new_doc_id = int(doc_itself.doc_id)-100
            heat_map_data[doc_itself.doc_id] ={
                'x': str(new_doc_id),
                'y': str(relevant_labels_all),
                'heat': len(relevant_labels_all)

            }
            metrics.append((doc_itself, compute_relevant_labels(relevant_labels_all)))

        """
        Now we have our metrics which is of the form
            metrics = [
            (
             Document object <-- which represents a page, [list of relevant labels]
            )
                
            ]

        We have to do one last consolidation, label congregation
        """
        # print(metrics)
        #print(compute_label_congregation(metrics))
        pdfs_with_labels.append((doc, compute_label_congregation(metrics)))
        data_metrics.append(heat_map_data)
        #print(compute_label_congregation(metrics))
        print("done", pdfs_with_labels)
    return pdfs_with_labels,data_metrics    
        

def search(doc_vector, label_vec):
    score = cosine_sim(label_vec, doc_vector)
    return score
