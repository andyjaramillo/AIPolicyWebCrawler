from typing import Dict, List, NamedTuple
from nltk.stem import SnowballStemmer
import nltk
from collections import Counter, defaultdict
import math
import string
from numpy.linalg import norm
from gensim.models import Word2Vec
from nltk.tokenize import word_tokenize
from collections import defaultdict
from nltk.stem import WordNetLemmatizer
import numpy as np
from parameters import config

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

def remove_stopwords_doc(doc: Document, labels):
    """
    Parse document text into an arrays of strings
    Remove stopwords and punctuation from each array 
    """

    sentences = []
    title_sentences=[]
    for sentence in doc.text:
        sentence_words = word_tokenize(sentence.lower())  # Convert to lowercase during tokenization
        sentence_words = [word for word in sentence_words if word.lower() not in stopwords and word.lower() not in string.punctuation]
        sentences.append(sentence_words)
    for element in doc.title:
        sentence_words = word_tokenize(element.lower())  # Convert to lowercase during tokenization
        sentence_words = [word for word in sentence_words if word.lower() not in stopwords and word.lower() not in string.punctuation]
        title_sentences.append(sentence_words)


    return Document(doc_id=doc.doc_id, created=doc.created, modified=doc.modified, url=doc.url ,
                    author=doc.author,
                    title=title_sentences, 
                    text = sentences)

def remove_stopwords(docs: List[Document], labels):
    return [remove_stopwords_doc(doc, labels) for doc in docs]



def process_docs_array(docs_array ,labels):

    processed_docs = remove_stopwords(docs_array ,labels)
    #processed_docs = stem_docs(processed_docs)
    return processed_docs


### Term-Document Matrix


def compute_doc_freqs(docs: List[Document]):
    '''
    Computes document frequency, i.e. how many documents contain a specific word
    '''
    freq = Counter()
    for doc in docs:
        words = set()
        for element_title in doc.title:
            for word in element_title:
            # include words in the title
                words.add(word)
        for element in doc.text:
            # add words in the text
            for word in element:
                words.add(word)
        for word in words:
            freq[word] += 1
    return freq


def compute_expo_tfidf(doc, doc_freqs, n, word_vectors):
    # we need doc frequency and term frequency
    """
    Compute TF-IDF using exponential decay on semantically related words tagged with '.X'
    """
    doc_list = []
    

    for sentence in doc.text:
        sentence_array = sentence
        important_term = False
        for i, word in enumerate(sentence_array):
            if '.X' in word:
                important_term = True

        # Now that the pivot point is denoted, we do exponential decay     
        tf_row = {}
        decay_value = config['decay_value']
        if important_term:
            x_indices = [i for i, word in enumerate(sentence_array) if ".X" in word]
            if x_indices:  # Check if x_indices is not empty
                for index, word in enumerate(sentence_array):
                    if ".X" in word:
                        tf_row[word.replace(".X", "")] = config['label_keyword_weight']
                    else:
                        decay_values = [1/(abs(x_indi - index) * decay_value) for x_indi in x_indices]
                        tf_row[word] = sum(decay_values) / len(decay_values)
            else:
                for word in sentence_array:
                    tf_row[word] = config['uniform_decay_value']
        else:
            for word in sentence_array:
                tf_row[word] = config['uniform_decay_value']

        doc_list.append(tf_row)

    # Incorporate doc.title into the weighting schema
    for i,sentence in enumerate(doc_list):
        for word, weight in sentence.items():
            if word in doc_freqs:
                doc_list[i][word] = (weight * math.log2(n / doc_freqs[word]))

    # Process doc.title
    title_array = doc.title
    for i, sentence in enumerate(title_array):
        for word in sentence:
            if word in doc_freqs:
                title_weight = config['title_weight'] * math.log2(n / doc_freqs[word])
                # Add title weight to corresponding words in doc_list
                for sentence in doc_list:
                    if word in sentence:
                        sentence[word] += title_weight


    return (doc, doc_list)


def compute_relevant_labels(docs_array): 
    """
    Filter to get two highest frequent labels from all pages.
    Return list of labels for the total pdf
    """
    labels = Counter()
    # Count the occurrences of each label
    for doc_map in docs_array:
        labels_list = doc_map[1]
        for label in labels_list:
            labels[label] += 1
    # Sort labels by count in descending order and select top 2
    # Assuming you have a dictionary named label_counts where keys are labels and values are counts

    # Get unique frequencies
    unique_frequencies = set(labels.values())

    # Sort unique frequencies in descending order
    sorted_frequencies = sorted(unique_frequencies, reverse=True)

    # Get the top 2 highest frequencies
    top_2_frequencies = sorted_frequencies[:2]

    # Get all labels with the top 2 highest frequencies
    top_labels = [label for label, count in labels.items() if count in top_2_frequencies]
    #top_labels = [label for label, _ in sorted_labels[:config['top_number_of_labels_taken']]]
  
    return top_labels


def compute_doc_label_congregation(metrics):
    """
    Filter list of label probablities per page based on tuned threshold value

    Data structure:
    {
    "label: "probability",
    "label": "probability"
    }
    """
    labels = dict()
    threshhold = config['doc_label_congregation_threshhold']
    for label, probability in metrics.items():
        if probability > threshhold:
            labels[label] = probability
    return labels


def compute_label_vector_map(labels, processed_docs):

    """
    Compute label vectors based on pdf corpus by:
        1. Use word2vec to compute vectors for entire pdf.
        2. For each label, compute tf-idf based on similarity scores above a given threshold
    """

    ## turn labels into vector representation using the whole corpu
    tokenized_labels = [word_tokenize(label.lower()) for label in labels]
    all_sentences  = []
    for doc in processed_docs:
        for sentence_array in doc.text:
            all_sentences.append(sentence_array)
        for arr in doc.title:
            all_sentences.append(arr)
    # Train Word2Vec model
    model = Word2Vec(sentences=all_sentences, vector_size=config['word2vec_vector_size'], window=config['word2vec_window_size'], min_count=1, workers=4)
    label_vectors = defaultdict(dict)
    for sublist in tokenized_labels:
        for label in sublist:
            if label in model.wv:
                label_vectors[label] = model.wv[label]


    ##now we do cosine sim and tf idf COMBINED
    label_vector_tfidf = defaultdict()
    for label_key, label_value in label_vectors.items():
        doc_tf_idf = defaultdict()
        for doc in processed_docs:
            for sentence_array in doc.text:
                for word in sentence_array:
                    vec1 = label_value
                    vec2 = model.wv[word]
                    result = cosine_sim(vec1, vec2)
                    if result > 0.5:
                        ##its similar enough, count it
                        word = word + '.X'
                        if doc.doc_id in doc_tf_idf:
                            doc_tf_idf[doc.doc_id] += config['text_weight']
                        else:
                            doc_tf_idf[doc.doc_id] = config['text_weight']
            for element in doc.title:
                for word in element:
                    vec1 = label_value
                    vec2 = model.wv[word]
                    result = cosine_sim(vec1, vec2)
                    if result > config['label_vector_cosine_threshhold']:
                        ##its similar enough, count it
                        if doc.doc_id in doc_tf_idf:
                            doc_tf_idf[doc.doc_id] += config['title_weight']
                        else:
                            doc_tf_idf[doc.doc_id] = config['title_weight']

        vector = doc_tf_idf.values()
        label_vector_tfidf[label_key] = vector
   

    return (label_vectors, model)



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
    num = sum(x * y for x, y in zip(x, y))
    if num == 0:
        return 0
    
    return num / ((norm(list(x))) * norm(list(y)))


def model(pdf_docs_array):
    """
        pdf_docs_array: a list of formatted pdfs
        Each pdf, rather than the raw doc, is an array of individual
        Document objects, and in each are the text and metadata
    """
    labels = read_labels_from_file('label.txt')
    multi_word = read_labels_from_file('multi_word_phrases.txt')
    pdf_with_labels = []
    for docs_array in pdf_docs_array:
        processed_docs = process_docs_array(docs_array, multi_word)
        doc_freqs = compute_doc_freqs(processed_docs)
        label_vector_map_tuple = compute_label_vector_map(labels, processed_docs)
        doc_vectors = [compute_expo_tfidf(doc, doc_freqs, len(processed_docs), label_vector_map_tuple[1]) for doc in processed_docs]
    

        metrics=[]
        list_of_label_lengths=[]
        for tuplet in doc_vectors:
            doc_itself = tuplet[0]
            sentence_vector_list = tuplet[1]

            """
            Data structure

            One Document => 
            [
                
            {sentence vector}
            {sentence vector}
            ]
            """
            doc_to_label_freq = dict()
            for label  in label_vector_map_tuple[0].keys():
                probabilities_map = dict()
                for sentence_map in sentence_vector_list:
                    label_vector = label_vector_map_tuple[0][label]
                    probability = cosine_sim(label_vector, sentence_map.values())
                    probabilities_map[label] = probability
                if len(probabilities_map) > 0:
                    label_average = sum(probabilities_map.values())/len(probabilities_map.values())

                    doc_to_label_freq[label] = label_average
                
            associated_labels = compute_doc_label_congregation(doc_to_label_freq)
            metrics.append((doc_itself, associated_labels))
            list_of_label_lengths.append(len(associated_labels))

        """
        [
        (Document --> [labels])
        ]
        """

        congregated_labels = compute_relevant_labels(metrics)
        

        pdf_with_labels.append(
            {
                "docs_array":docs_array, 
                "labels":congregated_labels,
                "docs_length": len(docs_array),
                "label_length" : len(congregated_labels),
                "list_of_label_lengths": list_of_label_lengths
            }
        )
       


    return pdf_with_labels
        

def search(doc_vector, label_vec):
    score = cosine_sim(label_vec, doc_vector)
    return score
