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
    '''
    tokenizes the text and title of a document and removes stopwords
    turns the sentences in text and title into a list of words (a doc is a list of sentences where each sentence is a list of words)
    returns a new Document object with tokenized(list of words) text and title
    '''
    sentences = []
    title_sentences=[]
    # turn the sentences into a list of words
    for sentence in doc.text:
        # print(sentence)
        sentence_words = word_tokenize(sentence)
        sentence_words = [word for word in sentence_words if word.lower() not in stopwords]
        sentences.append(sentence_words)
    for element in doc.title:        
        sentence_words = word_tokenize(element)
        sentence_words = [word for word in sentence_words if word.lower() not in stopwords]
        title_sentences.append(sentence_words)

    return Document(doc_id=doc.doc_id, created=doc.created, modified=doc.modified, url=doc.url ,
                    author=doc.author,
                    title=title_sentences, 
                    text = sentences)

def remove_stopwords(pdf: List[Document]):
    '''
    Removes stopwords from a each document in a PDF
    '''
    return [remove_stopwords_doc(doc) for doc in pdf]



def process_pdf(pdf):
    '''
    Processes a PDF document by removing stopwords and stemming words
    Each pdf is a list of Document objects
    returns a list of Document objects where the text and title have been processed ()
    '''
    processed_pdf = remove_stopwords(pdf)
    #processed_docs = stem_docs(processed_docs)
    return processed_pdf


### Term-Document Matrix


def compute_doc_freqs(pdf: List[Document]):
    '''
    Computes document frequency, i.e. how many documents contain a specific word
    Returns a dictionary where the key is the word and the value is the number of documents containing the word
    '''
    freq = Counter()
    for doc in pdf:
        # find the frequency of each unique word in the document for all documents in the pdf
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


def compute_expo_tfidf(doc, doc_freqs, n, terms):
    # we need doc frequency and term frequency

    doc_list = []
    
    for sentence in doc.text:
        sentence_array = sentence
        important_term = False
        for i, word in enumerate(sentence_array):
            for term in terms:
                if term in word:
                    sentence_array[i] += ".X"
                    important_term = True
            # if word in terms:
            #     print(word)
            #     sentence_array[i] += ".X"
            #     important_term = True
                

        # now that the pivot point is denoted, we do exponential decay     

        tf_row = {}
        decay_value = config['decay_value']
        if important_term == True:
             for index , word in enumerate(sentence_array):
                if ".X" in word:
                 tf_row[word] = config['label_keyword_weight']
                else:
                  #  tf_row[word] = 0.01
                    x_indices = [i for i, word in enumerate(sentence_array) if ".X" in word]
                    decay_values = [1/(abs(x_indi - index) * decay_value) for x_indi in x_indices]
                    tf_row[word] = sum(decay_values)/len(decay_values)

        else:
            for word in sentence_array:
                    tf_row[word] = config['uniform_decay_value']
        
        doc_list.append(tf_row)

    for i,sentence in enumerate(doc_list):
        for word,weight in sentence.items():
            if word in doc_freqs:
                doc_list[i][word] = (weight *  math.log2((n/doc_freqs[word])))
   
    return (doc, doc_list)
    
    
        
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


def compute_relevant_labels(docs_array): 
    labels = Counter()

    # Count the occurrences of each label
    for doc_map in docs_array:
        labels_list = doc_map[1]
        for label in labels_list:
            labels[label] += 1
    # Sort labels by count in descending order and select top 2
    sorted_labels = sorted(labels.items(), key=lambda x: x[1], reverse=True)
    top_labels = [label for label, _ in sorted_labels[:config['top_number_of_labels_taken']]]
    print(top_labels)
    return top_labels


def compute_doc_label_congregation(metrics):
    """
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


def compute_label_vector_map(labels, processed_pdf):
    ## turn labels into vector representation using the whole corpus
    # print(labels)
    tokenized_labels = [word_tokenize(label.lower()) for label in labels]
    # print(tokenized_labels)
    all_sentences  = []
    # combine all sentences in the pdf
    for doc in processed_pdf:
        # print(doc.text)
        for sentence_array in doc.text:
            all_sentences.append(sentence_array)
        for arr in doc.title:
            all_sentences.append(arr)
    # Train Word2Vec model in order to obtain 
    # train a model on a set of sentences (all_sentences).
    # The vector_size parameter determines the size of the word vectors.
    # The window parameter determines the number of context words to observe in each direction.
    # The min_count parameter specifies the minimum frequency a word must have to be included in the vocabulary.
    # The workers parameter determines the number of extra threads to use for training, which can speed up processing on multi-core machines
        # TODO try different vector size
        # TODO try different window size
    model = Word2Vec(sentences=all_sentences, vector_size=config['word2vec_vector_size'], window=config['word2vec_window_size'], min_count=1, workers=4)
    label_vectors = defaultdict(dict)
    for sublist in tokenized_labels:
        for label in sublist:
            if label in model.wv:

                label_vectors[label] = model.wv[label]
    # print(label_vectors)

    ##now we do cosine sim and tf idf COMBINED
    label_vector_tfidf = defaultdict()
    for label_key, label_value in label_vectors.items():
        # print("key and value", label_key, label_value)
        doc_tf_idf = defaultdict()
        for doc in processed_pdf:
            for sentence_array in doc.text:
                for word in sentence_array:
                    # represents the how 
                    vec1 = label_value
                    vec2 = model.wv[word]
                    result = cosine_sim(vec1, vec2)
                    if result > 0.5:
                        ##its similar enough, count it
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
   
    # for label, tokenized_label in zip(labels, all_sentences):
    #     # Check if there are sentences corresponding to the label
    #     if tokenized_label:
    #         # Initialize label vector
    #         label_vector = [0] * model.vector_size
    #         word_count = 0

    #         # Accumulate word vectors for words present in the Word2Vec model's vocabulary
    #         for word in tokenized_label:
    #             if word in model.wv:
    #                 label_vector += model.wv[word]
    #                 word_count += 1

    #         # Normalize label vector by dividing by the number of words present in the Word2Vec model's vocabulary
    #         if word_count > 0:
    #             label_vector /= word_count

    #         # Store label vector
    #         label_vectors[label] = label_vector
    # for label in labels:
    #     freq = dict()
    #     for doc in processed_docs:
    #         for element in doc.text:
    #             # add words in the text
    #             for word in element:
    #                 if label == word:
    #                     if doc.doc_id in freq:
    #                         freq[doc.doc_id] += 1
    #                     else:
    #                         freq[doc.doc_id] = 1
    #     label_vectors[label] = freq

    

# # Initialize lemmatizer
#     lemmatizer = WordNetLemmatizer()

#     for label in labels:
#         freq = defaultdict(int)  # Use defaultdict to simplify frequency counting
#         for doc in processed_docs:
#             for element in doc.text:
#                 for word in element:
#                     # Lemmatize the word to its root form
#                     lemma_word = lemmatizer.lemmatize(word)
#                     if label == lemma_word:
#                         freq[doc.doc_id] += 1
        # label_vectors[label] = freq
    # for label in labels:
    #     # Tokenize the label
    #     tokens = tokenizer.tokenize(label)
    #     # Convert tokens to token IDs
    #     token_ids = tokenizer.convert_tokens_to_ids(tokens)
    #     # Generate contextual embeddings
    #     with torch.no_grad():
    #         outputs = model(torch.tensor([token_ids]))
    #     # Extract contextual embeddings for the [CLS] token
    #     embedding = outputs[0][:, 0, :].numpy()
    #     # Update label_vectors with the embedding
    #     label_vectors[label]= embedding

    return label_vectors

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
    num = sum(x * y for x, y in zip(x, y))
    if num == 0:
        return 0
    
    return num / ((norm(list(x))) * norm(list(y)))


def model(pdf_array):
    """
        pdf_array: a list of formatted pdfs
        Each pdf, rather than the raw doc, is an array of individual
        Document objects, and in each are the text and metadata
    """
    labels = read_labels_from_file('label.txt')
    pdf_with_labels = []
    # create doc vectors for each document in the pdf
    for pdf in pdf_array:
        processed_pdf = process_pdf(pdf)
        pdf_doc_freqs = compute_doc_freqs(processed_pdf)
        label_vector_map = compute_label_vector_map(labels, processed_pdf)
        doc_vectors = [compute_expo_tfidf(doc, pdf_doc_freqs, len(processed_pdf), label_vector_map.keys()) for doc in processed_pdf]


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
            for label  in label_vector_map.keys():
                probabilities_map = dict()
                for sentence_map in sentence_vector_list:
                    label_vector = label_vector_map[label]
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
                "docs_array":pdf, 
                "labels":congregated_labels,
                "docs_length": len(pdf),
                "label_length" : len(congregated_labels),
                "list_of_label_lengths": list_of_label_lengths
            }
        )


    return pdf_with_labels


    #     heat_map_data = dict()
    #     for tuplet in doc_vectors:
    #         doc_itself = tuplet[0]
    #         doc_vector = tuplet[1]
    #         terms = tuplet[2]
    #         #tf_array = np.array([[tf_row[term] for term in terms if term in tf_row] for tf_row in doc_vector])
    #         tf_array = np.zeros((len(doc_vector), len(terms)))

    #         # Iterate over each document vector and update tf_array
    #         for i, tf_row in enumerate(doc_vector):
    #             for j, term in enumerate(terms):
    #                 if term in tf_row:
    #                     tf_array[i, j] = tf_row[term]  
    #                 # else:
    #                 #     tf_array[i, j] = 0  
    #      #   initial_centroids = np.array([doc_vector.toarray()[np.where(designated_labels_encoded == label)[0]].mean(axis=0) for label in range(num_clusters)])
         
    #         cntr, u, _, _, _, _, _ = fuzz.cluster.cmeans(tf_array.T, num_clusters, 2, error=0.005, maxiter=1000)
    #         cluster_to_encoded_index_map = {cluster_index: encoded_int for encoded_int, cluster_index in zip(designated_labels_encoded, np.argmax(u, axis=0))}
    #         threshold = 0.051 # Set the threshold for membership values
            
    #         relevant_labels_all = []

    #         for i in range(len(doc_vector)):  # Iterate over each data point      
    #             cluster_membership = u[:, i]  # Get the membership values of the data point for all clusters
    #             relevant_clusters = np.where(cluster_membership > threshold)[0] # Find clusters where membership values are above the threshold
    #             if len(relevant_clusters) > 0:
    #                 relevant_encoded_int = [cluster_to_encoded_index_map.get(cluster, 0) for cluster in relevant_clusters]
    #                 relevant_labels = [encoded_int_to_label_map[encoded_int] for encoded_int in relevant_encoded_int]
    #                 relevant_labels_all.append(relevant_labels)
    #         new_doc_id = int(doc_itself.doc_id)-100
    #         heat_map_data[doc_itself.doc_id] ={
    #             'x': str(new_doc_id),
    #             'y': str(relevant_labels_all),
    #             'heat': len(relevant_labels_all)

    #         }
    #         metrics.append((doc_itself, compute_relevant_labels(relevant_labels_all)))

    #     """
    #     Now we have our metrics which is of the form
    #         metrics = [
    #         (
    #          Document object <-- which represents a page, [list of relevant labels]
    #         )
                
    #         ]

    #     We have to do one last consolidation, label congregation
    #     """
    #     #print(metrics)
    #     #print(compute_label_congregation(metrics))
    #     pdfs_with_labels.append((docs_array, compute_label_congregation(metrics)))
    #     data_metrics.append(heat_map_data)
    #     #print(compute_label_congregation(metrics))
    # return pdfs_with_labels,data_metrics    
        

def search(doc_vector, label_vec):
    score = cosine_sim(label_vec, doc_vector)
    return score
