import numpy as np
from main import main
import json
from collections import defaultdict

"""
@params:
    data = [
    {
        "docs_array":docs_array, 
        "labels":congregated_labels,
        "docs_length": len(docs_array),
        "label_length" : len(congregated_labels)
        "list_of_label_lengths": list_of_label_lengths
    }
]

"""
def format_data(data):
    topics = [
              "virtual", 
              "reality",
              "vr",
              "artificial",
              "intelligence",
              "biotechnology",
              "chatgpt",
              "machine",
              "learning",
              "privacy",
              "health",
              "education",
              "cybersecurity",
              "robotics",
              "finance",
              "space",
              "autonomous",
              "vehicles"
             ]
    result_dict = defaultdict(list)
    for label in topics:
        for pdf in data:
            if label in pdf["labels"]:
                result_dict[label].append(pdf) 
    final_dict = defaultdict()
    for label_ in result_dict:
        average_length_of_docs = 0
        average_label_length = 0
        matrixLabels=[]
        for docs in result_dict[label_]:
            average_length_of_docs += docs["docs_length"] 
            average_label_length += docs["label_length"]
            matrixLabels.append(docs["list_of_label_lengths"]) 
        average_length_of_docs = int(average_length_of_docs / len(data))
        max_length = max(len(sublist) for sublist in matrixLabels)
        padded_list = [sublist + [0] * (max_length - len(sublist)) for sublist in matrixLabels]
        matrix = np.array(padded_list)
        transposed_matrix = matrix.T
        averageLabels=[]
        for doc_label_bins in transposed_matrix:
            bin_sum = 0
            for bin_ in doc_label_bins:
                bin_sum += bin_
            average_label_length_for_doc = bin_sum/len(doc_label_bins)
            averageLabels.append(average_label_length_for_doc)
        data_dict = []
        for i in range(average_length_of_docs):
            if i < len(averageLabels)-1:
                data_dict.append({"x": str(i), "y": str(0),"heat": averageLabels[i]})
            else:
                data_dict.append({"x": str(i), "y": str(0),"heat": 0})
        final_dict[label_] = data_dict
    pdf_names = ''
    pdf_urls = ''
    for pdf in data:
        pdf_names += ' '.join(pdf['docs_array'][0].title) + ';'
        pdf_urls += pdf['docs_array'][0].url + ','
    return json.dumps({
        'final_dict':final_dict,
        'names': pdf_names,
        'urls': pdf_urls
        })


