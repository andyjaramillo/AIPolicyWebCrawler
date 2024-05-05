import numpy as np
from main import main
import json

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
    average_length_of_docs = 0
    for dic in data:
        average_length_of_docs += dic["docs_length"] 
    average_length_of_docs = int(average_length_of_docs / len(data))

    matrixLabels=[]
    for pdf_dict in data:
        matrixLabels.append(pdf_dict["list_of_label_lengths"])
    
    matrix = np.array(matrixLabels)
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
    return json.dumps(data_dict)


