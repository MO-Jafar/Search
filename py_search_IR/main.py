import numpy as np
from tabulate import tabulate
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize

stop_words = set(stopwords.words("english"))
ps = PorterStemmer()


# Read from file
def read_text_files_from_folder(folder_path):
    documents = []
    for i in range(1, 4):
        with open(f"{folder_path}/doc{i}.txt", "r") as file:
            documents.append(file.read())
    return documents


def start_indexing():
    import string

    inv_index = {}
    docs = read_text_files_from_folder("docs")

    docIndex = 0
    for doc in docs:
        tokens = word_tokenize(doc)
        docs[docIndex] = tokens
        for term in tokens:
            if term in stop_words or term in string.digits:
                continue
            term = ps.stem(term)
            if term not in inv_index:
                inv_index[term] = {}
                inv_index[term]["df"] = 0
                inv_index[term]["cf"] = 0
                inv_index[term]["tfs"] = []
                inv_index[term]["docs"] = []
            if docIndex not in inv_index[term]["docs"]:
                inv_index[term]["tfs"].append(1)
                inv_index[term]["docs"].append(docIndex)
                inv_index[term]["df"] += 1
            else:
                ind = inv_index[term]["docs"].index(docIndex)
                inv_index[term]["tfs"][ind] += 1
            inv_index[term]["cf"] += 1

        docIndex += 1

    sorted_dict = dict(sorted(inv_index.items()))
    with open("index3.py", "w", encoding="utf-8") as file:
        file.write(f"index={sorted_dict}")
    meta = {"total_doc": len(docs), "tw": [len(item) for item in docs]}
    with open("index3.py", "a") as file:
        file.write(f"\nmeta={meta}")


# folder_path = "docs"  # Replace with the actual folder path
# documents = read_text_files_from_folder(folder_path)


def search(query):
    import index3
    import math

    query_vector = {term: 0 for term in index3.index}
    toknized_q = word_tokenize(query)

    for term in toknized_q:
        raw_t = term
        if term in stop_words:
            continue
        term = ps.stem(term)
        if term in index3.index:
            tf = toknized_q.count(raw_t) / len(toknized_q)
            idf = math.log2(index3.meta["total_doc"] / index3.index[term]["df"])
            query_vector[term] = tf * idf
    vectors = []
    # print(query_vector)
    for i in range(index3.meta["total_doc"]):
        vector = []
        for term in index3.index:
            if query_vector[term] == 0:
                vector.append(0)
                continue
            if i in index3.index[term]["docs"]:
                tf = (
                    index3.index[term]["tfs"][index3.index[term]["docs"].index(i)]
                ) / index3.meta["tw"][i]
                idf = math.log2(index3.meta["total_doc"] / index3.index[term]["df"])
                vector.append(tf * idf)
            else:
                vector.append(0)
        vectors.append(vector)
    query_vector = list(query_vector.values())
    print(query_vector)
    similarities = [np.dot(query_vector, vector) for vector in vectors]
    print(similarities)

    ranked = sorted(enumerate(similarities, 1), key=lambda x: x[1], reverse=True)
    headers = ["Rank", "Score", "Document"]
    data = []

    r = 0
    for rank, score in ranked:
        r = r + 1
        data.append([r, score, f"doc{rank}"])
    print(tabulate(data, headers, tablefmt="fancy_grid"))


# start_indexing()
# search("machine  Science")


import sys

print(sys.argv)
if sys.argv[1] == "buildindex":
    print("building index...")
    start_indexing()
elif sys.argv[1] == "search":
    query = input("Enter Search Query: ")
    search(query)

print(sys.argv)
