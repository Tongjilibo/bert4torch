"""
Example script to create elasticsearch documents.
"""
import argparse
import json
import pandas as pd
from tqdm import tqdm
from model import BertClient
bc = BertClient(batch_size=128, use_tqdm=False)


def create_document(doc, emb, index_name):
    return {
        '_op_type': 'index',
        '_index': index_name,
        'text': doc['text'],
        'title': doc['title'],
        'text_vector': emb
    }


def load_dataset(path):
    docs = []
    df = pd.read_csv(path, encoding='utf-8')
    for row in df.iterrows():
        series = row[1]
        doc = {
            'title': series.Title,
            'text': series.Description
        }
        if len(series.Description) < 50:
            continue
        docs.append(doc)
    return docs


def bulk_predict(docs, batch_size=256):
    """Predict bert embeddings."""
    for i in tqdm(range(0, len(docs), batch_size)):
        batch_docs = docs[i: i+batch_size]
        embeddings = bc.encode([doc['text'] for doc in batch_docs])
        for emb in embeddings:
            yield emb


def main(args):
    docs = load_dataset(args.data)
    with open(args.save, 'w', encoding='utf-8') as f:
        for doc, emb in zip(docs, bulk_predict(docs)):
            d = create_document(doc, emb, args.index_name)
            f.write(json.dumps(d, ensure_ascii=False) + '\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Creating elasticsearch documents.')
    parser.add_argument('--data', default='./example.csv', help='data for creating documents.')
    parser.add_argument('--save', default='documents.jsonl', help='created documents.')
    parser.add_argument('--index_name', default='jobsearch', help='Elasticsearch index name.')
    args = parser.parse_args()
    main(args)
