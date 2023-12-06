"""
Example script to index elasticsearch documents.
"""
import argparse
import json
from elasticsearch import Elasticsearch
from elasticsearch.helpers import bulk


def load_dataset(path):
    with open(path, 'r', encoding='utf-8') as f:
        return [json.loads(line) for line in f]


def main(args):
    client = Elasticsearch('localhost:9200')
    docs = load_dataset(args.data)
    print('doc len: ', len(docs))
    bulk(client, docs)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Indexing elasticsearch documents.')
    parser.add_argument('--data', default='documents.jsonl', help='Elasticsearch documents.')
    args = parser.parse_args()
    main(args)
