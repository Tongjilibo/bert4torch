"""
Example script to create elasticsearch index.
"""
import argparse

from elasticsearch import Elasticsearch


def main(args):
    client = Elasticsearch()
    client.indices.delete(index=args.index_name, ignore=[404])
    with open(args.index_file) as index_file:
        source = index_file.read().strip()
        client.indices.create(index=args.index_name, body=source)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Creating elasticsearch index.')
    parser.add_argument('--index_file', default='index.json', help='Elasticsearch index file.')
    parser.add_argument('--index_name', default='jobsearch', help='Elasticsearch index name.')
    args = parser.parse_args()
    main(args)
