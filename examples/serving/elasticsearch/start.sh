echo "create_index"
python src/create_index.py --index_file=src/index.json --index_name=jobsearch

echo "create_documents"
python src/create_documents.py --data=src/example.csv --save=src/documents.jsonl --index_name=jobsearch

echo "index_documents"
python src/index_documents.py --data=src/documents.jsonl

echo "start server"
python server.py