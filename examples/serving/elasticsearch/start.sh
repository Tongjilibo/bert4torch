echo "create_index"
python example/create_index.py --index_file=example/index.json --index_name=jobsearch

echo "create_documents"
python example/create_documents.py --data=example/example.csv --index_name=jobsearch

echo "index_documents"
python example/index_documents.py

echo "start server"
python app.py