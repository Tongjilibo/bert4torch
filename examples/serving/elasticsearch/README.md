## 插入index
```bash
$ python example/create_index.py --index_file=example/index.json --index_name=jobsearch
# index.json
{
  "settings": {
    "number_of_shards": 2,
    "number_of_replicas": 1
  },
  "mappings": {
    "dynamic": "true",
    "_source": {
      "enabled": "true"
    },
    "properties": {
      "title": {
        "type": "text"
      },
      "text": {
        "type": "text"
      },
      "text_vector": {
        "type": "dense_vector",
        "dims": 768
      }
    }
  }
}
```

## 调用模型生成embedding
```bash
$ python example/create_documents.py --data=example/example.csv --index_name=jobsearch
# example/example.csv
"Title","Description"
"Saleswoman","lorem ipsum"
"Software Developer","lorem ipsum"
"Chief Financial Officer","lorem ipsum"
"General Manager","lorem ipsum"
"Network Administrator","lorem ipsum"
```


### 插入es
```bash
$ python example/index_documents.py
```

### 7. Open browser

Go to <http://127.0.0.1:5000>.