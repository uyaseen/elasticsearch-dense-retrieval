# elasticsearch-dense-retrieval

This project demonstrates **semantic document search** (*document retrieval*) using [elasticsearch](https://www.elastic.co/) and [sentence-transformers](https://www.sbert.net/). In contrast to the traditional lexical search (e.g. [BM25](https://en.wikipedia.org/wiki/Okapi_BM25)), the semantic search can tolerate spelling mistakes as vector representations can capture notion of similarity between semantically similar words (thanks to word embeddings!). [Elasticsearch 7.3](https://www.elastic.co/guide/en/elasticsearch/reference/7.3/query-dsl-script-score-query.html#vector-functions) provides a `cosineSimilarity` function for vector fields, this enables convenient document retrieval based on vector similarity.

The demo application included in this project can enable the user to search health related questions based on the data scrapped from [NHS website](https://www.nhs.uk/conditions/).


## Usage

### 1. Download dataset

Download the NHS website data from [here](https://drive.google.com/drive/folders/1Remgw6uOrZ5UdCRfN675SpP9pat9rQVB?usp=sharing) and copy the `data` directory inside the project.

### 2. Run Docker containers

```bash
docker compose up
```

### 3. Run pipeline script to create index, process and index documents

```bash
conda create -n es python=3.8
conda activate es
pip install -r es/requirements.txt
python es/es_pipeline.py
```

### 4. Search

Open the search interface by opening http://localhost:8501/ in your browser.


### 5. Improvements

If you are not happy with the results, try experimenting with [pretrained models](https://www.sbert.net/docs/pretrained_models.html) relevant to your domain, or consider [adapting](https://www.sbert.net/examples/domain_adaptation/README.html) the "general domain" models to your target domain.

### Acknowledgements

I consulted the codebase of [bertsearch](https://github.com/Hironsan/bertsearch) & [pinecone-io/examples](https://github.com/pinecone-io/examples/) for this project. 