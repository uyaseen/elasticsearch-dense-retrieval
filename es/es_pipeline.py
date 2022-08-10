"""
Example script to create index, process and index documents.
"""
from elasticsearch import Elasticsearch
from elasticsearch.helpers import bulk
from sentence_transformers import SentenceTransformer
from utils import (
    load_nhs_documents,
    create_es_document,
    compute_embeddings,
    load_dataset,
)
import os
import json
import logging
import argparse


def create_index(client, index, settings, mappings):
    if not client.indices.exists(index):
        client.indices.create(
            index=index, settings=settings, mappings=mappings,
        )
        logging.info(f"{index} created")


def create_documents(data_path, processed_path, model_name, index):
    if not os.path.exists(processed_path):
        documents = load_nhs_documents(data_path)
        logging.info(f"{len(documents)} documents processed ...")
        model = SentenceTransformer(model_name)
        with open(processed_path, "w") as f:
            for doc, emb in zip(documents, compute_embeddings(model, documents)):
                d = create_es_document(doc, emb, index)
                f.write(json.dumps(d) + "\n")
        logging.info(f"{len(documents)} documents written to {processed_path} ...")


def index_documents(client, processed_path):
    doc_counts = int(
        client.cat.count("nhs_health", params={"format": "json"})[0]["count"]
    )
    if doc_counts == 0:
        docs = load_dataset(processed_path)
        bulk(client, docs)
        logging.info(f"{len(docs)} documents indexed ...")


def main(args):
    with open(args.config) as config_file:
        config = json.load(config_file)
    client = Elasticsearch("http://localhost:9200")
    create_index(
        client,
        config["es"]["index"],
        config["es"]["settings"],
        config["es"]["mappings"],
    )
    create_documents(
        config["data"]["path"],
        config["data"]["processed_path"],
        config["embeddings"]["model"],
        config["es"]["index"],
    )
    index_documents(client, config["data"]["processed_path"])


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser(description="Creating elastic search index")
    parser.add_argument(
        "--config",
        default="config.json",
        help="elasticsearch-dense-retrieval configurations",
    )
    args = parser.parse_args()
    main(args)
