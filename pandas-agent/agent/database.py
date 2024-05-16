import networkx as nx
from typing import List
import json
import chromadb
import chromadb.utils.embedding_functions as embedding_functions
import os
from chromadb.utils.batch_utils import create_batches
import yaml
from dotenv import load_dotenv, find_dotenv

with open("config.yaml") as stream:
    try:
        config_params = yaml.safe_load(stream)
    except yaml.YAMLError as exc:
        print(exc)

def build_docs_metadata(pandas_graph):
    docs = []
    metadata = []

    for node, attributes in pandas_graph.nodes(data=True):
        if attributes['type'] == 'function_node':
            docs.append(attributes['function_desc'])
            metadata.append(attributes)

        elif attributes['type'] == 'parent_node':
            docs.append(attributes['node_description'])
            attributes['name'] = node
            metadata.append(attributes)
    return docs, metadata

def build_database(docs, metadata, api_key):
    database_path = config_params["VECTORDB"]["BASE_DATABASE_PATH"]
    collection_name = config_params["VECTORDB"]["COLLECTION_NAME"]
    load_dotenv(find_dotenv(),override=True)
    emb_fn = embedding_functions.OpenAIEmbeddingFunction(
        api_key=api_key, model_name=config_params["VECTORDB"]["EMBEDDING_MODEL_NAME"]
    )

    client = chromadb.PersistentClient(path=database_path)
    pandas_collection = client.create_collection(
        name=collection_name, embedding_function=emb_fn
    )

    pandas_ids = [f"id{i}" for i in range(len(docs))]
    batches = create_batches(
        api=client, ids=pandas_ids, documents=docs, metadatas=metadata
    )
    for batch in batches:
        pandas_collection.add(ids=batch[0], documents=batch[3], metadatas=batch[2])
    return pandas_collection

def load_database(api_key):
    database_path = config_params["VECTORDB"]["BASE_DATABASE_PATH"]
    collection_name = config_params["VECTORDB"]["COLLECTION_NAME"]
    emb_fn = embedding_functions.OpenAIEmbeddingFunction(
        api_key=api_key, model_name=config_params["VECTORDB"]["EMBEDDING_MODEL_NAME"]
    )
    client = chromadb.PersistentClient(path=database_path)
    pandas_collection = client.get_collection(
        name=collection_name, embedding_function=emb_fn
    )
    return pandas_collection