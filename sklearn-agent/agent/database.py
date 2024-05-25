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

