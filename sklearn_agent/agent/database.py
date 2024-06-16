import json
import chromadb
import chromadb.utils.embedding_functions as embedding_functions
import os
from copy import deepcopy
import re
import ast
from chromadb.utils.batch_utils import create_batches
import yaml
from dotenv import load_dotenv, find_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter

with open("sklearn_agent/config.yaml") as stream:
    try:
        config_params = yaml.safe_load(stream)
    except yaml.YAMLError as exc:
        print(exc)


def clean_text(s: str):
    s = re.sub("\n", " ", s)
    s = json.dumps(s)
    return ast.literal_eval(s)


def build_graph_based_docs_metadata(sklearn_graph):
    text_splitter = RecursiveCharacterTextSplitter(
        separators=[
            "\n\n",
            "\n",
            " ",
            ".",
            ",",
            "\u200b",  # Zero-width space
            "\uff0c",  # Fullwidth comma
            "\u3001",  # Ideographic comma
            "\uff0e",  # Fullwidth full stop
            "\u3002",  # Ideographic full stop
            "",
        ],
        length_function=len,
        is_separator_regex=False,
        chunk_size=2500,
        chunk_overlap=400,
    )
    with open(
        config_params["PARENTS_SUMMARY"]["SUMMARY_JSON_FILE_PATH"], "r"
    ) as jsonfile:
        parent_summary_dict = json.load(jsonfile)

    embed_docs = []
    embed_metadata = []
    for node, attr in sklearn_graph.nodes(data=True):
        type = attr.get("type")
        if type == "function_node":
            ftext = attr["function_text"]
            if ftext == "":
                continue
            embed_docs.append(clean_text(attr["function_text"]))
            for k, v in attr.items():
                if not isinstance(v, str):
                    attr[k] = str(v)
            embed_metadata.append(attr)
        elif type == "parent_node":
            parent_summary = parent_summary_dict[node]
            parent_summary_text_split = text_splitter.split_text(parent_summary)
            for k, v in attr.items():
                if not isinstance(v, str):
                    attr[k] = str(v)
            attr.update({"name": node})
            for parent_splits in parent_summary_text_split:
                embed_docs.append(clean_text(parent_splits))
                embed_metadata.append(attr)
        elif type == "sub_level_node":
            for ctext in attr["child_texts"]:
                embed_docs.append(clean_text(ctext))
                sub_level_attr = deepcopy(attr)
                del sub_level_attr["child_texts"]
                for k, v in sub_level_attr.items():
                    if not isinstance(v, str):
                        sub_level_attr[k] = str(v)
                sub_level_attr.update({"name": node})
                embed_metadata.append(sub_level_attr)
    return embed_docs, embed_metadata


def build_docs_metadata():
    docs = []
    metadata = []
    with open(
        config_params["FUNCTION_CALLING_DATASET"]["FUNCTION_SAVE_DEST"], "r"
    ) as jsonfile:
        function_calling_data = json.load(jsonfile)
    for parent in function_calling_data:
        parent_data = function_calling_data[parent]
        parent_name = parent_data["name"]
        for sub_level in parent_data["functions"]:
            for sub_level_name, sub_level_funcs in sub_level.items():
                # defaults --> function definitions
                for funcs in sub_level_funcs:
                    function_text = funcs["function_definitions"]["function_text"]
                    function_text = function_text.replace("\n\n", " ")
                    function_text = function_text.replace("\n", " ")
                    function_text = function_text.replace("Examples", " ")
                    docs.append(function_text)
                    metadata.append(
                        {
                            "function_name": funcs["func_name"],
                            "function_url": funcs["func_url"],
                            "full_function": funcs["function_definitions"][
                                "full_function"
                            ],
                            "function_calling": str(funcs["function_calling"]),
                            "parent": parent_name,
                            "sub_level_name": sub_level_name,
                            "sub_level_trail": parent_name,
                            "function_trail": f"{parent_name}-->{sub_level_name}",
                            "parameters_names_desc": str(funcs["function_definitions"][
                                "parameter_names_desc"
                            ]),
                        }
                    )
    return docs, metadata


# @retry(wait=wait_random_exponential(min=1,max=60),stop=stop_after_attempt(6))
def build_database(docs, metadata, api_key):
    database_path = config_params["VECTORDB"]["BASE_DATABASE_PATH"]
    collection_name = config_params["VECTORDB"]["COLLECTION_NAME"]
    load_dotenv(find_dotenv(), override=True)
    emb_fn = embedding_functions.OpenAIEmbeddingFunction(
        api_key=api_key, model_name="text-embedding-3-small"
    )

    client = chromadb.PersistentClient(path=database_path)
    sklearn_collection = client.create_collection(
        name=collection_name, embedding_function=emb_fn
    )

    sklearn_ids = [f"id{i}" for i in range(len(docs))]
    batches = create_batches(
        api=client, ids=sklearn_ids, documents=docs, metadatas=metadata
    )
    for batch in batches:
        sklearn_collection.add(ids=batch[0], documents=batch[3], metadatas=batch[2])
    return sklearn_collection


def load_database(api_key):
    database_path = config_params["VECTORDB"]["BASE_DATABASE_PATH"]
    collection_name = config_params["VECTORDB"]["COLLECTION_NAME"]
    emb_fn = embedding_functions.OpenAIEmbeddingFunction(
        api_key=api_key, model_name=config_params["VECTORDB"]["EMBEDDING_MODEL_NAME"]
    )
    client = chromadb.PersistentClient(path=database_path)
    sklearn_collection = client.get_collection(
        name=collection_name, embedding_function=emb_fn
    )
    return sklearn_collection


def batch_build_database(embed_docs, embed_metadata):
    BATCH_SIZE = config_params["VECTORDB"]["BATCH_SIZE"]
    for start in range(0, len(embed_docs), BATCH_SIZE):
        end = min(start + BATCH_SIZE, len(embed_docs))
        curr_docs = embed_docs[start:end]
        curr_metadata = embed_metadata[start:end]
        build_database(start, curr_docs, curr_metadata, os.environ["OPENAI_API_KEY"])
        print(f"Done for {start}-{end}/{len(embed_docs)}")
