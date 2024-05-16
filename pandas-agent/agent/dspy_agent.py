import dspy
from dotenv import load_dotenv, find_dotenv
from tqdm import tqdm
import json
import yaml
import chromadb.utils.embedding_functions as embedding_functions
import os
from agent.utils import *

with open("config.yaml") as stream:
    try:
        config_params = yaml.safe_load(stream)
    except yaml.YAMLError as exc:
        print(exc)
load_dotenv(find_dotenv(), override=True)


class FirstSecondLevel(dspy.Signature):
    "You are given a list of keys and values separated by semicolon."
    "Based on the query, you have to output the key that is most relevant to the question separated by semicolon."
    "Be precise and output only the relevant key or keys from the provided keys only."
    "Don't include any other information"

    query = dspy.InputField(prefix="Query which you need to classify: ", format=str)
    keys_values = dspy.InputField(prefix="Keys and Values: ", format=str)
    output = dspy.OutputField(
        prefix="Relevant Key(s): ",
        format=str,
        desc="relevant keys separated by semicolon",
    )


emb_fn = embedding_functions.OpenAIEmbeddingFunction(
    api_key=os.environ["OPENAI_API_KEY"],
    model_name=config_params["VECTORDB"]["EMBEDDING_MODEL_NAME"],
)

llm = dspy.OpenAI()
dspy.settings.configure(lm=llm)


class PandasAgentChroma(dspy.Module):
    def __init__(self, collection):
        super(PandasAgentChroma, self).__init__()
        self.collection = collection
        self.firstSecondLevel = dspy.Predict(FirstSecondLevel)

    def __call__(self, *args, **kwargs):
        return super().__call__(*args, **kwargs)

    def forward(self, query: str):
        query_emb = emb_fn([query])[0]

        # Parent level querying
        parent_level = self.collection.query(
            query_embeddings=query_emb,
            where={
                "type": {"$eq": "parent_node"},
            },
            n_results=3,
        )
        parent_level_str = ""
        for parent_level_metadata in parent_level["metadatas"][0]:
            parent_level_str += f"{parent_level_metadata['name']}: {parent_level_metadata['node_description']}\n\n"

        parent_level_answer = self.firstSecondLevel(
            query=query, keys_values=parent_level_str
        ).output
        print(parent_level_str, parent_level_answer)
        trail_list = [parent_level_answer.split(";")]
        trail_list_pairs = generate_pairs_recursive(trail_list)

        trail_where_clause = get_trail_list_pairs(trail_list_pairs)

        function_level = self.collection.query(
            query_embeddings=query_emb,
            where={
                "$and": [
                    trail_where_clause,
                    {"type": {"$eq": "function_node"}},
                ]
            },
            n_results=5,
        )

        function_level_str = ""
        for function_level_metadata in function_level["metadatas"][0]:
            function_level_str += f"{function_level_metadata['function_name']}: {function_level_metadata['function_desc']}\n\n"
        print(function_level_str)
        function_level_answer = self.firstSecondLevel(
            query=query, keys_values=function_level_str
        ).output
        function_list = generate_pairs_recursive([function_level_answer.split(";")])
        function_where_clause = get_trail_list_pairs(function_list, "function_name")
        print(function_where_clause)
        functions = self.collection.get(
            where={
                "$and": [
                    function_where_clause,
                    {"type": {"$eq": "function_node"}},
                ]
            }
        )
        return functions
