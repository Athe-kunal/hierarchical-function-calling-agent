import dspy
from dotenv import load_dotenv, find_dotenv
from tqdm import tqdm
import json
import yaml
import chromadb.utils.embedding_functions as embedding_functions
import os
from agent.utils import *
from langchain_community.retrievers import BM25Retriever
from langchain.schema import Document

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

llm = dspy.OpenAI(model="gpt-3.5-turbo-0125",max_tokens=256)
dspy.settings.configure(lm=llm)


class PandasAgentChroma(dspy.Module):
    def __init__(self, collection):
        super().__init__()
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

class PandasAgentBM25(dspy.Module):
    def __init__(self, collection):
        super().__init__()
        self.collection = collection
        self.firstSecondLevel = dspy.Predict(FirstSecondLevel)
        parent_level = self.collection.get(
            where={
                "type": {"$eq": "parent_node"},
            }
        )
        self.parent_langchain_docs = []
        for doc,metadata in zip(parent_level['documents'],parent_level['metadatas']):
            self.parent_langchain_docs.append(Document(page_content=doc,metadata=metadata))

    def __call__(self, *args, **kwargs):
        return super().__call__(*args, **kwargs)
    
    def BM25RetrieverLangchain(self,query:str,type:str='parent',trail_where_clause:dict={}):

        assert type in ['parent','function'], "type must be 'parent' or 'function'"
        if type == 'function' and trail_where_clause=={}:
            raise ValueError("trail_where_clause must be a dict for function type")
        
        if type == 'parent':
            bm25_retriever = BM25Retriever.from_documents(
                self.parent_langchain_docs, k=5, preprocess_func=(lambda x: x.lower())
            )
            parent_bm25_docs = bm25_retriever.invoke(query.lower())
            return parent_bm25_docs
        elif type == 'function':
            function_level = self.collection.get(
            where={
                "$and": [
                    trail_where_clause,
                    {"type": {"$eq": "function_node"}},
                ]
            },
             )
            function_langchain_docs = []
            for doc,metadata in zip(function_level['documents'],function_level['metadatas']):
                function_langchain_docs.append(Document(page_content=doc,metadata=metadata))
            bm25_retriever = BM25Retriever.from_documents(
                function_langchain_docs, k=5, preprocess_func=(lambda x: x.lower())
            )
            function_bm25_docs = bm25_retriever.invoke(query.lower())
            return function_bm25_docs

    def forward(self, query: str):
        parent_bm25_docs = self.BM25RetrieverLangchain(query)
        parent_level_str = ""
        for parent_doc in parent_bm25_docs:
            parent_level_str+=f"{parent_doc.metadata['name']}: {parent_doc.metadata['node_description']}"
        
        parent_level_answer = self.firstSecondLevel(
            query=query, keys_values=parent_level_str
        ).output
        trail_list = [parent_level_answer.split(";")]
        trail_list_pairs = generate_pairs_recursive(trail_list)

        trail_where_clause = get_trail_list_pairs(trail_list_pairs)

        function_level_docs = self.BM25RetrieverLangchain(query,type='function',trail_where_clause=trail_where_clause)
        function_level_str = ""
        for function_doc in function_level_docs:
            function_level_str+=f"{function_doc.metadata['function_name']}: {function_doc.metadata['function_desc']}"
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
