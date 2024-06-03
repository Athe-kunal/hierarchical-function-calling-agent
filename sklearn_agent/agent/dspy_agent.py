import dspy
from dotenv import load_dotenv, find_dotenv
import yaml
import chromadb.utils.embedding_functions as embedding_functions
import os
from sklearn_agent.agent.utils import *
from langchain_community.retrievers import BM25Retriever
from langchain.schema import Document

with open("sklearn_agent/config.yaml") as stream:
    try:
        config_params = yaml.safe_load(stream)
    except yaml.YAMLError as exc:
        print(exc)
load_dotenv(find_dotenv(), override=True)


# class FirstSecondLevel(dspy.Signature):
#     "You are given a list of keys and values separated by semicolon."
#     "Based on the query, you have to output the key that is most relevant to the question separated by semicolon."
#     "Be precise and output only the relevant key or keys from the provided keys only."
#     "Don't include any other information and DON'T answer None or N/A"


#     query = dspy.InputField(prefix="Query which you need to classify: ", format=str)
#     keys_values = dspy.InputField(prefix="Keys and Values: ", format=str)
#     output = dspy.OutputField(
#         prefix="Relevant Key(s): ",
#         format=str,
#         desc="relevant keys separated by semicolon",
#     )
class FirstSecondLevel(dspy.Signature):
    """You are given a list of keys and their corresponding description separated by semicolon in the format keys: description.
    Based on the query, you have to classify the question to one of the key or keys that is relevant to the question.
    Be precise and output only the relevant key or keys and don't output their descriptions.
    Don't include any other information and DON'T answer None or N/A"""

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

llm = dspy.OpenAI(
    model=config_params["LEVEL_NODE_LLM"]["OPENAI_LLM_MODEL"], max_tokens=512
)
dspy.settings.configure(lm=llm)


# class SklearnAgentChroma(dspy.Module):
#     def __init__(self, collection):
#         super().__init__()
#         self.collection = collection
#         self.firstSecondLevel = dspy.Predict(FirstSecondLevel)

#     def __call__(self, *args, **kwargs):
#         return super().__call__(*args, **kwargs)

#     def forward(self, query: str):
#         query_emb = emb_fn([query])[0]

#         # Parent level querying
#         parent_level = self.collection.query(
#             query_embeddings=query_emb,
#             where={
#                 "type": {"$eq": "parent_node"},
#             },
#             n_results=3,
#         )
#         parent_level_str = ""
#         for parent_level_docs,parent_level_metadata in zip(parent_level['documents'][0],parent_level["metadatas"][0]):
#             parent_level_str += f"{parent_level_metadata['name']}: {parent_level_docs}\n\n"

#         parent_level_answer = self.firstSecondLevel(
#             query=query, keys_values=parent_level_str
#         ).output
#         print(parent_level_str, parent_level_answer)
#         trail_list = [parent_level_answer.split(";")]
#         trail_list = list(set(trail_list[0]))
#         trail_list_pairs = generate_pairs_recursive([trail_list])

#         trail_where_clause = get_trail_list_pairs(trail_list_pairs)

#         sub_level = self.collection.query(
#             query_embeddings=query_emb,
#             where={
#                 "$and": [
#                     trail_where_clause,
#                     {"type": {"$eq": "sub_level_node"}},
#                 ]
#             },
#             n_results=5,
#         )

#         sub_level_str = ""
#         for sub_level_docs,function_level_metadata in zip(sub_level['documents'][0],sub_level["metadatas"][0]):
#             sub_level_str += f"{function_level_metadata['name']}: {sub_level_docs}\n\n"
#         print(sub_level_str)
#         sub_level_answer = self.firstSecondLevel(
#             query=query, keys_values=sub_level_str
#         ).output
#         print(sub_level_answer)
#         sub_level_list = [sla.split("#")[-1] for sla in sub_level_answer.split(";")]
#         sub_level_list = list(set(sub_level_list))
#         function_list = generate_pairs_recursive([trail_list_pairs,sub_level_list])
#         function_where_clause = get_trail_list_pairs(function_list)
#         print(function_where_clause)
#         functions = self.collection.query(
#             query_embeddings=query_emb,
#             where={
#                 "$and": [
#                     function_where_clause,
#                     {"type": {"$eq": "function_node"}},
#                 ]
#             },
#             n_results=1
#         )
#         return functions['metadatas'][0]


class SklearnAgentChroma(dspy.Module):
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
            n_results=3,
        )
        parent_level_str = ""
        for parent_level_docs, parent_level_metadata in zip(
            parent_level["documents"][0], parent_level["metadatas"][0]
        ):
            if parent_level_docs in parent_level_str:
                continue
            parent_level_str += (
                f"{parent_level_metadata['parent']}: {parent_level_docs}\n\n"
            )

        parent_level_answer = self.firstSecondLevel(
            query=query, keys_values=parent_level_str
        ).output
        print(parent_level_str, parent_level_answer)
        trail_list = parent_level_answer.split(";")
        trail_list = list(set(trail_list))
        trail_list_pairs = generate_pairs_recursive([trail_list])

        trail_where_clause = get_trail_list_pairs(trail_list_pairs, "sub_level_trail")

        sub_level = self.collection.query(
            query_embeddings=query_emb,
            where=trail_where_clause,
            n_results=3,
        )

        sub_level_str = ""
        for sub_level_docs, function_level_metadata in zip(
            sub_level["documents"][0], sub_level["metadatas"][0]
        ):
            if sub_level_docs in sub_level_str:
                continue
            sub_level_str += f"{function_level_metadata['parent']}#{function_level_metadata['sub_level_name']}: {sub_level_docs}\n\n"
        print(sub_level_str)
        sub_level_answer = self.firstSecondLevel(
            query=query, keys_values=sub_level_str
        ).output
        print(sub_level_answer)
        sub_level_list = sub_level_answer.split(";")
        sub_level_list = [sbl.split("#")[-1] for sbl in sub_level_list]
        sub_level_list = list(set(sub_level_list))
        function_list = generate_pairs_recursive([trail_list_pairs, sub_level_list])
        function_where_clause = get_trail_list_pairs(function_list, "function_trail")
        print(function_where_clause)
        functions = self.collection.query(
            query_embeddings=query_emb, where=function_where_clause, n_results=1
        )
        return functions["metadatas"][0]


class SklearnAgentBM25(dspy.Module):
    def __init__(self, collection):
        super().__init__()
        self.collection = collection
        self.firstSecondLevel = dspy.Predict(FirstSecondLevel)
        all_docs = self.collection.get()
        self.langchain_docs = [Document(page_content=doc,metadata=meta) for doc,meta in zip(all_docs['documents'],all_docs['metadatas'])]


    def __call__(self, *args, **kwargs):
        return super().__call__(*args, **kwargs)

    def BM25RetrieverLangchain(
        self, query: str, node_type: str = "parent_node", trail_where_clause: dict = {}
    ):

        assert node_type in [
            "parent_node",
            "function_node",
            "sub_level_node",
        ], "type must be 'parent_node' or 'function_node' or 'sub_level_node'"
        if node_type != "parent_node" and trail_where_clause == {}:
            raise ValueError("trail_where_clause must be a dict for function type")

        if node_type == "parent_node":
            bm25_retriever = BM25Retriever.from_documents(
                self.langchain_docs, k=3, preprocess_func=(lambda x: x.lower())
            )
            parent_bm25_docs = bm25_retriever.invoke(query.lower())
            return parent_bm25_docs
        else:
            function_level = self.collection.get(
                where=trail_where_clause
            )
            function_langchain_docs = []
            for doc, metadata in zip(
                function_level["documents"], function_level["metadatas"]
            ):
                function_langchain_docs.append(
                    Document(page_content=doc, metadata=metadata)
                )
            if node_type == "function_node":
                k = 1
            else:
                k = 5
            bm25_retriever = BM25Retriever.from_documents(
                function_langchain_docs, k=k, preprocess_func=(lambda x: x.lower())
            )
            bm25_docs = bm25_retriever.invoke(query.lower())
            return bm25_docs

    def forward(self, query: str):
        parent_bm25_docs = self.BM25RetrieverLangchain(query=query)
        parent_level_str = ""
        for parent_doc in parent_bm25_docs:
            parent_level_str += (
                f"{parent_doc.metadata['parent']}: {parent_doc.page_content}\n\n"
            )

        parent_level_answer = self.firstSecondLevel(
            query=query, keys_values=parent_level_str
        ).output
        print(parent_level_str)
        print(parent_level_answer)
        trail_list = parent_level_answer.split(";")
        trail_list = list(set(trail_list))
        trail_list_pairs = generate_pairs_recursive([trail_list])

        trail_where_clause = get_trail_list_pairs(trail_list_pairs, "sub_level_trail")

        sub_level_docs = self.BM25RetrieverLangchain(query,"sub_level_node",trail_where_clause)

        sub_level_str = ""
        for sub_level in sub_level_docs:
            # if sub_level_docs in sub_level_str:
            #     continue
            function_level_metadata = sub_level.metadata
            sub_level_str += f"{function_level_metadata['parent']}#{function_level_metadata['sub_level_name']}: {sub_level.page_content}\n\n"
        print(sub_level_str)
        sub_level_answer = self.firstSecondLevel(
            query=query, keys_values=sub_level_str
        ).output
        print(sub_level_answer)
        sub_level_list = sub_level_answer.split(";")
        sub_level_list = [sla.split("#")[-1] for sla in sub_level_list]
        sub_level_list = list(set(sub_level_list))
        function_list = generate_pairs_recursive([trail_list_pairs, sub_level_list])
        function_where_clause = get_trail_list_pairs(function_list, "function_trail")
        print(function_where_clause)
        functions = self.BM25RetrieverLangchain(query,"function_node",function_where_clause)
        return functions[0].metadata