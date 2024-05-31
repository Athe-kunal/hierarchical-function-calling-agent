import networkx as nx
import json
import yaml
import chromadb.utils.embedding_functions as embedding_functions
from chromadb.utils.batch_utils import create_batches
from dotenv import load_dotenv, find_dotenv

with open("config.yaml") as stream:
    try:
        config_params = yaml.safe_load(stream)
    except yaml.YAMLError as exc:
        print(exc)


def function_text_to_req(s):
    star_idx = s.find(" *,")
    if star_idx == -1:
        return []
    req_str = s[s.find("(") + 1 : s.find("*")].strip()[:-1]
    req_list = [i.strip() for i in req_str.split(",")]
    return req_list


def add_openai_functions(data):
    for parent in data:
        for sub_level in data[parent]["functions"]:
            for func in sub_level["function_definitions"]:
                func_name = func["function_name"]
                function_calling = {
                    "name": func_name,
                    "descriptions": func["function_text"],
                }
                if func["parameter_names_desc"] != []:
                    properties_dict = {}
                    for params in func["parameter_names_desc"]:
                        type = params["param_type"]
                        if "int" in type:
                            type = "integer"
                            type_dict = {"type": type}
                        elif "str" in type:
                            type = "string"
                            type_dict = {"type": type}
                        elif "bool" in type:
                            type = "boolean"
                            type_dict = {"type": type}
                        elif "float" in type:
                            type = "float"
                            type_dict = {"type": type}
                        elif "float" in type:
                            type = "float"
                            type_dict = {"type": type}
                        elif "callable" in type or "object" in type:
                            type = "object"
                            type_dict = {"type": type}
                        elif "Union" in type or "list" in type or "array" in type:
                            type = "array"
                            type_dict = {"type": type}
                        elif (
                            "{" in type and "}" in type and "’" in type and "‘" in type
                        ):
                            list_params = type[type.find("{") + 1 : type.find("}")]
                            list_params = (
                                list_params.replace("’", "").replace("‘", "").split(",")
                            )
                            type_dict = {"type": "string", "enum": list_params}
                        else:
                            type_dict = {"type": type}
                        type_dict.update(
                            {
                                "description": params["param_type"]
                                + ". "
                                + params["param_desc"]
                            }
                        )

                        properties_dict.update({params["param_name"]: type_dict})

                    function_calling.update(
                        {
                            "parameters": {
                                "type": "object",
                                "properties": properties_dict,
                                "required": function_text_to_req(func["full_function"]),
                            }
                        }
                    )
                    func.update({"function_calling": function_calling})
                else:
                    func.update({"function_calling": {}})
    return data


def build_no_summary_graph():
    with open(
        config_params["FUNCTION_CALLING_DATASET"]["FUNCTION_SAVE_DEST"],
        "r",
        encoding="utf-8",
    ) as json_file:
        # Load the JSON data from the file
        data = json.load(json_file)
    pandas_graph = nx.DiGraph()

    parent_names = [
        (data[d]["name"], {"url": data[d]["url"], "type": "parent_node"}) for d in data
    ]
    pandas_graph.add_nodes_from(parent_names)

    for parent in data:
        parent_name = data[parent]["name"]
        for sub_level in data[parent]["functions"]:
            for func in sub_level["function_definitions"]:
                func_name = func["function_name"]
                if func_name not in pandas_graph.nodes:
                    pandas_graph.add_nodes_from(
                        [
                            (
                                func_name,
                                {
                                    "function_desc": func["function_text"],
                                    "function_url": func["function_url"],
                                    "trail": parent_name,
                                    "type": "function_node",
                                    "function_name": func["function_name"],
                                    "function_calling": str(func["function_calling"]),
                                    "parameter_names_desc": str(
                                        func["parameter_names_desc"]
                                    ),
                                },
                            )
                        ]
                    )
                    pandas_graph.add_edge(parent_name, func_name)
    return pandas_graph


def build_graph():
    with open(
        config_params["FUNCTION_CALLING_DATASET"]["FUNCTION_SAVE_DEST"],
        "r",
        encoding="utf-8",
    ) as json_file:
        # Load the JSON data from the file
        data = json.load(json_file)
    with open(
        config_params["PARENTS_SUMMARY"]["SUMMARY_JSON_FILE_PATH"],
        "r",
        encoding="utf-8",
    ) as json_file:
        # Load the JSON data from the file
        parent_summary_dict = json.load(json_file)
    pandas_graph = nx.DiGraph()

    parent_names = [
        (
            data[d]["name"],
            {
                "url": data[d]["url"],
                "type": "parent_node",
                "node_description": parent_summary_dict[data[d]["name"]],
            },
        )
        for d in data
    ]
    pandas_graph.add_nodes_from(parent_names)

    for parent in data:
        parent_name = data[parent]["name"]
        for sub_level in data[parent]["functions"]:
            for func in sub_level["function_definitions"]:
                func_name = func["function_name"]
                if func_name not in pandas_graph.nodes:
                    pandas_graph.add_nodes_from(
                        [
                            (
                                func_name,
                                {
                                    "function_desc": func["function_text"],
                                    "function_url": func["function_url"],
                                    "trail": parent_name,
                                    "type": "function_node",
                                    "function_name": func["function_name"],
                                    "function_calling": str(func["function_calling"]),
                                    "parameter_names_desc": str(
                                        func["parameter_names_desc"]
                                    ),
                                },
                            )
                        ]
                    )
                    pandas_graph.add_edge(parent_name, func_name)
    return pandas_graph


def generate_pairs(list1, list2):
    pairs = []
    for l1 in list1:
        for l2 in list2:
            curr_trail = l1
            curr_trail += f"-->{l2}"
            pairs.append(curr_trail)
    return [pairs]


def generate_pairs_recursive(trail_list):
    if len(trail_list) == 1:
        return trail_list[0]
    curr_pairs = generate_pairs(trail_list[-2], trail_list[-1])
    modified_trail_list = trail_list[:-2] + curr_pairs
    return generate_pairs_recursive(modified_trail_list)


def get_trail_list_pairs(trail_list_pairs, metadata_name="trail"):
    if len(trail_list_pairs) == 1:
        trail_where_clause = {metadata_name: {"$eq": trail_list_pairs[0]}}
    elif len(trail_list_pairs) > 1:
        trail_where_clause = {
            "$or": [{metadata_name: {"$eq": t}} for t in trail_list_pairs]
        }
    return trail_where_clause
