import networkx as nx


def function_text_to_req(s):
    star_idx = s.find(" *,")
    if star_idx == -1:
        return []
    req_str = s[s.find("(") + 1 : s.find("*")].strip()[:-1]
    req_list = [i.strip() for i in req_str.split(",")]
    return req_list


def add_function_calling(data):
    for parent in data:
        parent_data = data[parent]
        for sub_level in parent_data["functions"]:
            for _, sub_level_funcs in sub_level.items():
                for sub_level_func in sub_level_funcs:
                    function_definitions = sub_level_func["function_definitions"]
                    func_name = function_definitions["function_name"]
                    function_calling = {
                        "name": func_name,
                        "descriptions": function_definitions["function_text"],
                    }
                    if function_definitions["parameter_names_desc"] != []:
                        properties_dict = {}
                        for params in function_definitions["parameter_names_desc"]:
                            param_name = params["param_name"]
                            if param_name == "**params" or param_name == "**kwds":
                                continue
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
                                type = "number"
                                type_dict = {"type": type}
                            elif "float" in type:
                                type = "float"
                                type_dict = {"type": type}
                            elif (
                                "dict" in type
                                or "Sequence" in type
                                or "sequence" in type
                                or "iterable" in type
                                or "Array" in type
                                or "matrix" in type
                                or "Matrix" in type
                                or "Ignored" in type
                                or "tuple" in type
                                or "Tuples" in type
                                or "iterable" in type
                                or "Iterable" in type
                                or "indexable" in type
                                or "dict" in type
                                or "array" in type
                            ):
                                type = "array"
                                if "str" in type:
                                    type_dict = {"type": type,"items":{"type":"string"}}
                                else:
                                    type_dict = {"type": type,"items":{"type":"number"}}
                            elif (
                                "callable" in type
                                or "object" in type
                                or "Callable" in type
                                or "DataFrame" in type
                                or "Series" in type
                                or "instance" in type
                                or "matplotlib" in type
                            ):
                                type = "object"
                                type_dict = {"type": type}
                            elif "Union" in type or "list" in type:
                                type = "array"
                                type_dict = {"type": type,"items": {"type":"string"}}
                            elif "{" in type and "}" in type:
                                list_params = type[type.find("{") + 1 : type.find("}")]
                                if "’" in type and "‘" in type:
                                    list_params = (
                                        list_params.replace("’", "")
                                        .replace("‘", "")
                                        .split(",")
                                    )
                                elif "”" in type and "“" in type:
                                    list_params = (
                                        list_params.replace("“", "")
                                        .replace("”", "")
                                        .split(",")
                                    )
                                else:
                                    list_params = list_params.split(",")
                                list_params = [lp.strip() for lp in list_params]
                                type_dict = {"type": "string", "enum": list_params}
                            else:
                                type_dict = {"type": "object"}
                            params_desc = params["param_type"]+ ". "+ params["param_desc"]
                            if 'items' in type_dict:
                                type_dict['items'].update(
                                    {
                                        "description": params_desc
                                    }
                                )
                            else:
                                type_dict.update(
                                    {
                                        "description": params_desc
                                    }
                                )

                            properties_dict.update({params["param_name"]: type_dict})

                        function_calling.update(
                            {
                                "parameters": {
                                    "type": "object",
                                    "properties": properties_dict,
                                    "required": function_text_to_req(
                                        function_definitions["full_function"]
                                    ),
                                }
                            }
                        )
                        sub_level_func.update({"function_calling": function_calling})
                    else:
                        sub_level_func.update({"function_calling": {}})

    return data


def build_no_summary_graph(function_calling_data):
    sklearn_graph = nx.DiGraph()

    parent_names = [
        (
            function_calling_data[d]["name"],
            {"url": function_calling_data[d]["url"], "type": "parent_node"},
        )
        for d in function_calling_data
    ]
    sklearn_graph.add_nodes_from(parent_names)
    for parent in function_calling_data:
        parent_data = function_calling_data[parent]
        parent_name = parent_data["name"]
        for sub_level in parent_data["functions"]:
            for sub_level_name, sub_level_funcs in sub_level.items():
                # if sub_level_name == "defaults":
                unique_sub_level_name = parent_name + "#" + sub_level_name
                child_text_list = []
                child_nodes = []
                for sub_level_func in sub_level_funcs:
                    func_name = sub_level_func["function_definitions"]["function_name"]
                    if func_name not in sklearn_graph.nodes:
                        attribute_dict = {
                            "url": sub_level_func["func_url"],
                            "function_name": func_name,
                            "full_function": sub_level_func["function_definitions"][
                                "full_function"
                            ],
                            "function_text": sub_level_func["function_definitions"][
                                "function_text"
                            ],
                            "parameter_names_desc": sub_level_func[
                                "function_definitions"
                            ]["parameter_names_desc"],
                            "function_calling": sub_level_func["function_calling"],
                            "type": "function_node",
                            "trail": f"{parent_name}-->{sub_level_name}",
                        }
                        child_text_list.append(
                            sub_level_func["function_definitions"]["function_text"]
                            .replace("Examples", "")
                            .strip()
                        )

                        child_nodes.append((func_name, attribute_dict))
                sklearn_graph.add_nodes_from(
                    [
                        (
                            unique_sub_level_name,
                            {
                                "type": "sub_level_node",
                                "child_texts": child_text_list,
                                "trail": parent_name,
                            },
                        )
                    ]
                )
                sklearn_graph.add_edge(parent_name, unique_sub_level_name)
                sklearn_graph.add_nodes_from(child_nodes)
                for cn in child_nodes:
                    sklearn_graph.add_edge(unique_sub_level_name, cn[0])
    return sklearn_graph


def get_parents_dict(sklearn_graph):
    parent_dict = {
        node: {}
        for node, attr in sklearn_graph.nodes(data=True)
        if attr["type"] == "parent_node"
    }

    for node, attr in sklearn_graph.nodes(data=True):
        if attr["type"] == "sub_level_node":
            # parent_dict[attr['trail']].extend(attr['child_texts'])
            parent_trail = attr["trail"]
            if node not in parent_dict[parent_trail]:
                parent_dict[parent_trail].update({node: attr["child_texts"]})
            else:
                parent_dict[parent_trail][node].extend(attr["child_texts"])
    return parent_dict


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


def split_description(text_list, MAX_WORDS: int):
    split_s = []
    running_num_words = 0
    curr_func_string = ""
    for txt in text_list:
        num_words = len(txt.split(" "))
        running_num_words += num_words
        if running_num_words > MAX_WORDS:
            running_num_words = num_words
            split_s.append(curr_func_string)
            curr_func_string = txt
        else:
            curr_func_string += txt + "\n"
    if split_s == [] or split_s == [""]:
        split_s.append(curr_func_string)
    split_s = [s for s in split_s if s != ""]
    return split_s
