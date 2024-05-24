from tqdm import tqdm
import requests
from bs4 import BeautifulSoup

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
            for _,sub_level_funcs in sub_level.items():
                for sub_level_func in sub_level_funcs:
                    function_definitions = sub_level_func['function_definitions']
                    func_name = function_definitions["function_name"]
                    function_calling = {
                        "name": func_name,
                        "descriptions": function_definitions["function_text"],
                    }
                    if function_definitions["parameter_names_desc"] != []:
                        properties_dict = {}
                        for params in function_definitions["parameter_names_desc"]:
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
                            type_dict.update({"description": params['param_type'] + ". " + params["param_desc"]})

                            properties_dict.update({params["param_name"]: type_dict})

                        function_calling.update(
                            {
                                "parameters": {
                                    "type": "object",
                                    "properties": properties_dict,
                                    "required": function_text_to_req(function_definitions["full_function"]),
                                }
                            }
                        )
                        sub_level_func.update({"function_calling": function_calling})
                    else:
                        sub_level_func.update({"function_calling": {}})

    return data

