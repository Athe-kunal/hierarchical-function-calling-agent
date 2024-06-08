import re


def process_params(params_desc):
    params_desc_type = params_desc["type"]
    if "type" in params_desc:
        if "None" in params_desc_type and "Literal" in params_desc_type:
            result = re.findall(r"Literal\[(.*?)\]", params_desc_type)[0].split(", ")
            result = [
                value.strip('"') if value.strip() != "None" else None
                for value in result
            ]
            result = [value.replace("'", "") for value in result if value is not None]
            del params_desc["type"]
            # for res in result:
            # if res in function_options:
            #         params_desc['optional'] = False
            #         break
            params_desc.update({"type": "string", "enum": result})
        elif "None" not in params_desc_type and "Literal" in params_desc_type:
            result = re.findall(r"Literal\[(.*?)\]", params_desc["type"])[0].split(", ")
            result = [value.replace("'", "") for value in result]
            # for res in result:
            #     if res in function_options:
            #         params_desc['optional'] = False
            #         break
            del params_desc["type"]
            params_desc.update({"type": "string", "enum": result})

        elif "int" in params_desc_type:
            params_desc["type"] = "integer"
        elif "str" in params_desc_type:
            params_desc["type"] = "string"
        elif "float" in params_desc_type:
            params_desc["type"] = "number"
        elif "callable" in params_desc_type or "object" in params_desc_type:
            params_desc["type"] = "object"
        elif "Union" in params_desc_type or "List" in params_desc_type:
            params_desc["type"] = "array"
        elif "bool" in params_desc["type"]:
            params_desc.update({"type": "boolean", "enum": ["True", "False"]})
    return params_desc


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


def get_trail_list_pairs(trail_list_pairs):
    if len(trail_list_pairs) == 1:
        trail_where_clause = {"trail": {"$eq": trail_list_pairs[0]}}
    elif len(trail_list_pairs) > 1:
        trail_where_clause = {"$or": [{"trail": {"$eq": t}} for t in trail_list_pairs]}
    return trail_where_clause


def split_description(text_list, MAX_WORDS: int = 500):
    split_s = []
    running_num_words = 0
    curr_func_string = ""
    for txt in text_list:
        txt = txt.replace("\n", " ")
        txt = txt.replace("\n\n", " ")
        num_words = len(txt.split(" "))
        running_num_words += num_words
        if running_num_words > MAX_WORDS:
            running_num_words = num_words
            split_s.append(curr_func_string)
            curr_func_string = txt
        else:
            curr_func_string += txt + " "
    if split_s == [""] or split_s == []:
        split_s.append(curr_func_string)
    split_s = [s for s in split_s if s != ""]
    return split_s
