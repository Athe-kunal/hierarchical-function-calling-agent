import requests
from bs4 import BeautifulSoup
from tqdm import tqdm
import yaml
import json
from agent.utils import add_openai_functions

with open("config.yaml") as stream:
    try:
        config_params = yaml.safe_load(stream)
    except yaml.YAMLError as exc:
        print(exc)


def get_links(id_elem, base_func_url, class_name, first_level_name, url_id):
    curr_urls = []
    try:
        func_urls = id_elem.find_all(attrs={"class": class_name})
        for odd_url in func_urls:
            try:
                func_url = odd_url.find("a")["href"]
                curr_urls.append(base_func_url + func_url)
            except Exception as e:
                print(e)
                print(first_level_name)
                print(odd_url)
                print(url_id)
                print("-" * 100)
        return curr_urls
    except Exception as e:
        print(e, url_id)
        func_urls = id_elem.find(attrs={"class": class_name}).find("a")["href"]
        curr_urls.append(base_func_url + func_url)
        return curr_urls
    finally:
        return curr_urls


def scrape_pandas_website():
    base_url = "https://pandas.pydata.org/docs/reference/index.html"
    response = requests.get(base_url)
    soup = BeautifulSoup(response.text, "lxml")
    base_func_url = "https://pandas.pydata.org/docs/reference/"
    l1_elems = soup.find_all(class_="toctree-l1")
    req_l1_elems = l1_elems[-len(l1_elems) + 16 :]

    first_level = {}
    curr_parent = ""
    for parent_functions in req_l1_elems:
        for func in parent_functions.find_all("a"):
            href = func["href"]
            if "#" not in href and "api/pandas" not in href:
                first_level.update(
                    {
                        href: {
                            "functions": [],
                            "name": func.text,
                            "url": base_func_url + href,
                        }
                    }
                )
                curr_parent = href
            else:
                first_level[curr_parent]["functions"].append(
                    {"name": func.text, "url": base_func_url + href}
                )

    for first_level_name in first_level:
        parent_url = first_level[first_level_name]["url"]
        parent_soup = BeautifulSoup(requests.get(parent_url).text, "lxml")
        for idx, func in enumerate(first_level[first_level_name]["functions"]):
            url_id = func["url"].split("#")[-1]
            id_elem = parent_soup.find(attrs={"id": url_id})
            odd_urls = get_links(
                id_elem, base_func_url, "row-odd", first_level_name, url_id
            )
            even_urls = get_links(
                id_elem, base_func_url, "row-even", first_level_name, url_id
            )
            # all_urls.extend(odd_urls)
            # all_urls.extend(even_urls)
            if odd_urls == [] and even_urls == []:
                function_urls = [first_level[first_level_name]["functions"][idx]["url"]]
            else:
                function_urls = odd_urls + even_urls
            first_level[first_level_name]["functions"][idx][
                "function_urls"
            ] = function_urls

    function_def_dict, _ = get_param_data(first_level)
    function_def_dict = add_openai_functions(function_def_dict)
    # Serialize the dictionary to a JSON string
    json_data = json.dumps(function_def_dict, ensure_ascii=False)

    # Write the JSON string to a file with UTF-8 encoding
    with open(
        config_params["FUNCTION_CALLING_DATASET"]["FUNCTION_SAVE_DEST"],
        "w",
        encoding="utf-8",
    ) as json_file:
        json_file.write(json_data)

    print(
        "Data has been successfully written to",
        config_params["FUNCTION_CALLING_DATASET"]["FUNCTION_SAVE_DEST"],
    )


def get_param_data(first_level):
    not_worked = []
    for parent in first_level:
        parent_dict = first_level[parent]["functions"]
        for idx, sub_level in enumerate(parent_dict):
            parent_dict[idx].update({"function_definitions": []})
            for func_url in tqdm(sub_level["function_urls"]):
                func_response = requests.get(func_url)
                func_soup = BeautifulSoup(
                    func_response.content, "lxml", from_encoding="utf-8"
                )

                func_name = func_soup.find("h1").text.replace("#", "")  # remove #
                elem = func_soup.find(attrs={"class": "sig sig-object py"})
                try:
                    full_function = elem.text.replace("[source]#", "").replace("\n", "")
                    func_text = func_soup.find("dd").find("p").text
                    curr_dict = {
                        "function_name": func_name,
                        "full_function": full_function,
                        "function_text": func_text,
                        "parameter_names_desc": [],
                        "function_url": func_url,
                    }
                    em = func_soup.find_all(attrs={"class": "field-odd"})
                    if em[0].text == "Parameters:":
                        param_names = em[-1].find_all("dt")
                        desc_list = em[-1].find_all("dd")
                        for pn, dn in zip(param_names, desc_list):
                            try:
                                param_name = pn.strong.text
                                param_type = pn.find(attrs={"class": "classifier"}).text
                                if param_name == "**kwargs":
                                    continue
                                param_desc = dn.text
                                curr_dict["parameter_names_desc"].append(
                                    {
                                        "param_name": param_name,
                                        "param_type": param_type,
                                        "param_desc": param_desc,
                                    }
                                )
                            except Exception as e:
                                print(e, pn.text)
                except Exception as e:
                    not_worked.append((func_name, func_text, e, func_url))
                finally:
                    parent_dict[idx]["function_definitions"].append(curr_dict)
    return first_level, not_worked
