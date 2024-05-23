from tqdm import tqdm
import requests
from bs4 import BeautifulSoup

def get_param_data(first_level):
    not_worked = []
    pbar = tqdm(total=len(first_level.keys()))
    for parent in first_level:
        parent_dict = first_level[parent]["functions"]
        for sub_level in parent_dict:
            for _,sub_level_vals in sub_level.items():
                for sub_level in sub_level_vals:
                    func_url = sub_level["func_url"]
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
                                    pass
                    except Exception as e:
                        not_worked.append((func_name, func_text, e, func_url))
                    finally:
                        sub_level.update({"function_definitions":curr_dict})
        pbar.update(1)
    return first_level

