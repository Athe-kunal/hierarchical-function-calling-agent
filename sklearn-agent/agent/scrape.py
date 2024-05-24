import requests
from bs4 import BeautifulSoup
from tqdm import tqdm
from agent.utils import add_function_calling
import yaml
with open("config.yaml") as stream:
    try:
        config_params = yaml.safe_load(stream)
    except yaml.YAMLError as exc:
        print(exc)
def get_links(id_elem, class_name):
    def process_link(link):
        link = link.replace("..","")
        return "https://scikit-learn.org/stable"+link
    curr_urls = []
    try:
        func_urls = id_elem.find_all(attrs={"class": class_name})
        for url in func_urls:
            curr_url_dict = {}
            funcs_params = url.find_all('td')
            func_name = funcs_params[0].text
            curr_url_dict.update({"func_name":func_name})
            func_desc = funcs_params[1].text
            curr_url_dict.update({"func_desc":func_desc})
            try:
                func_url = url.find("a")["href"]
                curr_url_dict.update({"func_url":process_link(func_url)})
                curr_urls.append(curr_url_dict)
            except Exception as e:
                print(e)
                print(url)
                print("-" * 100)
        return curr_urls
    except Exception as e:
        func_urls = id_elem.find(attrs={"class": class_name})
        curr_url_dict = {}
        funcs_params = url.find_all('td')
        func_name = funcs_params[0].text
        curr_url_dict.update({"func_name":func_name})
        func_desc = funcs_params[1].text
        curr_url_dict.update({"func_desc":func_desc})
        func_url = func_urls.find("a")["href"]
        curr_url_dict.update({"func_url":process_link(func_url)})
        return [curr_url_dict]
    finally:
        return curr_urls

def get_odd_even_urls(id_elem):
    odd_urls = get_links(id_elem,"row-odd")
    even_urls = get_links(id_elem,"row-even")
    return odd_urls + even_urls

def scrape_sklearn_website():
    base_url = "https://scikit-learn.org/stable/api/index.html"
    response = requests.get(base_url)
    soup = BeautifulSoup(response.text, "lxml")
    l1_elems = soup.find_all(class_="toctree-l1")

    base_url = "https://scikit-learn.org/stable/api/index.html"
    base_parent_url = "https://scikit-learn.org/stable/"
    response = requests.get(base_url)
    soup = BeautifulSoup(response.text, "lxml")
    l1_elems = soup.find_all(class_="toctree-l1")

    first_level = {}
    curr_parent = ""
    for parent_functions in l1_elems:
        for func in parent_functions.find_all("a"):
            href = func["href"]
            if ".." not in href:
                first_level.update(
                    {
                        href: {
                            "functions": [],
                            "name": func.text,
                            "url": base_parent_url +"api/" + href,
                        }
                    }
                )
    
    for parent_name,first_vals in first_level.items():
        parent_page = requests.get(first_level[parent_name]['url'])
        parent_soup = BeautifulSoup(parent_page.content, 'lxml',from_encoding="utf-8")
        func_text = parent_soup.find('h1').text.replace("#","")
        parent_id = parent_name.rpartition(".")[0]
        if "<h2>" in str(parent_soup):
            all_urls_dict = []
            h2_elements = parent_soup.find_all("h2")
            default_func_table = parent_soup.find(id=f"module-{parent_id}").find(class_="autosummary longtable table autosummary",recursive=False)
            # tables = parent_soup.find_all(class_="autosummary longtable table autosummary")
            h2_elements = parent_soup.find_all('h2')
            h2_sections = []
            for idx,h2 in enumerate(h2_elements):
                curr_h2_sections = []
                if idx == len(h2_elements)-1:
                    for next_sib in h2.next_siblings:
                        curr_h2_sections.append(str(next_sib))
                else: 
                    for next_sib in h2.next_siblings:
                        if h2.next_sibling != h2_elements[idx+1]:
                            curr_h2_sections.append(str(next_sib))
                        else:
                            break
                section_text = "".join(curr_h2_sections)
                h2_sections.append(BeautifulSoup(section_text))
            if default_func_table is not None:
                default_urls = get_odd_even_urls(default_func_table)
                default_dict = {"defaults":default_urls}
                all_urls_dict.append(default_dict) 
                # Skip the first table as it is default table
                # tables = tables[1:]  
            assert len(h2_sections) == len(h2_elements), f"Assertion error for {parent_name}, number of heading 2 elements is {len(h2_elements)} and number of tables is {len(h2_sections)}"
            h2_elems_list = [h2.text.replace("#","") for h2 in h2_elements]
            for tb,h2 in zip(h2_sections,h2_elems_list):
                all_urls_dict.append({h2:get_odd_even_urls(tb)})
        else:
            tables = parent_soup.find(class_="autosummary longtable table autosummary")
            all_urls_dict = [{"defaults":get_odd_even_urls(tables)}]
        first_level[parent_name]['functions'] = all_urls_dict
    first_level_param_data = get_param_data(first_level)
    first_level_function_calling = add_function_calling(first_level_param_data)
    with open(
        config_params["FUNCTION_CALLING_DATASET"]["FUNCTION_SAVE_DEST"],
        "w",
        encoding="utf-8",
    ) as json_file:
        json_file.write(first_level_function_calling)

    return first_level_param_data

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
                        func_text = func_soup.find("dd").find_all("p")
                        func_text = " ".join([ft.text for ft in func_text])
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