import dspy
from dotenv import load_dotenv, find_dotenv
from tqdm import tqdm
import json
import yaml
from agent.utils import get_parents_dict

with open("config.yaml") as stream:
    try:
        config_params = yaml.safe_load(stream)
    except yaml.YAMLError as exc:
        print(exc)
load_dotenv(find_dotenv(), override=True)


class SummarizationGeneration(dspy.Signature):
    """You are given a list of descriptions of different functions separated by newline.
    Your task is to summarize all the text into coherent summary that covers all the functions descriptions.
    Make sure that no function description is left out of the final summary. Provide a detailed summary and cover all the function descriptions
    """

    function_descriptions = dspy.InputField(
        prefix="List of function descriptions: ",
        desc="list of function descriptions to be summarized",
    )
    summary = dspy.OutputField(
        prefix="Summary: ", desc="summary of all the function descriptions"
    )


summarization_llm = dspy.OpenAI(
    model=config_params["PARENTS_SUMMARY"]["OPENAI_LLM_MODEL"], max_tokens=4096
)
dspy.settings.configure(lm=summarization_llm)


class SummarizationPipeline(dspy.Module):
    def __init__(self, parent_node, parent_text, MAX_WORDS):
        self.parent_node = parent_node
        self.parent_text = parent_text
        self.summarization = dspy.Predict(SummarizationGeneration)
        self.MAX_WORDS = MAX_WORDS

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def split_description(self):
        split_s = []
        running_num_words = 0
        curr_func_string = ""
        for txt in self.parent_text:
            num_words = len(txt.split(" "))
            running_num_words += num_words
            if running_num_words > self.MAX_WORDS:
                running_num_words = num_words
                split_s.append(curr_func_string)
                curr_func_string = txt
            else:
                curr_func_string += txt + "\n"
        if split_s == [] or split_s == [""]:
            split_s.append(curr_func_string)
        split_s = [s for s in split_s if s != ""]
        return split_s

    def forward(self):
        if len(self.parent_text) == 0:
            return ""
        split_s = self.split_description()

        summaries = ""
        pbar = tqdm(total=len(split_s), desc=f"For {self.parent_node}")
        for desc in split_s:
            summaries += self.summarization(function_descriptions=desc).summary + " "
            pbar.update(1)
        return summaries


def run_summaries_agent(sklearn_graph, MAX_WORDS: int = 500):
    parent_dict = get_parents_dict(sklearn_graph)
    parent_summary_dict = {}
    for parent in parent_dict:
        if parent_summary_dict[parent] == "":
            print(f"Summarizing for {parent}")
            summ_pipeline = SummarizationPipeline(
                parent, parent_dict[parent], MAX_WORDS=MAX_WORDS
            )
            summary = summ_pipeline()
            parent_summary_dict[parent] = summary
    json.dump(
        parent_summary_dict,
        open(config_params["PARENTS_SUMMARY"]["SUMMARY_JSON_FILE_PATH"], "w"),
    )
    print(
        f"Summaries saved to {config_params['PARENTS_SUMMARY']['SUMMARY_JSON_FILE_PATH']}"
    )
    return parent_summary_dict
