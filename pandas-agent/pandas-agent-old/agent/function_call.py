from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
import ast


def run_function_calling(fcs, question: str):
    
    for meta in fcs["metadatas"]:
        function_call = ast.literal_eval(meta["function_calling"])
        break
    prompt = ChatPromptTemplate.from_messages([("human", "{input}")])

    model = ChatOpenAI(temperature=0).bind(
        functions=[function_call], function_call={"name": function_call["name"]}
    )
    runnable = prompt | model
    resp = runnable.invoke({"input": question})
    return resp


# def format_function(function_response):
    