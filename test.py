import os
import yaml
from langchain.agents import (
    create_json_agent,
    AgentExecutor
)
from langchain.agents.agent_toolkits import JsonToolkit
from langchain.chains import LLMChain
from langchain.llms.openai import OpenAI
from langchain.requests import TextRequestsWrapper
from langchain.tools.json.tool import JsonSpec

os.environ["OPENAI_API_TOKEN"] = "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"

with open("assets/search-results.json") as f:
    data = yaml.load(f, Loader=yaml.FullLoader)

json_spec = JsonSpec(dict_=data, max_value_length=4000)
json_toolkit = JsonToolkit(spec=json_spec)

llm = OpenAI(temperature=0.9, openai_api_key=os.environ["OPENAI_API_TOKEN"], streaming=True)

json_agent_executor = create_json_agent(
    llm=llm,
    toolkit=json_toolkit,
    verbose=True
)

# json_agent_executor('Meu cliente tem 500.000 reais e quer comprar um apartamento em Goiânia que esteja abaixo desse valor. Liste os empreendimentos disponíveis para compra. Busque pelo menos em 5 empreendimentos. Preciso de uma lista com o preço e id de 5 unidades.')

json_agent_executor('Quais os empreendimento entre 300.000 e 500.000 com 3 quartos')