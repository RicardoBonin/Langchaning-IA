from langchain.llms import OpenAI
import os

os.environ["OPENAI_API_TOKEN"] = "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"

llm = OpenAI(temperature=0.1, openai_api_key=os.environ["OPENAI_API_TOKEN"])

question = "Qual a variação do incc no mês de fevereiro de 2023?"

print(llm(question))
