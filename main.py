import os
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.text_splitter import CharacterTextSplitter
from langchain import OpenAI
from langchain.chains import RetrievalQA
from langchain.document_loaders import DirectoryLoader

os.environ["OPENAI_API_TOKEN"] = "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"

DIR_ARQUIVOS = 'src/assets/'

# Ler arquivos de uma determinada pasta
operador = DirectoryLoader(DIR_ARQUIVOS, glob='**/*.txt')

# Lê os arquivos de textos do diretório e armazena na variavel documentos
documentos = operador.load()

# A variável "divisorTexto" divide o texto em pedaços de 1000 caracteres.
divisorTexto = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)

# A variável "textoDivido" cria uma lista de textos divididos.
# {
#  text_content: '1000 caracteres',
#  metadata: {
#     "source": "src/assets/1.txt",
#   }
# }
#
textosDivididos = divisorTexto.split_documents(documentos)

# Ferramenta de vetorização do OpenAI para transformar textos/listas de textos em vetores de números

vetoresDeIncorporacao = OpenAIEmbeddings(
    openai_api_key=os.environ['OPENAI_API_TOKEN'])

# Buscador que tem capacidade de encontrar trechos de texto semanticamente semelhantes.

vetorDeBuscaDocs = Chroma.from_documents(textosDivididos, vetoresDeIncorporacao)

llm = OpenAI(temperature=0.9, openai_api_key=os.environ["OPENAI_API_TOKEN"])

botPersonalizado = RetrievalQA.from_chain_type(
    llm=llm, chain_type="refine", retriever=vetorDeBuscaDocs.as_retriever(search_kwargs={"k": 1}), input_key="question")

QUESTION = "Responda de maneira formal a pergunta a seguir: Qual a variação do incc no mês de fevereiro de 2023?"

print(f"\033[32m{botPersonalizado(QUESTION)}")
print(vetorDeBuscaDocs)