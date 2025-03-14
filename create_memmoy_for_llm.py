import os

from langchain_huggingface import HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

load_dotenv(find_dotenv())

HF_TOKEN = os.environ.get('HF_TOKEN')
HUGGINGFACE_REPO_ID = 'mistralai/Mistral-7B-Instruct-v0.3'

def laod_llm(huggingface_repo_id):
    llm = HuggingFaceEndpoint(
        repo_id = huggingface_repo_id,
        temperature = 0.5,
        model_kwargs = {'token':HF_TOKEN,'max_length':'512'}
    )
    return llm

CUSTOM_PROMT_TEMPLATE = '''
Use the pieces of information provided in the context to answer user's quesion.
If you don't know answer,just say that you don't know,don't try to make up an answer.
Don't provide anything out of context

Context:{context}
Question:{question}
Start the answer directly.No small talks please
'''

def set_custom_prompt(custom_promt_template):
    prompt = PromptTemplate(templates=custom_promt_template,input_varialbe=['context,'question'])
    return prompt

DB_FAISS_PATH = 'vectorstore/db_faiss'

embedding_model = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MinLM-L6-v2')
db = FAISS.load_local(DB_FAISS_PATH,embedding_model,allow_dangerous_deserialization=True)

qa_chain = RetrivalQA.from_chain_type(
    llm = load_llm(HUGGINGFACE_REPO_iD),
    chain_type='stuff',
    retriver_db.as_retriver(search_kwargs={'k':3}),
    return_source_documents = True,
    chain_type_kwargs={'prompt':set_custom_prompt(CUSTOM_PROMT_TEMPLATE)}
)


user_query = input('write querry here:')
response = qa_chain.invoke({'querry':user_query})
print('result':response['result'])
print('SOURCE DOCUMENTS',response['source_documents'])


