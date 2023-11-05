import os
from llama_index import (GPTVectorStoreIndex, SimpleDirectoryReader, ServiceContext, OpenAIEmbedding, PromptHelper, StorageContext, load_index_from_storage)
from llama_index.text_splitter import TokenTextSplitter
from llama_index.node_parser import SimpleNodeParser
from llama_index.llms import OpenAI
import textwrap
def index_create(filepath):
    llm = OpenAI(model='gpt-3.5-turbo', temperature=0, max_tokens=256)
    embed_model = OpenAIEmbedding()
    node_parser = SimpleNodeParser.from_defaults(
        text_splitter=TokenTextSplitter(chunk_size=1024, chunk_overlap=20)
    )
    prompt_helper = PromptHelper(
        context_window=4096,
        num_output=256,
        chunk_overlap_ratio=0.1,
        chunk_size_limit=None
    )

    service_context = ServiceContext.from_defaults(
        llm=llm,
        embed_model=embed_model,
        node_parser=node_parser,
        prompt_helper=prompt_helper
    )

    documents = SimpleDirectoryReader(input_files=[filepath]).load_data()
    index = GPTVectorStoreIndex.from_documents(documents, service_context=service_context)

    return index
#%%

def main():
    try:
        value = os.environ["OPENAI_API_KEY"]
    except:
        os.environ["OPENAI_API_KEY"] = input("Enter OPENAI API Key: ")

    index = index_create(input("Enter Filepath to Quarterly Report: ").strip('"'))
    query_engine = index.as_query_engine(streaming=False)
    company = query_engine.query("What company is this report for? Give me only the name of the company and nothing else.")

    more = True
    while(more):
        query = input(f"Enter question regarding {company}'s report: ")
        response = query_engine.query(query)
        print("\n", textwrap.fill(str(response), 100), "\n")
        more = False if(input("\nContinue y/n?: ") == "n") else True

if __name__ == "__main__":
    main()