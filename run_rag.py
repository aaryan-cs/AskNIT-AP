from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_ollama import ChatOllama

CHROMA_PATH = "chroma"

PROMPT_TEMPLATE = """
As the Student Query Assistance Bot for NIT Andhra Pradesh answer the question based on the following context representing the institute:

{context}

___

As the Student Query Assistance Bot for NIT Andhra Pradesh answer the question based on the above context in a short concise manner.: {question}
"""

def main():
    query_text = input()
    embedding_function = OllamaEmbeddings(model="nomic-embed-text")

    db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)

    results = db.similarity_search_with_relevance_scores(query_text, k=3)

    # print(results)
    
    context_text = "\n\n---\n\n".join(doc.page_content for doc, score in results)
    
    prompt = PROMPT_TEMPLATE.format(context=context_text, question=query_text)

    # print("\n", prompt)

    model = ChatOllama(model="llama3.1:latest")
    response_text = model.invoke(prompt)

    print("\n", response_text.content)

if __name__ == "__main__":
    while True:
        main()