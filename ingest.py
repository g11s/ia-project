from dotenv import load_dotenv
from langchain_community.document_loaders import DirectoryLoader,TextLoader,PyPDFDirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders.merge import MergedDataLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
import warnings
warnings.filterwarnings("ignore")

def load_and_split_documents(chunk_size, chunk_overlap):
    print ("Loading documents...")
    loader = DirectoryLoader(
        path='./project',
        glob="**/*.*",
        show_progress=True,
        use_multithreading=True,
        loader_cls=TextLoader
    )
    # brain_pdf = PyPDFDirectoryLoader(path="./brain/pdf",glob="**/*.pdf")
    list_loaders = MergedDataLoader([loader])

    print ("Documents loaded")

    print ("Creating chunks...")

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size = chunk_size,
        chunk_overlap = chunk_overlap,
        length_function = len,
    )

    chunks = list_loaders.load_and_split(text_splitter=text_splitter)
    print("Created", len(chunks), "chunks of data")
    return chunks

def main():
    load_dotenv()
    chunk_size = 1024
    chunk_overlap = 100

    chunks = load_and_split_documents(chunk_size, chunk_overlap)
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectorstore = FAISS.from_documents(documents = chunks, embedding=embeddings)
    vectorstore.save_local("faiss_vector_db")
    print("Success! Vector Store has been created!")


if __name__ == "__main__":
    main()