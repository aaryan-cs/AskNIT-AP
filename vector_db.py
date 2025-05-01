from langchain_community.document_loaders import DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
import os, shutil

DATA_PATH = "data"
CHROMA_PATH = "chroma"

def main():
    generate_data_store()

def generate_data_store():
    print("Starting data store generation")
    documents = load_documents()
    print("Loaded docs")
    chunks = split_text(documents)
    print("Split text into chunks")
    save_to_chroma(chunks)
    print("Saved to Chroma")

def load_documents():
    print("Loading directory loader")
    loader = DirectoryLoader(DATA_PATH, glob="*.txt")
    print("Loaded directory loader")
    documents = loader.load()
    filename_to_url = {
        "Admin_AdminStaff.txt": "https://nitandhra.ac.in/main/ofc_ad.php",
        "Admin_AssocDeans.txt": "https://nitandhra.ac.in/main/assoc_deans.php",
        "Admin_Deans.txt": "https://nitandhra.ac.in/main/deans.php",
        "data/Admin_HODs.txt": "https://nitandhra.ac.in/main/hod.php",
        "data/Admin_Registrar.txt": "https://nitandhra.ac.in/main/registrar.php",
        "data/Administration_DirectorProfile.txt": "https://nitandhra.ac.in/main/director_profile.php",
        "data/Amin_FormerDirectors.txt": "https://nitandhra.ac.in/main/formerprincipals.php",
        "data/AntiRagging.txt": "https://nitandhra.ac.in/main/anti_ragging.php",
        "data/BioTechFaculty.txt": "https://nitandhra.ac.in/dept/biot/",
        "data/ChemFaculty.txt": "https://nitandhra.ac.in/dept/chem/",
        "data/CivilFaculty.txt": "https://nitandhra.ac.in/dept/civil/",
        "data/CSEFaculty.txt": "https://nitandhra.ac.in/dept/cse/",
        "data/Clubs.txt": "https://nitandhra.ac.in/main/elcell.php",
        "data/FAQ_NIT_ANDHRA_PRADESH.txt": "https://nitandhra.ac.in/main/",
        "data/Home_AboutUS.txt": "https://nitandhra.ac.in/main/",
        "data/Home_VisitorProfile.txt": "https://nitandhra.ac.in/main/visitor.php",
        "data/HostelBlocksandCapacity.txt": "https://nitandhra.ac.in/main/hostels.php",
        "data/Library.txt": "https://nitandhra.ac.in/main/library.php",
        "data/PlacementNorms.txt": "https://nitandhra.ac.in/main/tnp.php",
        "data/PlacementRules.txt": "https://nitandhra.ac.in/main/tnp.php",
        "data/ResearchAreas.txt": "https://nitandhra.ac.in/main/pdf/annexure-A.pdf",
        "data/security_guidelines.txt": "https://nitandhra.ac.in/main/hostels.php"
        # All file-to-URL mappings are here
    }

    for doc in documents:
        filename = os.path.basename(doc.metadata.get('source', ''))
        doc.metadata['url'] = filename_to_url.get(filename, "Unknown")

    print("Loaded documents")
    return documents

def split_text(documents):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=450,
        chunk_overlap=50,
        length_function=len,
        add_start_index=True
    )

    chunks = text_splitter.split_documents(documents)
    print(f"Split {len(documents)} documents into {len(chunks)} chunks.")

    if len(chunks) > 10:
        document = chunks[10]
        print(document.page_content)
        print(document.metadata)


    return chunks
def save_to_chroma(chunks):
    if os.path.exists(CHROMA_PATH):
        shutil.rmtree(CHROMA_PATH)
    
    db = Chroma.from_documents(chunks, OllamaEmbeddings(model="nomic-embed-text"), persist_directory=CHROMA_PATH)
    print(f"Saved {len(chunks)} chunks to {CHROMA_PATH}.")

if __name__ == "__main__":
    main()