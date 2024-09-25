import os
from langchain.llms import OpenAI
from langchain.chat_models import ChatOpenAI
from langchain.schema import HumanMessage, SystemMessage
from langchain.vectorstores import Chroma
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chains import RetrievalQA
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import PyPDFLoader
from langchain.prompts import PromptTemplate
import json
import re
from langchain.vectorstores import FAISS
from utils.pdf_parser import extract_text_from_pdf
from langchain.docstore.document import Document
from langchain.chains.qa_with_sources import load_qa_with_sources_chain


# from langchain.retrievers import VectorstoreRetriever
# from langchain.vectorstores import VectorstoreRetriever


# Set up your OpenAI API key
os.environ["OPENAI_API_KEY"] = "key value here"

llm = ChatOpenAI(model_name="gpt-4", temperature=0)
# llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)

# llm = ChatOpenAI(model_name="gpt-4o-mini", temperature=0)

# chunk size for processing PDFs
chunk_size = 1000 


def load_documents_to_chroma(documents, chunk_size=1000):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=50)
    split_docs = text_splitter.split_documents(documents)

    # Initialize ChromaDB
    embeddings = OpenAIEmbeddings()
    vectorstore = Chroma(persist_directory="chroma_data", embedding_function=embeddings)

    # Add documents to Chroma
    vectorstore.add_documents(split_docs)
    return vectorstore


def create_rag_pipeline(vectorstore):
    # Define how to combine the retrieved documents into a response
    llm = ChatOpenAI(model_name="gpt-4", temperature=0)
    combine_documents_chain = load_qa_with_sources_chain(llm)

    # Create the RAG pipeline
    qa_chain = RetrievalQA.from_chain_type(
        retriever=vectorstore.as_retriever(),  
        chain_type="stuff",  
        combine_documents_chain=combine_documents_chain,
        llm=llm 
    )
    
    return qa_chain


def process_pdf_with_rag(pdf_path):
    # Step 1: Extract text from the PDF
    report_text = extract_text_from_pdf(pdf_path)
    if not report_text:
        return {"error": "Failed to extract text from PDF"}
    
    print("Extracted report_text:", report_text)

    # Step 2: Initialize document and load into Chroma vector store
    document = Document(page_content=report_text)
    try:
        vectorstore = load_documents_to_chroma([document])
    except Exception as e:
        return {"error": f"Failed to load documents into Chroma: {str(e)}"}
    
    # Step 3: Create RAG pipeline
    try:
        rag_chain = create_rag_pipeline(vectorstore)
    except Exception as e:
        return {"error": f"Failed to create RAG pipeline: {str(e)}"}
    
    # Step 4: Define financial prompt
    financial_prompt =  """
    You are a financial analyst. Extract key financial data from the following report and convert all values to INR. 
    Assume the exchange rate is 1 USD = 83 INR if required. All amounts in millions should be converted to actual values.
    
    ### Data to extract:
    1. **Earnings Per Share (EPS)** (INR)
    2. **Stock Price** (INR)
    3. **Net Income** (INR)
    4. **Total Investment** (INR)
    5. **Current Assets** (INR)
    6. **Current Liabilities** (INR)
    7. **Gross Profit** (INR)
    8. **Net Revenue** (INR)
    9. **Total Debt** (INR)
    10. **Total Shareholders' Equity** (INR)
    11. **Operating Cash Flow** (INR)
    
    ### Calculations:
    1. **Current Ratio** = Current Assets / Current Liabilities
    2. **Gross Profit Margin** = (Gross Profit / Net Revenue) * 100 (If Net Revenue is unavailable, return "Not Available")
    3. **Debt to Equity Ratio** = Total Debt / Total Shareholders' Equity
    4. **P/E Ratio** = Stock Price / EPS
    5. **ROI** = (Net Income / Total Investment) * 100

    ### Financial Data:
    {text}
    """
    
    # Step 5: Prepare the prompt
    prompt = PromptTemplate(template=financial_prompt, input_variables=["text"])

    # Step 6: Run the RAG pipeline
    try:
        result = rag_chain.run(prompt.render(text=report_text))
        print("Result obtained:", result)
        return result
    except Exception as e:
        return {"error": f"Failed to process RAG pipeline: {str(e)}"}

def pdf_processing_with_rag_for_financial_data(pdf_path):
    # Process the PDF and extract financial data
    financial_data = process_pdf_with_rag(pdf_path)
    
    # Return financial data or any errors encountered
    if "error" in financial_data:
        return financial_data
    
    return financial_data



#################################################################################################################################################
############################################### NORMAL IMPLEMENTATION ###########################################################################
#################################################################################################################################################

def extract_data_from_report(report_text):
    print('Report text being processed:', report_text)
    
    # Split the report text into chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    texts = text_splitter.split_text(report_text)

    # Convert text to Document objects for FAISS indexing
    docs = [Document(page_content=text) for text in texts]

    # Create embeddings and vector store
    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.from_documents(docs, embeddings)

    # Create a retriever to fetch relevant documents
    retriever = vectorstore.as_retriever()

    # Query the relevant financial data
    query = "Extract financial data and ratios from the financial report."
    relevant_docs = retriever.get_relevant_documents(query)

    # Concatenate retrieved texts
    retrieved_text = " ".join([doc.page_content for doc in relevant_docs])

    # Prepare the prompt with the extracted text
    template = """
    You are a financial analyst. Extract key financial data from the following report and convert all values to INR. 
    Assume the exchange rate is 1 USD = 83 INR if required. All amounts in millions should be converted to actual values.
    
    ### Data to extract:
    1. **Earnings Per Share (EPS)** (INR)
    2. **Stock Price** (INR)
    3. **Net Income** (INR)
    4. **Total Investment** (INR)
    5. **Current Assets** (INR)
    6. **Current Liabilities** (INR)
    7. **Gross Profit** (INR)
    8. **Net Revenue** (INR)
    9. **Total Debt** (INR)
    10. **Total Shareholders' Equity** (INR)
    11. **Operating Cash Flow** (INR)
    
    ### Calculations:
    1. **Current Ratio** = Current Assets / Current Liabilities
    2. **Gross Profit Margin** = (Gross Profit / Net Revenue) * 100 (If Net Revenue is unavailable, return "Not Available")
    3. **Debt to Equity Ratio** = Total Debt / Total Shareholders' Equity
    4. **P/E Ratio** = Stock Price / EPS
    5. **ROI** = (Net Income / Total Investment) * 100

    ### Financial Data:
    {text}
    """
    
    print('Retrieved text:', retrieved_text)
    prompt = template.format(text=retrieved_text)

    # LLM interaction
    try:
        messages = [
            {"role": "system", "content": "You are a financial assistant."},
            {"role": "user", "content": prompt}
        ]
        response = llm.invoke(messages)  # Use invoke instead of __call__
        if response.content:
            response_text = response.content
        else:
            raise ValueError("No content returned from LLM.")
    except Exception as e:
        return {"error": f"Failed to get response from LLM: {str(e)}"}

    # Convert the response text to a dictionary
    print("Response from LLM:", response_text)
    data_dict = convert_text_to_dict(response_text)
    return data_dict


def convert_text_to_dict(text):
    data_dict = {}
    lines = text.strip().split('\n')
    
    for line in lines:
        if ': ' in line:
            key, value = line.split(': ', 1)
            # Remove commas for conversion
            clean_value = value.replace(',', '')
            # Attempt to convert to float or keep as string
            try:
                if clean_value.lower() == "not available":
                    data_dict[key] = "Not Available"
                else:
                    data_dict[key] = float(clean_value)
            except ValueError:
                data_dict[key] = clean_value.strip()  # Keep as string if conversion fails

    return data_dict


def process_financial_report_using_rag_retriver(pdf_path):
    # Step 1: Extract text from PDF
    report_text = extract_text_from_pdf(pdf_path, chunk_size=1000)
    if not report_text:
        return {"error": "Failed to extract text from PDF"}

    # Step 2: Extract financial metrics using LangChain LLM
    print("report_text",report_text)
    financial_data = extract_data_from_report(report_text)
    if "error" in financial_data:
        return financial_data  # Return the error if data extraction failed
    
    return financial_data


# error handled :--  Please install it with `pip install faiss-gpu` (for CUDA supported GPU) or `pip install faiss-cpu