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
import chromadb


# from langchain.retrievers import VectorstoreRetriever
# from langchain.vectorstores import VectorstoreRetriever


# Set up your OpenAI API key
os.environ["OPENAI_API_KEY"] = "your key"

# llm = ChatOpenAI(model_name="gpt-4", temperature=0)
# llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)

llm = ChatOpenAI(model_name="gpt-4o-mini", temperature=0)

# chunk size for processing PDFs
chunk_size = 1000  #Each document will be divided into chunks of 1000 characters.


"""
Reason to use 1000 as chunk size :----

1. it is a common choice we can go for higher or lower as well
2. but this size is enough to capture meaingful context for the text and it is small in terms of memory for performing queries.
3. if we take smaller chunk, it will create more vector that makes search slower
4. longer chunks may result in less or fewer vector where we can lose individual embeddings



Reason to chose overlap as parameter :--

to ensure information close to chunk boundaries is not lost as we are doing question answering here, which lead to lose data



KNN --- index or ranking based on similarity --- most relavant chunks -- most similar weights --- context -- (advanced solution)


rewrite and refactoring  -- query solutions (open ended to user)
"""


def load_documents_to_chroma(documents, chunk_size=1000):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=50) #to split the input documents into smaller chunks
    split_docs = text_splitter.split_documents(documents)

    # Initialize ChromaDB
    embeddings = OpenAIEmbeddings()
    vectorstore = Chroma(persist_directory="chroma_data", embedding_function=embeddings)

    # Add documents to Chroma
    vectorstore.add_documents(split_docs)
    return vectorstore


def create_rag_pipeline(vectorstore):
    """
    load_qa_with_sources_chain(llm)
    where the LLM will generate an answer based on the retrieved documents and include the source of the information.

    chain_type="stuff", 
    This defines how the retrieved documents are processed.
    "Stuff" typically means that the documents are simply concatenated together and passed to the language model.
    
    """
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
    # print('Report text being processed:', report_text)
    print("came inside to work on retrival")
    # Initialize Chroma client
    client = chromadb.Client()
    
    # Split the report text into chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    texts = text_splitter.split_text(report_text)

    # Convert text to Document objects for Chroma indexing
    docs = [Document(page_content=text) for text in texts]

    # Create embeddings and vector store using Chroma
    embeddings = OpenAIEmbeddings()
    vectorstore = Chroma.from_documents(docs, embeddings, client=client, collection_name="financial_reports")

    # Create a retriever to fetch relevant documents
    retriever = vectorstore.as_retriever()

    # Query the relevant financial data
    query = "Extract financial data and ratios from the financial report."
    relevant_docs = retriever.get_relevant_documents(query)
    

    # Concatenate retrieved texts
    retrieved_text = " ".join([doc.page_content for doc in relevant_docs])
    print("getting retrived texts here :-- ",retrieved_text)
    # Improved prompt for better LLM interaction
    template = """You are the CFO of a company, tasked with analyzing the financial ratios from the provided financial report in Indian Rupees (INR). Your goal is to extract key financial data, compute essential financial ratios, assess the company's financial health, and provide investment risk insights.

        ### Data Extraction:
        Extract the following financial data from the provided text and **convert all currency values to INR**. **Remove any non-INR currency symbols** and **convert amounts in millions to their actual values** (e.g., 1 million becomes 1,000,000). Assume any non-INR amounts should be converted based on an exchange rate you specify (e.g., 1 USD = 83 INR). Include the conversion in the report.

        - **Earnings Per Share (EPS) (INR)**
        - **Stock Price (INR)**
        - **Net Income (e.g., Net Profit, Net Earnings, Income After Tax) (INR)**
        - **Total Investment (e.g., Total Capital, Capital Investment, Total Funds, Total Capital Employed, Capital Expenditure) (INR)**:
        - Look for terms such as **"Total Capital"**, **"Capital Investment"**, **"Total Funds"**, **"Total Capital Employed"**, **"Capital Expenditure"**, or **similar terms** that indicate total investment.
        - If none of these terms are directly available, compute **Total Investment** as the sum of **Total Debt** and **Total Shareholders' Equity**.
        - Use this value to calculate **ROI (Return on Investment)** using the formula: 
            - ROI = (Net Income / Total Investment) * 100.
        - If the term or value is missing, mark as "Not Available".
        - **Current Assets (INR)**
        - **Current Liabilities (INR)**
        - **Gross Profit (INR)**
        - **Net Revenue (INR)** (If missing, mark as "Not Available")
        - **Total Debt (INR)**
        - **Total Shareholders' Equity (INR)**
        - **Operating Cash Flow (INR)**

        **Note**: If amounts are in millions, convert them to their actual values. If values are in USD or any non-INR currency, convert them to INR at an exchange rate you specify. Remove all currency symbols or signs and convert them to INR as float numbers.

        Text:
        {text}

        ### Financial Ratio Calculations:
        Using the extracted financial data (in INR), compute the following ratios:

        - **Current Ratio**: `Current Ratio = Current Assets / Current Liabilities`
        - **Gross Profit Margin**: If `Net Revenue` is available, `Gross Profit Margin = (Gross Profit / Net Revenue) * 100`, otherwise return "Not Available".
        - **Debt to Equity Ratio**: `D/E Ratio = Total Debt / Total Shareholders' Equity`
        - **Operating Cash Flow**: Report this as a standalone value.
        - **Price to Earnings (P/E) Ratio**: `P/E Ratio = Stock Price / Earnings Per Share (EPS)`
        - **Return on Investment (ROI)**: If both `Net Income` and `Total Investment` are available, `ROI = (Net Income / Total Investment) * 100`. If either is missing, return "Not Available".

        ### Health and Investment Risk Assessment:
        Based on the calculated ratios, assess whether the investment is risky or not. Provide insight into the **health ratio for the IT industry** and how the company's ratios compare to industry benchmarks.

        ### Output Format:
        Answer in this format:

        ```plaintext
        EPS (INR): <value>
        Stock Price (INR): <value>
        Net Income (INR): <value>
        Total Investment (INR): <value> (If missing, mark as "Not Available")
        Current Assets (INR): <value>
        Current Liabilities (INR): <value>
        Gross Profit (INR): <value>
        Net Revenue (INR): <value> (If missing, mark as "Not Available")
        Total Debt (INR): <value>
        Total Shareholders' Equity (INR): <value>
        Operating Cash Flow (INR): <value>
        Current Ratio: <value>
        Gross Profit Margin: <value> (If missing, return "Not Available")
        D/E Ratio: <value>
        P/E Ratio: <value>
        ROI: <value> (If either Net Income or Total Investment is missing, return "Not Available")
        Investment Risk: <Risk Assessment - e.g., "Risky" or "Not Risky">
        Industry Health Ratio: <Industry-specific ratio analysis>
        Summary: <Provide a brief summary of the company's financial health and performance based on the above ratios and suggest if it aligns with typical IT industry health benchmarks.> 
    """
    
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
    # print("report_text",report_text)
    financial_data = extract_data_from_report(report_text)
    if "error" in financial_data:
        return financial_data  # Return the error if data extraction failed
    
    return financial_data


# error handled :--  Please install it with `pip install faiss-gpu` (for CUDA supported GPU) or `pip install faiss-cpu

{
  "Current Assets (INR)": "Not Available",
  "Current Liabilities (INR)": "Not Available",
  "Current Ratio": "Not Available",
  "D/E Ratio": "Not Available",
  "EPS (INR)": "Not Available",
  "Gross Profit (INR)": 146052000000.0,
  "Gross Profit Margin": 68.94,
  "Industry Health Ratio": "Not Available",
  "Investment Risk": "Risky",
  "Net Income (INR)": 72361000000.0,
  "Net Revenue (INR)": 211915000000.0,
  "Operating Cash Flow (INR)": "Not Available",
  "P/E Ratio": "Not Available",
  "ROI": "Not Available",
  "Stock Price (INR)": "Not Available",
  "Summary": "The company's financial health appears to be under pressure with a significant net income but missing key metrics such as total investment current assets and liabilities. The gross profit margin is strong indicating effective cost management but the lack of comprehensive data raises concerns about overall financial stability. Given the missing data and the potential for volatility in the IT sector the investment is assessed as risky. Further analysis is needed to align with industry benchmarks.",
  "Total Debt (INR)": "Not Available",
  "Total Investment (INR)": "Not Available",
  "Total Shareholders' Equity (INR)": "Not Available"
}