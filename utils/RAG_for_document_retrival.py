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


# from langchain.retrievers import VectorstoreRetriever

# from langchain.vectorstores import VectorstoreRetriever


# Set up your OpenAI API key
os.environ["OPENAI_API_KEY"] = "key_here"

# llm = ChatOpenAI(model_name="gpt-4", temperature=0)
llm = ChatOpenAI(model_name="gpt-4o-mini", temperature=0)

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
    retriever = vectorstore.as_retriever()
    rag_chain = RetrievalQA(llm=llm, retriever=retriever)
    return rag_chain

def process_pdf_with_rag(pdf_path):
    report_text = extract_text_from_pdf(pdf_path)
    if not report_text:
        return {"error": "Failed to extract text from PDF"}

    # Assume documents are loaded into a Chroma database
    document = Document(page_content=report_text)
    vectorstore = load_documents_to_chroma([document])

    # Create the RAG pipeline
    rag_chain = create_rag_pipeline(vectorstore)

    # Define the financial data extraction prompt
    financial_prompt = """
        You are the CFO of a company, tasked with analyzing the financial ratios from the provided financial report in Indian Rupees (INR). Your goal is to extract key financial data, compute essential financial ratios, assess the company's financial health, and provide investment risk insights.

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
    prompt = PromptTemplate(template=financial_prompt, input_variables=["text"])

    # Run the RAG pipeline on the report text
    result = rag_chain.run(prompt.render(text=report_text))
    print("getting result as ", result)
    return result

def pdf_processing_with_rag_for_financial_data(pdf_path):
    # Step 1: Extract text from PDF
    report_text = process_pdf_with_rag(pdf_path)
    if not report_text:
        return {"error": "Failed to extract text from PDF"}

    # Step 2: Extract financial metrics using LangChain LLM
    financial_data = report_text 
    if "error" in financial_data:
        return financial_data  # Return the error if data extraction failed
    
    return financial_data




#################################################################################################################################################
############################################### NORMAL IMPLEMENTATION ###########################################################################
#################################################################################################################################################


def validate_financial_data(data_dict):
    # List of required fields that should not be missing
    required_fields = [
        'EPS (INR)', 'Stock Price (INR)', 'Net Income (INR)', 'Total Investment (INR)', 
        'Current Assets (INR)', 'Current Liabilities (INR)', 'Gross Profit (INR)', 
        'Net Revenue (INR)', 'Total Debt (INR)', "Total Shareholders' Equity (INR)", 
        'Operating Cash Flow (INR)', 'Current Ratio', 'Gross Profit Margin', 
        'D/E Ratio', 'P/E Ratio', 'ROI'
    ]
    
    # Check if any required field is missing
    missing_fields = [field for field in required_fields if data_dict.get(field) is None]
    if missing_fields:
        return False, f"Missing fields: {', '.join(missing_fields)}"
    
    # Validate that numeric fields are floats and greater than or equal to zero
    for key, value in data_dict.items():
        if key not in ['Investment Risk', 'Summary']:  # Skip non-numeric fields
            if not isinstance(value, float) or value < 0:
                return False, f"Invalid value for {key}: {value}"
    
    return True, None


def extract_data_from_report(report_text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    texts = text_splitter.split_text(report_text)

    # Convert text to Document objects
    docs = [Document(page_content=text) for text in texts]

    # Create embeddings
    embeddings = OpenAIEmbeddings()

    # Create FAISS vector store
    vectorstore = FAISS.from_documents(docs, embeddings)

    # Create retriever
    retriever = vectorstore.as_retriever()

    # Query relevant documents
    query = "Extract financial data and ratios from the financial report."
    relevant_docs = retriever.get_relevant_documents(query)

    # Concatenate retrieved text
    retrieved_text = " ".join([doc.page_content for doc in relevant_docs])

    # Prepare the prompt
    template = """
        You are the CFO of a company, tasked with analyzing the financial ratios from the provided financial report in Indian Rupees (INR). Your goal is to extract key financial data, compute essential financial ratios, assess the company's financial health, and provide investment risk insights.

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
        response = llm(messages)
        response_text =  response.content
    except Exception as e:
        return {"error": f"Failed to get response from LLM: {str(e)}"}
    
    # Convert the response text to a dictionary
    print("getting respose from llm is :-- ", response_text)
    data_dict = convert_text_to_dict(response_text)
    # print("Extracted Data Dictionary:", data_dict)
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
    report_text = extract_text_from_pdf(pdf_path, chunk_size=chunk_size)
    if not report_text:
        return {"error": "Failed to extract text from PDF"}

    # Step 2: Extract financial metrics using LangChain LLM
    # print("getting report text here is :-- ", report_text)
    financial_data = extract_data_from_report(report_text)
    # print('getting financial data here is', financial_data)
    if "error" in financial_data:
        return financial_data  # Return the error if data extraction failed
    
    return financial_data
