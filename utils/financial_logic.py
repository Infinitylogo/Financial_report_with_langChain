import os
from langchain.llms import OpenAI
from utils.pdf_parser import extract_text_from_pdf
# from langchain_community.llms import ollama
from utils.pdf_parser import extract_text_from_pdf
from langchain.chat_models import ChatOpenAI
import json
import re


# # Set up your OpenAI API key
os.environ["OPENAI_API_KEY"] = "my_key"
# Initialize LangChain with OpenAI LLM
# llm = OpenAI(temperature=0)

# # Initialize LangChain with ChatOpenAI
# # Initialize LangChain with OpenAI
llm = OpenAI(model_name="gpt-4o-mini", temperature=0)


# llm = ChatOpenAI(model_name="gpt-4", temperature=0)


# def convert_to_float(value_str):
#     # Remove currency symbols
#     value_str = re.sub(r'[^\d.,]', '', value_str)
#     # Replace comma with dot for decimal
#     value_str = value_str.replace(',', '.')
#     # Convert to float
#     try:
#         value = float(value_str)
#     except ValueError:
#         value = None
#     return value

# def process_financial_data(output):
#     # Find amounts in millions and convert
#     def convert_millions(value):
#         if value is not None:
#             if 'million' in value.lower():
#                 value = re.sub(r'[^\d.,]', '', value.lower())
#                 value = float(value.replace(',', '.'))
#                 return value * 1_000_000
#             else:
#                 return convert_to_float(value)
#         return None

#     # Extract the values from the output (assuming the format is strictly followed)
#     eps = re.search(r'EPS: ([\w\s\d.,]+)', output)
#     stock_price = re.search(r'Stock Price: ([\w\s\d.,]+)', output)
#     net_income = re.search(r'Net Income|Net Profit|Net Earnings|Income After Tax: ([\w\s\d.,]+)', output)
#     total_investment = re.search(r'Total Investment|Total Capital|Capital Investment|Total Funds: ([\w\s\d.,]+)', output)
    
#     # Convert to float and process
#     print(net_income)
#     eps_value = convert_millions(eps.group(1)) if eps else 0
#     stock_price_value = convert_millions(stock_price.group(1)) if stock_price else 0
#     net_income_value = convert_millions(net_income.group(1)) if net_income else 0
#     total_investment_value = convert_millions(total_investment.group(1)) if total_investment else 0
    
#     return {
#         'EPS': eps_value,
#         'Stock Price': stock_price_value,
#         'Net Income': net_income_value,
#         'Total Investment': total_investment_value
#     }


def extract_data_from_report(report_text):
    # LangChain prompt to extract financial metrics
    template = """
    Extract the following financial data from the text and format them as floats. Remove any currency symbols and convert amounts in millions to actual numbers:

    1. Earnings Per Share (EPS)
    2. Stock Price
    3. Net Income (e.g., Net Profit, Net Earnings, Income After Tax)
    4. Total Investment (e.g., Total Capital, Capital Investment, Total Funds)

    Text:
    {text}

    Answer in this format:
    EPS: <value>
    Stock Price: <value>
    Net Income: <value>
    Total Investment: <value>
    
    Note: If amounts are in millions, convert them to their actual values (e.g., 1 million becomes 1000000). Remove all currency symbols or signs. Provide the values as float numbers.
    """
    # template = """
    # Extract the following financial information from the report text, including variations and alternative terms:
    
    # 1. Earnings Per Share (EPS) - Look for terms like "Earnings Per Share" or simply "EPS".
    # 2. Stock Price - Look for phrases like "Stock Price", "Share Price", or "Price Per Share".
    # 3. Net Income (e.g., Net Profit, Net Earnings, Income After Tax) - Look for variations such as "Net Profit", "Net Earnings", "Net Income", "Income After Tax", or "Profit After Tax".
    # 4. Total Investment (e.g., Total Capital, Capital Investment, Total Funds) - Identify terms like "Total Capital", "Capital Investment", "Total Investment", or "Total Funds".

    # After identifying these values, format them as follows:
    
    # - Remove any currency symbols (e.g., $, €, etc.).
    # - Convert amounts in millions to their full value (e.g., 1 million should be converted to 1,000,000).
    # - Ensure all values are returned as floats for further calculations.

    # Here's the text from the financial report:
    
    # {text}

    # Provide the extracted and formatted values as follows:
    
    # EPS: <value>
    # Stock Price: <value>
    # Net Income: <value>
    # Total Investment: <value>
    
    # Additional Notes:
    # - If no value is found for any of the fields, return "Not Found" as the value.
    # - Ensure that large numbers are correctly converted and any commas or periods are handled according to their proper use in numeric formatting.
    # """

    prompt = template.format(text=report_text)
    
    # Make request to the LLM
    response = llm(prompt)
    
    # Print raw response for debugging
    print("Getting response here is", response)
    
    # Extract the text part of the response
    response_text = response #response['choices'][0]['text'].strip()

    print("response text i am getting here is :-- ", response_text)

    data_dict = {}
    
    # Define regex patterns for extraction
    patterns = {
        "EPS": r"EPS:\s*([\d.]+)",
        "Stock Price": r"Stock Price:\s*([\d.]+)",
        "Net Income": r"Net Income:\s*([\d.]+)",
        "Total Investment": r"Total Investment:\s*([\d.]+)"
    }
    
    for key, pattern in patterns.items():
        match = re.search(pattern, response)
        if match:
            data_dict[key] = float(match.group(1)) or 0
    
    return data_dict

def calculate_pe_ratio(eps, stock_price):
    return stock_price / eps if eps > 0 else 0

def calculate_roi(net_income, total_investment):
    return (net_income / total_investment) * 100 if total_investment > 0 else 0

def classify_risk(pe_ratio, roi):
    if pe_ratio is not None:
        if pe_ratio > 25 or roi < 5:
            return "Risky"
        elif pe_ratio < 10 or roi > 20:
            return "Not Risky"
        else:
            return "Moderate Risk"
    else:
        return "Insufficient Data"

def process_financial_report(pdf_path):
    # Step 1: Extract text from PDF
    report_text = extract_text_from_pdf(pdf_path)
    
    # Step 2: Extract financial metrics using LangChain LLM
    financial_data = extract_data_from_report(report_text)

    print("getting financial data here is",financial_data)
    
    print("type here is ",isinstance(financial_data, dict))


    eps = float(financial_data.get('EPS', 0))  
    stock_price = float(financial_data.get('Stock Price', 0))  
    net_income = float(financial_data.get('Net Income', 0))
    total_investment = float(financial_data.get('Total Investment', 0))

    print("getting values here as" ,eps, stock_price, net_income, total_investment)
    
    # Step 3: Calculate P/E Ratio and ROI
    pe_ratio = calculate_pe_ratio(eps, stock_price)
    roi = calculate_roi(net_income, total_investment)
    
    # Step 4: Classify risk based on P/E ratio and ROI
    risk = classify_risk(pe_ratio, roi)
    
    return {
        "P/E Ratio": pe_ratio,
        "ROI": roi,
        "Risk": risk
    }



#RAG EXPERIMENTATION>>>>>>>>>>>>>>>>>>>>>>>>>>>

# import os
# from langchain.llms import OpenAI
# from langchain.vectorstores import Chroma
# from langchain.chains import RetrievalQA
# from langchain.embeddings import OpenAIEmbeddings
# from utils.pdf_parser import extract_text_from_pdf
# import json
# import re

# # Set up your OpenAI API key
# os.environ["OPENAI_API_KEY"] = ""

# # Initialize ChromaDB
# embeddings = OpenAIEmbeddings()
# # vector_store = Chroma(embedding_function=embeddings.embed_query)

# vector_store = Chroma(embedding_function=embeddings.embed_documents)

# # Initialize LangChain with OpenAI LLM
# llm = OpenAI(model_name="gpt-4", temperature=0)

# # Initialize RetrievalQA with LangChain and ChromaDB
# retrieval_qa = RetrievalQA(llm=llm, retriever=vector_store.as_retriever())

# def extract_data_from_report(report_text):
#     # Retrieve relevant context from ChromaDB
#     context = retrieval_qa.retrieve(report_text)

#     # Use LangChain LLM to extract financial metrics
#     template = """
#     Extract the following financial data from the text:

#     1. Earnings Per Share (EPS)
#     2. Stock Price
#     3. Net Income
#     4. Total Investment

#     Context:
#     {context}

#     Text:
#     {text}

#     Answer in this format:
#     EPS: <value>
#     Stock Price: <value>
#     Net Income: <value>
#     Total Investment: <value>
#     """
#     prompt = template.format(context=context, text=report_text)
    
#     # Make request to the LLM
#     response = llm(prompt)
    
#     # Print raw response for debugging
#     print("Getting response here is", response)
    
#     # Extract the text part of the response
#     # response_text = response['choices'][0]['text'].strip()

#     # Process response to extract data into a dictionary
#     data_dict = {}
    
#     # Define regex patterns for extraction
#     patterns = {
#         "EPS": r"EPS:\s*([\d.]+)",
#         "Stock Price": r"Stock Price:\s*([\d.]+)",
#         "Net Income": r"Net Income:\s*([\d.]+)",
#         "Total Investment": r"Total Investment:\s*([\d.]+)"
#     }
    
#     for key, pattern in patterns.items():
#         match = re.search(pattern, response)
#         if match:
#             data_dict[key] = float(match.group(1))
    
#     return data_dict

# # The rest of your functions (calculate_pe_ratio, calculate_roi, classify_risk) remain unchanged

# def process_financial_report(pdf_path):
#     # Step 1: Extract text from PDF
#     report_text = extract_text_from_pdf(pdf_path)
    
#     # Step 2: Extract financial metrics using LangChain LLM
#     financial_data = extract_data_from_report(report_text)

#     print("getting financial data here is", financial_data)
    
#     eps = float(financial_data.get('EPS', 0))  
#     stock_price = float(financial_data.get('Stock Price', 0))  
#     net_income = float(financial_data.get('Net Income', 0))
#     total_investment = float(financial_data.get('Total Investment', 0))
    
#     print("getting values here as", eps, stock_price, net_income, total_investment)
    
#     # Step 3: Calculate P/E Ratio and ROI
#     pe_ratio = calculate_pe_ratio(eps, stock_price)
#     roi = calculate_roi(net_income, total_investment)
    
#     # Step 4: Classify risk based on P/E ratio and ROI
#     risk = classify_risk(pe_ratio, roi)
    
#     return {
#         "P/E Ratio": pe_ratio,
#         "ROI": roi,
#         "Risk": risk
#     }
