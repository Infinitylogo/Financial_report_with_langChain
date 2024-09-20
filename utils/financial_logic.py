import os
from langchain.llms import OpenAI
from utils.pdf_parser import extract_text_from_pdf
# from langchain_community.llms import ollama
from utils.pdf_parser import extract_text_from_pdf
from langchain.chat_models import ChatOpenAI
import json
import re
chunk_size = 10 #to control how many pages are processed in each iteration


# # Set up your OpenAI API key
os.environ["OPENAI_API_KEY"] = "open_api_key"
# Initialize LangChain with OpenAI LLM
# llm = OpenAI(temperature=0)

# # Initialize LangChain with ChatOpenAI
# # Initialize LangChain with OpenAI
llm = OpenAI(model_name="gpt-4o-mini", temperature=0)


# llm = ChatOpenAI(model_name="gpt-4", temperature=0)

def validate_financial_data(data_dict):
    required_fields = ['EPS', 'Stock Price', 'Net Income', 'Total Investment']
    missing_fields = [field for field in required_fields if data_dict.get(field) is None]
    
    if missing_fields:
        return False, f"Missing fields: {', '.join(missing_fields)}"
    
    # Validate all fields are floats and greater than zero where applicable
    for key, value in data_dict.items():
        if not isinstance(value, float) or value < 0:
            return False, f"Invalid value for {key}: {value}"
    
    return True, None

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
    
    try:
        # Make request to the LLM
        response = llm(prompt)
        print("LLM response:", response)
    except Exception as e:
        return {"error": f"Failed to get response from LLM: {str(e)}"}
    
    # Extract the text part of the response
    response_text = response
    data_dict = {}
    
    # Define regex patterns for extraction
    patterns = {
        "EPS": r"EPS:\s*([\d.]+)",
        "Stock Price": r"Stock Price:\s*([\d.]+)",
        "Net Income": r"Net Income:\s*([\d.]+)",
        "Total Investment": r"Total Investment:\s*([\d.]+)"
    }
    
    # Extract financial data using regex patterns
    for key, pattern in patterns.items():
        match = re.search(pattern, response_text)
        if match:
            data_dict[key] = float(match.group(1))
        else:
            data_dict[key] = None
    
    # Validate extracted data
    is_valid, error_message = validate_financial_data(data_dict)
    if not is_valid:
        return {"error": error_message}
    
    return data_dict

def calculate_pe_ratio(eps, stock_price):
    if eps <= 0:
        return {"error": "EPS must be greater than zero for P/E ratio calculation."}
    return stock_price / eps

def calculate_roi(net_income, total_investment):
    if total_investment <= 0:
        return {"error": "Total Investment must be greater than zero for ROI calculation."}
    return (net_income / total_investment) * 100

def classify_risk(pe_ratio, roi):
    if pe_ratio is None or roi is None:
        return "Insufficient Data"
    
    if pe_ratio > 25 or roi < 5:
        return "Risky"
    elif pe_ratio < 10 or roi > 20:
        return "Not Risky"
    else:
        return "Moderate Risk"

def process_financial_report(pdf_path):
    # Step 1: Extract text from PDF
    report_text = extract_text_from_pdf(pdf_path, chunk_size=chunk_size)
    if not report_text:
        return {"error": "Failed to extract text from PDF"}

    # Step 2: Extract financial metrics using LangChain LLM
    financial_data = extract_data_from_report(report_text)
    if "error" in financial_data:
        return financial_data  # Return the error if data extraction failed

    eps = float(financial_data.get('EPS', 0))  
    stock_price = float(financial_data.get('Stock Price', 0))  
    net_income = float(financial_data.get('Net Income', 0))
    total_investment = float(financial_data.get('Total Investment', 0))

    # Check if the extracted values are valid
    if eps <= 0 or stock_price <= 0 or net_income <= 0 or total_investment <= 0:
        return {"error": "Invalid financial data extracted"}

    # Step 3: Calculate P/E Ratio and ROI
    pe_ratio = calculate_pe_ratio(eps, stock_price)
    if isinstance(pe_ratio, dict) and "error" in pe_ratio:
        return pe_ratio  # Return error if P/E ratio calculation failed

    roi = calculate_roi(net_income, total_investment)
    if isinstance(roi, dict) and "error" in roi:
        return roi  # Return error if ROI calculation failed

    # Step 4: Classify risk based on P/E ratio and ROI
    risk = classify_risk(pe_ratio, roi)

    return {
        "P/E Ratio": pe_ratio,
        "ROI": roi,
        "Risk": risk
    }