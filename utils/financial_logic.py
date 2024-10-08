import os
from langchain.llms import OpenAI
from utils.pdf_parser import extract_text_from_pdf
# from langchain_community.llms import ollama
from utils.pdf_parser import extract_text_from_pdf
from langchain.chat_models import ChatOpenAI
from langchain.schema import HumanMessage, SystemMessage
import json
import re
chunk_size = 10 #to control how many pages are processed in each iteration


# # Set up your OpenAI API key
os.environ["OPENAI_API_KEY"] = "your key"
# Initialize LangChain with OpenAI LLM
# llm = OpenAI(temperature=0)

# # Initialize LangChain with ChatOpenAI
# # Initialize LangChain with OpenAI
# llm = OpenAI(model_name="gpt-4o-mini", temperature=0)

llm = ChatOpenAI(model_name="gpt-4o-mini", temperature=0)
# llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)


# llm = ChatOpenAI(model_name="gpt-4", temperature=0)

def validate_financial_data(data_dict):
    # List of required fields that should not be missing
    required_fields = ['EPS (INR)', 'Stock Price (INR)', 'Net Income (INR)', 'Total Investment (INR)', 
                       'Current Assets (INR)', 'Current Liabilities (INR)', 'Gross Profit (INR)', 
                       'Net Revenue (INR)', 'Total Debt (INR)', 'Total Shareholders\' Equity (INR)', 
                       'Operating Cash Flow (INR)', 'Current Ratio', 'Gross Profit Margin', 
                       'D/E Ratio', 'P/E Ratio', 'ROI']
    
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
    # LangChain prompt to extract financial metrics
    old_template = """
        You are the CFO of a company, tasked with analyzing the financial ratios from the provided financial report in Indian Rupees (INR). Your goal is to extract key financial data, compute essential financial ratios, assess the company's financial health, and provide investment risk insights. Here's what needs to be done:

        ### Data Extraction:
        Extract the following financial data from the provided text and **convert all currency values to INR**. **Remove any non-INR currency symbols** and **convert amounts in millions to their actual values** (e.g., 1 million becomes 1,000,000). Assume any non-INR amounts should be converted based on an exchange rate you specify (e.g., 1 USD = 83 INR). Include the conversion in the report.

        1. **Earnings Per Share (EPS) (INR)**
        2. **Stock Price (INR)**
        3. **Net Income (e.g., Net Profit, Net Earnings, Income After Tax) (INR)**
        4. **Total Investment (e.g., Total Capital, Capital Investment, Total Funds, Total Expenditure, Total Capital Employed) (INR)**: **Look for alternate terms or phrasing that might indicate Total Investment and ensure extraction.**
            - If **Total Investment** is not directly available, attempt to calculate it using the formula: `Total Investment = Net Income / ROI` where ROI is provided, or estimate if other relevant data is present. (If missing, mark as "Not Available")
        5. **Current Assets (INR)**
        6. **Current Liabilities (INR)**
        7. **Gross Profit (INR)**
        8. **Net Revenue (INR)** (If missing, mark as "Not Available")
        9. **Total Debt (INR)**
        10. **Total Shareholders' Equity (INR)**
        11. **Operating Cash Flow (INR)**
        12. **Return on Investment (ROI)**:
            - If **Total Investment** and **Net Income** are available, calculate `ROI = (Net Income / Total Investment) * 100`.
            - If **ROI** is not provided, attempt to calculate it using other available metrics (e.g., reverse-calculating from Total Investment or Net Income).

        **Note**: If amounts are in millions, convert them to their actual values (e.g., 1 million becomes 1,000,000). If values are in USD or any non-INR currency, convert them to INR at an exchange rate you specify (e.g., 1 USD = 83 INR). Remove all currency symbols or signs and convert them to Indian Rupees (INR) as float numbers.

        Text:
        {text}

        ### Financial Ratio Calculations:
        Using the extracted financial data (in INR), compute the following ratios:

        - **Current Ratio**: `Current Ratio = Current Assets / Current Liabilities`
        - **Gross Profit Margin**: If `Net Revenue` is available, `Gross Profit Margin = (Gross Profit / Net Revenue) * 100`, otherwise return "Not Available".
        - **Debt to Equity Ratio**: `D/E Ratio = Total Debt / Total Shareholders' Equity`
        - **Operating Cash Flow**: Report this as a standalone value.
        - **Price to Earnings (P/E) Ratio**: `P/E Ratio = Stock Price / Earnings Per Share (EPS)`
        - **Return on Investment (ROI)**: 
            - If both **Net Income** and **Total Investment** are available, `ROI = (Net Income / Total Investment) * 100`.
            - If either is missing, attempt to reverse-calculate **Total Investment** or **ROI** based on other available data.

        ### Health and Investment Risk Assessment:
        Based on the calculated ratios, assess whether the investment is risky or not. Additionally, provide insight into the **health ratio for the IT industry** and how the company's ratios compare to industry benchmarks.

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
        ```
    """

    up_template = """
        You are the CFO of a company, tasked with analyzing the financial ratios from the provided financial report in Indian Rupees (INR). Your goal is to extract key financial data, compute essential financial ratios, assess the company's financial health, and provide investment risk insights. Here's what needs to be done:

        ### Data Extraction:
        Extract the following financial data from the provided text and **convert all currency values to INR**. **Remove any non-INR currency symbols** and **convert amounts in millions to their actual values** (e.g., 1 million becomes 1,000,000). Assume any non-INR amounts should be converted based on an exchange rate you specify (e.g., 1 USD = 83 INR). Include the conversion in the report.

        1. **Earnings Per Share (EPS) (INR)**
        2. **Stock Price (INR)**
        3. **Net Income (e.g., Net Profit, Net Earnings, Income After Tax) (INR)**
        4. **Total Investment (e.g., Total Capital, Capital Investment, Total Funds, Capital Employed, Capital Deployed, Total Expenditure)**:
        - **Look for a wide range of terms** such as "Total Capital," "Capital Investment," "Total Expenditure," "Capital Employed," "Total Funds," "Capital Deployed," or variations that may indicate Total Investment.
        - Use contextual clues to infer investments or expenditures related to company funding.
        - If Total Investment is still missing, attempt to **reverse-calculate it** using other financial values or industry-standard ROI estimates. 
        - **If Total Investment is unavailable**, return "Not Available."
        5. **Current Assets (INR)**
        6. **Current Liabilities (INR)**
        7. **Gross Profit (INR)**
        8. **Net Revenue (INR)** (If missing, mark as "Not Available")
        9. **Total Debt (INR)**
        10. **Total Shareholders' Equity (INR)**
        11. **Operating Cash Flow (INR)**
        12. **Return on Investment (ROI)**:
            - If **Total Investment** and **Net Income** are available, calculate `ROI = (Net Income / Total Investment) * 100`.
            - If **ROI** is not provided and **Total Investment** is missing, **estimate Total Investment** using the formula: 
            `Total Investment = Net Income / (ROI percentage / 100)`. Use an industry-standard ROI if specific ROI is unavailable.

        ### Financial Ratio Calculations:
        Using the extracted financial data (in INR), compute the following ratios:

        - **Current Ratio**: `Current Ratio = Current Assets / Current Liabilities`
        - **Gross Profit Margin**: If `Net Revenue` is available, `Gross Profit Margin = (Gross Profit / Net Revenue) * 100`, otherwise return "Not Available".
        - **Debt to Equity Ratio**: `D/E Ratio = Total Debt / Total Shareholders' Equity`
        - **Operating Cash Flow**: Report this as a standalone value.
        - **Price to Earnings (P/E) Ratio**: `P/E Ratio = Stock Price / Earnings Per Share (EPS)`
        - **Return on Investment (ROI)**:
            - If **Net Income** and **Total Investment** are available, `ROI = (Net Income / Total Investment) * 100`.
            - If **Total Investment** is missing but ROI is provided, attempt to **reverse-calculate Total Investment** using available Net Income.

        ### Health and Investment Risk Assessment:
        Based on the calculated ratios, assess whether the investment is risky or not. Additionally, provide insight into the **health ratio for the IT industry** and how the company's ratios compare to industry benchmarks.

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
        ```
    """

    third_template = """
        You are the CFO of a company, tasked with analyzing the financial ratios from the provided financial report in Indian Rupees (INR). Your goal is to extract key financial data, compute essential financial ratios, assess the company's financial health, and provide investment risk insights.

        ### Data Extraction:
        Extract the following financial data from the provided text and **convert all currency values to INR**. **Remove any non-INR currency symbols** and **convert amounts in millions to their actual values** (e.g., 1 million becomes 1,000,000). Assume any non-INR amounts should be converted based on an exchange rate you specify (e.g., 1 USD = 83 INR). Include the conversion in the report.

        - **Earnings Per Share (EPS) (INR)**
        - **Stock Price (INR)**
        - **Net Income (e.g., Net Profit, Net Earnings, Income After Tax) (INR)**
        - **Total Investment (e.g., Total Capital, Capital Investment, Total Funds, Total Capital Employed, Capital Expenditure) (INR)**: Ensure that terms like "Total Capital Employed" or "Total Capital Expenditure" are included in the extraction process to find `Total Investment`. (If missing, mark as "Not Available")
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
    
    messages = [
            SystemMessage(content="You are a financial assistant."),
            HumanMessage(content=template.format(text=report_text))
        ]
    
    try:
        # Make request to the ChatOpenAI model
        response = llm(messages)
        # print("getting response from the llm is {}".format(response))
        response_text = response.content  # Get the content directly from AIMessage object
        print("LLM response:", response_text)
    except Exception as e:
        return {"error": f"Failed to get response from LLM: {str(e)}"}
    
    # Process the response
    # response_text = response['choices'][0]['message']['content']
    # print("Response text:\n", response_text)
    
    data_dict = financial_data_dict = convert_text_to_dict(response_text)
    print("getting dict here is ",data_dict)
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

def process_financial_report(pdf_path):
    # Step 1: Extract text from PDF
    report_text = extract_text_from_pdf(pdf_path, chunk_size=chunk_size)
    if not report_text:
        return {"error": "Failed to extract text from PDF"}
    print("getting report text here is:--", report_text)
    # Step 2: Extract financial metrics using LangChain LLM
    financial_data = extract_data_from_report(report_text)
    if "error" in financial_data:
        return financial_data  # Return the error if data extraction failed
    
    return financial_data



# pevious templates :--
 # template = """
    #     You are the CFO of a company, tasked with analyzing the financial ratios from the provided financial report in Indian Rupees (INR). Your goal is to extract key financial data, compute essential financial ratios, assess the company's financial health, and provide investment risk insights. Here's what needs to be done:

    #     ### Data Extraction:
    #     Extract the following financial data from the provided text and **convert all currency values to INR**. **Remove any non-INR currency symbols** and **convert amounts in millions to their actual values** (e.g., 1 million becomes 1,000,000). Assume any non-INR amounts should be converted based on an exchange rate you specify (e.g., 1 USD = 83 INR). Include the conversion in the report.

    #     1. **Earnings Per Share (EPS) (INR)**
    #     2. **Stock Price (INR)**
    #     3. **Net Income (e.g., Net Profit, Net Earnings, Income After Tax) (INR)**
    #     4. **Total Investment (e.g., Total Capital, Capital Investment, Total Funds) (INR)** (If missing, mark as "Not Available")
    #     5. **Current Assets (INR)**
    #     6. **Current Liabilities (INR)**
    #     7. **Gross Profit (INR)**
    #     8. **Net Revenue (INR)** (If missing, mark as "Not Available")
    #     9. **Total Debt (INR)**
    #     10. **Total Shareholders' Equity (INR)**
    #     11. **Operating Cash Flow (INR)**

    #     **Note**: If amounts are in millions, convert them to their actual values (e.g., 1 million becomes 1,000,000). If values are in USD or any non-INR currency, convert them to INR at an exchange rate you specify (e.g., 1 USD = 83 INR). Remove all currency symbols or signs and convert them to Indian Rupees (INR) as float numbers.

    #     Text:
    #     {text}

    #     ### Financial Ratio Calculations:
    #     Using the extracted financial data (in INR), compute the following ratios:

    #     - **Current Ratio**: `Current Ratio = Current Assets / Current Liabilities`
    #     - **Gross Profit Margin**: If `Net Revenue` is available, `Gross Profit Margin = (Gross Profit / Net Revenue) * 100`, otherwise return "Not Available".
    #     - **Debt to Equity Ratio**: `D/E Ratio = Total Debt / Total Shareholders' Equity`
    #     - **Operating Cash Flow**: Report this as a standalone value.
    #     - **Price to Earnings (P/E) Ratio**: `P/E Ratio = Stock Price / Earnings Per Share (EPS)`

    #     - **Return on Investment (ROI)**: 
    #     - If both `Net Income` and `Total Investment` are available, `ROI = (Net Income / Total Investment) * 100`. 
    #     - If either is missing, return "Not Available".

    #     ### Health and Investment Risk Assessment:
    #     Based on the calculated ratios, assess whether the investment is risky or not. Additionally, provide insight into the **health ratio for the IT industry** and how the company's ratios compare to industry benchmarks.

    #     ### Output Format:
    #     Answer in this format:

    #     ```plaintext
    #     EPS (INR): <value>
    #     Stock Price (INR): <value>
    #     Net Income (INR): <value>
    #     Total Investment (INR): <value> (If missing, mark as "Not Available")
    #     Current Assets (INR): <value>
    #     Current Liabilities (INR): <value>
    #     Gross Profit (INR): <value>
    #     Net Revenue (INR): <value> (If missing, mark as "Not Available")
    #     Total Debt (INR): <value>
    #     Total Shareholders' Equity (INR): <value>
    #     Operating Cash Flow (INR): <value>

    #     Current Ratio: <value>
    #     Gross Profit Margin: <value> (If missing, return "Not Available")
    #     D/E Ratio: <value>
    #     P/E Ratio: <value>
    #     ROI: <value> (If either Net Income or Total Investment is missing, return "Not Available")
    #     Investment Risk: <Risk Assessment - e.g., "Risky" or "Not Risky">
    #     Industry Health Ratio: <Industry-specific ratio analysis>
    #     Summary: <Provide a brief summary of the company's financial health and performance based on the above ratios and suggest if it aligns with typical IT industry health benchmarks.>    
    #     """
    
    # template1 = """
    # You are the CFO of a company, tasked with analyzing the financial ratios from the provided financial report in Indian Rupees (INR). Your goal is to extract key financial data, compute essential financial ratios, assess the company's financial health, and provide investment risk insights. Here's what needs to be done:

    # ### Data Extraction:
    # Extract the following financial data from the provided text and **convert all currency values to INR**. **Remove any non-INR currency symbols** and **convert amounts in millions to their actual values** (e.g., 1 million becomes 1,000,000). Assume any non-INR amounts should be converted based on an exchange rate you specify (e.g., 1 USD = 83 INR). Include the conversion in the report.

    # 1. **Earnings Per Share (EPS) (INR)**
    # 2. **Stock Price (INR)**
    # 3. **Net Income (e.g., Net Profit, Net Earnings, Income After Tax) (INR)**
    # 4. **Total Investment (e.g., Total Capital, Capital Investment, Total Funds) (INR)** (If missing, mark as "Not Available")
    # 5. **Current Assets (INR)**
    # 6. **Current Liabilities (INR)**
    # 7. **Gross Profit (INR)**
    # 8. **Net Revenue (INR)** (If missing, mark as "Not Available")
    # 9. **Total Debt (INR)**
    # 10. **Total Shareholders' Equity (INR)**
    # 11. **Operating Cash Flow (INR)**

    # **Note**: If amounts are in millions, convert them to their actual values (e.g., 1 million becomes 1,000,000). If values are in USD or any non-INR currency, convert them to INR at an exchange rate you specify (e.g., 1 USD = 83 INR). Remove all currency symbols or signs and convert them to Indian Rupees (INR) as float numbers. Make sure to include both `Total Investment` and `ROI` calculations even if data for one or both is missing.

    # Text:
    # {text}

    # ### Financial Ratio Calculations:
    # Using the extracted financial data (in INR), compute the following ratios:

    # - **Current Ratio**: `Current Ratio = Current Assets / Current Liabilities`
    # - **Gross Profit Margin**: If `Net Revenue` is available, `Gross Profit Margin = (Gross Profit / Net Revenue) * 100`, otherwise return "Not Available".
    # - **Debt to Equity Ratio**: `D/E Ratio = Total Debt / Total Shareholders' Equity`
    # - **Operating Cash Flow**: Report this as a standalone value.
    # - **Price to Earnings (P/E) Ratio**: `P/E Ratio = Stock Price / Earnings Per Share (EPS)`

    # - **Return on Investment (ROI)**: 
    # - If both `Net Income` and `Total Investment` are available, `ROI = (Net Income / Total Investment) * 100`. 
    # - If either is missing, return "Not Available" for ROI.

    # ### Health and Investment Risk Assessment:
    # Based on the calculated ratios, assess whether the investment is risky or not. Additionally, provide insight into the **health ratio for the IT industry** and how the company's ratios compare to industry benchmarks.

    # ### Output Format:
    # Answer in this format:

    # ```plaintext
    # EPS (INR): <value>
    # Stock Price (INR): <value>
    # Net Income (INR): <value>
    # Total Investment (INR): <value> (If missing, mark as "Not Available")
    # Current Assets (INR): <value>
    # Current Liabilities (INR): <value>
    # Gross Profit (INR): <value>
    # Net Revenue (INR): <value> (If missing, mark as "Not Available")
    # Total Debt (INR): <value>
    # Total Shareholders' Equity (INR): <value>
    # Operating Cash Flow (INR): <value>
    # Current Ratio: <value>
    # Gross Profit Margin: <value> (If missing, return "Not Available")
    # D/E Ratio: <value>
    # P/E Ratio: <value>
    # ROI: <value> (If either Net Income or Total Investment is missing, return "Not Available")
    # Investment Risk: <Risk Assessment - e.g., "Risky" or "Not Risky">
    # Industry Health Ratio: <Industry-specific ratio analysis>
    # Summary: <Provide a brief summary of the company's financial health and performance based on the above ratios and suggest if it aligns with typical IT industry health benchmarks.>    
    # """



#     {
#   "Current Assets (INR)": 184257000000.0,
#   "Current Liabilities (INR)": 104149000000.0,
#   "Current Ratio": 1.77,
#   "D/E Ratio": 0.23,
#   "EPS (INR)": 9.68,
#   "Gross Profit (INR)": 146052000000.0,
#   "Gross Profit Margin": 68.94,
#   "Industry Health Ratio": "The IT industry typically has a current ratio above 1.5 a gross profit margin around 60-70% and a D/E ratio below 0.5 indicating a healthy financial position.",
#   "Investment Risk": "Not Risky",
#   "Net Income (INR)": 72361000000.0,
#   "Net Revenue (INR)": 211915000000.0,
#   "Operating Cash Flow (INR)": 87582000000.0,
#   "P/E Ratio": 26.0,
#   "ROI": 35.14,
#   "Stock Price (INR)": 252.59,
#   "Summary": "The company's financial health appears strong with a current ratio of 1.77 indicating good short-term liquidity a gross profit margin of 68.94% suggesting efficient cost management and a low D/E ratio of 0.23 indicating low financial leverage. The P/E ratio of 26.00 is reasonable for the IT sector reflecting growth expectations. Overall these ratios align well with typical IT industry benchmarks suggesting that the company is not considered a risky investment.",
#   "Total Debt (INR)": 47237000000.0,
#   "Total Investment (INR)": 205753000000.0,
#   "Total Shareholders' Equity (INR)": 206223000000.0}