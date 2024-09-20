# try:
#     # Make request to the LLM
#     response = llm(prompt)
#     print("LLM response:", response)
# except Exception as e:
#     return {"error": f"Failed to get response from LLM: {str(e)}"}

# # Extract the text part of the response
# response_text = response
# data_dict = {}

# # Define regex patterns for extraction
# patterns = {
#     "EPS": r"EPS:\s*([\d.]+)",
#     "Stock Price": r"Stock Price:\s*([\d.]+)",
#     "Net Income": r"Net Income:\s*([\d.]+)",
#     "Total Investment": r"Total Investment:\s*([\d.]+)"
# }

# # Extract financial data using regex patterns
# for key, pattern in patterns.items():
#     match = re.search(pattern, response_text)
#     if match:
#         data_dict[key] = float(match.group(1))
#     else:
#         data_dict[key] = None

# # Validate extracted data
# is_valid, error_message = validate_financial_data(data_dict)
# if not is_valid:
#     return {"error": error_message}

# return data_dict

######################################## PREVIOUS CODE ###########################################################

# def calculate_pe_ratio(eps, stock_price):
#     if eps <= 0:
#         return {"error": "EPS must be greater than zero for P/E ratio calculation."}
#     return stock_price / eps

# def calculate_roi(net_income, total_investment):
#     if total_investment <= 0:
#         return {"error": "Total Investment must be greater than zero for ROI calculation."}
#     return (net_income / total_investment) * 100

# def classify_risk(pe_ratio, roi):
#     if pe_ratio is None or roi is None:
#         return "Insufficient Data"
    
#     if pe_ratio > 25 or roi < 5:
#         return "Risky"
#     elif pe_ratio < 10 or roi > 20:
#         return "Not Risky"
#     else:
#         return "Moderate Risk"

# def process_financial_report(pdf_path):
#     # Step 1: Extract text from PDF
#     report_text = extract_text_from_pdf(pdf_path, chunk_size=chunk_size)
#     if not report_text:
#         return {"error": "Failed to extract text from PDF"}

#     # Step 2: Extract financial metrics using LangChain LLM
#     financial_data = extract_data_from_report(report_text)
#     if "error" in financial_data:
#         return financial_data  # Return the error if data extraction failed

#     eps = float(financial_data.get('EPS', 0))  
#     stock_price = float(financial_data.get('Stock Price', 0))  
#     net_income = float(financial_data.get('Net Income', 0))
#     total_investment = float(financial_data.get('Total Investment', 0))

#     # Check if the extracted values are valid
#     if eps <= 0 or stock_price <= 0 or net_income <= 0 or total_investment <= 0:
#         return {"error": "Invalid financial data extracted"}

#     # Step 3: Calculate P/E Ratio and ROI
#     pe_ratio = calculate_pe_ratio(eps, stock_price)
#     if isinstance(pe_ratio, dict) and "error" in pe_ratio:
#         return pe_ratio  # Return error if P/E ratio calculation failed

#     roi = calculate_roi(net_income, total_investment)
#     if isinstance(roi, dict) and "error" in roi:
#         return roi  # Return error if ROI calculation failed

#     # Step 4: Classify risk based on P/E ratio and ROI
#     risk = classify_risk(pe_ratio, roi)

#     return {
#         "P/E Ratio": pe_ratio,
#         "ROI": roi,
#         "Risk": risk
#     }



################################################ TEMPLATE USED BEFORE FOR EXPERIMENTATION ####################################

#first one >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
# """
# Extract the following financial data from the text and format them as floats. Remove any currency symbols and convert amounts in millions to actual numbers:

# 1. Earnings Per Share (EPS)
# 2. Stock Price
# 3. Net Income (e.g., Net Profit, Net Earnings, Income After Tax)
# 4. Total Investment (e.g., Total Capital, Capital Investment, Total Funds)

# Text:
# {text}

# Answer in this format:
# EPS: <value>
# Stock Price: <value>
# Net Income: <value>
# Total Investment: <value>

# Note: If amounts are in millions, convert them to their actual values (e.g., 1 million becomes 1000000). Remove all currency symbols or signs. Provide the values as float numbers.
# """
# second prompt >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
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



#>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> WORKING TEMPLATE >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

template = """
        You are the CFO of a company, tasked with analyzing the financial ratios from the provided financial report in Indian Rupees (INR). Your goal is to extract key financial data, compute essential financial ratios, assess the company's financial health, and provide investment risk insights. Here's what needs to be done:

        ### Data Extraction:
        Extract the following financial data from the provided text and **convert all currency values to INR**. **Remove any non-INR currency symbols** and **convert amounts in millions to their actual values** (e.g., 1 million becomes 1,000,000). Assume any non-INR amounts should be converted based on an exchange rate of your choice, which you must specify.

        1. **Earnings Per Share (EPS) (INR)**
        2. **Stock Price (INR)**
        3. **Net Income (e.g., Net Profit, Net Earnings, Income After Tax) (INR)**
        4. **Total Investment (e.g., Total Capital, Capital Investment, Total Funds) (INR)**
        5. **Current Assets (INR)**
        6. **Current Liabilities (INR)**
        7. **Gross Profit (INR)**
        8. **Net Revenue (INR)**
        9. **Total Debt (INR)**
        10. **Total Shareholders' Equity (INR)**
        11. **Operating Cash Flow (INR)**

        **Note**: If amounts are in millions, convert them to their actual values (e.g., 1 million becomes 1,000,000). Remove all currency symbols or signs and convert them to Indian Rupees (INR) as float numbers.

        Text:
        {text}

        ### Financial Ratio Calculations:
        Using the extracted financial data (in INR), compute the following ratios:

        - **Current Ratio**: `Current Ratio = Current Assets / Current Liabilities`
        - **Gross Profit Margin**: `Gross Profit Margin = (Gross Profit / Net Revenue) * 100`
        - **Debt to Equity Ratio**: `D/E Ratio = Total Debt / Total Shareholders' Equity`
        - **Operating Cash Flow**: Report this as a standalone value.
        - **Price to Earnings (P/E) Ratio**: `P/E Ratio = Stock Price / Earnings Per Share (EPS)`
        - **Return on Investment (ROI)**: `ROI = Net Income / Total Investment`

        ### Health and Investment Risk Assessment:
        Based on the calculated ratios, assess whether the investment is risky or not. Additionally, provide insight into the **health ratio for the IT industry** and how the company's ratios compare to industry benchmarks.

        ### Output Format:
        Answer in this format:

        ```plaintext
        EPS (INR): <value>
        Stock Price (INR): <value>
        Net Income (INR): <value>
        Total Investment (INR): <value>
        Current Assets (INR): <value>
        Current Liabilities (INR): <value>
        Gross Profit (INR): <value>
        Net Revenue (INR): <value>
        Total Debt (INR): <value>
        Total Shareholders' Equity (INR): <value>
        Operating Cash Flow (INR): <value>

        Current Ratio: <value>
        Gross Profit Margin: <value>
        D/E Ratio: <value>
        P/E Ratio: <value>
        ROI: <value>

        Investment Risk: <Risk Assessment - e.g., "Risky" or "Not Risky">
        Industry Health Ratio: <Industry-specific ratio analysis>
        Summary: <Provide a brief summary of the company's financial health and performance based on the above ratios and suggest if it aligns with typical IT industry health benchmarks.>    
        """