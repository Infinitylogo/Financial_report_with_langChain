# Financial Report Analysis with LangChain

This project utilizes LangChain and OpenAI's GPT-4 model to extract and analyze financial data from company PDF reports. The objective is to retrieve key financial metrics, calculate ratios (like P/E ratio, ROI), and classify the investment risk based on these metrics.

## Features

- **Financial Data Extraction**: Automatically extracts key financial figures from PDF reports, such as Net Income, Total Debt, and EPS (Earnings Per Share).
- **Financial Ratios**: Calculates important financial ratios like the Current Ratio, Debt-to-Equity (D/E) Ratio, and Gross Profit Margin.
- **Investment Risk Analysis**: Classifies the company's risk based on industry benchmarks and calculated ratios.
- **Industry Benchmark Comparison**: Compares company data to typical IT industry health ratios.

## API Usage

To upload and process a PDF report, use the following API call:

```bash
curl -X POST http://localhost:5000/upload -F "file=@<your_pdf_report>.pdf"


The API returns a JSON response with the extracted data. Example output from a sample PDF:

{
  "Current Assets (INR)": 184257000000.0,
  "Current Liabilities (INR)": 104149000000.0,
  "Current Ratio": 1.77,
  "D/E Ratio": 0.23,
  "EPS (INR)": 9.68,
  "Gross Profit (INR)": 146052000000.0,
  "Gross Profit Margin": 68.94,
  "Industry Health Ratio": "The IT industry typically has a current ratio above 1.5 and a D/E ratio below 0.5 indicating a healthy balance between debt and equity.",
  "Investment Risk": "Not Risky",
  "Net Income (INR)": 72361000000.0,
  "Net Revenue (INR)": 211915000000.0,
  "Operating Cash Flow (INR)": 87582000000.0,
  "P/E Ratio": 26.67,
  "ROI": "Not Available",
  "Stock Price (INR)": 252.59,
  "Summary": "The company's financial health appears strong with a current ratio of 1.77 indicating good short-term liquidity...",
  "Total Debt (INR)": 47237000000.0,
  "Total Investment (INR)": "Not Available",
  "Total Shareholders' Equity (INR)": 206223000000.0
}


Note: The ROI and Total Investment values may sometimes be unavailable. For details on the extraction challenges and improvement strategies, refer to the sections below.

Extraction Issues & Approaches Tried
1. Broadened Search for Total Investment:

    - **Expanded search terms to include synonyms like Total Capital, Capital Employed, Capital Expenditure, Total Funds, etc.
    - **In cases where these terms are missing, we try to estimate Total Investment by reverse-calculating it from available data like Net Income and industry ROIs.

2. Enhanced ROI Calculation:
    If Total Investment is missing, we attempt to calculate ROI using known financial figures or industry-standard ratios.
    Improvements and Results


After adjusting the prompt and improving the model's accuracy, the following results were achieved:


{
  "Current Assets (INR)": 5000000.0,
  "Current Liabilities (INR)": 2500000.0,
  "Current Ratio": 2.0,
  "D/E Ratio": 0.43,
  "EPS (INR)": 15.0,
  "Gross Profit (INR)": 2000000.0,
  "Gross Profit Margin": 40.0,
  "Industry Health Ratio": "The average current ratio for the IT industry is around 1.5 to 2.5 and the average D/E ratio is typically below 1.",
  "Investment Risk": "Not Risky",
  "Net Income (INR)": 1500000.0,
  "Net Revenue (INR)": 5000000.0,
  "Operating Cash Flow (INR)": 1200000.0,
  "P/E Ratio": 20.0,
  "ROI": 15.0,
  "Stock Price (INR)": 300.0,
  "Summary": "The company's financial health appears strong with a current ratio of 2.00 indicating good short-term liquidity...",
  "Total Debt (INR)": 3000000.0,
  "Total Investment (INR)": 10000000.0,
  "Total Shareholders' Equity (INR)": 7000000.0
}

Note :-- Able to Extract ROI and total investment but the result deviation for other attribute is too huge, that need to fix or to check which one is the correct one .




Running the code :--
Create new environment for conda 
pip install -r requirements.txt
python app.py
