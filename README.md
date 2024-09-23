# Financial_report_with_langChain

The results I have been able to achieve for the PDF are as follows :--

curl -X POST http://localhost:5000/upload -F "file=@2023_Annual_Report.pdf"

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
  "Summary": "The company's financial health appears strong with a current ratio of 1.77 indicating good short-term liquidity and a low D/E ratio of 0.23 suggesting a conservative approach to leverage. The gross profit margin of 68.94% is robust reflecting efficient operations. The P/E ratio of 26.67 is reasonable for the IT sector indicating that the stock is fairly valued relative to its earnings. Overall the company's performance aligns well with typical IT industry health benchmarks suggesting it is not a risky investment.",
  "Total Debt (INR)": 47237000000.0,
  "Total Investment (INR)": "Not Available",
  "Total Shareholders' Equity (INR)": 206223000000.0

}

I am encountering issues with extracting and calculating the overall ROI and Total Investment for certain cases.

For the results, check the Results folder where you will find a screenshot of the outputs received.
