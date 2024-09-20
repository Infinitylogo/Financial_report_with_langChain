
pip install -r requirements.txt
sh run.sh


#for docker:--

docker run -p 5000:5000 financial-analysis-app


for ollama :--

using mistral here :--
pip install ollama 
ollama pull llama2

ollama create financial_analyser -f ./fina_analyser


Results:--- 

curl -X POST http://localhost:5000/upload -F "file=@2023_Annual_Report.pdf"
{
  "P/E Ratio": 37.731404958677686,
  "ROI": 65.0365803239201,
  "Risk": "Risky"}


curl -X POST http://localhost:5000/upload -F "file=@apple_statements.pdf"
{
  "P/E Ratio": 19.166210045662098,
  "ROI": 9.593962332467738,
  "Risk": "Moderate Risk"}


data download links:--
https://www.apple.com/newsroom/pdfs/fy2024-q1/FY24_Q1_Consolidated_Financial_Statements.pdf





# results i am getting here is :---

curl -X POST http://localhost:5000/upload -F "file=@apple_statements.pdf"
{
  "Current Assets (INR)": 143692000000.0,
  "Current Liabilities (INR)": 133973000000.0,
  "Current Ratio": 1.07,
  "D/E Ratio": 3.77,
  "EPS (INR)": 182.77,
  "Gross Profit (INR)": 54855000000.0,
  "Gross Profit Margin": 45.9,
  "Industry Health Ratio": "The IT industry typically has a current ratio of around 1.5 a gross profit margin of 40-60% and a D/E ratio of less than 1.5.",
  "Investment Risk": "Risky",
  "Net Income (INR)": 2818368000.0,
  "Net Revenue (INR)": 119575000000.0,
  "Operating Cash Flow (INR)": 39895000000.0,
  "P/E Ratio": 1.0,
  "ROI": "Not Available",
  "Stock Price (INR)": 182.77,
  "Summary": "The company's financial health shows a current ratio of 1.07 indicating it has slightly more current assets than current liabilities which is a positive sign but below the industry average. The gross profit margin of 45.9% is healthy and aligns well with industry standards. However the D/E ratio of 3.77 suggests a high level of debt compared to equity indicating potential risk. The P/E ratio of 1.00 is unusually low which may suggest undervaluation or concerns about future earnings. Overall while the company shows strong gross profitability the high debt levels and low P/E ratio indicate investment risks that potential investors should consider.",
  "Total Debt (INR)": 279414000000.0,
  "Total Investment (INR)": "Not Available",
  "Total Shareholders' Equity (INR)": 74100000000.0}



curl -X POST http://localhost:5000/upload -F "file=@2023_Annual_Report.pdf"
{
  "Current Assets (INR)": 184257000000.0,
  "Current Liabilities (INR)": 104149000000.0,
  "Current Ratio": 1.77,
  "D/E Ratio": 0.23,
  "EPS (INR)": 9.68,
  "Gross Profit (INR)": 146052000000.0,
  "Gross Profit Margin": 69.0,
  "Industry Health Ratio": "The IT industry typically has a current ratio above 1.5 and a D/E ratio below 0.5 indicating a healthy financial position.",
  "Investment Risk": "Not Risky",
  "Net Income (INR)": 72361000000.0,
  "Net Revenue (INR)": 211915000000.0,
  "Operating Cash Flow (INR)": 87582000000.0,
  "P/E Ratio": 26.0,
  "ROI": "Not Available",
  "Stock Price (INR)": 252.59,
  "Summary": "The company's financial health appears strong with a current ratio of 1.77 indicating good short-term liquidity and a low D/E ratio of 0.23 suggesting low financial leverage. The gross profit margin of 69% is robust reflecting efficient cost management and strong pricing power. The P/E ratio of 26 indicates that the stock may be fairly valued compared to its earnings. Overall the company's ratios align well with typical IT industry health benchmarks suggesting it is not a risky investment.",
  "Total Debt (INR)": 47237000000.0,
  "Total Investment (INR)": "Not Available",
  "Total Shareholders' Equity (INR)": 206223000000.0}
