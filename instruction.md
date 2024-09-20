
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
