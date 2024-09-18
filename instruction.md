
pip install -r requirements.txt
python app.py
curl -X POST -F "file=@financial_report.pdf" http://127.0.0.1:5000/upload


curl -X POST -F "file=@financial_report_50_companies.pdf" http://127.0.0.1:5000/upload


#for docker:--

docker run -p 5000:5000 financial-analysis-app


for ollama :--

using mistral here :--
pip install ollama 
ollama pull llama2

ollama create financial_analyser -f ./fina_analyser

