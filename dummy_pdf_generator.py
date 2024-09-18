from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas

def generate_sample_pdf(filename):
    c = canvas.Canvas(filename, pagesize=letter)
    width, height = letter
    
    companies = [
        {"name": "Alpha Corp", "eps": 5.5, "stock_price": 120, "net_income": 1200000, "investment": 5000000},
    #     {"name": "Beta Inc", "eps": 4.2, "stock_price": 95, "net_income": 850000, "investment": 4500000},
    #     # Add more companies
    #     {"name": "Zeta Ltd", "eps": 3.1, "stock_price": 80, "net_income": 450000, "investment": 2500000},
    ]
    
    y = height - 40
    for company in companies:
        c.drawString(100, y, f"Company: {company['name']}")
        y -= 20
        c.drawString(100, y, f"Earnings Per Share (EPS): {company['eps']}")
        y -= 20
        c.drawString(100, y, f"Stock Price: {company['stock_price']}")
        y -= 20
        c.drawString(100, y, f"Net Income: {company['net_income']}")
        y -= 20
        c.drawString(100, y, f"Total Investment: {company['investment']}")
        y -= 40
    
    c.save()

# Generate the sample PDF
generate_sample_pdf("financial_report_50_companies.pdf")