import os
from quart import Quart, request, jsonify
from werkzeug.utils import secure_filename
from utils.pdf_parser import extract_text_from_pdf
from utils.financial_logic import process_financial_report
import asyncio

app = Quart(__name__)

# Configuration for uploads
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'pdf'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Utility function to check allowed files
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Route for home page (optional, for testing)
@app.route('/')
async def index():
    return '''
    <!doctype html>
    <title>Upload PDF</title>
    <h1>Upload PDF for Financial Analysis</h1>
    <form method=post enctype=multipart/form-data>
      <input type=file name=file>
      <input type=submit value=Upload>
    </form>
    '''

# API route to upload PDF and process the file
@app.route('/upload', methods=['POST'])
async def upload_file():
    # Check if the file part is in the request
    if 'file' not in (await request.files):
        return jsonify({"error": "No file part in the request"}), 400

    file = (await request.files)['file']
    
    # Check if a file was selected
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    # Check if the file is allowed (PDF)
    if not allowed_file(file.filename):
        return jsonify({"error": "Invalid file type, only PDFs are allowed"}), 400

    try:
        # Secure the filename and save the file asynchronously
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        await file.save(file_path)
    except Exception as e:
        return jsonify({"error": f"Failed to save file: {str(e)}"}), 500

    # Process the PDF file for financial analysis asynchronously
    try:
        response = await process_financial_report_async(file_path)
        
        # Check if processing returned any errors
        if "error" in response:
            return jsonify({"error": response["error"]}), 400
        
        # If processing is successful
        return jsonify(response), 200
    except Exception as e:
        return jsonify({"error": f"An error occurred during processing: {str(e)}"}), 500

async def process_financial_report_async(pdf_path):
    # Run the synchronous function in a separate thread to avoid blocking
    return await asyncio.to_thread(process_financial_report, pdf_path)

# Create upload folder if it doesn't exist
if __name__ == '__main__':
    os.makedirs(UPLOAD_FOLDER, exist_ok=True)
    app.run(debug=True)
