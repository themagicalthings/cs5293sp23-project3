import os
from project3 import read_pdf


def test_read_pdf():
    # Define the folder containing the PDF files
    pdf_folder = "smartcity/"

    # Iterate through all PDF files in the folder
    for file_name in os.listdir(pdf_folder):
        if file_name.endswith(".pdf"):
            file_path = os.path.join(pdf_folder, file_name)

            # Read the PDF file
            text = read_pdf(file_path)

            # Print the text for debugging
            print(f"File name: {file_name}")
            print(f"Text length: {len(text)}")
            print(text[:100])

            # Assert that the text is not empty
            assert text.strip() != ""

