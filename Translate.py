#3. Edited to accept any length of text.


#### TO translate more than 4685 tokens ####
import os
import PyPDF2
import docx
import textract
# from googletrans import Translator

from defaultValues import TRANSLATE_DIRECTORY, SOURCE_DIRECTORY, DE_TRANSLATOR_DIRECTORY

from TranslatedutToEnglish import my_DE_translator

# Function to split text into smaller chunks
def split_text(text, max_chunk_length):
    chunks = []
    current_chunk = ""
    words = text.split()
    for word in words:
        if len(current_chunk) + len(word) + 1 <= max_chunk_length:
            current_chunk += " " + word if current_chunk else word
        else:
            chunks.append(current_chunk)
            current_chunk = word
    if current_chunk:
        chunks.append(current_chunk)
    return chunks

# Function to translate text and handle text chunks
def translate_text(text, src_language, max_chunk_length=500):
    # translator = Translator()
    chunks = split_text(text, max_chunk_length)
    translated_chunks = []
    for chunk in chunks:
        translated_chunk = my_DE_translator(chunk)
        translated_chunks.append(translated_chunk)
    return "\n".join(translated_chunks)

def is_text_file(file_path):
    text_extensions = ['.txt', '.doc', '.docx', '.pdf']
    file_ext = os.path.splitext(file_path)[1].lower()
    return file_ext in text_extensions

def extract_text_from_pdf(pdf_path):
    with open(pdf_path, 'rb') as pdf_file:
        pdf_reader = PyPDF2.PdfReader(pdf_file)
        text = ""
        for page_num in range(len(pdf_reader.pages)):
            page = pdf_reader.pages[page_num]
            text += page.extract_text()
    return text

def extract_text_from_docx(docx_path):
    doc = docx.Document(docx_path)
    paragraphs = [para.text for para in doc.paragraphs]
    return '\n'.join(paragraphs)

def translate_to_english(text, src_language):
    # translator = Translator()
    translated_text = my_DE_translator(text)
    return translated_text

def extract_text_and_translate(file_path, src_language='auto', output_folder= SOURCE_DIRECTORY):
    if not is_text_file(file_path):
        return

    if file_path.lower().endswith('.txt'):
        with open(file_path, 'r', encoding='utf-8') as txt_file:
            text = txt_file.read()
    elif file_path.lower().endswith(('.doc', '.docx')):
        text = extract_text_from_docx(file_path)
    elif file_path.lower().endswith('.pdf'):
        text = extract_text_from_pdf(file_path)
    else:
        text = textract.process(file_path, encoding='utf-8')

    translated_text = translate_text(text, src_language)

    # Create an output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)

    # Create a .txt file with the translated text
    file_name = os.path.splitext(os.path.basename(file_path))[0]
    output_file_path = os.path.join(output_folder, f"{file_name}_translated.txt")
    with open(output_file_path, 'w', encoding='utf-8') as txt_file:
        txt_file.write(translated_text)

def translate_files_in_folder(input_folder, src_language='auto'):
    if not os.path.exists(input_folder):
        print(f"Folder '{input_folder}' not found.")
        return

    for root, dirs, files in os.walk(input_folder):
        for file_name in files:
            file_path = os.path.join(root, file_name)
            extract_text_and_translate(file_path, src_language)


if __name__ == "__main__":
    ROOT_DIRECTORY = os.path.dirname(os.path.realpath(__file__))
    input_folder = TRANSLATE_DIRECTORY
    translate_files_in_folder(input_folder)
    print("\n The translation process is completed\n")
