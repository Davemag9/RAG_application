
DATA_FOLDER_ABSOLUTE_PATH = '' # <-- enter your path

PATH_PREPROCESSED_DOCS = f'{DATA_FOLDER_ABSOLUTE_PATH}/preprocessed_text.json'
PATH_METADATA = f'{DATA_FOLDER_ABSOLUTE_PATH}/metadata.json'
PATH_BASE = f'{DATA_FOLDER_ABSOLUTE_PATH}/faiss_base.faiss'


import os
if __name__ == '__main__':
    current_file_path = os.path.abspath(__file__)
    print(current_file_path)
