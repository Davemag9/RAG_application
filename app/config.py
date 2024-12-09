DATA_FOLDER_ABSOLUTE_PATH = '' # <-- enter your path

PATH_PREPROCESSED_DOCS = f'{DATA_FOLDER_ABSOLUTE_PATH}/data/preprocessed_text.json'
PATH_METADATA = f'{DATA_FOLDER_ABSOLUTE_PATH}/data/metadata.json'
PATH_BASE = f'{DATA_FOLDER_ABSOLUTE_PATH}/data/faiss_base.faiss'


import os
if __name__ == '__main__':
    current_file_path = os.path.abspath(__file__)
    directory_path = os.path.dirname(current_file_path)
    directory_path = directory_path.replace("\\", "/")
    print(directory_path)