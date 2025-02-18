import gdown
import zipfile
import os

class DataDownloader:
    def __init__(self, file_id, output_zip='dataset.zip', output_dir='dataset'):
        self.file_id = file_id
        self.output_zip = output_zip
        self.output_dir = output_dir

    def download_and_extract(self):
        if not os.path.exists(self.output_zip):
            gdown.download(f"https://drive.google.com/uc?id={self.file_id}", output=self.output_zip, quiet=False)
        with zipfile.ZipFile(self.output_zip, 'r') as zip_ref:
            zip_ref.extractall(self.output_dir)

        print(f"Data extracted to {self.output_dir}")
