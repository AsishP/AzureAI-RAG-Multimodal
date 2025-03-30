import json
import sys
import io
import os
import fitz  # PyMuPDF
from azure.storage.blob import BlobServiceClient
from azure.identity import DefaultAzureCredential
from azure.storage.blob import ContainerClient
from dotenv import load_dotenv
load_dotenv()

BLOB_SERVICE_SAS_URL = os.getenv('BLOB_SERVICE_SAS_URL')
BLOB_SAS_URI = os.getenv('BLOB_SAS_URI')
BLOB_RAWDATA_DIR = os.getenv('BLOB_RAWDATA_DIR')
BLOB_CONTAINER_NAME = os.getenv('BLOB_CONTAINER_NAME')
BLOB_TEXT_DIR_FILE_PREFIX = os.getenv('BLOB_TEXT_DIR_FILE_PREFIX')
BLOB_IMAGES_DIR_FILE_PREFIX = os.getenv('BLOB_IMAGES_DIR_FILE_PREFIX')
PAGE_TEXT_CHUNK_WORD_SIZE = int(os.getenv('PAGE_TEXT_CHUNK_WORD_SIZE'))
MIN_IMAGE_SIZE = int(os.getenv('MIN_IMAGE_SIZE'))
MIN_VECTOR_GRAPHIC_SIZE = int(os.getenv('MIN_VECTOR_GRAPHIC_SIZE'))
EXTRACT_VECTOR_GRAPHICS = bool(os.getenv('EXTRACT_VECTOR_GRAPHICS'))

# Validate environment variables
required_vars = {
    'BLOB_SAS_URI': BLOB_SAS_URI,
    'BLOB_RAWDATA_DIR': BLOB_RAWDATA_DIR,
    'BLOB_CONTAINER_NAME': BLOB_CONTAINER_NAME,
    'BLOB_TEXT_DIR_FILE_PREFIX': BLOB_TEXT_DIR_FILE_PREFIX,
    'BLOB_IMAGES_DIR_FILE_PREFIX': BLOB_IMAGES_DIR_FILE_PREFIX,
    'PAGE_TEXT_CHUNK_WORD_SIZE': PAGE_TEXT_CHUNK_WORD_SIZE,
    'MIN_IMAGE_SIZE': MIN_IMAGE_SIZE,
    'MIN_VECTOR_GRAPHIC_SIZE': MIN_VECTOR_GRAPHIC_SIZE,
    'EXTRACT_VECTOR_GRAPHICS': EXTRACT_VECTOR_GRAPHICS
}

for var_name, var_value in required_vars.items():
    if var_value is None:
        print(f"Error: Environment variable {var_name} is not set.")
        sys.exit(1)

PAGE_TEXT_CHUNK_WORD_SIZE = int(PAGE_TEXT_CHUNK_WORD_SIZE)  # Convert to int after validation

config = {
    'container_name': BLOB_CONTAINER_NAME,
    'text_dir_prefix': BLOB_TEXT_DIR_FILE_PREFIX,
    'image_dir_prefix': BLOB_IMAGES_DIR_FILE_PREFIX,
    'chunk_size': PAGE_TEXT_CHUNK_WORD_SIZE,
    'min_img_size': MIN_IMAGE_SIZE,
    'min_vector_graphic_size': MIN_VECTOR_GRAPHIC_SIZE,
    'extract_vector_graphics': EXTRACT_VECTOR_GRAPHICS
}

class ImgExtractor:
    
    @staticmethod
    def run_extractor(url, config):
        blob, container, filepath, filename = ImgExtractor.parse_blob_url(url)
        filename_without_extention = filename.split('.')[0].replace(' ', '-')
        client = ImgExtractor.get_client(BLOB_SERVICE_SAS_URL)
        stream = ImgExtractor.download_blob_to_stream(client, container, filepath)
        ImgExtractor.extract_and_upload(client, blob, filename_without_extention, stream, config)

    @staticmethod
    def parse_blob_url(url):
        # Not the best way of doing it, but it works.
        blob = f"{url.split('.net')[0]}.net"
        container = url.split('.net/')[1].split('/')[0]
        filepath = '/'.join(url.split('.net/')[1].split('/')[1:])
        filename = url.split('/')[-1]
        
        return blob, container, filepath, filename

    @staticmethod
    def download_blob_to_stream(blob_service_client: BlobServiceClient, container_name, filpath):
        print('>>> downloading file from blob')
        blob_client = blob_service_client.get_blob_client(container=container_name, blob=filpath)
        stream = io.BytesIO()
        num_bytes = blob_client.download_blob().readinto(stream)
        print(f"Number of bytes: {num_bytes}")
        print('<<< downloading file from blob')
        return stream

    @staticmethod
    def upload_blob_file(blob_service_client: BlobServiceClient, container_name: str, file_name, data, overwrite=True):
        print('>>> uploading file from blob')
        blob_client = blob_service_client.get_blob_client(container=container_name, blob=file_name)
        # data = b"Sample data for blob"

        # Upload the blob data - default blob type is BlockBlob
        blob_client.upload_blob(data, blob_type="BlockBlob", overwrite=overwrite)
        print('<<< uploading file from blob')

    @staticmethod
    def get_client(account_url):
        # credential = DefaultAzureCredential()
        # service = BlobServiceClient(account_url=account_url, credential=credential)

        service = BlobServiceClient(account_url=account_url) 

        return service
    
    @staticmethod
    def chunk_page(strl, length):
        return (' '.join(strl[i:length + i]) for i in range(0, len(strl), length))

    @staticmethod
    def extract_and_upload(service, blob, filename_without_extention, pdf_stream, config):        
        dest_container_name = config['container_name']
        chunk_size = config['chunk_size']
        text_dir_prefix = config['text_dir_prefix']
        image_dir_prefix = config['image_dir_prefix']
        min_img_size = config['min_img_size']
        min_vector_graphic_size = config['min_vector_graphic_size']
        extract_vector_graphics = config['extract_vector_graphics']
        
        print('>>> extracting images from file')
        # Open the PDF file
        pdf = fitz.open(stream=pdf_stream)
        data_list = []
        
        # Iterate over PDF pages
        for page_num in range(len(pdf)):
            page = pdf[page_num]
            
            image_url = []
            
            def get_vector_graphics():
                # extract vector graphic objects
                bboxes = page.cluster_drawings()

                # Iterate through each bounding box
                for i, bbox in enumerate(bboxes):
                    # Get the pixmap for the bounding box
                    pix = page.get_pixmap(clip=bbox)
                    
                    # Save the pixmap as an image
                    filename = f"sample-{page_num + 1}-{i + 1}.png"
                    pix.save(filename)
                    pix
                    print(f"Saved {filename}")
            
            # Extract images
            for img_num, img in enumerate(page.get_images(full=True)):
                xref = img[0]
                base_image = pdf.extract_image(xref)
                image_bytes = base_image["image"]
                
                file_name = f"image_{page_num + 1}_{img_num + 1 }.png"
                full_file_path = f'{image_dir_prefix}/{filename_without_extention}/{file_name}'
                
                # Filter out very small images that are likely an empty image or a small icon.
                if sys.getsizeof(image_bytes) > min_img_size:
                    image_url.append(f'{blob}/{dest_container_name}/{full_file_path}')
                    print(f'>>> uploading image: {file_name}')
                    ImgExtractor.upload_blob_file(service, dest_container_name, full_file_path, image_bytes)
                    
            if extract_vector_graphics == 'true':
                drawings = page.cluster_drawings()

                for vg_num, drawing in enumerate(drawings):
                    file_name = f"image_vg_{page_num + 1}_{vg_num + 1 }.png"
                    full_file_path = f'{image_dir_prefix}/{filename_without_extention}/{file_name}'
                    pix = page.get_pixmap(clip=drawing)
                    image_bytes = pix.tobytes()

                    # Filter out very small vetor draving that are likely an empty image or a small icon.
                    if sys.getsizeof(image_bytes) > min_vector_graphic_size:
                        image_url.append(f'{blob}/{dest_container_name}/{full_file_path}')
                        print(f'>>> uploading image: {file_name}')
                        ImgExtractor.upload_blob_file(service, dest_container_name, full_file_path, image_bytes)

            page_text = page.get_text()
            word_list = page_text.split(' ')
            for chunk in ImgExtractor.chunk_page(word_list, chunk_size):
                data_list.append({'content': {'chunk': chunk, 'imgurl': image_url}})
        
        print(f'>>> uploading text: {file_name}')
        full_file_path = f'{text_dir_prefix}/{filename_without_extention}.json'
        
        # Upload content and metadata
        ImgExtractor.upload_blob_file(service, dest_container_name, full_file_path, json.dumps(data_list))

        # Close the PDF after extraction
        pdf.close()
        print('<<< extracting images from file')


def list_blobs_with_sas(sas_url, folder_name):

    container_client = ContainerClient.from_container_url(sas_url)
    blob_uris = []

    print(f'>>> Listing blobs in the folder: {folder_name}')
    for blob in container_client.list_blobs(name_starts_with=folder_name):
        blob_uri = f"{sas_url.split('?')[0]}/{blob.name}"
        blob_uris.append(blob_uri)
        print(blob_uri)

    print(f'<<< Listing blobs in the folder: {folder_name}')
    return blob_uris

## Start the process
print('Python HttpRequest trigger function processing the event.')
blobUris = list_blobs_with_sas(BLOB_SAS_URI, BLOB_RAWDATA_DIR)
for blobUri in blobUris:
    print(f'>>> Extracted {blobUri}')
    ImgExtractor.run_extractor(blobUri, config)
    