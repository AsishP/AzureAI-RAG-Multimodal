import json
import sys
import io
import os
import fitz  # PyMuPDF
from azure.storage.blob import BlobServiceClient
from azure.identity import DefaultAzureCredential
from azure.storage.blob import ContainerClient
from openai import AzureOpenAI
from dotenv import load_dotenv
import base64
from mimetypes import guess_type
load_dotenv()

# Load environment variables
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
AZURE_OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
AZURE_OPENAI_ENDPOINT = os.getenv("OPENAI_BASE_URL")
OPEN_AI_MODEL = os.getenv("OPENAI_DEPLOYMENT_NAME")
API_VERSION = os.getenv("API_VERSION")
BLOB_SAS_TOKEN = os.environ.get("BLOB_SAS_TOKEN")

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

## Main Class for all the Image Extraction and Uploading
class ImgExtractor:
    
    #Extract file details from the blob URL
    @staticmethod
    def run_extractor(url, config):
        blob, container, filepath, filename = ImgExtractor.parse_blob_url(url)
        filename_without_extention = filename.split('.')[0].replace(' ', '-')
        client = ImgExtractor.get_client(BLOB_SERVICE_SAS_URL)
        stream = ImgExtractor.download_blob_to_stream(client, container, filepath)
        ImgExtractor.extract_and_upload(client, blob, filename_without_extention, stream, config)

    # Parse the blob URL to extract the blob, container, filepath, and filename
    @staticmethod
    def parse_blob_url(url):
        # Not the best way of doing it, but it works.
        blob = f"{url.split('.net')[0]}.net"
        container = url.split('.net/')[1].split('/')[0]
        filepath = '/'.join(url.split('.net/')[1].split('/')[1:])
        filename = url.split('/')[-1]
        
        return blob, container, filepath, filename

    # Download the blob to a stream
    @staticmethod
    def download_blob_to_stream(blob_service_client: BlobServiceClient, container_name, filpath):
        print('>>> downloading file from blob')
        blob_client = blob_service_client.get_blob_client(container=container_name, blob=filpath)
        stream = io.BytesIO()
        num_bytes = blob_client.download_blob().readinto(stream)
        print(f"Number of bytes: {num_bytes}")
        print('<<< downloading file from blob')
        return stream
    
    # Upload the blob file to Azure Blob Storage. Use overwrite=True to overwrite the file if it already exists.
    @staticmethod
    def upload_blob_file(blob_service_client: BlobServiceClient, container_name: str, file_name, data, overwrite=True):
        blob_client = blob_service_client.get_blob_client(container=container_name, blob=file_name)
        # data = b"Sample data for blob"

        # Upload the blob data - default blob type is BlockBlob
        if not overwrite and blob_client.exists():
            print(f"Blob {file_name} already exists. Skipping upload.")
            return
        blob_client.upload_blob(data, blob_type="BlockBlob", overwrite=overwrite)
        print('>>> uploading file from blob')

    # Get the BlobServiceClient using the SAS URL
    @staticmethod
    def get_client(account_url):
        ## If you want to use the DefaultAzureCredential, uncomment the following lines:
        # credential = DefaultAzureCredential()
        # service = BlobServiceClient(account_url=account_url, credential=credential)

        # Using SAS URL for authentication
        service = BlobServiceClient(account_url=account_url) 

        return service
    
    # Retuns the blob URL with SAS token
    @staticmethod
    def generate_sas_token(file_name):
        return f"{file_name}?{BLOB_SAS_TOKEN}"

    # Reads the blob as base64 and returns the base64 string
    @staticmethod
    def read_blob_as_base64(blob_service_client:BlobServiceClient, blobfilename: str):
        
        print(f'>>> Reading blob {blobfilename} as base64')
        blob, container, filepath, filename = ImgExtractor.parse_blob_url(blobfilename)
        blob_client = blob_service_client.get_blob_client(container=BLOB_CONTAINER_NAME, blob=filepath)
        stream = io.BytesIO()
        blob_client.download_blob().readinto(stream)
        stream.seek(0)
        # Guess the MIME type of the image based on the file extension
        mime_type, _ = guess_type(blobfilename)
        if mime_type is None:
            mime_type = 'application/octet-stream'  # Default MIME type if none is found
        base64_string = base64.b64encode(stream.read()).decode('utf-8')
        stream.close()
        print(f'<<< Blob {blobfilename} converted to base64')
        return f"data:{mime_type};base64,{base64_string}"
    
    # Chunks the page text into smaller parts of specified length
    @staticmethod
    def chunk_page(strl, length):
        return (' '.join(strl[i:length + i]) for i in range(0, len(strl), length))
    
    ## Extract Image Caption with GPT models
    @staticmethod
    def understand_image_with_gptv(service, image_path, caption):
        client = AzureOpenAI(
            api_key=AZURE_OPENAI_API_KEY,  
            api_version=API_VERSION,
            base_url=AZURE_OPENAI_ENDPOINT
        )

        data_base64 = ImgExtractor.read_blob_as_base64(service, image_path)

        # We send both image caption and the image body to GPTv for better understanding
        if caption != "" and caption != 'None':
            response = client.chat.completions.create(
                    model=OPEN_AI_MODEL,
                    messages=[
                        { "role": "system", "content": "You are a helpful assistant." },
                        { "role": "user", "content": [  
                            { 
                                "type": "text", 
                                "text": f"Describe this image (note: it has image caption: {caption}):" 
                            },
                            { 
                                "type": "image_url",
                                "image_url": {
                                    "url": f"{data_base64}"
                                }
                            }
                        ] } 
                    ],
                    max_tokens=800
                )

        else:
            response = client.chat.completions.create(
                model=OPEN_AI_MODEL,
                messages=[
                    { "role": "system", "content": "You are a helpful assistant." },
                    { "role": "user", "content": [  
                        { 
                            "type": "text", 
                            "text": "Describe this image:" 
                        },
                        { 
                            "type": "image_url",
                            "image_url": {
                                "url": f"{data_base64}"
                            }
                        }
                    ] } 
                ],
                max_tokens=800
            )

        img_description = response.choices[0].message.content
        
        return img_description
    
    # Extracts vector graphics from the PDF page and saves them as images
    @staticmethod
    def get_vector_graphics(page, page_num):
        # Extract vector graphic objects
        bboxes = page.cluster_drawings()

        # Iterate through each bounding box
        for i, bbox in enumerate(bboxes):
            # Get the pixmap for the bounding box
            pix = page.get_pixmap(clip=bbox)
            
            # Save the pixmap as an image
            filename = f"sample-{page_num + 1}-{i + 1}.png"
            pix.save(filename)
            print(f"Saved {filename}")
            return filename


    # Checks if a given imgurl exists in a JSON file stored in Azure Blob Storage.
    #
    # Parameters:
    # - blob_service_client (BlobServiceClient): The BlobServiceClient instance.
    # - container_name (str): The name of the container.
    # - json_file_path (str): The path to the JSON file in the blob storage.
    # - imgurl (str): The image URL to check.
    #
    # Returns:
    # - bool: True if the imgurl exists in the JSON file, False otherwise.
    @staticmethod
    def check_imgurl_in_json(blob_service_client: BlobServiceClient, container_name: str, json_file_path: str, imgurl: str) -> bool:
        
        print(f'>>> Checking if imgurl exists in JSON file: {json_file_path}')
        blob_client = blob_service_client.get_blob_client(container=container_name, blob=json_file_path)
        
        try:
            # Download the JSON file content
            stream = io.BytesIO()
            blob_client.download_blob().readinto(stream)
            stream.seek(0)
            json_data = json.load(stream)
            stream.close()

            # Search for the imgurl in the JSON data
            for entry in json_data:
                if 'content' in entry and 'imagedata' in entry['content']:
                    for image_data in entry['content']['imagedata']:
                        if image_data.get('imgurl') == imgurl:
                            print(f'<<< imgurl found in JSON file: {imgurl}')
                            return True
        except Exception as e:
            print(f"Error reading JSON file: {e}")
            return False
        
        print(f'<<< imgurl not found in JSON file: {imgurl}')
        return False

    # 
    # Main Function
    # Extracts images and text from a PDF file, uploads them to Azure Blob Storage, and generates captions for the images using GPT.
    # The function takes the following parameters:
    # - service: The BlobServiceClient instance.
    # - blob: The base URL of the blob storage account.
    # - filename_without_extention: The name of the PDF file without its extension.
    # - pdf_stream: The stream of the PDF file.
    # - config: A dictionary containing configuration parameters for the extraction process.
    #
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

        full_json_file_path = f'{text_dir_prefix}/{filename_without_extention}.json'
        
        # Iterate over PDF pages
        for page_num in range(len(pdf)):
            page = pdf[page_num]
            
            image_url = []
            imagedata = []
            
            # Extract images
            for img_num, img in enumerate(page.get_images(full=True)):
                xref = img[0]
                base_image = pdf.extract_image(xref)
                image_bytes = base_image["image"]
                
                file_name = f"image_{page_num + 1}_{img_num + 1 }.png"
                full_image_file_path = f'{image_dir_prefix}/{filename_without_extention}/{file_name}'
                
                # Filter out very small images that are likely an empty image or a small icon.
                if sys.getsizeof(image_bytes) > min_img_size:
                    image_url.append(f'{blob}/{dest_container_name}/{full_image_file_path}')
                    if(ImgExtractor.check_imgurl_in_json(service, dest_container_name, full_json_file_path, image_url[-1]) == False):
                        print(f'>>> uploading image: {file_name}')
                        ImgExtractor.upload_blob_file(service, dest_container_name, full_image_file_path, image_bytes, False)
                        imagedata.append({
                            'imgurl': f'{blob}/{dest_container_name}/{full_image_file_path}',
                            'caption': ImgExtractor.understand_image_with_gptv(service, f'{blob}/{dest_container_name}/{full_image_file_path}', base_image["cs-name"])
                        })
                    else:
                        print(f'>>> image already exists: {file_name}. Skipping upload.')
                    
                if extract_vector_graphics == 'true':
                    drawings = page.cluster_drawings()

                    for vg_num, drawing in enumerate(drawings):
                        file_name = f"image_vg_{page_num + 1}_{vg_num + 1 }.png"
                        full_image_file_path = f'{image_dir_prefix}/{filename_without_extention}/{file_name}'
                        pix = page.get_pixmap(clip=drawing)
                        image_bytes = pix.tobytes()

                        # Filter out very small vetor draving that are likely an empty image or a small icon.
                        if sys.getsizeof(image_bytes) > min_vector_graphic_size:
                            image_url.append(f'{blob}/{dest_container_name}/{full_image_file_path}')
                            if(ImgExtractor.check_imgurl_in_json(service, dest_container_name, full_json_file_path, image_url[-1]) == False):
                                print(f'>>> uploading image: {file_name}')
                                ImgExtractor.upload_blob_file(service, dest_container_name, full_image_file_path, image_bytes, False)
                                imagedata.append({
                                    'imgurl': f'{blob}/{dest_container_name}/{full_image_file_path}',
                                    'caption': ImgExtractor.understand_image_with_gptv(service, f'{blob}/{dest_container_name}/{full_image_file_path}', base_image["cs-name"])
                                })
                            else:
                                print(f'>>> image already exists: {file_name}. Skipping upload.')
                
            page_text = page.get_text()
            word_list = page_text.split(' ')
            for chunk in ImgExtractor.chunk_page(word_list, chunk_size):
                data_list.append({'content': {'chunk': chunk, 'imgurl': image_url, 'imagedata': imagedata}})
            
            # Upload content and metadata
            ImgExtractor.upload_blob_file(service, dest_container_name, full_json_file_path, json.dumps(data_list))
        
        print(f'>>> uploading text: {file_name}')
        full_json_file_path = f'{text_dir_prefix}/{filename_without_extention}.json'
        
        # Upload content and metadata
        ImgExtractor.upload_blob_file(service, dest_container_name, full_json_file_path, json.dumps(data_list))

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
    