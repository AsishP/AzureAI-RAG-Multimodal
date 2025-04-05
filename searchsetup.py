import os
import json
import requests
from dotenv import load_dotenv
import sys
load_dotenv()

SUBSCRIPTION_ID_HERE = os.getenv('SUBSCRIPTION_ID_HERE')
RESOURCE_GROUP_NAME_HERE = os.getenv('RESOURCE_GROUP_NAME_HERE')
STORAGE_ACCOUNT_NAME_HERE = os.getenv('STORAGE_ACCOUNT_NAME_HERE')
CONTAINER_NAME_HERE = os.getenv('CONTAINER_NAME_HERE')
TEXTDATA_QUERY_HERE = os.getenv('TEXTDATA_QUERY_HERE')

AOAI_URI_HERE = os.getenv('OPENAI_EMBEDDINGBASE_URL')
AOAI_API_KEY_HERE = os.getenv('OPENAI_EMBEDDINGAPI_KEY')
AOAI_DEPLOYMENT_ID_HERE = os.getenv('OPENAI_EMBEDDINGDEPLOYMENT')
AOAI_MODEL_NAME_HERE = os.getenv('OPENAI_EMBEDDINGMODEL')

VECTOR_INDEX_NAME_HERE = os.getenv('VECTOR_INDEX_NAME_HERE')
AZURE_SEARCH_ENDPOINT = os.getenv('AZURE_SEARCH_ENDPOINT_2')
AZURE_SEARCHKEY = os.getenv('AZURE_SEARCHKEY_2')
AZURE_SEARCH_API_VERSION = os.getenv('AZURE_SEARCH_API_VERSION')


required_vars = {
    'SUBSCRIPTION_ID_HERE': SUBSCRIPTION_ID_HERE,
    'RESOURCE_GROUP_NAME_HERE': RESOURCE_GROUP_NAME_HERE,
    'STORAGE_ACCOUNT_NAME_HERE': STORAGE_ACCOUNT_NAME_HERE,
    'VECTOR_INDEX_NAME_HERE': VECTOR_INDEX_NAME_HERE,
    'AZURE_SEARCH_ENDPOINT': AZURE_SEARCH_ENDPOINT,
    'AZURE_SEARCHKEY': AZURE_SEARCHKEY,
    'CONTAINER_NAME_HERE': CONTAINER_NAME_HERE,
    'AOAI_URI_HERE': AOAI_URI_HERE,
    'AOAI_API_KEY_HERE': AOAI_API_KEY_HERE,
    'AOAI_DEPLOYMENT_ID_HERE': AOAI_DEPLOYMENT_ID_HERE,
    'AOAI_MODEL_NAME_HERE': AOAI_MODEL_NAME_HERE,
    'AZURE_SEARCH_API_VERSION': AZURE_SEARCH_API_VERSION,
    'TEXTDATA_QUERY_HERE': TEXTDATA_QUERY_HERE
}

for var_name, var_value in required_vars.items():
    if var_value is None:
        print(f"Error: Environment variable {var_name} is not set.")
        sys.exit(1)

def replace_placeholder_in_json(data, placeholder_name, placeholder_value):
    # Replace the placeholder
    for key, value in data.items():
        if isinstance(value, str) and placeholder_name in value:
            data[key] = value.replace(placeholder_name, placeholder_value)
        elif isinstance(value, dict):
            replace_placeholder_in_json(value, placeholder_name, placeholder_value)
        elif isinstance(value, list):
            for item in value:
                if isinstance(item, dict):
                    replace_placeholder_in_json(item, placeholder_name, placeholder_value)
    return data
   
def update_search_from_json(search_control_plane,json_file_path):
    headers = {
        "Content-Type": "application/json",
        "api-key": AZURE_SEARCHKEY
    }

    # Read the JSON file
    try:
        with open(json_file_path, 'r', encoding='utf-8') as file:
            payload = json.load(file)

            if search_control_plane == "datasource":
                payload = replace_placeholder_in_json(payload, "SUBSCRIPTION_ID_HERE", SUBSCRIPTION_ID_HERE)  # Replace with your subscription ID
                payload = replace_placeholder_in_json(payload, "RESOURCE_GROUP_NAME_HERE", RESOURCE_GROUP_NAME_HERE)  # Replace with your resource group name
                payload = replace_placeholder_in_json(payload, "STORAGE_ACCOUNT_NAME_HERE", STORAGE_ACCOUNT_NAME_HERE)  # Replace with your storage account name
                payload = replace_placeholder_in_json(payload, "CONTAINER_NAME_HERE", CONTAINER_NAME_HERE)  # Replace with your container name
                payload = replace_placeholder_in_json(payload, "TEXTDATA_QUERY_HERE", TEXTDATA_QUERY_HERE)  # Replace with your text data query
                payload = replace_placeholder_in_json(payload, "INDEX_NAME_HERE", VECTOR_INDEX_NAME_HERE)  # Replace with your index name

                data_source_name = f"{VECTOR_INDEX_NAME_HERE}-datasource"

                response = requests.put(
                    url=f"{AZURE_SEARCH_ENDPOINT}/datasources/{data_source_name}?api-version={AZURE_SEARCH_API_VERSION}",
                    headers=headers,
                    json=payload
                )

                if response.status_code == 201:
                    print(f"Data source '{data_source_name}' created successfully.")
                    return True
                else:
                    print(f"Failed to create data source '{data_source_name}'. Status code: {response.status_code}, Response: {response.text}")
                return False

            elif search_control_plane == "skillset":
                payload = replace_placeholder_in_json(payload, "AOAI_URI_HERE", AOAI_URI_HERE)  # Replace with your AOAI URI
                payload = replace_placeholder_in_json(payload, "AOAI_API_KEY_HERE", AOAI_API_KEY_HERE)  # Replace with your AOAI API key
                payload = replace_placeholder_in_json(payload, "AOAI_DEPLOYMENT_ID_HERE", AOAI_DEPLOYMENT_ID_HERE)  # Replace with your AOAI deployment ID
                payload = replace_placeholder_in_json(payload, "AOAI_MODEL_NAME_HERE", AOAI_MODEL_NAME_HERE)  # Replace with your AOAI model name
                payload = replace_placeholder_in_json(payload, "INDEX_NAME_HERE", VECTOR_INDEX_NAME_HERE)  # Replace with your index name

                skillset_name = f"{VECTOR_INDEX_NAME_HERE}-skillset"

                response = requests.put(
                    url=f"{AZURE_SEARCH_ENDPOINT}/skillsets/{skillset_name}?api-version={AZURE_SEARCH_API_VERSION}",
                    headers=headers,
                    json=payload
                )

                if response.status_code == 201:
                    print(f"Skillset '{skillset_name}' created successfully.")
                    return True
                else:
                    print(f"Failed to create skillset '{skillset_name}'. Status code: {response.status_code}, Response: {response.text}")
                return False

            elif search_control_plane == "index":
                payload = replace_placeholder_in_json(payload, "AOAI_URI_HERE", AOAI_URI_HERE)  # Replace with your AOAI URI
                payload = replace_placeholder_in_json(payload, "AOAI_API_KEY_HERE", AOAI_API_KEY_HERE)  # Replace with your AOAI API key
                payload = replace_placeholder_in_json(payload, "AOAI_DEPLOYMENT_ID_HERE", AOAI_DEPLOYMENT_ID_HERE)  # Replace with your AOAI deployment ID
                payload = replace_placeholder_in_json(payload, "AOAI_MODEL_NAME_HERE", AOAI_MODEL_NAME_HERE)  # Replace with your AOAI model name
                payload = replace_placeholder_in_json(payload, "INDEX_NAME_HERE", VECTOR_INDEX_NAME_HERE)  # Replace with your index name 

                index_name = VECTOR_INDEX_NAME_HERE

                response = requests.put(
                    url=f"{AZURE_SEARCH_ENDPOINT}/indexes/{index_name}?api-version={AZURE_SEARCH_API_VERSION}",
                    headers=headers,
                    json=payload
                )

                if response.status_code == 201:
                    print(f"Index '{index_name}' created successfully.")
                    return True
                else:
                    print(f"Failed to create index '{index_name}'. Status code: {response.status_code}, Response: {response.text}")
                return False
            elif search_control_plane == "indexer":
                payload = replace_placeholder_in_json(payload, "INDEX_NAME_HERE", VECTOR_INDEX_NAME_HERE)  # Replace with your index name 
                indexer_name = f"{VECTOR_INDEX_NAME_HERE}-indexer"

                response = requests.put(
                    url=f"{AZURE_SEARCH_ENDPOINT}/indexers/{indexer_name}?api-version={AZURE_SEARCH_API_VERSION}",
                    headers=headers,
                    json=payload
                )

                if response.status_code == 201:
                    print(f"Indexer '{indexer_name}' created successfully.")
                    return True
                else:
                    print(f"Failed to create indexer '{indexer_name}'. Status code: {response.status_code}, Response: {response.text}")
                return False
    except FileNotFoundError:
        print(f"Error: File '{json_file_path}' not found.")
        return False
    except json.JSONDecodeError as e:
        print(f"Error decoding JSON from file '{json_file_path}': {e}")
        return False

def run_indexer(indexer_name):
    headers = {
        "Content-Type": "application/json",
        "api-key": AZURE_SEARCHKEY
    }

    response = requests.post(
        url=f"{AZURE_SEARCH_ENDPOINT}/indexers/{indexer_name}/run?api-version={AZURE_SEARCH_API_VERSION}",
        headers=headers
    )

    if response.status_code == 202:
        print(f"Indexer '{indexer_name}' run successfully.")
    else:
        print(f"Failed to run indexer '{indexer_name}'. Status code: {response.status_code}, Response: {response.text}")


# Define the directory containing the JSON files
directory = os.path.join(os.getcwd(), "search-helpers")

# Create a dictionary to store JSON file names and their paths
json_files = {}
search_creation_success = True

# Iterate through all files in the directory
for filename in os.listdir(directory):
    if filename.endswith(".json"):
        file_name_without_extension = os.path.splitext(filename)[0]
        file_path = os.path.join(directory, filename)
        json_files[file_name_without_extension] = file_path

for control_plane in ["datasource", "index", "skillset", "indexer"]:
    if control_plane in json_files and search_creation_success:
        print(f"Processing {control_plane} JSON file: {json_files[control_plane]}")
        search_creation_success = update_search_from_json(control_plane, json_files[control_plane])

print("All JSON files processed.")

if search_creation_success:
    indexer_name = f"{VECTOR_INDEX_NAME_HERE}-indexer"
    run_indexer(indexer_name)
    print("Indexer run command sent.")
else:
    print("Failed to create search components. Please check the logs for details.")