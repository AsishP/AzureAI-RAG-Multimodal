import os
import json
import requests

# Define the directory containing the JSON files
directory = os.path.join(os.getcwd(), "search-helpers")

# Create a dictionary to store JSON file names and their paths
json_files = {}

# Iterate through all files in the directory
for filename in os.listdir(directory):
    if filename.endswith(".json"):
        file_name_without_extension = os.path.splitext(filename)[0]
        file_path = os.path.join(directory, filename)
        json_files[file_name_without_extension] = file_path

# Print the dictionary
print("JSON files and their paths:")
print(json_files)

def create_data_source_from_json(search_url, api_key, json_file_path):
    """
    Creates a data source in Azure AI Search service using a JSON file as input.

    :param search_url: The URL of the Azure AI Search service.
    :param api_key: The API key for the Azure AI Search service.
    :param json_file_path: The path to the JSON file containing the data source configuration.
    :return: The response from the Azure AI Search service.
    """
    headers = {
        "Content-Type": "application/json",
        "api-key": api_key
    }

    # Read the JSON file
    try:
        with open(json_file_path, 'r', encoding='utf-8') as file:
            payload = json.load(file)
    except FileNotFoundError:
        print(f"Error: File '{json_file_path}' not found.")
        return None
    except json.JSONDecodeError as e:
        print(f"Error decoding JSON from file '{json_file_path}': {e}")
        return None

    data_source_name = payload.get("name")
    if not data_source_name:
        print("Error: JSON file must contain a 'name' field for the data source.")
        return None

    response = requests.put(
        url=f"{search_url}/datasources/{data_source_name}?api-version=2021-04-30-Preview",
        headers=headers,
        json=payload
    )

    if response.status_code == 201:
        print(f"Data source '{data_source_name}' created successfully.")
    else:
        print(f"Failed to create data source '{data_source_name}'. Status code: {response.status_code}, Response: {response.text}")

    return response