# AzureAI-RAG-Multimodal
Azure AI Solutions with RAG with Multimodal data such as Images, Text and retrieving Multimodal Results

## Step 1
Create an Azure Storage Account and a Container to store the PDF content to be parsed.  
> How to: [Create an Azure Storage Account](https://learn.microsoft.com/en-us/azure/storage/common/storage-account-create?tabs=azure-portal)

Create two folders in it as below:
- raw_data
- prepared_data

Create a SAS URI to the blob container with the permissions shown below:
![alt text](Images/blobsasaccess.png)
> [Create SAS tokens for Azure Storage](https://learn.microsoft.com/en-us/azure/ai-services/translator/document-translation/how-to-guides/create-sas-tokens?tabs=Containers)

## Step 2
Create an Azure AI Search Service (if not done before, use the below reference quickstart)
> [Quickstart: Create an Azure Cognitive Search service in the Azure portal](https://learn.microsoft.com/en-us/azure/search/search-create-service-portal)

## Step 3
Copy the .env-sample file and rename to .env file
Fill the values in .env file with values from Azure



