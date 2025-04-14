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
Create an Azure Open AI resource (recommendeded:Use Azure AI Foundry Project) and deploy the below models in the same region as Azure AI Search above.

1. text-embedding-3-small
2. gpt-4o-mini

## Step 4
Copy the .env-sample file and paste it in the same folder. Then rename the copied file to .env file
Fill the values in .env file with values from Azure Resources where not available.

## Step 5
Upload the PDF documents to raw_data folder.
Run the split-pdf.py file to split the PDF into JSON file and Images
Make sure the prepared_data folder has a Text folder with a JSON file and Images folder has images in it

## Step 6
Run the searchsetup.py to create the Data Sources, Skills, Indexer and Index in the Azure AI Search instance
Check the Indexer run has finised successfully and there are documents crawled in the Index

## Step 7
Run the Streamlit debugger to run app.py which will open the web page with the Search box.
In case of issue, please check the Terminal in VS code or your IDE for more details.



