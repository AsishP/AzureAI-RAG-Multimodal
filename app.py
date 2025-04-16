import streamlit as st
import dotenv
import os
import json
import random, string
import time
import requests

from openai import AzureOpenAI
from azure.identity import DefaultAzureCredential
from azure.storage.blob import BlobServiceClient
from azure.search.documents.models import VectorizedQuery
from azure.search.documents import SearchClient
from azure.core.credentials import AzureKeyCredential

st.set_page_config(layout="wide")
dotenv.load_dotenv()

# Load environment variables
endpoint = os.environ.get("OPENAI_BASE_URL")
api_key = os.environ.get("OPENAI_API_KEY")
deployment = os.environ.get("OPENAI_DEPLOYMENT_NAME")
api_version = os.environ.get("OPENAI_API_VERSION")
deployment_embedding = os.environ.get("OPENAI_EMBEDDINGMODEL")

ai_vision_endpoint = os.environ.get("AI_VISION_ENDPOINT")
ai_vision_apiversion = os.environ.get("AI_VISION_API_VERSION")
ai_vision_api_key = os.environ.get("AI_VISION_API_KEY")

search_endpoint = os.environ.get("AZURE_SEARCH_ENDPOINT")
search_index = os.environ.get("AZURE_SEARCH_INDEX")
search_index_image = os.environ.get("AZURE_SEARCH_INDEX_IMAGES")
search_api_key = os.environ.get("AZURE_SEARCHKEY")
search_semantic_config = os.environ.get("SEARCH_SEMANTIC_CONFIGURATION")
search_query_type = "vector" if os.environ.get("SEARCH_QUERY_TYPE") == "vector" else "simple"

blobl_sas_token = os.environ.get("BLOB_SAS_TOKEN")
blob_folderPath = os.environ.get("BLOB_FOLDER_PATH")

###
# Streamlit configuration
###
def open_avatare_image(file_path):
    data = None
    with open(file_path, "rb") as f:
        data = f.read()
    return data

# Configuration settings on the Streamlit sidebar
with st.sidebar:
    on = st.toggle("Settings")
    if on:
        system_prompt = st.sidebar.text_area(
            "System Prompt", st.session_state['default_prompt'], height=100
        )
        seed_message = {"role": "system", "content": system_prompt}
        st.session_state["messages"] = [seed_message]

        c1, c2 = st.sidebar.columns(2)
        with c1:
            st.write('#### Azure Open AI Configuration')
            max_response_help = "Set a limit on the number of tokens per model response. The API supports a maximum of MaxTokensPlaceholderDoNotTranslate tokens shared between the prompt (including system message, examples, message history, and user query) and the model's response. One token is roughly 4 characters for typical English text."
            values = st.slider("Max response", 0, 4096, 
                               st.session_state["llm_config_params"]['max_response'],
                               help=max_response_help)
            st.session_state["llm_config_params"]['max_response'] = values
            # st.write("Values:", values)
            
            temperatur_help = "Controls randomness. Lowering the temperature means that the model will produce more repetitive and deterministic responses. Increasing the temperature will result in more unexpected or creative responses. Try adjusting temperature or Top P but not both."
            values = st.slider("Temperature", 0.0, 1.0,
                               st.session_state["llm_config_params"]['temperatur'], 
                               help=temperatur_help)
            st.session_state["llm_config_params"]['temperatur'] = values
            
            top_p_help = "Similar to temperature, this controls randomness but uses a different method. Lowering Top P will narrow the model’s token selection to likelier tokens. Increasing Top P will let the model choose from tokens with both high and low likelihood. Try adjusting temperature or Top P but not both."
            values = st.slider("Top P", 0.0, 1.0, 
                               st.session_state["llm_config_params"]['top_p'], 
                               help=top_p_help)
            st.session_state["llm_config_params"]['top_p'] = values
            
            stop_sequences_help = "Make the model end its response at a desired point. The model response will end before the specified sequence, so it won't contain the stop sequence text. For ChatGPT, using <|im_end|> ensures that the model response doesn't generate a follow-up user query. You can include as many as four stop sequences."
            values = st.text_input(
                label="Stop sequences", 
                value=st.session_state["llm_config_params"]['stop_sequences'], 
                placeholder='Stop sequences', 
                help=stop_sequences_help,
                disabled=True
            )
            st.session_state["llm_config_params"]['stop_sequences'] = values
            
            frequency_penalty_help = "Reduce the chance of repeating a token proportionally based on how often it has appeared in the text so far. This decreases the likelihood of repeating the exact same text in a response."
            values = st.slider("Frequency penalty", 0.0, 2.0, 
                               st.session_state["llm_config_params"]['frequency_penalty'],
                               help=top_p_help)
            st.session_state["llm_config_params"]['frequency_penalty'] = values
            
            presence_penalty_help = "Reduce the chance of repeating any token that has appeared in the text at all so far. This increases the likelihood of introducing new topics in a response."
            values = st.slider("Presence penalty", 0.0, 2.0, 
                               st.session_state["llm_config_params"]['presence_penalty'], 
                               help=presence_penalty_help)
            st.session_state["llm_config_params"]['presence_penalty'] = values
        with c2:
            st.write('#### AI Search Configuration')
            values = st.text_input(label="Filter", value=st.session_state["search_config_params"]['filter'])
            st.session_state["search_config_params"]['filter'] = values
            
            values = st.text_input(label="Strictness", value=st.session_state["search_config_params"]['strictness'])
            st.session_state["search_config_params"]['strictness'] = values
            
            values = st.text_input(label="Top N Documents", value=st.session_state["search_config_params"]['top_n_documents'])
            st.session_state["search_config_params"]['top_n_documents'] = values

        c1, c2 = st.sidebar.columns(2)
        with c1:
            clear_button = st.button("Clear Conversation", key="clear")
            if clear_button:
                st.session_state["generated"] = []
                st.session_state["past"] = []
                st.session_state["messages"] = [seed_message]
                st.session_state["number_tokens"] = []
                st.session_state["model_name"] = []
                st.session_state["cost"] = []
                st.session_state["total_cost"] = 0.0
                st.session_state["total_tokens"] = []
        with c2:
            download_conversation_button = st.download_button(
                "Download Conversation",
                data=json.dumps(st.session_state["generated"]),
                file_name=f"conversation.json",
                mime="text/json",
            )


# Initialise session state variables for Streamlit
if "generated" not in st.session_state:
    st.session_state["generated"] = []
if "past" not in st.session_state:
    st.session_state["past"] = []
if "messages" not in st.session_state:
    st.session_state["messages"] = []
if "model_name" not in st.session_state:
    st.session_state["model_name"] = []
if "llm_config_params" not in st.session_state:
    st.session_state["llm_config_params"] = {
        'max_response': 4000,
        'temperatur': 0.7,
        'top_p': 1.0,
        'stop_sequences': None,
        'frequency_penalty': 0.0,
        'presence_penalty': 0.0
    }
if "search_config_params" not in st.session_state:
    st.session_state["search_config_params"] = {
        'filter': None,
        'strictness': 1,
        'top_n_documents': 50
    }        
if "cost" not in st.session_state:
    st.session_state["cost"] = []
if "total_tokens" not in st.session_state:
    st.session_state["total_tokens"] = 0
if "prompt_tokens" not in st.session_state:
    st.session_state["prompt_tokens"] = 0 
if "completion_tokens" not in st.session_state:
    st.session_state["completion_tokens"] = 0
if "total_cost" not in st.session_state:
    st.session_state["total_cost"] = 0.0
if "references" not in st.session_state:
    st.session_state["references"] = []
if "refs_count" not in st.session_state:
    st.session_state["refs_count"] = 0
if "references_toggle" not in st.session_state:
    st.session_state["references_toggle"] = False
if 'avatar_user' not in st.session_state:
    st.session_state['avatar_user'] = open_avatare_image('images/avatar_user.png')
if 'avatar_ai' not in st.session_state:
    st.session_state['avatar_ai'] = open_avatare_image('images/avatar_ai.png')
if 'default_prompt' not in st.session_state:
    st.session_state['default_prompt'] = 'You are an AI assistant that helps people find information.'
if 'aoai_client' not in st.session_state:
    print(endpoint)
    print(api_key)
    print(api_version)
    st.session_state['aoai_client'] = AzureOpenAI(
        azure_endpoint=endpoint,
        # azure_ad_token_provider=token_provider,
        api_key=api_key,
        api_version=api_version,
    )

def randomword(length):
    letters = string.ascii_lowercase
    return ''.join(random.choice(letters) for i in range(length))

# Get the Azure Blob Storage client
def get_client(account_url):
    credential = DefaultAzureCredential()
    service = BlobServiceClient(account_url=account_url, credential=credential)

    return service

# Get SaS token url for the blob
def generate_sas_token(file_name):
    #return f"{file_name}"
    return f"{file_name}?{blobl_sas_token}"


def get_chat_msg_count():
    return len(st.session_state["references"])

####
############### RAG Search with Images in Citations ##########
####

# Traverse the completion response and extract the citations and information of the documents
def generate_response_data(completion):
    content = completion["choices"][0]["message"]["content"]
    context = completion["choices"][0]["message"]["context"]
    total_tokens = completion["usage"]["total_tokens"]
    prompt_tokens = completion["usage"]["prompt_tokens"]
    completion_tokens = completion["usage"]["completion_tokens"]

    citations = {'all': [], 'selected': []}
    for idx, citation in enumerate(context["citations"]):
        citation_reference = f"[doc{idx + 1}]"
        imgurls = []
        if citation["url"] != "":
            for img in json.loads(citation["url"]):
                imgurls.append(generate_sas_token(img))

        filepath = citation["filepath"]
        title = citation["title"]
        snippet = citation["content"]
        chunk_id = citation["chunk_id"]
        replaced_html = f":blue-background[{citation_reference}]"
        content = content.replace(citation_reference, replaced_html)

        citation_obj = {
            "citations": "Citations",
            "citation_reference": citation_reference,
            "imgurls": imgurls,
            "filepath": filepath,
            "title": title,
            "snippet": snippet,
            "chunk_id": chunk_id,
            "replaced_html": replaced_html,
        }
        citations['all'].append(citation_obj)
        
        if citation_reference in content:
            citations['selected'].append(citation_obj)
            
    return content, total_tokens, prompt_tokens, completion_tokens, citations

# Expand the Citation data to create the dataframe
def get_citation_data():
    data = []
    for r in refs:
        if r["imgurls"]:
            data.append({})
        print(f'{r["citation_reference"]}, {r["imgurls"]}')


def get_refs():
    refs = None
    if st.session_state["references"]:
        refs = st.session_state["references"][-1]
    return refs

# Render the references in the chat
def render_references(refs, render_type, citation_type):
    refs = refs[citation_type]
    tab_names = [r["citation_reference"] for r in refs]
    
    if render_type == 'tabs':
        if len(tab_names) > 0:
            tab_array = st.tabs(tab_names)
            for i, tab in enumerate(tab_array):
                with tab:
                    ref = refs[i]
                    st.write(f"## {ref['citations']}")
                    filename = ref["filepath"].split("/")[-1]
                    st.write(f"### [{ref['title']}]({ref['filepath']})")
                    st.write(ref["snippet"])
                    for img in ref["imgurls"]:
                        caption = ''
                        try:
                            page, img_num = img.split('image_')[1].split('.png')[0].split('_')
                            pdf_name = ref['title'].split('.')[0].replace('-', ' ')
                            caption = f'{pdf_name} (Page {page}, Picture {img_num})'
                        except Exception as e:
                            pass
                        st.image(img, use_column_width="auto", caption=caption)
                    
    elif render_type == 'expenders':
        for i, tab in enumerate(tab_names):
            ref = refs[i]
            with st.expander(f":blue-background[{tab_names[i]}] [{ref['title']}]({ref['filepath']})"):
                st.write(f"#### {ref['citations']}")
                st.write(ref["snippet"])
                for img in ref["imgurls"]:
                    caption = ''
                    try:
                        page, img_num = img.split('image_')[1].split('.png')[0].split('_')
                        pdf_name = ref['title'].split('.')[0].replace('-', ' ')
                        caption = f'{pdf_name} (Page {page}, Picture {img_num})'
                    except Exception as e: 
                        pass
                    st.image(img, use_column_width="auto", caption=caption)
                    

## Main LLM function call to get results from Azure OpenAI
def llm_request(msg):
    client = st.session_state['aoai_client']
    completion = client.chat.completions.create(
        model=deployment,
        messages=msg,
        max_tokens=st.session_state["llm_config_params"]['max_response'],
        temperature=st.session_state["llm_config_params"]['temperatur'],
        top_p=st.session_state["llm_config_params"]['top_p'],
        frequency_penalty=st.session_state["llm_config_params"]['frequency_penalty'],
        presence_penalty=st.session_state["llm_config_params"]['presence_penalty'],
        stop=st.session_state["llm_config_params"]['stop_sequences'],
        extra_body={
            "data_sources": [
                {
                    "type": "azure_search",
                    "parameters": {
                        "filter": st.session_state["search_config_params"]['filter'],
                        "strictness": st.session_state["search_config_params"]['strictness'],
                        "top_n_documents": st.session_state["search_config_params"]['top_n_documents'],
                        "semantic_configuration": search_semantic_config,
                        "query_type": search_query_type,
                        "endpoint": search_endpoint,
                        "index_name": search_index,
                        "authentication": {"type": "api_key","key": search_api_key},
                        "embedding_dependency": {
                            "type": "deployment_name",
                            "deployment_name": deployment_embedding
                        }
                    },
                }
            ]
        },
    )

    return completion

# LLM function call to get results
def get_llm_completion(prompt):
    st.session_state["messages"].append({"role": "user", "content": prompt})
    try:

        completion = llm_request(st.session_state["messages"])
        response = completion.choices[0].message.content

    except Exception as e:
        st.write(e)
        response = f"The API could not handle this content: {json.dumps(e, default=str)}"
        print('>>> error???')
    st.session_state["messages"].append({"role": "assistant", "content": response})

    return json.loads(completion.to_json())


# Fetch the response from the LLM
def generate_response(prompt):
    st.session_state["messages"].append({"role": "user", "content": prompt})
    try:

        completion = llm_request(st.session_state["messages"])
        response = completion.choices[0].message.content

    except Exception as e:
        st.write(e)
        response = f"The API could not handle this content: {str(e)}"
    st.session_state["messages"].append({"role": "assistant", "content": response})

    total_tokens = completion.usage.total_tokens
    prompt_tokens = completion.usage.prompt_tokens
    completion_tokens = completion.usage.completion_tokens

    return response, total_tokens, prompt_tokens, completion_tokens

###
################ Image Search ################
###

# Generate embeddings for the text using Azure AI Vision
def generate_embeddings_text(text):

    print(f"Generating embeddings...")
    print(f"Input text: {text}")
    print(f"Input AI_VISION_ENDPOINT: {ai_vision_endpoint}")

    url = f"{ai_vision_endpoint}/computervision/retrieval:vectorizeText?api-version={ai_vision_apiversion}"

    headers = {
        "Content-Type": "application/json",
        "Ocp-Apim-Subscription-Key": ai_vision_api_key,
    }

    data = {"text": text}

    s = time.time()
    response = requests.post(url, headers=headers, json=data)
    print(f"Time taken florence text embedding: {time.time() - s}")

    if response.status_code == 200:
        # print(f"Embeddings: {response.json()}")
        embeddings = response.json()["vector"]
        return embeddings
    else:
        print(f"Error: {response.status_code} - {response.text}")
        return None


## Call the Azure OpenAI API to get the rephrased query
def ask_openai(query):
    print(f"Asking OpenAI...")
    print(f"Input query: {query}")

    client = st.session_state['aoai_client']

    chat_completion = client.chat.completions.create(
        messages=[
            {
                "role": "system",
                "content": (
                    "You are helpful AI assistant."
                ),
            },
            {
                "role": "user",
                "content": (
                    "Convert a user query into a textual representation capturing central "
                    "semantic meanings which is most suitable for finding best results in a search. "
                    "Output only a final query not more than 200 tokens size. Here is the original query: "
                    f"{query}"
                ),
            },
        ],
        model=deployment,
        max_tokens=200,
    )

    return chat_completion.choices[0].message.content

## Image Search function
def ImageSearch(query, max_images=5):
    
    try:
        user_query = query

        query = ask_openai(user_query)
        print(f"Rephrased query: {query}")

        # Generate embeddings for the query
        vector_query = VectorizedQuery(
            vector=generate_embeddings_text(query),
            k_nearest_neighbors=max_images,
            fields="image_vector",
        )

        ssrc = time.time()
        # Perform vector search
        print(f"Creating search client. AI_SEARCH_SERVICE_ENDPOINT: {search_endpoint}, AI_SEARCH_INDEX_NAME: {search_index_image}")
        
        search_client = SearchClient(
            endpoint=search_endpoint,
            index_name=search_index_image,
            credential= AzureKeyCredential(search_api_key)
        )

        results = search_client.search(
            search_text=query, vector_queries=[vector_query], select=["title", "chunk", "imageUrl"]
        )

        print(f"Time taken for florence search: {time.time() - ssrc}")
        # Print the search results
        output = []

        for result in results:
            
            print(f"Result: {result}")

            if "imageUrl" in result and result["imageUrl"]:
                image_url = result["imageUrl"]
            else:
                image_url = blob_folderPath + result["title"]

            sas_url = generate_sas_token(image_url)

            ### Convert to Base64 image if URL is not accessible
            ## Check if the request was successful
            # if response.status_code == 200:
            #     # Convert the image data to base64
            #     base64_image = base64.b64encode(response.content).decode('utf-8')
            #     print(f"Base64 Image: {base64_image}")
            # else:
            #     base64_image = f"Failed to download image. Status code: {response.status_code}, Reason: {response.reason}"

            output.append(
                {
                    "Title": result["chunk"] if result["chunk"] else result["title"],
                    "ImageURL": image_url,
                    # "Image": base64_image,
                    "Image": sas_url,
                    "Score": result["@search.score"],
                }
            )
    except Exception as e:
            st.write(e)
            response = f"The API could not handle this content: {json.dumps(e, default=str)}"
            print('>>> error???')
            return json.dumps([])
    return json.dumps(output)


#### STreamlit - Main app Rendering ####
st.title("Azure Open AI RAG with Images")

container = st.container()
with container:
    response_container = st.container()
tabs = st.tabs(["Chat with Data", "Image Search"])

with tabs[0]:
    with st.form(key="form", clear_on_submit=True):
        user_input = st.text_area("Type a new question...", key="input", height=100)
        submit_button = st.form_submit_button(label="Send")
            
    if submit_button and user_input:
        completion = get_llm_completion(user_input)
        content, total_tokens, prompt_tokens, completion_tokens, citations = generate_response_data(completion)

        st.session_state["references"].append(citations)
        st.session_state["past"].append(user_input)
        st.session_state["generated"].append(content)
        sentiment_mapping = [":material/thumb_down:", ":material/thumb_up:"]
        st.session_state["model_name"].append(deployment)

with tabs[1]:
    st.write("### Image Search")
    image_query = st.text_input("Search for images", key="image_search_input")
    search_button = st.button("Search Images", key="image_search_button")
    
    if search_button and image_query:
        # Placeholder for image search logic
        st.write(f"Searching for images related to: {image_query}")
        # Add logic to fetch and display images here
        image_results = ImageSearch(image_query)
        if image_results:
            image_results = json.loads(image_results)
            cols = st.columns(3)  # Adjust the number of columns as needed
            for idx, result in enumerate(image_results):
                with cols[idx % 3]:
                    st.image(result["Image"], caption=result["Title"], use_column_width=True)

if st.session_state["generated"]:
    with response_container:
        for i in range(len(st.session_state["generated"])):
            with st.chat_message('user', avatar=st.session_state['avatar_user']):
                st.write(st.session_state["past"][i], unsafe_allow_html=True)
                
            with st.chat_message('user', avatar=st.session_state['avatar_ai']):
                st.write(
                    st.session_state["generated"][i] + '\n\n --- \n\n' + '<p style="text-align: right; font-size:0.8em"> AI-generated content may be incorrect </p>',
                    unsafe_allow_html=True
                )
                
            refs = st.session_state["references"][i]
            render_references(refs, 'expenders', 'selected')


with st.sidebar:
    refs = get_refs()
    if get_refs():
        refs_count = len(get_refs()['selected'])
        on = st.toggle(f"{refs_count} References", help="Toggle on to see citations", value =True)
        if on:
            render_references(refs, 'tabs', 'selected')

