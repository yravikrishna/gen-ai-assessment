

# get embeddings for a list of texts

def get_embeddings_wrapper(texts, project, location):
    import time
    import tqdm  # to show a progress bar
    from vertexai.language_models import TextEmbeddingModel
    from google.cloud import aiplatform

    aiplatform.init(project=project, location=location)

    BATCH_SIZE=5
    model_name="textembedding-gecko@002"
    model = TextEmbeddingModel.from_pretrained(model_name)
    embs = []
    for i in tqdm.tqdm(range(0, len(texts), BATCH_SIZE)):
        time.sleep(1)  # to avoid the quota error
        result = model.get_embeddings(texts[i : i + BATCH_SIZE])
        embs = embs + [e.values for e in result]
    return embs

def load_file(text_chunk_file_path):
    import json
    data=[]
    with open(text_chunk_file_path, "r") as file:
        for line in file:
            entry=json.loads(line)
            data.append(entry)
    return data

def generate_context(ids,data):
    concatenated_names = ''
    for id in ids:
        for entry in data:
            if entry['id'] == int(id):
                concatenated_names += entry['text'] + "\n" 
    return concatenated_names.strip()

def ask_gemini(question, context):
    from vertexai.generative_models import GenerationConfig, GenerativeModel

    model = GenerativeModel("gemini-1.0-pro")

    # You will need to change the code below to ask Gemni to
    # answer the user's question based on the data retrieved
    # from their search
    prompt = f"""Answer the question given in the contex below:
            Context: {context}?\n
            Question: {question} \n
            Answer:
            """
    response = model.generate_content(prompt).text
    return response

def search_vector_database(question):
    from google.cloud import aiplatform

    # 1. Convert the question into an embedding
    temp_list=[]
    temp_list.append(question)
    test_embeddings = get_embeddings_wrapper(temp_list)

    # 2. Search the Vector database for the 5 closest embeddings to the user's question
    #create index
    BUCKET_URI="gs://ravi-rag"
    my_index = aiplatform.MatchingEngineIndex.create_tree_ah_index(
        display_name="RAG400Lab-Index",
        contents_delta_uri=BUCKET_URI,
        dimensions=768,
        approximate_neighbors_count=20,
        distance_measure_type="DOT_PRODUCT_DISTANCE",
    )
    # create IndexEndpoint
    my_index_endpoint = aiplatform.MatchingEngineIndexEndpoint.create(
    display_name="assessment-index-endpoint",
    public_endpoint_enabled=True,
    )
    DEPLOYED_INDEX_ID = "Deployed_RAG400Lab_Index"
    # deploy the Index to the Index Endpoint
    my_index_endpoint.deploy_index(index=my_index, deployed_index_id=DEPLOYED_INDEX_ID)
    
    response = my_index_endpoint.find_neighbors(
                deployed_index_id=DEPLOYED_INDEX_ID,
                queries=test_embeddings,
                num_neighbors=5,
    )
    # 3. Get the IDs for the five embeddings that are returned
    matching_ids = [neighbor.id for sublist in response for neighbor in sublist]

    # 4. Get the five documents from Firestore that match the IDs
    text_chunk_file_path="text_chunks.json"
    data=load_file(text_chunk_file_path)
    context=generate_context(matching_ids, data)
    # 5. Concatenate the documents into a single string and return it

    #data = ""
    return context
