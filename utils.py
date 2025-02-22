import os
import base64
from pdf2image import convert_from_path
import pandas as pd
def store_image_pdf(pdf_path, base_store_path):
    """Generates images from the pages of a PDF file

    Args:
        pdf_path (str): Path to the pdf file
        base_store_path (str): Path to store the images 
    """
    pages = convert_from_path(pdf_path)
    for count, page in enumerate(pages):
        store_path = os.path.join(base_store_path, f'page_{count}.png')
        page.save(store_path, 'PNG')

def encode_image(image_path):
    """Function to encode images to base64

    Args:
        image_path (str): Path to the image

    Returns:
        base64: File converted to base64
    """
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")

def sumarizer_for_image(image_path, client):
    """Creates a summary of the images, this summary will be used as context for the models

    Args:
        image_path (str): Path to image to summarize
        client (openai.OpenAI): OpenAI client

    Returns:
        str: Response generated by openAI
    """
    response = client.chat.completions.create(
        model='gpt-4o-mini',
        messages=[
            {"role": "user", "content": [
                {"type": "text", "text": "Can you give me a description in spanish of the products that you can find in the image, all the information that you can obtain from the image."},
                {"type": "image_url", "image_url": {
                    "url": f"data:image/png;base64,{encode_image(image_path)}"}
                }
            ]}
        ],
        temperature=0.0,
    )
    content = response.choices[0].message.content
    return content

def prompt_to_llm(query, faiss_index):
    """Prompt definition and context adition to generate answer from LLM 

    Args:
        query (str): Input message from the user
        faiss_index (langchain_community.vectorstores.faiss.FAISS): FAISS instance with documents and embeddings

    Returns:
        str: Prompt that will be sent to the LLM
    """
    results = faiss_index.similarity_search(query, k = 3)
    knowdledge = ''
    for result in results:
        knowdledge += result.page_content + '\n' + 'Esto se encuentra en la pagina ' + str(result.metadata['page']) + '\n'

        knowdledge = knowdledge.replace('\n', ' ')
        knowdledge = knowdledge.replace('  ', ' ')

    prompt_definition = f''' Using the context below and just this context, answer the query, if there is a question that is not related to the context you can
    answer "No tengo conocimiento en este tema, porfavor hazme preguntas sobre los productos":

    Context: {knowdledge}
    Query: {query}
    '''
    return prompt_definition

def context_generation(client):
    """Generates images from the PDF file pages and uses openAI client to find a summary of the information of these images

    Args:
        client (openai.OpenAI): Client that generates summary of images in spanish
    """
    GENERAL_PATH = os.curdir
    INFORMATION_PATH = os.path.join(GENERAL_PATH, 'df_info.csv')
    if os.path.exists(INFORMATION_PATH) == False:
        image_store_path = os.path.join(GENERAL_PATH, 'images')
        if not os.path.exists(image_store_path):
            os.makedirs(image_store_path)
            
        
        pdf_path = os.path.join(GENERAL_PATH, 'Bruno_child_offers.pdf')
        store_image_pdf(pdf_path,image_store_path)
        df_info = pd.DataFrame(columns=['path','page'])
        for path in os.listdir(image_store_path):
            df_info.loc[len(df_info.index)] = [path, path.split('_')[1].split('.')[0]]
        
        sumarization_array = []
        for path in os.listdir(image_store_path):
            sumarization_array.append(sumarizer_for_image(os.path.join(image_store_path, path), client))
        if 'sumarization' not in df_info.columns:
            df_info['sumarization'] = sumarization_array
        df_info.to_csv(INFORMATION_PATH, index=False)
    else:
        print('Summary generation already exist')
        df_info = pd.read_csv(INFORMATION_PATH, index_col=False)
    return df_info