import streamlit as st
import pandas as pd
from pypdf import PdfReader
import io
import requests
import json
import base64
from langchain.docstore.document import Document
from collections import Counter
import copy
from streamlit.runtime.scriptrunner.script_run_context import get_script_run_ctx
import os
import glob
from langchain.vectorstores import Chroma
from src.constants import *
from src import llama2
from chromadb.config import Settings
from langchain.chains import RetrievalQA
from streamlit_extras.colored_header import colored_header
from streamlit_chat import message
from langchain.text_splitter import RecursiveCharacterTextSplitter

os.environ['CHROMA_CACHE_DIR'] = 'chroma_cache'
vec = llama2.Llama2Embeddings()

class ST_CONFIG:
    user_bg_color = '#77ff77'
    user_icon = 'https://tse2-mm.cn.bing.net/th/id/OIP-C.LTTKrxNWDr_k74wz6jKqBgHaHa?w=203&h=203&c=7&r=0&o=5&pid=1.7'
    robot_bg_color = '#ccccee'
    robot_icon = 'https://ts1.cn.mm.bing.net/th/id/R-C.5302e2cc6f5c7c4933ebb3394e0c41bc?rik=z4u%2b7efba5Mgxw&riu=http%3a%2f%2fcomic-cons.xyz%2fwp-content%2fuploads%2fStar-Wars-avatar-icon-C3PO.png&ehk=kBBvCvpJMHPVpdfpw1GaH%2brbOaIoHjY5Ua9PKcIs%2bAc%3d&risl=&pid=ImgRaw&r=0'
    default_mode = 'KowledgeBase QA'
    defalut_kb = ''


def init_session():
    st.session_state.setdefault('history', [])
    st.session_state.setdefault('something', None)
    st.session_state.setdefault('vector_store', None)
    st.session_state.setdefault('pdf', None)
    # dataframe to store filenames with tags and keywords
    st.session_state.setdefault('df_files', pd.DataFrame(columns=['FID', 'Filename', 'Tags', 'Keywords']))
    # dataframe to store information for each paragraph
    st.session_state.setdefault('df_pdata',
                                pd.DataFrame(
                                    columns=['PID', 'Content', 'Tag1', 'Tag2', 'Tag3', 'Filename', 'Page', 'Keywords'])
                                )
    st.session_state.setdefault('tag_button', None)
    st.session_state.setdefault('tag_files', None)
    ctx = get_script_run_ctx()
    st.session_state['session_id'] = ctx.session_id
    st.session_state['vs_folder'] = 'sessions/' + ctx.session_id + '/'
    if not os.path.exists(st.session_state['vs_folder']):
        os.makedirs(st.session_state['vs_folder'])
    st.session_state['chroma_setting'] = Settings(
        persist_directory=st.session_state['vs_folder'],
        chroma_db_impl='duckdb+parquet',
        anonymized_telemetry=False
    )
    db = Chroma(persist_directory=st.session_state['vs_folder'], embedding_function=vec,
                client_settings=st.session_state['chroma_setting'])
    retriever = db.as_retriever(search_kwargs={"k": target_source_chunks})
    llm = llama2.Llama2LLM()
    st.session_state['qa'] = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever,
                                     return_source_documents=True)

## generated stores AI generated responses
if 'generated' not in st.session_state:
    st.session_state['generated'] = ["How may I help you?"]
## past stores User's questions
if 'past' not in st.session_state:
    st.session_state['past'] = ['Hi!']


def json_tags(obj):
    '''
    Convert tagger response in json to tags and keywords
    '''
    # extract tags from json response
    taxo = obj[0]['IAB taxonomy categories'][0]
    tags = []
    kwds = []

    i = 1
    while i < len(taxo):
        if taxo[i][1] is not None:
            tags.append(taxo[i][1])
        i += 1

    for x in obj[0]['keyword/phrase']:
        if len(x[1]) > 3:
            kwds.append(x[1])

    return '_'.join(tags), '_'.join(kwds)


def get_tags(x):
    data = {'body': x}
    r = requests.post(url=tag_url, json=json.dumps(data))
    response = r.json()
    ans_tag, ans_kwd = '', ''
    if 'error' in response[0]:
        return '', ''
    try:
        ans_tag, ans_kwd = json_tags(response)
    except Exception as e:
        print(x)
        print(response)
        print(e)
    return ans_tag, ans_kwd

def does_vectorstore_exist(persist_directory: str) -> bool:
    """
    Checks if vectorstore exists
    """
    if os.path.exists(os.path.join(persist_directory, 'index')):
        if os.path.exists(os.path.join(persist_directory, 'chroma-collections.parquet')) and os.path.exists(os.path.join(persist_directory, 'chroma-embeddings.parquet')):
            list_index_files = glob.glob(os.path.join(persist_directory, 'index/*.bin'))
            list_index_files += glob.glob(os.path.join(persist_directory, 'index/*.pkl'))
            # At least 3 documents are needed in a working vectorstore
            if len(list_index_files) > 3:
                return True
    return False

def update_para(texts, metadatas):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    _metadatas = metadatas or [{}] * len(texts)
    documents = []
    for i, text in enumerate(texts):
        index = -1
        for chunk in text_splitter.split_text(text):
            metadata = copy.deepcopy(_metadatas[i])
            index = text.find(chunk, index + 1)
            metadata['tags'], metadata['keywords'] = get_tags(chunk.replace('\n', ' '))
            metadata["start_index"] = index
            new_doc = Document(page_content=chunk, metadata=metadata)
            documents.append(new_doc)
            row = ['', chunk]
            tags_list = metadata['tags'].split('_')
            for i in range(3):
                if i < len(tags_list):
                    row.append(tags_list[i])
                else:
                    row.append('')
            row.append(metadata['filename'])
            row.append(metadata['page'])
            row.append(metadata['keywords'])
            st.session_state['df_pdata'].loc[len(st.session_state['df_pdata'])] = row
    return documents

def update_vs(texts):
    if does_vectorstore_exist(st.session_state['vs_folder']):
        db_update = Chroma(persist_directory=st.session_state['vs_folder'], embedding_function=vec,
                           client_settings=st.session_state['chroma_setting'])
        db_update.add_documents(texts)
    else:
        db_update = Chroma.from_documents(texts, vec, persist_directory=st.session_state['vs_folder'],
                                          client_settings=st.session_state['chroma_setting'])

    db_update.persist()

@st.cache_resource(show_spinner='Processing...')
def update_file(_pdf_doc, filename):
    '''
    This function takes the most recent uploaded information as input.
    1. split the text into paragraphs with filename, page, location information
    2. get tags for each paragraph
    3. update df_files with this document
    4. update p_data with paragraphs in this document
    5. update vector store
    6. return the files with tags to be shown in the sidebar
    '''
    texts, metadatas = [], []
    for i in range(len(pdf_doc.pages)):
        texts.append(pdf_doc.pages[i].extract_text())
        metadatas.append({'page': i + 1, 'filename': filename})

    paras = update_para(texts, metadatas)
    all_tags, all_kwds = [], []
    for x in paras:
        x_meta = x.metadata
        if len(x_meta['tags']) > 1:
            all_tags.append(x_meta['tags'])
            all_kwds.extend(x_meta['keywords'].split('_'))

    tags_top2 = [x[0] for x in Counter(all_tags).most_common(2)]
    # print(tags_top2)
    # print(st.session_state['df_pdata'].shape)
    kwds_top5 = [x[0] for x in Counter(all_kwds).most_common(5)]
    st.session_state['df_files'].loc[len(st.session_state['df_files'])] = [filename + '_001', filename, '_'.join(tags_top2),
                                                                   '_'.join(kwds_top5)]
    st.session_state['count'] += 1

    update_vs(paras)


def update_button(word):
    st.session_state['tag_button'] = word
    df_word = st.session_state['df_files'][st.session_state['df_files']['Tags'].str.contains(word)]
    st.session_state['tag_files'] = df_word['Filename'].tolist()

def get_text():
    input_text = st.text_input("You: ", "", key="input")
    return input_text

def generate_response(prompt):
    res = st.session_state['qa'](prompt)
    reply, docs = res['result'], res['source_documents']
    template = "\n\n<p style='font-size:small;color: blue;'>[{source}]</p>\n\n<p style='font-size:small;'>{content}</p>"
    # template = "\n{source}\n\n{content}"
    # Print the relevant sources used for the answer
    for document in docs:
        reply += template.format(source = document.metadata["filename"] + ' : page ' + str(document.metadata["page"]),
                                 content = document.page_content.replace('\n',"</p><p style='font-size:small;'>"))
    # print(reply)
    return reply

@st.cache_resource(show_spinner='Thinking...')
def update_response(prompt):
    response = generate_response(prompt)
    st.session_state.past.append(prompt)
    st.session_state.generated.append(response)


webui_title = """
# LLM-KowledgeBase QA
"""

# main ui
st.set_page_config(webui_title, layout='wide')
init_session()

# Create dummy data
# st.session_state['df_files'].loc[st.session_state['df_files'].shape[0]] = ['MAS_abc','MAS', '_'.join(['finance','finance_fintech']), '_'.join(['AI','ChatGPT'])]

# Sidebar for upload pdf and show file structure for all uploaded files with tags
st.session_state['count'] = 0
with st.sidebar:
    # upload file
    uploaded_file = st.file_uploader("Upload a PDF file", type="pdf")
    if uploaded_file is not None:
        # Read the uploaded PDF file as a byte stream
        st.session_state['pdf'] = uploaded_file.read()
        bytes_data = uploaded_file.getvalue()
        filename = uploaded_file.name
        pdf_bytes_io = io.BytesIO(bytes_data)
        pdf_doc = PdfReader(pdf_bytes_io)
        update_file(pdf_doc, filename)
        for idx, row in st.session_state['df_files'].iterrows():
            st.header(row['Filename'])
            tags_tmp = row['Tags'].split('_')
            idx = 0
            while idx < len(tags_tmp):
                st.write('-' + tags_tmp[idx] + ' : ' + tags_tmp[idx + 1])
                idx += 2

# main body
preview_tab, qa_tab = st.tabs(["Preview", "Q&A"])

with preview_tab:
    preview_left, keyword_files = st.columns(2, gap='large')
    with preview_left:
        st.header("Preview")
        if st.session_state['pdf'] is not None:
            pdf_base64 = base64.b64encode(st.session_state['pdf']).decode("utf-8")
            st.markdown(f'<embed src="data:application/pdf;base64,{pdf_base64}" width="100%" height="500px">',
                        unsafe_allow_html=True)
            st.divider()
            st.header("Tags:")
            tags = st.session_state['df_files'].iloc[-1]['Tags'].split('_')
            # print(tags)
            tags = list(set(tags))
            cols_tag = st.columns(len(tags))
            buttons_tag = [None] * len(tags)
            for idx, word in enumerate(tags):
                cols_tag[idx].button(word, key='tag-' + str(idx), on_click=update_button, args=(word,))

    with keyword_files:
        st.header('Tags:')
        if st.session_state['tag_files'] is not None:
            st.header(st.session_state['tag_button'])
            for f in st.session_state['tag_files']:
                st.write('-' + f)


with qa_tab:
    input_container = st.container()
    colored_header(label='', description='', color_name='blue-30')
    response_container = st.container()
    with input_container:
        user_input = get_text()

    with response_container:
        if user_input:
            update_response(user_input)
            # print(len(st.session_state['generated']))

        if st.session_state['generated']:
            for i in range(len(st.session_state['generated'])):
                message(st.session_state['past'][i], is_user=True, key=str(i) + '_user')
                message(st.session_state["generated"][i], key=str(i), allow_html=True)
