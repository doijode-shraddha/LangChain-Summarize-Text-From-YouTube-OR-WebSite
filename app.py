## Importing Necessary Libraries

import validators, streamlit as st
from langchain.prompts import PromptTemplate
from langchain_groq import ChatGroq
from langchain.chains.summarize import load_summarize_chain
from langchain_community.document_loaders import UnstructuredURLLoader
from youtube_transcript_api import YouTubeTranscriptApi
from langchain.docstore.document import Document

## Setting up the Streamlit APP

st.set_page_config(page_title="LangChain--Summarize Text From YouTube OR WebSite", page_icon="🦜")
st.title("🦜 LangChain: Summarize Text From YouTube OR WebSite")
st.subheader("Summarize URL")

# to get the Groq API key input
with st.sidebar:
    groq_api_key = st.text_input("GROQ-API-KEY", value="", type="password")
    
# Function to get URL input
generic_uri = st.text_input("URL", label_visibility="collapsed")

# Updated Groq Model
llm = ChatGroq(model="gemma2-9b-it", groq_api_key=groq_api_key)

# Prompt Template
prompt_template = """
Provide a summary of the following content in 500 words:
Content: {text}
"""
prompt = PromptTemplate(template=prompt_template, input_variables=['text'])

# Function to get YouTube Transcript
def get_youtube_transcript(url):
    try:
        video_id = url.split("v=")[-1]  # Extract the video ID from the URL
        transcript = YouTubeTranscriptApi.get_transcript(video_id)
        text = " ".join([entry["text"] for entry in transcript])
        return text
    except Exception as e:
        return f"Error fetching transcript: {e}"

# Button to Summarize the Content from YouTube or WebSite
if st.button("Summarize the Content from YouTube or WebSite"):
    if not groq_api_key.strip() or not generic_uri.strip():
        st.error("Please provide the required information to get started")
    elif not validators.url(generic_uri):
        st.error("Please enter a valid URL. It may be a YouTube video URL or a WebSite URL")
    else:
        try:
            with st.spinner("Waiting..."):
                # Loading the YouTube or Website data
                if "youtube.com" in generic_uri or "youtu.be" in generic_uri:
                    docs = [Document(page_content=get_youtube_transcript(generic_uri))]
                else:
                    loader = UnstructuredURLLoader(urls=[generic_uri],
                                                   ssl_verify=False,
                                                   headers={"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 13_5_1) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/116.0.0.0 Safari/537.36"})
                    docs = loader.load()

                ## Chain for Summarization
                chain = load_summarize_chain(llm, chain_type="stuff", prompt=prompt)
                output_summary = chain.run(docs)
                st.success(output_summary)
        except Exception as e:
            st.exception(f"Exception:{e}")