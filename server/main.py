from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
import os
from typing import List
import shutil
from dotenv import load_dotenv

load_dotenv()


app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["http://localhost:3000"], allow_methods=["*"], allow_headers=["*"])

PERSIST_DIR = "./chroma_db"  # Directory for vector store

@app.post("/upload-pdfs")
async def upload_pdfs(files: List[UploadFile] = File(...)):
    # Clear previous DB if needed
    if os.path.exists(PERSIST_DIR):
        shutil.rmtree(PERSIST_DIR)
    
    docs = []
    for file in files:
        if file.filename.endswith(".pdf"):
            # Save temp file
            temp_path = f"./temp_{file.filename}"
            with open(temp_path, "wb") as buffer:
                shutil.copyfileobj(file.file, buffer)
            loader = PyPDFLoader(temp_path)
            docs.extend(loader.load())
            os.remove(temp_path)  # Clean up
    
    if not docs:
        return {"error": "No valid PDFs uploaded"}
    
    # Split documents
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = text_splitter.split_documents(docs)
    
    # Embed and store
    embeddings = OpenAIEmbeddings()
    vectorstore = Chroma.from_documents(documents=splits, embedding=embeddings, persist_directory=PERSIST_DIR)
    vectorstore.persist()
    
    return {"message": f"Indexed {len(splits)} chunks from {len(files)} PDFs"}


from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, AIMessage
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from typing import TypedDict, Annotated, Sequence
from operator import add
import operator
from pydantic import BaseModel, Field
from langchain.tools.retriever import create_retriever_tool


# State for graph
class AgentState(TypedDict):
    messages: Annotated[Sequence[AIMessage], add]
    documents: list  # Retrieved docs

# Global retriever (loaded on startup or per request)
embeddings = OpenAIEmbeddings()
db = Chroma(persist_directory=PERSIST_DIR, embedding_function=embeddings)
retriever = db.as_retriever(search_kwargs={"k": 4})

# Retriever tool
@tool
def retrieve_docs(query: str) -> str:
    """Retrieve relevant docs from PDFs."""
    docs = retriever.invoke(query)
    return "\n\n".join([doc.page_content for doc in docs])

tools = [retrieve_docs]
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

# Node: Decide to retrieve or respond
def route_query(state: AgentState):
    prompt = ChatPromptTemplate.from_template(
        "Given user question: {question}\nShould you retrieve context? Respond 'retrieve' or 'respond'."
    )
    chain = prompt | llm
    msg = chain.invoke({"question": state["messages"][-1].content})
    return {"next": msg.content.lower()}

# Node: Retrieve
def retrieve(state: AgentState):
    tool_node = ToolNode(tools)
    result = tool_node.invoke(state)
    return {"messages": result["messages"], "documents": result["intermediate_steps"]}

# Node: Grade docs (simple binary relevance)
class Grade(BaseModel):
    binary_score: str = Field(description="yes/no if relevant")

def grade_docs(state: AgentState):
    prompt = ChatPromptTemplate.from_template(
        'Docs: {docs}\nQuestion: {question}\nIs relevant? Output JSON: {{"binary_score": "yes/no"}}'
    )
    chain = prompt | llm | JsonOutputParser(pydantic_object=Grade)
    scores = []
    for doc in state["documents"]:
        score = chain.invoke({"docs": doc[1], "question": state["messages"][-1].content})
        scores.append(score)
    relevant = [doc for doc, score in zip(state["documents"], scores) if score["binary_score"] == "yes"]
    return {"documents": relevant} if relevant else {"next": "rewrite"}

# Node: Rewrite question if irrelevant
def rewrite_question(state: AgentState):
    prompt = ChatPromptTemplate.from_template(
        "Rewrite question for better retrieval: {question}\nRewritten:"
    )
    chain = prompt | llm
    rewritten = chain.invoke({"question": state["messages"][-1].content}).content
    return {"messages": [HumanMessage(content=rewritten)]}

# Node: Generate answer
def generate_answer(state: AgentState):
    prompt = ChatPromptTemplate.from_template(
        """Answer based on context only. If unsure, say so.
        Context: {context}
        Question: {question}"""
    )
    chain = prompt | llm
    context = "\n\n".join([doc[1] for doc in state["documents"]])
    answer = chain.invoke({"context": context, "question": state["messages"][-1].content})
    return {"messages": [AIMessage(content=answer.content)]}

# Build graph
workflow = StateGraph(AgentState)
workflow.add_node("route", route_query)
workflow.add_node("retrieve", retrieve)
workflow.add_node("grade", grade_docs)
workflow.add_node("rewrite", rewrite_question)
workflow.add_node("generate", generate_answer)

# Edges
workflow.set_entry_point("route")
workflow.add_conditional_edges("route", lambda x: x["next"], {"retrieve": "retrieve", "respond": "generate"})
workflow.add_edge("retrieve", "grade")
workflow.add_conditional_edges("grade", lambda x: "rewrite" if not x["documents"] else "generate", {"rewrite": "rewrite", "generate": "generate"})
workflow.add_edge("rewrite", "retrieve")
workflow.add_edge("generate", END)

graph = workflow.compile()


from pydantic import BaseModel

class ChatRequest(BaseModel):
    message: str

@app.post("/chat")
async def chat(request: ChatRequest):
    state = {"messages": [HumanMessage(content=request.message)], "documents": []}
    result = graph.invoke(state)
    return {"response": result["messages"][-1].content}






if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)