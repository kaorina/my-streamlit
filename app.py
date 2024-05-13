import os

import streamlit as st
from dotenv import load_dotenv
from langchain import hub
from langchain.agents import AgentExecutor, create_openai_tools_agent, load_tools
from langchain.memory import ConversationBufferMemory
from langchain_community.callbacks import StreamlitCallbackHandler
from langchain_community.chat_message_histories import StreamlitChatMessageHistory
from langchain_openai import ChatOpenAI

load_dotenv()

#st.title("langchain-streamlit-app")

#input_num = st.number_input('Input a number', value=0)
#result = input_num ** 2
#st.write('Result: ', result)

#def create_agent_chain():
#  chat = ChatOpenAI(
#    model_name=os.environ["OPENAI_API_MODEL"],
#    temperature=os.environ["OPENAI_API_TEMPERATURE"],
#    streaming=True,
#  )
#
#  agent_kwargs = {
#    "extra_prompt_messages": [MessagesPlaceholder(variable_name="memory")],
#  }
#  memory = ConversationBufferMemory(memory_key="memory", return_messages=True)
#  
#  tools = load_tools(["ddg-search", "wikipedia"])
#  return initialize_agent(
#    tools, 
#    chat, 
#    agent=AgentType.OPENAI_FUNCTIONS,
#    agent_kwargs=agent_kwargs,
#    memory=memory,
#  )
#  #prompt = hub.pull("hwchase17/openai-tools-agent")
#  #agent = create_openai_tools_agent(chat, tools, prompt)
#  #return AgentExecutor(agent=agent, tools=tools)
#
#if "messages" not in st.session_state:
#  st.session_state.messages = []
#
#for message in st.session_state.messages:
#  with st.chat_message(message["role"]):
#    st.markdown(message["content"])  
#
#prompt = st.chat_input("What is up?")
#
#if prompt:
#  st.session_state.messages.append({"role" : "user", "content" : prompt})
#
#  with st.chat_message("user"):
#    st.markdown(prompt)
#  with st.chat_message("assistant"):
#    #response = "Hello"
#    #st.markdown(response)
#    #chat = ChatOpenAI(
#    #  model_name=os.environ["OPENAI_API_MODEL"],
#    #  temperature=os.environ["OPENAI_API_TEMPERATURE"],
#    #)
#    #messages = [HumanMessage(content=prompt)]
#    #response = chat(messages)
#    callback = StreamlitCallbackHandler(st.container())
#    #agent_chain = create_agent_chain()
#    if "agent_chain" not in st.session_state:
#      st.session_state.agent_chain = create_agent_chain()
#
#    response = st.session_state.agent_chain.run(input=prompt, callback=[callback])
#    st.markdown(response)
#
#  st.session_state.messages.append({"role" : "assistant", "content" : response})


def create_agent_chain(history):
  chat = ChatOpenAI(
    model_name=os.environ["OPENAI_API_MODEL"],
    temperature=os.environ["OPENAI_API_TEMPERATURE"],
  )

  tools = load_tools(["ddg-search", "wikipedia"])

  prompt = hub.pull("hwchase17/openai-tools-agent")

  memory = ConversationBufferMemory(
    chat_memory=history, memory_key="chat_history", return_messages=True
  )

  agent = create_openai_tools_agent(chat, tools, prompt)
  return AgentExecutor(agent=agent, tools=tools, memory=memory)

st.title("langchain-streamlit-app")

# StreamlitChatMessageHistoryを使用することで、
# st.session_stateを使った会話履歴の管理を自前で実装する必要がなくなりました。
history = StreamlitChatMessageHistory()

for message in history.messages:
  st.chat_message(message.type).write(message.content)

prompt = st.chat_input("What is up?")

if prompt:
  with st.chat_message("user"):
    st.markdown(prompt)

  with st.chat_message("assistant"):
    callback = StreamlitCallbackHandler(st.container())

    agent_chain = create_agent_chain(history)
    response = agent_chain.invoke(
      {"input": prompt},
      {"callbacks": [callback]},
    )

    st.markdown(response["output"])
