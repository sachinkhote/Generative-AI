import os 
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate , MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage

load_dotenv()

llm = ChatGoogleGenerativeAI(model= 'gemini-2.5-flash', temperature= 0.7)

prompt = ChatPromptTemplate.from_messages([('system','you are helpful assistant. Remember everything the user tells you.'),
                                           MessagesPlaceholder(variable_name='chat_history'),
                                           ('human','{input}')])

chain = prompt | llm

chat_history= []

print('-'*50)
print('chatbots with memory')
print('-'*50)
print('I will remember everything you tell me!')
print("Type 'exit' to quit. \n")

while True:
    user_input= input('You:')

    if user_input.lower() == 'exit':
        print('\n Goodbye!')
        break
    if user_input.strip() =='':
        continue

    response =chain.invoke({'input':user_input, 'chat_history':chat_history})       # pass history every time

    answer = response.content

    chat_history.append(HumanMessage(content=user_input))           
    chat_history.append(AIMessage(content=answer))                          # after every conversation turn, append adds both user message and AI response to history list.

    print(f"\n Assistant: {answer} \n")

    # on every next conversation, everyhting previously asked and answered by llm is resent to llm again.