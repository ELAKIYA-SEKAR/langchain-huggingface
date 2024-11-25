#HuggingFace Endpoint Usage
import os                                                                                                                                                                                                          
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEndpoint
from langchain import PromptTemplate, LLMChain

load_dotenv()
hf_token=os.getenv("HUGGING_FACE_TOKEN")
os.environ["HUGGINGFACEHUB_API_TOKEN"]=hf_token
repo_id="Qwen/Qwen2.5-Coder-32B-Instruct"
llm=HuggingFaceEndpoint(repo_id=repo_id,max_length=128,temperature=0.7,token=hf_token)
#print(llm.invoke("what is machine learning?"))
question="Who played the most innings in the history of Cricket?"

template="""Question:{question}
Answer: Let's think step by step,"""
prompt=PromptTemplate(template=template,input_variables=['question'])
print(prompt)
llm_chain=LLMChain(llm=llm,prompt=prompt)
print(llm_chain.invoke(question))

#hugging face pipleine
from langchain_huggingface import HuggingFacePipeline
from transformers import AutoModelForCausalLM,AutoTokenizer,pipeline
model_id="openai-community/gpt2"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id)  

pipe=pipeline("text-generation",model=model,tokenizer=tokenizer,max_new_tokens=100)
hf=HuggingFacePipeline(pipeline=pipe)
#print(hf)
print(hf.invoke("Langchain is a company"))
gpu_llm = HuggingFacePipeline.from_model_id(
    model_id="gpt2",
    task="text-generation",
    device=-1,  # replace with device_map="auto" to use the accelerate library.
    pipeline_kwargs={"max_new_tokens": 100},
)

#LangChain with GPU
from langchain_core.prompts import PromptTemplate

template = """Question: {question}

Answer: Let's think step by step."""
prompt = PromptTemplate.from_template(template)
chain=prompt|gpu_llm

question="What is artificial intelligence?"
print(chain.invoke({"question":question}))

#device=0- gpu
#device=-1 -cpu