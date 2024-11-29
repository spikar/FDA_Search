from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI
from langchain.output_parsers import PydanticOutputParser
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
from prompts import multiquery_prompt as template
from configs import model_multiquery, temperature
from typing import List
from dotenv import load_dotenv
load_dotenv()


class ConsolidatedInformation(BaseModel):
    SubQuestions: List[str] = Field(description="List of sub-questions related to main question")

output_parser = PydanticOutputParser(pydantic_object=ConsolidatedInformation)
format_instructions = output_parser.get_format_instructions()

prompt_decomposition = ChatPromptTemplate.from_template(template,partial_variables={"format_instructions":format_instructions})

# LLM
llm = ChatOpenAI(model=model_multiquery,temperature=temperature)

# Chain
generate_queries_decomposition = (
    prompt_decomposition 
    | llm 
    | output_parser 
    | (lambda x: x.SubQuestions)
)
