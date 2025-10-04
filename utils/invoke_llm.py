import os

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.schema.output_parser import StrOutputParser
from langchain.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
load_dotenv()
llm=ChatOpenAI(
    model=os.environ.get("CUSTOM_MODEL"),
    base_url=os.environ.get("API_BASE"),
    api_key=os.environ.get("CUSTOM_API_KEY")
)
def invoke_llm(text):
    sys_prompt=ChatPromptTemplate.from_template(
        """你是一名专业的文本摘要助手。请阅读以下文本，并在确保保留核心信息与逻辑完整的前提下，对其进行简洁、精炼、通顺的摘要。  
    要求：  
    1. 用简洁、自然的语言重述原文重点。  
    2. 删除冗余、重复或无关内容。  
    3. 不添加主观评价或额外信息。  
    4. 输出应短小、清晰、逻辑流畅。
    
    以下是需要摘要的文本：
    ---
    {text}
    ---
    请输出摘要。
    """
    )
    parser=StrOutputParser()
    chain=sys_prompt | llm | parser
    result=chain.invoke({'text':text})
    return result

if __name__=='__main__':
    text="这里是一个需要摘要的示例文本。它包含了一些冗余的信息，比如这句话，以及一些重复的内容，比如这句话，还有一些无关的内容，比如天气预报。但是，它也有一些核心的信息，比如项目的目标是开发一个高效的算法，这个信息非常重要。"
    summary=invoke_llm(text)
    print(summary)