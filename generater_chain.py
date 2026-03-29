
from openai import OpenAI
from tinydb import TinyDB
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from clip_embeddings import CLIPEmbeddings
from langchain_core.runnables import RunnableLambda
from langchain_community.vectorstores import Chroma

class GeneratorChain:
    """
    Agent class for generating kurti images
    """
    def __init__(self, openai_api_key, openrouter_token, tinydb_path="vision_data.json", vectorstore_collection="images"):
        self.client = OpenAI(api_key=openai_api_key)
        self.openrouter_token = openrouter_token
        self.clip_embeddings = CLIPEmbeddings()
        self.db = TinyDB(tinydb_path)
        self.vectorstore_collection = vectorstore_collection
        #self.initialize_vectorstore()
        self.initialize_llm()
        self.initialize_prompt()
      

    def initialize_vectorstore(self):
        self.vectorstore = Chroma(
            collection_name=self.vectorstore_collection,
            embedding_function=self.clip_embeddings
        )
        self.retriever = self.vectorstore.as_retriever()

    def initialize_llm(self):
        self.llm = ChatOpenAI(
            openai_api_base="https://openrouter.ai/api/v1",
            openai_api_key=self.openrouter_token,
            model="meta-llama/llama-3-8b-instruct",
            temperature=0.2
        )

    def initialize_prompt(self):
        self.prompt = PromptTemplate(
            input_variables=["context", "description", "view", "market_place"],
            template="""
                You are an expert fashion stylist and ecommerce image prompt engineer specializing in Indian marketplaces like {market_place}.
                Your task is to generate a highly detailed and optimized prompt for an AI image generation model.
                You must pass the locked context to generated prompt as it is.

                Locked Context:
                {context}

                Kurti Description:
                {description}

                Goal:
                Create a realistic {view} image of a female model wearing the given kurti, suitable for ecommerce listing.

                Return ONLY the final prompt. Do not add explanations."""
        )
    


    def get_context(self, inputs):
        #retrieve attributes from tinydb
        records = self.db.all()
        context = records[0]["attributes"]["raw"]
        
        print("---------context---------\n", context)
        return {
            "context": context,
            "description": inputs["description"],
            "view": inputs["view"],
            "market_place": inputs["market_place"],
        }

    def prompt_parser(self, prompt):
        """
        Parse and clean the LLM response to extract only the final prompt
        
        Args:
            inputs: Dictionary containing the LLM response
            
        Returns:
            Dictionary with cleaned prompt only
        """
        print("\n---------prompt---------\n", prompt)
        llm_response = prompt
        
        if not llm_response:
            return {"error": "No LLM response to parse"}
        
        # Remove common explanatory phrases and prefixes
        
        response_lines = ""
        try:
            response_lines = llm_response.split('\n')
        except:
            llm_response = llm_response.content
            response_lines = llm_response.split('\n')
        cleaned_lines = []
        
        # Skip lines that are explanations or headers
        skip_phrases = [
            "Here is the optimized prompt:",
            "Here's the optimized prompt:",
            "Optimized prompt:",
            "Final prompt:",
            "Generated prompt:",
            "Here is your final prompt:",
            "Here's your final prompt:",
            "This prompt is optimized",
            "The optimized prompt is:",
            "Final optimized prompt:",
            "Here is the highly detailed and optimized prompt:",
            "I'll create a highly detailed and optimized prompt",
            "Let me create a highly detailed and optimized prompt",
            "Based on the context",
            "Here's a detailed prompt",
            "Here is a prompt",
            "This prompt",
            "The prompt",
            "Here is your prompt",
            "Here's your prompt",
            "Final prompt for image generation:",
            "Optimized prompt for image generation:",
        ]
        
        # Process each line
        for line in response_lines:
            line = line.strip()
            
            # Skip if line contains skip phrases
            if any(phrase.lower() in line.lower() for phrase in skip_phrases):
                continue
            
            # Skip if line is too short or looks like a header
            if len(line) < 10:
                continue
            
            # Keep the line if it looks like actual prompt content
            cleaned_lines.append(line)
        
        # Join the cleaned lines
        final_prompt = '\n'.join(cleaned_lines).strip()
        
        # Additional cleanup - remove any remaining prefixes
        prefixes_to_remove = [
            "Prompt:", "Here is the prompt:", "Final prompt:",
            "Optimized prompt:"
        ]
        
        for prefix in prefixes_to_remove:
            if final_prompt.startswith(prefix):
                final_prompt = final_prompt[len(prefix):].strip()
        
        print("\n---------cleaned prompt---------\n", final_prompt)
        
        return {
            "cleaned_prompt": final_prompt,
            "original_length": len(llm_response),
            "cleaned_length": len(final_prompt),
            "lines_removed": len(response_lines) - len(cleaned_lines)
        }
        

    def chain(self):
        return (
            RunnableLambda(self.get_context) 
            | self.prompt 
            | self.llm
            | RunnableLambda(self.prompt_parser)
        )

    def generate_with_reference(self, prompt, img):
        print("---------prompt---------\n", prompt)
        result = self.client.images.edit(
            model="gpt-image-1",
            image=img,
            prompt=prompt,
            size="1024x1024"
        )
        return result.data[0].b64_json
        pass
    
    
    def invoke(self, inputs):
        return self.chain().invoke(inputs)