from litellm import completion
import os

class LLM:

    def __init__ (self, API_KEY): 
        os.environ["COHERE_API_KEY"] = API_KEY
    
    def generate_answer(self, prompt, temperature=0.5, max_tokens=1000):
        response = completion(
            model="command-r-plus",
            messages=[{"content": prompt, "role": "user"}],
            temperature=temperature,
            max_tokens=max_tokens
        )

        return response.choices[0].message.content
    

# llm = LLM('3uTKE448T558Qmem6pBSbvW54nHBR4FP6Xnn6jCl')
# print(llm.generate_answer(""))