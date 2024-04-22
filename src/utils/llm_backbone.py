import logging
import time
from openai import OpenAI
from litellm import completion, embedding, batch_completion # import litellm for calling multiple llms using the same input/output format


class LLM_Backbone():
   def __init__(self, args):
      self.client = OpenAI()
      self.embedding_model = args.embedding_model
      self.completion_model = args.model_name
      self.max_attempt = 5 # number of attempts to get the completion
      
   def get_embeddings(self, texts: list):
      embeddings = []
      texts_per_batch = 2000
      text_chunks = [texts[i:i + texts_per_batch] for i in range(0, len(texts), texts_per_batch)]
      
      attempt = 0
      while attempt < self.max_attempt:
         try:
            for chunk in text_chunks:
               chunk_embeddings = self.client.embeddings.create(
                  model=self.embedding_model, 
                  input=chunk
                  ) # return [item['embedding'] for item in _['data']]
               embeddings.extend([item.embedding for item in chunk_embeddings.data])
            return embeddings
         except Exception as e:
            logging.error(f"Error occurred: {e}")
            attempt += 1
            time.sleep(1)
            
   def get_completion(self, prompt: dict):
      messages = [
         {"role": "system", "content": prompt["system"]},
         *prompt["examples"],
         {"role": "user", "content": prompt["prompt"]}
      ]
         
      attempt = 0
      while attempt < self.max_attempt:
         try:
               _ = self.client.chat.completions.create(
                  model=self.completion_model, 
                  messages=messages, 
                  temperature=0, 
                  top_p=0, 
                  logprobs=False
                  )
               # return _['choices'][0]['message']['content']
               return _.choices[0].message.content
         except Exception as e:
               logging.error(f"Error occurred: {e}")
               attempt += 1
               time.sleep(1)
               
   def get_log_probs(self, log_probs: list):
      scores = []
      for item in log_probs:
        top_logprobs = item[0]["top_logprobs"]
        match = False
        for i in range(len(top_logprobs)):
            if top_logprobs[i]["token"] in [" A", "A", "A "]:
                scores.append(top_logprobs[i]["logprob"])
                match = True
                break
        if not match:
            scores.append(-10000.0)
      return scores
   
   def get_batch_completion(self, prompt: dict, input_batch: list):
      """
      for item in log_probs:
         if item["token"] == "A":
               print(item['logprob'])
      """
      
      messages = []
      for item in input_batch:
         messages.append(
               [
                  {"role": "system", "content": prompt["system"]},
                  *prompt["examples"],
                  {"role": "user", "content": item}
               ]
         )
      attempt = 0
      while attempt < 5: 
         try:
               _ = batch_completion(
                  model=self.completion_model, 
                  messages=messages, 
                  temperature=0, 
                  top_p=0, 
                  logprobs=True,
                  top_logprobs=5
                  )
               contents = [_[i]['choices'][0]['message']['content'] for i in range(len(_))]
               log_probs = [_[i]['choices'][0]['logprobs']['content'] for i in range(len(_))]
               return contents, log_probs
               
         except Exception as e:
               logging.error(f"Error occurred: {e}")
               attempt += 1
               time.sleep(1)
               
               
async def get_embedding(session, texts, model="text-embedding-3-small"):
    api_url = f"https://api.openai.com/v1/embeddings"
    headers = {
        "Authorization": f"Bearer {config['OPENAI_API_KEY']}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": model,
        "input": texts
    }

    async with session.post(api_url, headers=headers, json=payload) as response:
        if response.status == 200:
            response_data = await response.json()
            return [item['embedding'] for item in response_data['data']]
        else:
            return None


async def query_api(session, args, prompt):
    api_url = f"https://api.openai.com/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {config['OPENAI_API_KEY']}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": args.model_name,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0,
    }
    
    attempt = 0
    while attempt < 3: # retry 3 times if exception occurs
        try: 
            async with session.post(api_url, headers=headers, json=payload) as response:
                response_data = await response.json()
                response_content = response_data['choices'][0]['message']['content']
                logging.info(f"PROMPT: {prompt}")
                logging.info("===" * 50)
                logging.info(f"RECEIVED RESPONSE: {response_content}")
                return {"prompt": prompt, "response": response_content}
        except Exception as e:
            logging.error(f"Error occurred: {e}")
            attempt += 1
            await asyncio.sleep(1)
    
    logging.error(f"Failed to get response for query {prompt} after 3 attempts")
    raise APIQueryError("Failed to get a valid response after all retries.")
