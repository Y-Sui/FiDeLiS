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
