import requests
import time
import json

class LlamaFileChat:
    def __init__(self, host='localhost', port=8080, api_key=None):
        self.llamafile_host = host
        self.llamafile_port = port
        self.api_key = api_key
        # Initialize the conversation with the system message
        self.conversation = [
            {
                "role": "system",
                "content": "You are a helpful assistant responding in a friendly, casual, and jokey tone."
            }
        ]

    def call_llamafile_api(self, messages, stream=True):
        try:
            llamafile_api_url = f"http://{self.llamafile_host}:{self.llamafile_port}/v1/chat/completions"
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key or 'no-key-required'}"
            }
            payload = {
                "model": "LLaMA_CPP",
                "messages": messages,
                "stream": stream
            }
            start_time = time.time()
            response = requests.post(llamafile_api_url, headers=headers, json=payload, stream=stream)
            
            if response.status_code != 200:
                print(f"Error: Received status code {response.status_code}")
                print(f"Response Content: {response.content.decode('utf-8')}")
                return None

            if stream:
                def generate():
                    first_chunk_time_recorded = False
                    for chunk in response.iter_content(chunk_size=None):
                        if chunk:
                            if not first_chunk_time_recorded:
                                first_chunk_time = time.time()
                                ttft = first_chunk_time - start_time
                                print(f"\nTime to First Token: {ttft:.2f} seconds\n")
                                first_chunk_time_recorded = True
                            decoded_chunk = chunk.decode('utf-8')
                            # Split the chunk into lines in case multiple data entries are in the chunk
                            lines = decoded_chunk.strip().split('\n')
                            for line in lines:
                                if line.startswith('data: '):
                                    data_str = line[len('data: '):]
                                    if data_str.strip() == '[DONE]':
                                        return
                                    try:
                                        data = json.loads(data_str)
                                        delta_content = data.get('choices', [{}])[0].get('delta', {}).get('content', '')
                                        yield delta_content
                                    except json.JSONDecodeError as e:
                                        print(f"JSONDecodeError: {e} - Line Content: {data_str}")
                                else:
                                    pass  # Ignore irrelevant lines
                return generate()
            else:
                response_json = response.json()
                ttft = time.time() - start_time
                print(f"\nTime to First Token: {ttft:.2f} seconds\n")
                return response_json.get('choices', [{}])[0].get('message', {}).get('content', '')
        except Exception as e:
            print(f"Error while calling Llamafile API: {e}")
            return None

def main():
    chat = LlamaFileChat()
    print("Welcome to Llamafile Chat! Type 'exit' to quit.\n")
    while True:
        user_input = input("You: ")
        if user_input.lower() in ['exit', 'quit']:
            print("Goodbye!")
            break
        chat.conversation.append({"role": "user", "content": user_input})
        print("Assistant:", end=' ', flush=True)
        response_generator = chat.call_llamafile_api(chat.conversation)
        if response_generator:
            assistant_response = ''
            for chunk in response_generator:
                print(chunk, end='', flush=True)
                assistant_response += chunk
            print()
            chat.conversation.append({"role": "assistant", "content": assistant_response})
        else:
            print("Failed to get a response from Llamafile API.")

if __name__ == "__main__":
    main()