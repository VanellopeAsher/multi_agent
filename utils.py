import os
import time
import base64
import jsonlines
from openai import OpenAI
from constants import PLATFORM, API_KEY_MAP, BASE_URL_MAP 
import random
import tenacity

class LLM:
    """
    The language model class for generating text using different platforms.

    Args:
        model_name (str): The name of the model.
        platform (PLATFORM): The platform to use.

    Attributes:
        model_name (str): The name of the model.
        platform (PLATFORM): The platform to use.
        client (openai.Client): The OpenAI client.
        completion_tokens (int): The number of tokens used for completion.
        prompt_tokens (int): The number of tokens used for prompt.

    Methods:
        generate: Generate text based on the prompt.
        clear_history: Clear the history of generated text.
        save_history: Save the history of generated text to a file.
    
    """
    def __init__(self, model_name: str, platform: PLATFORM = 'openai', api_key: str = None):
        """
        Initializes an instance of the class.

        Args:
            model_name (str): The name of the model.
            platform (PLATFORM, optional): The platform to use. Defaults to 'openai'.
            api_key (str, optional): The API key for the platform. Defaults to None (use the environment variable).
        """
        self.model_name = model_name
        self.platform = platform
        self.api_key = api_key
        
        self.client = self._init_client(platform)

        self.completion_tokens = 0
        self.prompt_tokens = 0
        self.history = []
        
        # Add retry configuration
        self.max_retries = 5
        self.min_delay = 1  # Minimum delay of 1 second
        self.max_delay = 30  # Maximum delay of 30 seconds

    def _init_client(self, platform: PLATFORM):
        """
        Initialize the OpenAI client with the API key and base URL.

        Args:
            platform (PLATFORM): The platform to use.
        
        Returns:
            openai.Client: The OpenAI client.
        """
        assert platform in API_KEY_MAP, f"Platform {platform} is not supported."
        if self.api_key:
            api_key = self.api_key
        else:
            api_key = os.environ.get(API_KEY_MAP[platform])
        assert api_key, f"API key for platform {platform} is not found."
        base_url = BASE_URL_MAP[platform]
        client = OpenAI(
            api_key=api_key, 
            base_url=base_url
        )
        return client

    @tenacity.retry(
        wait=tenacity.wait_random_exponential(min=1, max=30),
        stop=tenacity.stop_after_attempt(5),
        retry=tenacity.retry_if_exception_type((Exception))
    )
    def generate(
        self, 
        prompt: list[dict]|str, 
        model:str|None = None, 
        temperature: float = 1.0
    ):
        """
        Generate text based on the prompt with retry mechanism.

        Args:
            prompt (str): The prompt for the model.
            model (str|None): The model to use. If None, use the default model.
            temperature (float): The temperature for sampling.
        
        Returns:
            str: The generated text.
        """
        try:
            if model is None:
                model = self.model_name

            if isinstance(prompt, str):
                messages = [{'role': 'user', 'content': prompt}]
            else:
                messages = prompt 
            for m in messages:
                m.update({"time": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())})
                self.history.append(m)

            completion = self.client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature
            )
            self.completion_tokens += completion.usage.completion_tokens
            self.prompt_tokens += completion.usage.prompt_tokens

            response = completion.choices[0].message.content
            self.history.append({'role': 'assistant', 'content': response, 'time': time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())})

            # Add random delay after success to avoid frequent requests
            time.sleep(random.uniform(0.5, 1))

            return response

        except Exception as e:
            raise
    
    def get_vision_response(self, image_path: str, prompt: str, model: str = None):
        """
        Generate text based on the image and prompt.

        Args:
            image_path (str): The path to the image.
            prompt (str): The prompt for the model.
            model (str|None): The model to use. If None, use the default model.
        """
        # Function to encode the image
        def encode_image(image_path):
            with open(image_path, "rb") as image_file:
                return base64.b64encode(image_file.read()).decode("utf-8")
            
        # Using the default model if not specified
        if model is None:
            model = self.model_name
            
        # Getting the Base64 string
        base64_image = encode_image(image_path)

        # Getting the messages
        messages = [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{base64_image}",
                            },
                        },
                        {"type": "text", "text": prompt},
                    ],
                }
            ]
        completion = self.client.chat.completions.create(
            model=model,
            messages=messages
        )
        self.completion_tokens += completion.usage.completion_tokens
        self.prompt_tokens += completion.usage.prompt_tokens

        response = completion.choices[0].message.content
        self.history.append({'role': 'assistant', 'content': response, 'time': time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())})

        return response

    def clear_history(self):
        """
        Clear the history of generated text.
        """
        self.history = []

    def save_history(self, path: str):
        """
        Save the history of generated text to a file.

        Args:
            path (str): The path to save the history.
        """
        with jsonlines.open(path, 'w') as writer:
            for item in self.history:
                writer.write(item)

    def __repr__(self):
        return f"LLM(model_name={self.model_name}, platform={self.platform})"