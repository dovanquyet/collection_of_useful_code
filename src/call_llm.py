
import openai, together, google.generativeai as genai
import asyncio, time
from joblib import Memory

'''
This file provides wrapper functions to call LLMs from various sources

It supports:
- thinking models:
    for OpenAI API, use max_completion_tokens instead of max_tokens
        https://platform.openai.com/docs/api-reference/chat/create
    for other source, max_tokens is fine

- caching: use joblib
- async request: use asyncio, support parallel requests to accelerate the LLM inference
    for OpenAI API: use AsyncOpenAI
        https://community.openai.com/t/using-asynchronous-client-with-asyncopenai/624114
        https://gist.github.com/dafajon/fa3519b9f6b1eb9507aea30b38989dcd
        https://github.com/openai/openai-python/blob/main/examples/async_demo.py
    for TogetherAI API: similar to OpenAI, we can use AsyncTogether
        https://www.together.ai/blog/python-sdk-v1#:~:text=(embeddings)-,Async%20Support,-We%20now%20have

NOTE:
    AsyncOpenAI cannot be used with caching, but we can call openAI models with Together AI SDK!
    Thus, everything is merged to Together AI code!
    For caching, as joblib hashes all variables (even the client), but hash by value thus okay!
    IMPORTANT! OpenAI releases a new API named Responses, https://platform.openai.com/docs/guides/responses-vs-chat-completions
    Should use this one for agentic tasks!

    Avoid using system prompt!
    https://community.openai.com/t/is-role-system-content-you-are-a-helpful-assistant-redundant-in-chat-api-calls/191229
    https://docs.together.ai/docs/prompting-deepseek-r1#:~:text=No%20system%20prompt%3A%20Avoid%20adding%20a%20system%20prompt%3B%20all%20instructions%20should%20be%20contained%20within%20the%20user%20prompt.
'''


# Create a memory object, specifying the cache directory
cachedir = './cache_llm_call'
memory_llm = Memory(cachedir, verbose=0)
MAX_TRAILS = 3
ERROR_LOG = 'API Error'
SYSTEM_PROMPT = None # "You are a helpful assistant. Follow the instruction as closely as possible.",


# Helper func
def prepare_messages(prompt, system_prompt):
    if system_prompt is None:
        messages = [
            {"role": "user", "content": prompt},
        ]
    else:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
        ]
    return messages

# OpenAI models
@memory_llm.cache(ignore=['openai_client', 'timeout_failure'])
def call_openai_responses_api(
    prompt='',
    *,
    openai_client=openai,
    model_name="gpt-4.1-nano",
    system_prompt=SYSTEM_PROMPT,
    messages=None,
    max_tokens=64,
    timeout_failure=1,
    return_num_reasoning_tokens=False,
    return_all_metadata=False,
    **kwargs
):
    ''' Can replace `openai` in this function with `openai_client` defined with Together SDK
    openai_client = Together(
        api_key=userdata.get('OPENAI_API_KEY'),
        base_url="https://api.openai.com/v1/"
    )
    '''
    trails = MAX_TRAILS
    while trails > 0:
        trails -= 1
        try:
            if messages is None:
                messages = prepare_messages(prompt, system_prompt)
            response = openai_client.responses.create(
                model=model_name,
                max_output_tokens=max_tokens,
                input=messages,
                **kwargs
            )
            # output_text = response.output_text.strip()
            output_text = response.output[0].content[0].text.strip()
            if return_num_reasoning_tokens:
                num_reasoning_tokens = dict(response.usage.output_tokens_details).get('reasoning_tokens', -1)
                stop_reason = response.output[0].status
                return {
                    "num_reasoning_tokens": num_reasoning_tokens,
                    "stop_reason": stop_reason,
                    "output_text": output_text,
                }
            elif return_all_metadata:
                return response
            else:
                return output_text

        except Exception as e:
            print(e)
            time.sleep(timeout_failure)
    return ERROR_LOG

@memory_llm.cache(ignore=['openai_client', 'timeout_failure'])
def call_openai_api(
    prompt='',
    *,
    openai_client=openai,
    model_name="gpt-4.1-nano",
    system_prompt=SYSTEM_PROMPT,
    messages=None,
    max_tokens=64,
    timeout_failure=1,
    return_num_reasoning_tokens=False,
    return_all_metadata=False,
    **kwargs
):
    ''' Can replace `openai` in this function with `openai_client` defined with Together SDK
    openai_client = Together(
        api_key=userdata.get('OPENAI_API_KEY'),
        base_url="https://api.openai.com/v1/"
    )
    '''
    trails = MAX_TRAILS
    while trails > 0:
        trails -= 1
        try:
            if messages is None:
                messages = prepare_messages(prompt, system_prompt)
            completion = openai_client.chat.completions.create(
                model=model_name,
                max_completion_tokens=max_tokens,
                messages=messages,
                **kwargs
            )
            output_text = completion.choices[0].message.content.strip()
            if return_num_reasoning_tokens:
                num_reasoning_tokens = dict(completion.usage.completion_tokens_details).get('reasoning_tokens', -1)
                stop_reason = completion.choices[0].finish_reason.strip()
                return {
                    "num_reasoning_tokens": num_reasoning_tokens,
                    "stop_reason": stop_reason,
                    "output_text": output_text,
                }
            elif return_all_metadata:
                return completion
            else:
                return output_text
        except Exception as e:
            print(e)
            time.sleep(timeout_failure)
    return ERROR_LOG

@memory_llm.cache(ignore=['openai_async_client', 'timeout_failure'])
async def call_openai_api_async(
    prompts=[],
    *,
    openai_async_client=None,
    model_name="gpt-4.1-nano",
    system_prompt=SYSTEM_PROMPT,
    list_of_messages=None,
    max_tokens=64,
    timeout_failure=1,
    return_num_reasoning_tokens=False,
    return_all_metadata=False,
    **kwargs
):
    '''Calls the OpenAI API asynchronously.
    Only allow multiple inputs as a list. For single input, use call_openai_api instead.

    Examples:
    await call_openai_api_async(
        prompts=sentences,
        openai_async_client=openai_async_client,
        model_name='gpt-4o-mini',
        max_tokens=20
    )

    Note:
        Have to use AsyncOpenAI client: openai_async_client = AsyncOpenAI(api_key=userdata.get('OPENAI_API_KEY'))
        Error log when using await on ChatCompletion: "object ChatCompletion can't be used in 'await' expression"

    Since AsyncOpenAI does not support cache, we use Together AI SDK for OpenAI models
    openai_async_client = AsyncTogether(
        api_key=userdata.get('OPENAI_API_KEY'),
        base_url="https://api.openai.com/v1/"
    )
    '''
    async def call_single_async(prompt='', messages=None):
        trails = MAX_TRAILS
        while trails > 0:
            trails -= 1
            try:
                if messages is None:
                    messages = prepare_messages(prompt, system_prompt)
                completion = await openai_async_client.chat.completions.create(
                    model=model_name,
                    max_completion_tokens=max_tokens,
                    messages=messages,
                    **kwargs
                )
                output_text = completion.choices[0].message.content.strip()
                if return_num_reasoning_tokens:
                    num_reasoning_tokens = dict(completion.usage.completion_tokens_details).get('reasoning_tokens', -1)
                    stop_reason = completion.choices[0].finish_reason.strip()
                    return {
                        "num_reasoning_tokens": num_reasoning_tokens,
                        "stop_reason": stop_reason,
                        "output_text": output_text,
                    }
                elif return_all_metadata:
                    return completion
                else:
                    return output_text
            except Exception as e:
                print(e)
                time.sleep(timeout_failure)
        return ERROR_LOG

    if list_of_messages is None:
        responses = await asyncio.gather(*[call_single_async(prompt=p) for p in prompts])
    else:
        responses = await asyncio.gather(*[call_single_async(messages=m) for m in list_of_messages])

    return responses


# Opensource models
@memory_llm.cache(ignore=['together_client', 'timeout_failure'])
def call_together_api(
    prompt='',
    *,
    together_client=None,
    model_name='meta-llama/Llama-3.2-3B-Instruct-Turbo',
    system_prompt=SYSTEM_PROMPT,
    messages=None,
    max_tokens=64,
    timeout_failure=1,
    **kwargs
):
    trails = MAX_TRAILS
    while trails > 0:
        trails -= 1
        try:
            if messages is None:
                messages = prepare_messages(prompt, system_prompt)
            return together_client.chat.completions.create(
                model=model_name,
                max_tokens=max_tokens,
                messages=messages,
                **kwargs
            ).choices[0].message.content.strip()
        except Exception as e:
            print(e)
            time.sleep(timeout_failure)
    return ERROR_LOG

@memory_llm.cache(ignore=['together_async_client', 'timeout_failure'])
async def call_together_api_async(
    prompts=[],
    *,
    together_async_client=None,
    model_name='meta-llama/Llama-3.2-3B-Instruct-Turbo',
    system_prompt=SYSTEM_PROMPT,
    list_of_messages=None,
    max_tokens=64,
    timeout_failure=1,
    **kwargs
):
    '''Calls the TogetherAI API asynchronously.
    Only allow multiple inputs as a list. For single input, use call_together_api instead.
    https://www.together.ai/blog/python-sdk-v1
    '''
    async def call_single_async(prompt='', messages=None):
        trails = MAX_TRAILS
        while trails > 0:
            trails -= 1
            try:
                if messages is None:
                    messages = prepare_messages(prompt, system_prompt)
                response = await together_async_client.chat.completions.create(
                    model=model_name,
                    max_tokens=max_tokens,
                    messages=messages,
                    **kwargs
                )
                return response.choices[0].message.content.strip()
            except Exception as e:
                print(e)
                time.sleep(timeout_failure)
        return ERROR_LOG

    if list_of_messages is None:
        responses = await asyncio.gather(*[call_single_async(prompt=p) for p in prompts])
    else:
        responses = await asyncio.gather(*[call_single_async(messages=m) for m in list_of_messages])
    return responses


# Google models
@memory_llm.cache(ignore=['timeout_failure'])
def call_googleai_api(
    prompt,
    *,
    model_name='gemini-1.5-flash',
    temperature=0.7,
    max_tokens=64,
    timeout_failure=60
):
    model = genai.GenerativeModel(model_name)
    while True:
        try:
            response = model.generate_content(
                prompt,
                generation_config=genai.types.GenerationConfig(
                    candidate_count=1, # Only one candidate for now.
                    # stop_sequences=['x'],
                    max_output_tokens=max_tokens,
                    temperature=0.0),
                safety_settings={
                    'harassment':'block_none',
                    'hate_speech':'block_none',
                    'sexual':'block_none',
                    'dangerous':'block_none'
                }
            )
            # print(response.prompt_feedback)
            return response.parts[0].text
        except Exception as e:
            try:
                response = model.generate_content(prompt,
                    safety_settings={
                        'harassment':'block_none',
                        'hate_speech':'block_none',
                        'sexual':'block_none',
                        'dangerous':'block_none'
                    }
                )
                if len(response.candidates) > 0:
                    return response.parts[0].text
                else:
                    return ' '
            except Exception as e:
                print(e)
                time.sleep(timeout_failure)

#---# Usage
# output = await call_openai_api_async(
#     prompts=['Write a 4-line poem'],
#     openai_async_client=openai_async_client,
#     max_tokens=1024,
#     model_name='o4-mini',
#     reasoning_effort='high'
# )

# output = call_openai_api(
#     prompt='Write a 4-line poem',
#     max_tokens=1024,
#     model_name='o4-mini',
#     reasoning_effort='high'
# )

# output = await call_openai_api_async(
#     prompts=['Write a 4-line poem.'],
#     openai_async_client=openai_async_client,
#     max_tokens=128,
#     model_name='gpt-4.1-nano',
#     return_num_reasoning_tokens=True
# )

# output = await call_together_api_async(
#     prompts=['Write a 4-line poem'],
#     together_async_client=together_async_client,
#     max_tokens=128,
# )

# output = call_together_api(
#     prompt='Write a 4-line poem',
#     together_client=together_client,
#     max_tokens=128,
# )

# output

# !pip install -qU openai together google-generativeai
from call_llm import *
from call_llm import (
    call_openai_api,
    call_openai_api_async,
    call_together_api,
    call_together_api_async,
    ERROR_LOG as API_ERROR_LOG,
)
import openai, together, google.generativeai as genai, asyncio
from openai import AsyncOpenAI, OpenAI
from together import Together, AsyncTogether
from google.colab import userdata

## openai
openai.api_key = userdata.get('OPENAI_API_KEY')
openai_client = OpenAI(api_key=userdata.get('OPENAI_API_KEY'))
openai_async_client = AsyncTogether(
    api_key=userdata.get('OPENAI_API_KEY'),
    base_url="https://api.openai.com/v1/"
)
call_openai_api_async_wo_client = call_openai_api_async
call_openai_api_async = lambda **kwargs: call_openai_api_async_wo_client(openai_async_client=openai_async_client, **kwargs)
# openai_async_client = AsyncOpenAI(api_key=userdata.get('OPENAI_API_KEY'))
# model_name = "gpt-4o-2024-08-06" #@param ["gpt-4o-2024-08-06", "gpt-4o-mini", "gpt-3.5-turbo"]

## togetherai
together_client = Together(api_key=userdata.get('TOGETHER_API_KEY'))
together_async_client = AsyncTogether(api_key=userdata.get('TOGETHER_API_KEY'))
call_together_api_wo_client = call_together_api
call_together_api = lambda **kwargs: call_together_api_wo_client(together_client=together_client, **kwargs)
call_together_api_async_wo_client = call_together_api_async
call_together_api_async = lambda **kwargs: call_together_api_async_wo_client(together_async_client=together_async_client, **kwargs)
# model_name = "meta-llama/Meta-Llama-3.1-405B-Instruct-Turbo" #@param ["meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo", "meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo", "meta-llama/Meta-Llama-3.1-405B-Instruct-Turbo", "meta-llama/Llama-3.2-3B-Instruct-Turbo"]

## googleai
# genai.configure(api_key=userdata.get('GOOGLE_GENAI_KEY'))

## demo run
# print(call_openai_api(prompt='Write a short poem', model_name='gpt-4o-mini'), end='\n\n')
# print(call_together_api(prompt='Write a short poem'), end='\n\n')