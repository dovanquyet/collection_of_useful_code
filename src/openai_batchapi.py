import sys, os, asyncio
import re, time, copy, json, random, datetime
import numpy as np
import pandas as pd
from tqdm import tqdm
from typing import Callable, Union, Literal, List
from joblib import Memory

# create a memory object, specifying the cache directory
# cachedir = 'data/backtran-cache'
cachedir = './cachedir'
memory = Memory(cachedir, verbose=0)

# work with **kwargs
# @memory.cache()
# def sum(a, b, **kwargs):
#     time.sleep(10)
#     return a + b
# sum(1,2,c=3)

def sleep_with_tqdm(seconds, progress_bar=True):
    desc = f"Sleeping {seconds} seconds ..."
    if progress_bar:
        for i in tqdm(range(seconds), desc=desc):
            time.sleep(1)
    else:
        print(desc)
        time.sleep(seconds)

def get_current_time():
    return datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")


# https://platform.openai.com/docs/guides/batch
def form_batch_api_request(key1, model_name, messages, max_tokens, **kwargs):
    # return json.dumps({"custom_id": key1, "method": "POST", "url": "/v1/chat/completions", "body": {"model": model_name, "messages": messages, "max_completion_tokens": max_tokens}})
    return json.dumps({
        "custom_id": key1,
        "method": "POST",
        "url": "/v1/chat/completions",
        "body": {
            "model": model_name,
            "messages": messages,
            "max_completion_tokens": max_tokens,
            **kwargs
        }
    })

# form_batch_api_request('k1', 'o3', [{'role': 'user', 'content': 'Hello'}], 64, reasoning_effort='low')
# {"custom_id": "k1", "method": "POST", "url": "/v1/chat/completions", "body": {"model": "o3", "messages": [{"role": "user", "content": "Hello"}], "max_completion_tokens": 64, "reasoning_effort": "low"}}
# {"custom_id": "request-1", "method": "POST", "url": "/v1/chat/completions", "body": {"model": "gpt-3.5-turbo-0125", "messages": [{"role": "system", "content": "You are a helpful assistant."},{"role": "user", "content": "Hello world!"}],"max_tokens": 1000}}
# {"custom_id": "request-2", "method": "POST", "url": "/v1/chat/completions", "body": {"model": "gpt-3.5-turbo-0125", "messages": [{"role": "system", "content": "You are an unhelpful assistant."},{"role": "user", "content": "Hello world!"}],"max_tokens": 1000}}

#---#
def get_output_of_batch_api_request(
    key1,
    response,
    return_num_reasoning_tokens=False,
    return_all_metadata=False,
):
    response = json.loads(response)
    ord = response['custom_id']
    if ord.startswith(key1):
        ord = int(ord[len(key1):])
        response = response['response']['body']
        output_text = response['choices'][0]['message']['content']
        if return_num_reasoning_tokens:
            num_reasoning_tokens = response['usage']['completion_tokens_details'].get('reasoning_tokens', -1)
            stop_reason = response['choices'][0]['finish_reason']
            return {
                "ord": ord,
                "num_reasoning_tokens": num_reasoning_tokens,
                "stop_reason": stop_reason,
                "output_text": output_text,
            }
        else:
            return {
                'ord': ord,
                'output_text': output_text,
            }
    else:
        print(f'custom_id {ord} does not match! Return None')
        return None

# , "finish_reason": "stop"}], "usage": {"prompt_tokens": 60054, "completion_tokens": 26712, "total_tokens": 86766, "prompt_tokens_details": {"cached_tokens": 0, "audio_tokens": 0}, "completion_tokens_details": {"reasoning_tokens": 17984, "audio_tokens": 0, "accepted_prediction_tokens": 0, "rejected_prediction_tokens": 0}}, "service_tier": "default", "system_fingerprint": "fp_8b28473bdb"}}, "error": null}
# {"id": "batch_req_123", "custom_id": "request-2", "response": {"status_code": 200, "request_id": "req_123", "body": {"id": "chatcmpl-123", "object": "chat.completion", "created": 1711652795, "model": "gpt-3.5-turbo-0125", "choices": [{"index": 0, "message": {"role": "assistant", "content": "Hello."}, "logprobs": null, "finish_reason": "stop"}], "usage": {"prompt_tokens": 22, "completion_tokens": 2, "total_tokens": 24}, "system_fingerprint": "fp_123"}}, "error": null}
# {"id": "batch_req_456", "custom_id": "request-1", "response": {"status_code": 200, "request_id": "req_789", "body": {"id": "chatcmpl-abc", "object": "chat.completion", "created": 1711652789, "model": "gpt-3.5-turbo-0125", "choices": [{"index": 0, "message": {"role": "assistant", "content": "Hello! How can I assist you today?"}, "logprobs": null, "finish_reason": "stop"}], "usage": {"prompt_tokens": 20, "completion_tokens": 9, "total_tokens": 29}, "system_fingerprint": "fp_3ba"}}, "error": null}



#---#
print(f'Using Batch API for {model_name}')
list_of_summaries = [{} for _ in range(len(llm_messages_all_calls))]
request_strings = []
for ord, messages in enumerate(llm_messages_all_calls):
    request_strings.append(form_batch_api_request(key1+str(ord), model_name, messages, max_tokens))

batch_api_dir = 'data/generation-and-analysis/batch-api/'
request_file_name = f'request-{key1}.jsonl'
batch_file_name = f'batch-{key1}.json'
response_file_name = f'response-{key1}.jsonl'
if request_file_name not in os.listdir(batch_api_dir):
    with open(batch_api_dir+request_file_name, 'w') as f:
        f.write('\n'.join(request_strings))
        f.close()

batch_input_file = openai_client.files.create(
    file=open(batch_api_dir+request_file_name, "rb"),
    purpose="batch"
)
if batch_file_name not in os.listdir(batch_api_dir):
    print(f'Creating Batch {key1}')
    batch = openai_client.batches.create(
        input_file_id=batch_input_file.id,
        endpoint="/v1/chat/completions",
        completion_window="24h",
        metadata={
            "description": key1
        }
    )
    batch = dict(batch)
    batch.pop('request_counts')
    json.dump(batch, open(batch_api_dir+batch_file_name, 'w'))
else:
    print(f'Retrieving Batch {key1}')
    batch = json.load(open(batch_api_dir+batch_file_name, 'r'))
    batch = openai_client.batches.retrieve(batch['id'])
    status = batch.status
    print(key1, status, batch)
    if batch.finalizing_at == None:
        print(f'In progress {(round(time.time()) - batch.in_progress_at)//60} mins')
    else:
        print(f'Completed after {(batch.finalizing_at - batch.in_progress_at)//60} mins')
        if batch.error_file_id != None:
            error_file = openai_client.files.content(batch.error_file_id)
            print(error_file.text)
        if batch.output_file_id != None:
            file_response = openai_client.files.content(batch.output_file_id)
            with open(batch_api_dir+response_file_name, 'w') as f:
                json.dump(file_response.text, f)
                f.close()
            for response in file_response.text.splitlines():
                output = get_output_of_batch_api_request(
                    key1=key1,
                    response=response,
                    return_num_reasoning_tokens=return_num_reasoning_tokens
                )
                if output:
                    if return_num_reasoning_tokens:
                        list_of_summaries[output['ord']] = output
                    else:
                        list_of_summaries[output['ord']] = output['output_text']
    batch = dict(batch)
    batch.pop('request_counts')
    json.dump(batch, open(batch_api_dir+batch_file_name, 'w'))