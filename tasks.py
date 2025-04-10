import time
import ollama
from benchmarks import log_info, resource_usage, evaluate

# 1) Get each response from the 3 different models
def get_response(model, question):
    memory_before, cpu_before = resource_usage()
    start_time = time.time()
    response = ollama.chat(
        model = model,
        messages=[
            {'role': 'user', 'content': question}
        ]
    )
    end_time = time.time()
    memory_after, cpu_after = resource_usage()
    
    result_time = end_time - start_time
    avg_memory = (memory_before + memory_after) / 2
    avg_cpu = (cpu_before + cpu_after) / 2
    
    evaluation = evaluate(response['message']['content'], question, "Q&A", model)
    log_info(model, question, response, "Q&A", result_time, avg_memory, avg_cpu, evaluation)
    return response['message']['content']

# 2) Text Summarization
# Method to process and produce a summary response for the models
def text_summarization(model, text):
    memory_before, cpu_before = resource_usage()
    start_time = time.time()
    prompt = (
        f"Summarize the following text about Jewish delis in NYC, focusing on their history, culture, and current challenges:\n\n{text}\n\nSummary:"
    )
    response = ollama.chat(
        model = model,
        messages=[
            {'role': 'user', 'content': prompt}
        ]
    )
    end_time = time.time()
    memory_after, cpu_after = resource_usage()
    
    result_time = end_time - start_time
    avg_memory = (memory_before + memory_after) / 2
    avg_cpu = (cpu_before + cpu_after) / 2
    
    evaluation = evaluate(response['message']['content'], prompt, "Summarization", model)
    log_info(model, prompt, response, "Summarization", result_time, avg_memory, avg_cpu, evaluation)
    return response['message']['content']

# 3) Simple Code Generation
def code_generator(model, user_input):
    memory_before, cpu_before = resource_usage()
    start_time = time.time()
    prompt = (
        f"Generate code based off of the prompt:\n\n{user_input}\n\nCode:"
    )
    response = ollama.chat(
        model = model,
        messages=[
            {'role': 'user', 'content': prompt}
        ]
    )
    end_time = time.time()
    memory_after, cpu_after = resource_usage()
    
    result_time = end_time - start_time
    avg_memory = (memory_before + memory_after) / 2
    avg_cpu = (cpu_before + cpu_after) / 2
    
    evaluation = evaluate(response['message']['content'], prompt, "Code Generator", model)
    log_info(model, prompt, response, "Code Generator", result_time, avg_memory, avg_cpu, evaluation)
    return response['message']['content']

# 4) Creative Writing
def writer(model, story_prompt):
    memory_before, cpu_before = resource_usage()
    start_time = time.time()
    prompt = (
        f"Generate a short story based off of a unique prompt:\n\n{story_prompt}\n\nStory:"
    )
    response = ollama.chat(
        model = model,
        messages=[
            {'role': 'user', 'content': prompt}
        ]
    )
    end_time = time.time()
    memory_after, cpu_after = resource_usage()
    
    result_time = end_time - start_time
    avg_memory = (memory_before + memory_after) / 2
    avg_cpu = (cpu_before + cpu_after) / 2
    
    evaluation = evaluate(response['message']['content'], prompt, "Writer", model)
    log_info(model, prompt, response, "Writer", result_time, avg_memory, avg_cpu, evaluation)
    return response['message']['content']

# 5) Multilingual Capabilities
def multilingual(model, lang_prompt):
    memory_before, cpu_before = resource_usage()
    start_time = time.time()
    prompt = (
        f"Generate paragraphs in 3 SPECIFIED languages and translate them all to English. Show both original language and translation to English:\n\n{lang_prompt}\n\nLanguage:"
    )
    response = ollama.chat(
        model = model,
        messages=[
            {'role': 'user', 'content': prompt}
        ]
    )
    end_time = time.time()
    memory_after, cpu_after = resource_usage()
    
    result_time = end_time - start_time
    avg_memory = (memory_before + memory_after) / 2
    avg_cpu = (cpu_before + cpu_after) / 2
    
    evaluation = evaluate(response['message']['content'], prompt, "Multilingual", model)
    log_info(model, prompt, response, "Multilingual", result_time, avg_memory, avg_cpu, evaluation)
    return response['message']['content']