import time
import psutil
import ollama

# Export prompts and outputs to .txt file
log = []

def log_info(model, prompt, response, task, result_time, memory, cpu, evaluation):
    output = response['message']['content']
    log.append({
        "model": model,
        "task": task,
        "prompt": prompt,
        "output": output,
        "result_time": result_time,
        "memory": memory,
        "cpu": cpu,
        "evaluation": evaluation
    })

# Measure resources of computer during active Ollama session
def resource_usage():
    process = psutil.Process()
    memory = process.memory_info().rss / (1024*1024)
    
    cpu = psutil.cpu_percent(interval=0.1)
    
    # Measure CPU usage within Ollama execution
    for func in psutil.process_iter(['pid', 'name']):
        if 'ollama' in func.info['name'].lower():
            try:
                ollama_exec = psutil.Process(func.info['pid'])
                ollama_cpu = ollama_exec.cpu_percent(interval=0)
                cpu = (cpu + ollama_cpu) / 2
                break
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass
    
    return memory, cpu

# Evaluate quality of response via LLM
def evaluate(response, prompt, task, model):
    eval_prompt = (
        f"As a grader, rate the following {task} response for its quality and accuracy of truthful information."
        f"Prompt: {prompt}\n\nResponse: {response}\n\nProvide a letter grade from F to A and a explanation behind decision."
        f"Only grade on the F to A scale and do not use numerical values. Please grade accordingly like a human."
    )
    eval_response = ollama.chat(
        model = model,
        messages=[
            {'role': 'user', 'content': eval_prompt}
        ]
    )
    return eval_response['message']['content']

def write_log(filename="ollama_log.txt"):
# Write info from functions to the file
    with open("ollama_log.txt", "w", encoding = "utf-8") as f:
        for entry in log:
            f.write(f"Model: {entry['model']}\n")
            f.write(f"Task: {entry['task']}\n")
            f.write(f"Prompt: {entry['prompt']}\n")
            f.write(f"Output: {entry['output']}\n")
            f.write(f"Time (s): {entry['result_time']:.2f}\n")
            f.write(f"Memory Usage: {entry['memory']:.2f}\n")
            f.write(f"CPU Usage: {entry['cpu']:.2f}\n\n")
            f.write(f"Evaluation: {entry['evaluation']}\n")
            f.write(f"="*60 + "\n\n")