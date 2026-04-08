import time

def log_run(query, result, start_time):
    latency = time.time() - start_time
    log = {
        "query": query,
        "mode": result,
        "latency": latency
    }

    print(log)