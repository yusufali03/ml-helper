# main.py

import argparse
from multiprocessing import Process, Pipe
from cli import parse_task
from plugin_factory import get_plugin
from logger import log_task

def worker(task_type, params, conn):
    try:
        plugin = get_plugin(task_type)
        result = plugin.run(params)
        conn.send({"status": "success", "result": result})
    except Exception as e:
        conn.send({"status": "error", "error": str(e)})
    finally:
        conn.close()

def execute_task(task_str: str):
    task_type, params = parse_task(task_str)
    parent_conn, child_conn = Pipe()
    p = Process(target=worker, args=(task_type, params, child_conn))
    p.start()
    outcome = parent_conn.recv()
    p.join()
    return task_type, params, outcome

def main():
    parser = argparse.ArgumentParser(
        description="ml-helper: console assistant for ML tasks"
    )
    parser.add_argument(
        "task",
        nargs="+",
        help="Task command, e.g. build ROC curve from metrics.json"
    )
    args = parser.parse_args()
    task_str = " ".join(args.task)

    try:
        task_type, params, outcome = execute_task(task_str)

        if outcome["status"] == "success":
            result = outcome["result"]
            print(f"[Done] {result}")
            log_task(task_type, params, result.get("status", "success"), result.get("details", ""))
        else:
            error_msg = outcome["error"]
            print(f"[Error] {error_msg}")
            log_task(task_str, params, "error", error_msg)

    except Exception as e:
        # Fallback for parsing errors, etc.
        print(f"[Error] {e}")
        log_task(task_str, {}, "error", str(e))

if __name__ == "__main__":
    main()
