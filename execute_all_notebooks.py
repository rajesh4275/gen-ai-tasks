import os
import sys
import glob
import time
import asyncio
import nbformat
from nbconvert.preprocessors import ExecutePreprocessor

if sys.platform == "win32":
    try:
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    except Exception:
        pass

# Ensure .venv\Scripts is on PATH
repo_dir = os.path.abspath(os.path.dirname(__file__))
venv_scripts = os.path.join(repo_dir, ".venv", "Scripts")
os.environ["PATH"] = venv_scripts + os.pathsep + os.environ.get("PATH", "")

notebooks = [
    "ChromaDB_CRUD.ipynb",
    "Cosine_Similarity_Mathematically.ipynb",
    "HuggingFace_Embeddings_Scratch_to_Intermediate.ipynb",
    "Intro_to_Vector_Databases_ChromaDB.ipynb",
    "RAG-Document-Loading-and-Chunking.ipynb",
    "word_embeddings.ipynb"
]

results = {}

for nb_name in notebooks:
    nb_path = os.path.join(repo_dir, nb_name)
    if not os.path.exists(nb_path):
        print(f"Skipping {nb_name} (not found)")
        continue
    
    print("\n" + "=" * 60)
    print(f"Executing: {nb_name}")
    print("=" * 60)
    start_time = time.time()
    
    with open(nb_path, "r", encoding="utf-8") as f:
        nb = nbformat.read(f, as_version=4)
        
    ep = ExecutePreprocessor(timeout=900, kernel_name="gen-ai-tasks")
    success = False
    err_msg = ""
    
    try:
        ep.preprocess(nb, {"metadata": {"path": repo_dir}})
        success = True
        elapsed = time.time() - start_time
        print(f"--> Succeeded in {elapsed:.2f} seconds")
    except Exception as e:
        err_msg = str(e)
        elapsed = time.time() - start_time
        print(f"--> FAILED in {elapsed:.2f} seconds with error:\n{err_msg}")
    
    # Save notebook (with all executed cells/outputs up to failure or completion)
    with open(nb_path, "w", encoding="utf-8") as f:
        nbformat.write(nb, f)
    print(f"--> Saved updated notebook to {nb_name}")
    
    # Count stats
    total_code = sum(1 for c in nb.cells if c.cell_type == "code")
    with_outputs = sum(1 for c in nb.cells if c.cell_type == "code" and c.outputs)
    
    results[nb_name] = {
        "success": success,
        "elapsed_sec": round(elapsed, 2),
        "total_code_cells": total_code,
        "cells_with_output": with_outputs,
        "error": err_msg
    }

print("\n" + "=" * 60)
print("EXECUTION SUMMARY")
print("=" * 60)
for nb_name, res in results.items():
    status = "SUCCESS" if res["success"] else "FAILED"
    print(f"[{status}] {nb_name} | {res['cells_with_output']}/{res['total_code_cells']} code cells produced output ({res['elapsed_sec']}s)")
    if not res["success"]:
        print(f"    Error: {res['error'][:200]}...")
