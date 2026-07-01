import asyncio
import contextvars
from concurrent.futures import ThreadPoolExecutor

cv = contextvars.ContextVar("cv", default=None)

def sync_task(name):
    val = cv.get()
    print(f"[{name}] ContextVar in thread: {val}")
    return val

async def main():
    cv.set("hello_world")
    print(f"[Main] ContextVar set to: {cv.get()}")
    
    # 1. Using asyncio.to_thread (uses ThreadPoolExecutor under the hood in Python 3.9+)
    val = await asyncio.to_thread(sync_task, "asyncio.to_thread")
    
    # 2. Using loop.run_in_executor
    loop = asyncio.get_running_loop()
    with ThreadPoolExecutor() as pool:
        val2 = await loop.run_in_executor(pool, sync_task, "run_in_executor")

if __name__ == "__main__":
    asyncio.run(main())
