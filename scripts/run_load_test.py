import asyncio
import time
import statistics
import sys
import httpx

# Pool of test queries derived from actual IFSCA regulatory documents
QUERY_POOL = [
    "What is the minimum capital for an IFSC Banking Unit (IBU)?",
    "What does the term 'acceptance of deposits' mean under IFSCA?",
    "Can an IBU accept deposits from Indian residents?",
    "What is the Liquidity Coverage Ratio for a Banking Unit?",
    "What is considered an international financial services centre product under the IFSCA Act?",
    "What constitutes a high-risk jurisdiction under the Global In-House rules?",
    "What is the processing timeframe for a TechFin application?",
    "Are actions taken before the repeal of previous regulations still valid?",
    "What is the definition of a subsidiary company under the Companies Act?",
    "What are the permitted TechFin services under the framework?"
]

async def simulate_user(client: httpx.AsyncClient, user_id: int, query: str):
    url = "http://localhost:8000/api/qa"
    params = {"query": query}
    
    start_time = time.perf_counter()
    ttft = None
    total_time = None
    success = False
    tokens_received = 0
    
    try:
        async with client.stream("GET", url, params=params, timeout=60.0) as response:
            if response.status_code != 200:
                print(f"User {user_id} failed with status code {response.status_code}")
                return False, None, None
            
            async for line in response.aiter_lines():
                # Detect the first token
                if line.startswith("event: token") and ttft is None:
                    ttft = (time.perf_counter() - start_time) * 1000.0 # convert to ms
                
                if line.startswith("data:"):
                    tokens_received += 1
                
                # Stream completes on 'done' or end of stream
                if line.startswith("event: done"):
                    success = True
                    break
            
            total_time = (time.perf_counter() - start_time) * 1000.0 # convert to ms
            if ttft is None:
                # Fallback if stream was empty or failed early
                ttft = total_time
                
            success = True
    except Exception as e:
        print(f"User {user_id} request error: {str(e)}")
        
    return success, ttft, total_time

async def run_load_test(concurrency_level: int):
    print(f"\n--- Running Load Test with Concurrency Level: {concurrency_level} ---")
    
    async with httpx.AsyncClient() as client:
        tasks = []
        for i in range(concurrency_level):
            # Pick a query from the pool
            query = QUERY_POOL[i % len(QUERY_POOL)]
            tasks.append(simulate_user(client, i + 1, query))
            
        start_test = time.perf_counter()
        results = await asyncio.gather(*tasks)
        end_test = time.perf_counter()
        
        # Process results
        successes = [r for r in results if r[0]]
        ttfts = [r[1] for r in results if r[0] and r[1] is not None]
        total_times = [r[2] for r in results if r[0] and r[2] is not None]
        
        success_rate = (len(successes) / concurrency_level) * 100
        print(f"Total Test Duration: {end_test - start_test:.2f}s")
        print(f"Success Rate: {success_rate:.1f}% ({len(successes)}/{concurrency_level})")
        
        if ttfts:
            print("\nTime to First Token (TTFT) in ms:")
            print(f"  Min:  {min(ttfts):.2f}ms")
            print(f"  Mean: {statistics.mean(ttfts):.2f}ms")
            print(f"  Max:  {max(ttfts):.2f}ms")
            print(f"  P50:  {statistics.median(ttfts):.2f}ms")
            if len(ttfts) >= 2:
                print(f"  P90:  {statistics.quantiles(ttfts, n=10)[8]:.2f}ms")
                print(f"  P95:  {statistics.quantiles(ttfts, n=20)[18]:.2f}ms")
        
        if total_times:
            print("\nTotal Query Latency in ms:")
            print(f"  Min:  {min(total_times):.2f}ms")
            print(f"  Mean: {statistics.mean(total_times):.2f}ms")
            print(f"  Max:  {max(total_times):.2f}ms")
            print(f"  P50:  {statistics.median(total_times):.2f}ms")
            if len(total_times) >= 2:
                print(f"  P90:  {statistics.quantiles(total_times, n=10)[8]:.2f}ms")
                print(f"  P95:  {statistics.quantiles(total_times, n=20)[18]:.2f}ms")

async def main():
    # Allow passing concurrency level via command argument
    concurrency_levels = [5, 10, 20]
    if len(sys.argv) > 1:
        try:
            concurrency_levels = [int(sys.argv[1])]
        except ValueError:
            pass
            
    for level in concurrency_levels:
        await run_load_test(level)

if __name__ == "__main__":
    asyncio.run(main())
