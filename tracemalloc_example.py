import tracemalloc


def analyze_memory_usage():
    tracemalloc.start()

    # Simulate memory-intensive operations
    data = []
    for i in range(100_000):
        data.append({
            'id': i,
            'value': f"item_{i}",
            'metadata': {'timestamp': i * 1000}
        })

    # Get memory snapshot
    snapshot = tracemalloc.take_snapshot()
    top_stats = snapshot.statistics('lineno')

    print("\nTop 3 memory allocations:")
    for stat in top_stats[:3]:
        print(f"{stat.count} blocks: {stat.size / 1024 / 1024:.1f} MB")
        print(f"    {stat.traceback.format()[0]}")

    tracemalloc.stop()


if __name__ == "__main__":
    analyze_memory_usage()