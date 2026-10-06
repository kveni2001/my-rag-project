# quick_start.py

"""
Quick start script to test the pipeline interactively.
"""

from injector import RAGInjector
import json


def main():
    print("🚀 RAG Data Injection Pipeline - Quick Start")
    print("=" * 50)

    # Initialize (uses defaults from config.py)
    injector = RAGInjector()

    while True:
        print("\n📋 Menu:")
        print("  1. Inject a file")
        print("  2. Inject a directory")
        print("  3. Search")
        print("  4. View stats")
        print("  5. Exit")

        choice = input("\nChoice (1-5): ").strip()

        if choice == "1":
            path = input("Enter file path: ").strip()
            result = injector.inject_file(path)
            print(json.dumps(result, indent=2))

        elif choice == "2":
            path = input("Enter directory path: ").strip()
            result = injector.inject_directory(path)
            print(json.dumps(result, indent=2))

        elif choice == "3":
            query = input("Enter search query: ").strip()
            top_k = int(input("How many results? (default 5): ").strip() or "5")
            results = injector.search(query, top_k=top_k)

            print(f"\n🔍 Found {len(results)} results:\n")
            for i, r in enumerate(results, 1):
                print(f"{'─' * 50}")
                print(f"  #{i} | Score: {r['similarity_score']:.4f}")
                print(f"  Source: {r['metadata'].get('filename')}")
                print(f"  Content: {r['content'][:200]}...")

        elif choice == "4":
            stats = injector.get_stats()
            print(json.dumps(stats, indent=2))

        elif choice == "5":
            print("👋 Goodbye!")
            break


if __name__ == "__main__":
    main()