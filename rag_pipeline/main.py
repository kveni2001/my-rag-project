# main.py

import argparse
import json
from injector import RAGInjector

def main():
    parser = argparse.ArgumentParser(description="RAG Data Injection Pipeline")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # ── inject command ──
    inject_parser = subparsers.add_parser("inject", help="Inject documents into vector DB")
    inject_parser.add_argument("path", help="File or directory path")
    inject_parser.add_argument("--recursive", action="store_true", default=True)

    # ── search command (Vector DB only) ──
    search_parser = subparsers.add_parser("search", help="Search the vector DB directly")
    search_parser.add_argument("query", help="Search query")
    search_parser.add_argument("--top-k", type=int, default=5)

    # ── ask command (Full RAG with LLM) ──
    ask_parser = subparsers.add_parser("ask", help="Ask the AI a question about your documents")
    ask_parser.add_argument("query", help="Your question")
    ask_parser.add_argument("--top-k", type=int, default=5)

    # ── stats & clear commands ──
    subparsers.add_parser("stats", help="Show pipeline stats")
    subparsers.add_parser("clear", help="Clear the vector store")

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return

    if args.command == "inject":
        injector = RAGInjector()
        import os
        if os.path.isfile(args.path):
            result = injector.inject_file(args.path)
        elif os.path.isdir(args.path):
            result = injector.inject_directory(args.path, recursive=args.recursive)
        else:
            print(f"❌ Path not found: {args.path}")
            return
        print(f"\n📊 Result: {json.dumps(result, indent=2)}")

    elif args.command == "search":
        injector = RAGInjector()
        results = injector.search(args.query, top_k=args.top_k)
        print(f"\n🔍 Search results for: '{args.query}'\n")
        for i, r in enumerate(results, 1):
            print(f"{'─' * 60}")
            print(f"  Result #{i} | Score: {r['similarity_score']:.4f}")
            print(f"  Source: {r['metadata'].get('filename', 'unknown')}")
            print(f"  Content: {r['content'][:300]}...")
            print()

    elif args.command == "ask":
        from generator import RAGGenerator
        generator = RAGGenerator()
        generator.ask(args.query, top_k=args.top_k)

    elif args.command == "stats":
        injector = RAGInjector()
        stats = injector.get_stats()
        print(f"\n📈 Pipeline Stats:\n{json.dumps(stats, indent=2)}")

    elif args.command == "clear":
        injector = RAGInjector()
        confirm = input("⚠️  Are you sure you want to clear the vector store? (yes/no): ")
        if confirm.lower() == "yes":
            injector.vector_store.delete_collection()
            print("✅ Vector store cleared")

if __name__ == "__main__":
    main()