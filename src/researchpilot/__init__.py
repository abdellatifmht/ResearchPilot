from .graph import run_research_pilot


def main():
    try:
        query = input("Enter your research query: ")
    except EOFError:
        return

    result = run_research_pilot(query)

    print("Research Agent Result:")
    if isinstance(result, dict):
        print(result.get("research_output", ""))
        print("\nSources:")
        for source in result.get("sources") or []:
            if isinstance(source, dict):
                print(f"- {source.get('url', '')}")
        print("\nAnalysis Report:")

        analysis_report = result.get("analysis_report") or {}
        summaries = analysis_report.get("source_summaries", [])

        if summaries:
            print("\nSource Summaries:")
            for summary in summaries[:5]:
                if isinstance(summary, dict):
                    print(f"- {summary.get('source_url', '')}: {summary.get('summary', '')}")
        else:
            print("No source summaries available.")

        atomic_facts = analysis_report.get("atomic_facts", [])
        if atomic_facts:
            print("\nAtomic Facts:")
            for fact in atomic_facts[:5]:
                if isinstance(fact, dict):
                    print(f"- {fact.get('statement', '')}")
                    print(f"  Category: {fact.get('category', '')}")
                    print(f"  Source URLs: {', '.join(fact.get('source_urls', []))}")
        else:
            print("No atomic facts available.")
        
        contradictions = analysis_report.get("contradictions", [])
        if contradictions:
            print("\nContradictions:")
            for contradiction in contradictions[:5]:
                if isinstance(contradiction, dict):
                    print(f"- {contradiction.get('statement', '')}")
        else:
            print("\nNo contradictions available.")

        path = result.get("export_path")
        if path:
            print(f"\nReport exported to: {path}")
        else:
            print("\nNo report export path available.")

    else:
        print(result)

if __name__ == "__main__":
    main()
