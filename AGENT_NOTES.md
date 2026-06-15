# Agent Notes — Future Improvements & Known Limitations

## Ingestion Pipeline (Phase 1)
- **Cross-document Reference Parsing**: The current matching in `corpus_resolver.py` scopes reference resolution to the same document. In later phases, we should parse references referencing other documents (e.g. "under the Act" or "under the SandBox framework") and resolve them globally.
- **Embedding Model MLEB Caveat**: We are using `nomic-embed-text:v1.5` as a zero-overhead local baseline. In Phase 3, we should evaluate upgrading to `BAAI/bge-m3` which supports dense sparse hybrid search to improve Recall@10.
- **Visual Table Structure Parsing**: Docling parses tables into layout blocks. Currently, table content is indexed as raw text. In later phases, we can keep the markdown representation of tables separate to improve structural Q&A reasoning.
