Your task is to update the documentation based on the latest changes.
Starting point is the last git commit. Minimally use git git status && git diff HEAD && git status --porcelain to find out.
From there, analyse the latest changes in the codebase etc.:
- Firstly, change the CLAUDE.md.
- Then go to the implementation details (IMPLEMENTATION.md, SCHEMA.md) within the docs/ folder and make changes accordingly.
- In the end, reflect all critical changes also in README.md, if necessary.

CRITICAL RULES:
- Keep README.md < 15.000 chars (general)
- Keep CLAUDE.md < 25.000 chars (use references to docs/ folder documentation in the form e. g. [docs/architecture.md](docs/architecture.md) @docs/architecture.md Full architecture diagram, state objects, data flow)
- Keep IMPLEMENTATION.md, SCHEMA.md < 35.000 chars
- Keep each single doc in docs/ folder < 35.000 chars