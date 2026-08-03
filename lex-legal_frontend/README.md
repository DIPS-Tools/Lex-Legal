# LexGuard Vue frontend

The frontend is a Vue 3 and Vite single-page application with four routes:

- `/` — legal compliance assistant landing page
- `/legal-compliance` — streaming legal query workspace
- `/import` — PDF/Markdown knowledge-base ingestion
- `/legal-query-monitor` — browser-local query activity monitor

Legacy `.html` paths redirect through equivalent Vue Router aliases.

## Page templates

Each route keeps its editable Vue markup in a matching file under
`templates/`:

- `templates/index.html` — overview page
- `templates/legal-compliance.html` — legal query page
- `templates/import.html` — knowledge import page
- `templates/legal-query-monitor.html` — query monitor page

The corresponding files under `src/views/` contain page behavior and import
these files with Vue's external `<template src="...">` feature. The template
files are Vue template fragments, not standalone HTML documents, so Vue
directives such as `v-if`, `v-for`, `v-model`, and `@click` remain functional.
Shared navigation and service-status markup remains in `src/App.vue`.
The Vite configuration inlines these fragments before Vue compilation and
invalidates their owning view during hot reload, avoiding compiler state errors
when a template is edited while the development server is running.

## Backend services

Start the APIs from the repository root:

```bash
uvicorn app.api.import_server:app --host 0.0.0.0 --port 8007
uvicorn app.api.query_server:app --host 0.0.0.0 --port 8008
```

The frontend derives the API hostname from the browser URL and uses ports
`8007` and `8008`. Override either endpoint when necessary:

```bash
VITE_IMPORT_API_BASE_URL=http://127.0.0.1:8007
VITE_QUERY_API_BASE_URL=http://127.0.0.1:8008
```

## Development

```bash
cd lex-legal_frontend
npm ci
npm run dev -- --host 0.0.0.0 --port 5176
```

Open `http://10.22.12.250:5176/`.

## Production build

```bash
npm run build
```
