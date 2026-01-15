# BRICK - Native Desktop IDE

**B**ound **R**epresentation **I**nference & **C**ompilation **K**ernel.

A system where physical constraints (Bound) and symbolic models (Representation) are resolved via agentic logic (Inference) to produce deterministic hardware artifacts (Compilation) through a reactive core (Kernel).

## Architecture

- **Frontend**: React + TypeScript + Tailwind CSS
- **Backend**: Rust (Tauri)
- **Compiler Kernel**: Rust
- **Agent Orchestrator**: Python Sidebar (Environment Agent)
- **Simulation Engine**: Three.js + PostProcessing

## Development

### Prerequisites

- Node.js 18+
- Rust 1.70+
- npm or yarn

### Running in Development Mode

```bash
# Install frontend dependencies
cd frontend
npm install

# Run Tauri dev mode (starts both Rust backend and React frontend)
npm run tauri:dev
```

This will:
1. Start the Vite dev server on port 3000
2. Compile the Rust backend
3. Launch the native window with hot reload

### Building for Production

```bash
cd frontend
npm run tauri:build
```

Output locations:
- **macOS**: `src-tauri/target/release/bundle/macos/BRICK OS.app`
- **Windows**: `src-tauri/target/release/bundle/msi/BRICK OS.msi`
- **Linux**: `src-tauri/target/release/bundle/appimage/brick.AppImage`

## Project Structure

```
brick/
├── frontend/              # React UI
│   ├── src/
│   │   ├── components/    # Modular UI components
│   │   ├── utils/         # Shared utilities
│   │   ├── App.jsx        # Main app component
│   │   └── index.jsx      # Entry point
│   ├── package.json
│   └── vite.config.js
├── src-tauri/             # Rust backend
│   ├── src/
│   │   └── main.rs        # Tauri entry point + IPC commands
│   ├── Cargo.toml         # Rust dependencies
│   └── tauri.conf.json    # Tauri configuration
└── README.md
```

## Available IPC Commands

The Rust backend exposes these commands to the frontend:

- `read_brick_file(path)` - Read file contents
- `write_brick_file(path, content)` - Write file contents
- `list_directory(path)` - List directory contents
- `compile_isa(source)` - Compile ISA source (placeholder)
- `get_system_info()` - Get system information

### Usage Example

```typescript
import { invoke } from '@tauri-apps/api/tauri';

// Read a file
const content = await invoke('read_brick_file', { path: '/path/to/file.brick' });

// Get system info
const info = await invoke('get_system_info');
console.log(info); // { platform: "macos", arch: "x86_64", ... }
```

## Features

- ✅ Native desktop application (no Electron bloat)
- ✅ 4-panel resizable IDE layout
- ✅ File system access via Rust backend
- ✅ Cross-platform (macOS, Windows, Linux)
- ✅ Hot reload in development
- 🔄 Compiler kernel (coming soon)
- 🔄 Agent orchestration (coming soon)
- 🔄 Physics simulation engine (coming soon)

## Performance

- **Bundle Size**: ~3MB (vs ~300MB for Electron)
- **Memory**: <100MB idle
- **Startup**: <500ms

## License

MIT
