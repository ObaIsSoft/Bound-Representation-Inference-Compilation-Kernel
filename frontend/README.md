# BRICK OS Frontend - Component Structure

## Directory Structure

```
frontend/
├── src/
│   ├── components/
│   │   ├── shared/
│   │   │   ├── TabButton.jsx          # Reusable tab button
│   │   │   └── PanelHeader.jsx        # Panel header with icon
│   │   ├── layout/
│   │   │   ├── ActivityBar.jsx        # Left navigation rail
│   │   │   └── Header.jsx             # Top status bar
│   │   ├── design/
│   │   │   └── DesignLibrary.jsx      # File explorer panel
│   │   ├── simulation/
│   │   │   └── SimulationBay.jsx      # 3D rendering panel
│   │   ├── control/
│   │   │   ├── ControlDeck.jsx        # Right control panel
│   │   │   ├── ProposalCard.jsx       # Agent proposal card
│   │   │   ├── ConstraintCard.jsx     # Parameter constraint
│   │   │   └── IntentInput.jsx        # Semantic input
│   │   └── terminal/
│   │       ├── LowerPanel.jsx         # Bottom panel container
│   │       ├── TerminalTab.jsx        # Shell terminal
│   │       ├── DiagnosticsTab.jsx     # System diagnostics
│   │       ├── ConsoleTab.jsx         # vHIL console
│   │       ├── OutputTab.jsx          # Build logs
│   │       └── KCLTab.jsx             # KCL code viewer
│   ├── utils/
│   │   └── constants.js               # Shared constants
│   ├── App.jsx                        # Main application
│   ├── index.jsx                      # React entry point
│   └── index.css                      # Global styles
├── index.html
├── package.json
├── vite.config.js
├── tailwind.config.js
└── postcss.config.js
```

## Component Hierarchy

```
App
├── Header
├── ActivityBar
├── DesignLibrary
├── SimulationBay
├── ControlDeck
│   ├── ProposalCard (multiple)
│   ├── ConstraintCard (multiple)
│   └── IntentInput
└── LowerPanel
    ├── TerminalTab
    ├── DiagnosticsTab
    ├── ConsoleTab
    ├── OutputTab
    └── KCLTab
```

## Running the Application

```bash
# Install dependencies
npm install

# Start development server
npm run dev

# Build for production
npm run build

# Preview production build
npm run preview
```

## Features

- **Modular Architecture**: Each component has a single responsibility
- **Resizable Panels**: Left, right, and bottom panels are resizable
- **Interactive Terminal**: Functional shell with command history
- **Agent Proposals**: Approve/deny agent suggestions
- **Live Simulation**: Toggle simulation running state
- **Multiple Tabs**: Switch between different views in bottom panel

## Technology Stack

- **React 18**: Component framework
- **Vite**: Build tool and dev server
- **Tailwind CSS**: Utility-first styling
- **Lucide React**: Icon library
