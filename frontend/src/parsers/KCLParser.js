
/**
 * BRICK OS: KCL Parser (Lite)
 * 
 * A lightweight client-side parser for KCL (KittyCAD Language).
 * Used for:
 * 1. Syntax highlighting / Validation
 * 2. Extracting variables for UI controls
 * 3. Simple preview generation (Box/Cylinder)
 */

export class KCLParser {
    constructor() {
        this.variables = {};
        this.ast = { type: 'Program', body: [] };
    }

    /**
     * Parses KCL source code and extracts variables + simple geometry
     */
    parse(source) {
        this.variables = {};
        this.ast = { type: 'Program', body: [] };

        const lines = source.split('\n');

        for (let i = 0; i < lines.length; i++) {
            const line = lines[i].trim();
            if (!line || line.startsWith('//')) continue;

            // 1. Variable Declaration
            // const width = 10
            // let height = 5 + 2
            if (line.match(/^(const|let)\s+([a-zA-Z0-9_]+)\s*=\s*(.+)/)) {
                this._parseVariable(line);
            }
            // 2. Function Call (startSketchOn, extrude)
            // For now, we just identify them, we don't fully build a scene graph yet
            else if (line.includes('|>')) {
                // Pipe operator - function chain
                // TODO: Parse chain
            }
        }

        return {
            variables: this.variables,
            ast: this.ast
        };
    }

    _parseVariable(line) {
        const match = line.match(/^(const|let)\s+([a-zA-Z0-9_]+)\s*=\s*(.+)/);
        if (!match) return;

        const name = match[2];
        const expr = match[3];

        // Evaluate expression
        const value = this._evaluateExpression(expr);
        this.variables[name] = value;
    }

    _evaluateExpression(expr) {
        // Remove comments if any
        let cleanExpr = expr.split('//')[0].trim();

        try {
            // Replace known variables with values
            for (const [key, val] of Object.entries(this.variables)) {
                // Regex to verify word boundary to avoid partial replacement
                const regex = new RegExp(`\\b${key}\\b`, 'g');
                cleanExpr = cleanExpr.replace(regex, val);
            }

            // Basic Math Eval (using Function constructor is safer than eval, but still risky if input is purely from user)
            // For a localized parser, this is acceptable for prototype.
            // In prod, use a math parser library.

            // Check for allowed characters to prevent injection
            if (!/^[0-9+\-*/().\s]+$/.test(cleanExpr) && !Number.isNaN(parseFloat(cleanExpr))) {
                // Try parsing text (might be a function call result? ignored for now)
                return cleanExpr;
            }

            return new Function('return ' + cleanExpr)();

        } catch (e) {
            console.warn(`[KCLParser] Failed to eval: ${expr}`, e);
            return 0;
        }
    }

    // Helper to extract parameters from valid KCL for UI Generation
    static extractParameters(code) {
        const parser = new KCLParser();
        const result = parser.parse(code);
        return result.variables;
    }
}
