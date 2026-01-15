const API_BASE_URL = "http://localhost:8000/api";

/**
 * Checks backend health.
 */
export const checkBackendHealth = async () => {
    try {
        const response = await fetch(`${API_BASE_URL}/health`);
        return await response.json();
    } catch (error) {
        console.error("Backend health check failed:", error);
        return null;
    }
};

/**
 * Sends a design intent to the Orchestrator for compilation.
 * @param {string} userIntent - Natural language description.
 * @param {string} projectId - Current project ID.
 * @returns {Promise<object>} The final agent state (BOM, Geometry, etc).
 */
export const compileDesign = async (userIntent, projectId = "demo-1") => {
    try {
        const response = await fetch(`${API_BASE_URL}/compile`, {
            method: "POST",
            headers: {
                "Content-Type": "application/json",
            },
            body: JSON.stringify({
                user_intent: userIntent,
                project_id: projectId,
            }),
        });

        if (!response.ok) {
            throw new Error(`API Error: ${response.statusText}`);
        }

        return await response.json();
    } catch (error) {
        console.error("Compile Design failed:", error);
        throw error;
    }
};
