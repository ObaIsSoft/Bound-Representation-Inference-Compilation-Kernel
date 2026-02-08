
import axios from 'axios';

// Default to localhost if env var is not set (development convenience)
const baseURL = import.meta.env.VITE_API_BASE_URL || 'http://localhost:8000/api';

const apiClient = axios.create({
    baseURL,
    headers: {
        // 'Content-Type': 'application/json', // Let axios set this automatically
    },
});

// Response interceptor for standardized error handling
apiClient.interceptors.response.use(
    (response) => response.data,
    (error) => {
        const message = error.response?.data?.detail || error.message || 'An unknown error occurred';
        console.error('[API Error]:', message, error);
        return Promise.reject({ message, status: error.response?.status, original: error });
    }
);

export default apiClient;
