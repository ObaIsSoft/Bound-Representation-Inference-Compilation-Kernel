import logging
import io
import os
from openai import OpenAI

logger = logging.getLogger(__name__)

class STTAgent:
    """
    STTAgent handles speech-to-text transcription for BRICK OS.
    Uses OpenAI Whisper API for real, high-fidelity transcription.
    """
    def __init__(self):
        self.name = "STTAgent"
        self.api_key = os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            logger.warning("STTAgent: OPENAI_API_KEY not set. Transcription will fail.")
        
        self.client = OpenAI(api_key=self.api_key)

    def transcribe(self, audio_data: bytes, filename: str = "audio.wav") -> str:
        """
        Transcribes audio bytes into a text string using OpenAI Whisper.
        """
        if not audio_data:
            return ""

        if not self.api_key:
            logger.error("STTAgent: Cannot transcribe without API Key.")
            return "[Error: API Key Missing]"

        logger.info(f"STTAgent: Sending {len(audio_data)} bytes to Whisper API")

        try:
            # Whisper API requires a file-like object with a name attribute
            audio_file = io.BytesIO(audio_data)
            audio_file.name = filename
            
            transcript = self.client.audio.transcriptions.create(
                model="whisper-1", 
                file=audio_file,
                response_format="text"
            )
            
            # OpenAI response_format="text" returns the string directly
            result = transcript.strip() if isinstance(transcript, str) else str(transcript)
            logger.info(f"STTAgent: Transcription successful: {result[:50]}...")
            return result
            
        except Exception as e:
            logger.error(f"STTAgent: Transcription failed: {e}")
            return f"[Transcription Error: {str(e)}]"

# Singleton instance (DEPRECATED: Use Registry)
# _stt_agent = STTAgent()

def get_stt_agent():
    """
    Get STTAgent via Global Registry to ensure Observability Wrapper is applied.
    """
    from agent_registry import registry
    return registry.get_agent("STTAgent")
