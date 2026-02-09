# Landing Page & Requirements Gathering - Implementation Plan V2

**Updates**: 100MB file limit for 3D files, Voice input JSON flow documented

---

## Part 1: File Upload (UPDATED - 100MB Limit)

### File Size & Types

| Category | Extensions | Max Size | Extraction Method |
|----------|------------|----------|-------------------|
| **3D CAD Files** | `.stl`, `.step`, `.stp`, `.obj`, `.fbx`, `.gltf`, `.glb` | **100MB** | Parse metadata (dimensions, volume, triangles) |
| **PDF Documents** | `.pdf` | 50MB | Text extraction (pdfplumber) |
| **Images** | `.jpg`, `.png`, `.gif`, `.bmp`, `.tiff`, `.webp` | 20MB | OCR (pytesseract) + dimensions |
| **Word Documents** | `.docx` | 20MB | Text extraction (python-docx) |
| **Spreadsheets** | `.xlsx`, `.xls`, `.csv` | 20MB | Convert to markdown tables |
| **Text/Code** | `.txt`, `.md`, `.json`, `.yaml`, `.xml`, `.py`, `.js`, `.ts`, `.c`, `.cpp`, `.h`, `.java`, `.go`, `.rs` | 10MB | Direct read |

### 3D File Special Handling

For 3D files (STL, STEP, OBJ), extract:
- Bounding box dimensions (X, Y, Z)
- Volume estimate
- Triangle/mesh count
- File format version
- Surface area (if calculable)

```python
# backend/services/file_extractor.py - 3D file parsing
async def extract_3d_metadata(file_path: str, ext: str) -> Dict[str, Any]:
    """Extract metadata from 3D CAD files."""
    metadata = {"type": "3d_model", "format": ext.upper()}
    
    if ext == '.stl':
        # Read STL header and calculate bounds
        import numpy as np
        # Parse binary or ASCII STL
        # Extract: vertices, normals, triangle count
        # Calculate: bounding box, volume, surface area
        
    elif ext in ['.step', '.stp']:
        # STEP files - read header, extract entities
        # Parse: PRODUCT, SHAPE, dimensions
        
    elif ext == '.obj':
        # Wavefront OBJ - parse vertices, faces
        # Count: v, vn, vt, f lines
        
    return metadata
```

---

## Part 2: Voice Input Flow (JSON Structure)

### Current Voice Input Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                         VOICE INPUT FLOW                             │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  1. USER CLICKS MIC (Landing.tsx)                                    │
│     ↓                                                                │
│  2. VoiceRecorder.tsx ACTIVATES                                      │
│     - Request microphone: navigator.mediaDevices.getUserMedia()      │
│     - Start MediaRecorder (audio/webm format)                        │
│     - Record audio chunks → Blob                                     │
│     ↓                                                                │
│  3. USER STOPS RECORDING                                             │
│     - mediaRecorder.stop()                                           │
│     - Blob created: new Blob(chunks, {type: 'audio/webm'})           │
│     - Status → 'playback'                                            │
│     ↓                                                                │
│  4. USER CLICKS "TRANSCRIBE"                                         │
│     ↓                                                                │
│  5. FRONTEND SENDS REQUEST                                           │
│                                                                      │
│     POST /api/stt/transcribe                                         │
│     Content-Type: multipart/form-data                                │
│                                                                      │
│     FormData:                                                        │
│     ├── audio: Blob (audio/webm)                                     │
│     └── format: "webm"                                               │
│                                                                      │
│     ↓                                                                │
│  6. BACKEND PROCESSES                                                │
│     - Receive UploadFile                                             │
│     - STTAgent.transcribe(audio_bytes)                               │
│     - OpenAI Whisper API call                                        │
│     ↓                                                                │
│  7. RESPONSE JSON                                                    │
│                                                                      │
│     {                                                                │
│       "text": "I want to design a titanium bracket...",              │
│       "success": true                                                │
│     }                                                                │
│                                                                      │
│     ↓                                                                │
│  8. USER CONFIRMS TRANSCRIPTION                                      │
│     ↓                                                                │
│  9. NAVIGATE TO REQUIREMENTS                                         │
│                                                                      │
│     navigate('/requirements', {                                       │
│       state: {                                                        │
│         userIntent: transcription,  // "I want to design..."          │
│         llmProvider: 'groq'                                          │
│       }                                                               │
│     })                                                                │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

### Voice Input JSON Schemas

#### Request: Frontend → Backend
```typescript
// POST /api/stt/transcribe
// Content-Type: multipart/form-data

interface STTRequest {
  // FormData fields:
  audio: Blob;        // audio/webm MIME type
  format: string;     // "webm" (also supports "wav", "mp3", "mp4", "mpeg", "mpga", "m4a", "ogg")
}
```

#### Response: Backend → Frontend
```typescript
// POST /api/stt/transcribe
// Content-Type: application/json

interface STTResponse {
  text: string;       // Transcribed text
  success: boolean;   // true if transcription succeeded
}

// Error response (HTTP 400/500)
interface STTErrorResponse {
  detail: string;     // Error message
}
```

#### Navigation State: Landing → Requirements
```typescript
// navigate('/requirements', { state: {...} })

interface RequirementsNavigationState {
  userIntent: string;      // The transcribed text
  llmProvider: string;     // Selected LLM (e.g., 'groq', 'openai', 'anthropic')
  // Future additions:
  // uploadedFiles?: string[];  // File IDs from uploads
  // fileNames?: string[];      // Original filenames
}
```

### VoiceRecorder Component State Machine

```
┌─────────┐    startRecording()     ┌─────────────┐
│  IDLE   │ ──────────────────────► │  RECORDING  │
│  [🎤]   │                         │  [⏹] Red    │
└─────────┘                         │  Timer: 0:00 │
     ▲                              └─────────────┘
     │                                    │
     │ onCancel()                    stopRecording()
     │                                    │
     │                                    ▼
     │                              ┌─────────────┐
     │                              │  PLAYBACK   │
     └───────────────────────────── │  [▶][🔄]    │
         reRecord()                 │  [Transcribe]
                                    └─────────────┘
                                           │
                                    transcribeAudio()
                                           │
                                           ▼
                                    ┌─────────────┐
                                    │TRANSCRIBING │
                                    │ [spinner]   │
                                    └─────────────┘
                                           │
                                           ▼
                                    ┌─────────────┐
                                    │  PLAYBACK   │
                                    │  w/ Text    │
                                    │  [🔄][✓]    │
                                    └─────────────┘
                                           │
                                    confirmTranscription()
                                           │
                                           ▼
                                    onTranscriptionComplete(text)
```

### Backend STTAgent Implementation

```python
# backend/agents/stt_agent.py
import logging
import io
import os
from openai import OpenAI

logger = logging.getLogger(__name__)

class STTAgent:
    """
    STTAgent handles speech-to-text transcription using OpenAI Whisper API.
    Supports: webm, wav, mp3, mp4, mpeg, mpga, m4a, ogg, oga, wav, weba
    """
    
    def __init__(self):
        self.name = "STTAgent"
        self.api_key = os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            logger.warning("STTAgent: OPENAI_API_KEY not set. Transcription will fail.")
        
        self.client = OpenAI(api_key=self.api_key)

    def transcribe(self, audio_data: bytes, filename: str = "audio.webm") -> str:
        """
        Transcribes audio bytes into text using OpenAI Whisper.
        
        Args:
            audio_data: Raw audio bytes
            filename: Filename with extension (used for format detection)
            
        Returns:
            Transcribed text or error message
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
                response_format="text",  // Returns plain text directly
                language="en"  // Optional: auto-detect if not specified
            )
            
            # response_format="text" returns string directly
            result = transcript.strip() if isinstance(transcript, str) else str(transcript)
            logger.info(f"STTAgent: Transcription successful: {result[:50]}...")
            return result
            
        except Exception as e:
            logger.error(f"STTAgent: Transcription failed: {e}")
            return f"[Transcription Error: {str(e)}]"


def get_stt_agent():
    """Get STTAgent via Global Registry."""
    from agent_registry import registry
    return registry.get_agent("STTAgent")
```

### Backend Endpoint

```python
# backend/main.py - /api/stt/transcribe

from fastapi import UploadFile, File

@app.post("/api/stt/transcribe")
async def transcribe_audio(file: UploadFile = File(...)):
    """
    Transcribes uploaded audio file using STTAgent (OpenAI Whisper).
    
    Supported formats: webm, wav, mp3, mp4, mpeg, mpga, m4a, ogg, oga, weba
    Max size: 25MB (Whisper API limit)
    """
    from agents.stt_agent import get_stt_agent
    stt_agent = get_stt_agent()
    
    # Read audio content
    audio_content = await file.read()
    if not audio_content:
        raise HTTPException(status_code=400, detail="Empty audio file")
    
    # Check size (Whisper limit is 25MB)
    if len(audio_content) > 25 * 1024 * 1024:
        raise HTTPException(status_code=400, detail="Audio file exceeds 25MB limit")
    
    # Transcribe
    transcript = stt_agent.transcribe(audio_content, filename=file.filename)
    
    # Return JSON response
    return {
        "text": transcript,
        "success": "[Error" not in transcript
    }
```

---

## Part 3: File Upload + Voice Integration

### Complete Landing Page Flow

```
┌─────────────────────────────────────────────────────────────────────┐
│                         LANDING PAGE                                 │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  User Input Options:                                                 │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐               │
│  │   TEXT       │  │   VOICE      │  │  FILES       │               │
│  │   [Type]     │  │   [🎤]       │  │  [📎 3]      │               │
│  └──────────────┘  └──────────────┘  └──────────────┘               │
│                                                                      │
│  Combined Payload:                                                   │
│  {                                                                   │
│    text: "Design a bracket...",                                      │
│    voiceTranscript: "I need a titanium bracket...",                  │
│    files: [                                                          │
│      { id: "file_abc123", name: "specs.pdf", size: 2048 },           │
│      { id: "file_def456", name: "model.stl", size: 52428800 },       │
│      { id: "file_ghi789", name: "sketch.png", size: 1024 }           │
│    ]                                                                 │
│  }                                                                   │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

### Navigation State with Files

```typescript
// Updated navigation state
interface RequirementsNavigationState {
  // From text/voice input
  userIntent: string;
  llmProvider: string;
  
  // From file upload (NEW)
  uploadedFiles: string[];      // File IDs from /api/files/upload
  fileNames: string[];          // Original filenames for display
  fileMetadata: {               // Extracted metadata
    id: string;
    name: string;
    type: string;
    size: number;
    extractedDims?: { x: number; y: number; z: number };
    extractedText?: string;
  }[];
  
  // Combined input (for context)
  combinedContext: string;      // Text + voice + file summaries
}

// Example:
const navigationState = {
  userIntent: "Design a titanium bracket for aerospace",
  llmProvider: "groq",
  uploadedFiles: ["file_abc123", "file_def456"],
  fileNames: ["specs.pdf", "model.stl"],
  fileMetadata: [
    {
      id: "file_abc123",
      name: "specs.pdf", 
      type: "application/pdf",
      size: 2048,
      extractedText: "Material: Ti-6Al-4V, Mass: 2.5kg..."
    },
    {
      id: "file_def456",
      name: "model.stl",
      type: "application/sla",
      size: 52428800,  // 50MB
      extractedDims: { x: 0.15, y: 0.08, z: 0.04 }  // 15cm x 8cm x 4cm
    }
  ],
  combinedContext: `
    User wants to design a titanium bracket for aerospace.
    Voice: "I need a titanium bracket that can handle 100kg load"
    Files:
    - specs.pdf: Material Ti-6Al-4V, Mass 2.5kg
    - model.stl: Dimensions 15cm x 8cm x 4cm
  `
};
```

---

## Part 4: Updated File Upload Endpoint (100MB)

```python
# backend/main.py - /api/files/upload

from fastapi import UploadFile, File, HTTPException
from typing import List
import uuid
import os

# Size limits per file type (bytes)
FILE_SIZE_LIMITS = {
    '3d': 100 * 1024 * 1024,      # 100MB for STL, STEP, OBJ, etc.
    'pdf': 50 * 1024 * 1024,       # 50MB for PDFs
    'image': 20 * 1024 * 1024,     # 20MB for images
    'document': 20 * 1024 * 1024,  # 20MB for DOCX, XLSX
    'text': 10 * 1024 * 1024,      # 10MB for text/code
}

# File type categories
FILE_CATEGORIES = {
    '3d': ['.stl', '.step', '.stp', '.obj', '.fbx', '.gltf', '.glb', '.3mf', '.ply'],
    'pdf': ['.pdf'],
    'image': ['.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff', '.webp', '.svg'],
    'document': ['.docx', '.xlsx', '.xls', '.csv', '.pptx'],
    'text': ['.txt', '.md', '.json', '.yaml', '.yml', '.xml', 
             '.py', '.js', '.ts', '.jsx', '.tsx', '.c', '.cpp', 
             '.h', '.hpp', '.java', '.go', '.rs', '.swift', '.kt']
}

def get_file_category(ext: str) -> str:
    """Get category for file extension."""
    ext = ext.lower()
    for category, extensions in FILE_CATEGORIES.items():
        if ext in extensions:
            return category
    return 'text'  # Default

def get_size_limit(ext: str) -> int:
    """Get size limit for file extension."""
    category = get_file_category(ext)
    return FILE_SIZE_LIMITS.get(category, 10 * 1024 * 1024)

@app.post("/api/files/upload")
async def upload_files(
    files: List[UploadFile] = File(...),
    session_id: Optional[str] = None
):
    """
    Upload files for requirements processing.
    
    Supports up to 6 files with varying size limits:
    - 3D files (STL, STEP, OBJ, etc.): 100MB
    - PDFs: 50MB
    - Images: 20MB
    - Documents: 20MB
    - Text/Code: 10MB
    
    Returns:
        file_ids: List of unique file IDs
        files: Metadata including extracted content/preview
    """
    if len(files) > 6:
        raise HTTPException(400, "Maximum 6 files allowed")
    
    results = []
    
    for file in files:
        # Get file extension
        ext = os.path.splitext(file.filename)[1].lower()
        category = get_file_category(ext)
        size_limit = get_size_limit(ext)
        
        # Read content
        content = await file.read()
        
        # Validate size
        if len(content) > size_limit:
            raise HTTPException(
                400, 
                f"{file.filename} ({len(content)/1024/1024:.1f}MB) exceeds "
                f"{category} file limit ({size_limit/1024/1024:.0f}MB)"
            )
        
        # Generate file ID
        file_id = f"file_{session_id or 'temp'}_{uuid.uuid4().hex[:8]}"
        
        # Save to temp storage
        temp_path = f"/tmp/brick_uploads/{file_id}{ext}"
        os.makedirs(os.path.dirname(temp_path), exist_ok=True)
        with open(temp_path, "wb") as f:
            f.write(content)
        
        # Extract content based on file type
        extracted = None
        metadata = {
            "category": category,
            "size_bytes": len(content),
            "extension": ext
        }
        
        if category == '3d':
            # Extract 3D metadata
            from services.file_extractor import extract_3d_metadata
            metadata.update(await extract_3d_metadata(temp_path, ext))
            extracted = f"[3D Model: {metadata.get('format', ext)}]"
            
        elif category == 'image':
            # OCR for images
            from services.file_extractor import extract_image_ocr
            extracted = await extract_image_ocr(temp_path)
            
        else:
            # Text extraction for other types
            from services.file_extractor import extract_file_content
            extracted = await extract_file_content(temp_path, file.content_type)
        
        results.append({
            "file_id": file_id,
            "filename": file.filename,
            "content_type": file.content_type,
            "category": category,
            "size": len(content),
            "size_formatted": f"{len(content)/1024/1024:.2f}MB" if len(content) > 1024*1024 else f"{len(content)/1024:.1f}KB",
            "metadata": metadata,
            "extracted_preview": extracted[:1000] + "..." if extracted and len(extracted) > 1000 else extracted
        })
    
    return {
        "file_ids": [r["file_id"] for r in results],
        "files": results,
        "total_size": sum(r["size"] for r in results),
        "total_size_formatted": f"{sum(r['size'] for r in results)/1024/1024:.2f}MB"
    }
```

---

## Summary of Changes

| Feature | V1 Plan | V2 Update |
|---------|---------|-----------|
| Max file size | 10MB | **100MB for 3D files** |
| Supported 3D | None | **STL, STEP, OBJ, FBX, GLTF, GLB** |
| Voice input | Mentioned | **Full JSON flow documented** |
| 3D metadata | None | **Dimensions, volume, triangles** |
| File categories | Basic | **5 categories with different limits** |

### Voice Input Status
✅ **Fully implemented and working**
- VoiceRecorder component records audio (webm)
- Sends to `/api/stt/transcribe`
- STTAgent calls OpenAI Whisper API
- Returns transcription JSON
- Passes to Requirements page via navigation state

### File Upload Status
🆕 **Needs implementation**
- New FileUploadZone component
- Updated `/api/files/upload` endpoint (100MB for 3D)
- 3D metadata extraction service
- Integration with Requirements page
