# Landing Page & Requirements Gathering - Implementation Plan

## Overview
Enable multi-file upload (up to 6 files) on Landing page, with file content extracted and added to the conversation context in Requirements Gathering page.

---

## Part 1: Landing Page - File Upload Feature

### Current State
- Voice input: ✅ UI exists
- Text input: ✅ Functional  
- File upload: ❌ Placeholder only (line 138: "Note: File uploads need separate handling")

### New UI Design

```
┌───────────────────────────────────────────────────────────────┐
│                         BRICK OS                              │
│                                                               │
│              Good evening, what would you like to             │
│                     design today?                             │
│                                                               │
│    ╭───────────────────────────────────────────────────╮     │
│    │  [🎤]  Type your message or speak...              │     │
│    ╰───────────────────────────────────────────────────╯     │
│                                                               │
│    ┌─────────────────────────────────────────────────────┐   │
│    │  📎 Attach files (up to 6):                         │   │
│    │                                                     │   │
│    │  [Drop files here or click to browse]               │   │
│    │                                                     │   │
│    │  📄 requirements.pdf    🖼️ sketch.jpg              │   │
│    │  📊 specs.xlsx          📋 notes.txt               │   │
│    │  ❌ Clear all                                       │   │
│    └─────────────────────────────────────────────────────┘   │
│                                                               │
│    [🚀 Start Design Session]                                  │
│                                                               │
└───────────────────────────────────────────────────────────────┘
```

### Frontend Implementation

#### 1. File Upload Component (new)
**File**: `frontend/src/components/file/FileUploadZone.tsx`

```typescript
interface FileUploadZoneProps {
  files: File[];
  onFilesChange: (files: File[]) => void;
  maxFiles?: number;
  acceptedTypes?: string[];
}

// Features:
// - Drag & drop zone
// - Click to browse
// - File type validation (PDF, images, TXT, DOCX, XLSX, CSV, JSON)
// - File size limit (10MB per file)
// - Thumbnail preview for images
// - Remove individual files
// - "Clear all" button
// - Max 6 files enforcement
```

#### 2. Landing Page Updates
**File**: `frontend/src/pages/Landing.tsx`

```typescript
// New state
const [attachedFiles, setAttachedFiles] = useState<File[]>([]);
const [isUploading, setIsUploading] = useState(false);
const [uploadProgress, setUploadProgress] = useState<number[]>([]);

// Modified submit handler
const handleTextSubmit = async (message: string, options: InputOptions) => {
  // 1. Upload files first (if any)
  let fileIds: string[] = [];
  if (attachedFiles.length > 0) {
    fileIds = await uploadFiles(attachedFiles);
  }
  
  // 2. Navigate to requirements with file context
  navigate('/requirements', {
    state: {
      userIntent: message,
      llmProvider: options.llmProvider,
      uploadedFiles: fileIds,  // Pass file IDs
      fileNames: attachedFiles.map(f => f.name)
    }
  });
};
```

#### 3. File Upload API Client
**File**: `frontend/src/utils/fileUpload.ts`

```typescript
export async function uploadFiles(
  files: File[],
  onProgress?: (progress: number[]) => void
): Promise<string[]> {
  const formData = new FormData();
  files.forEach((file, index) => {
    formData.append(`file_${index}`, file);
  });
  
  const response = await apiClient.post('/files/upload', formData, {
    headers: { 'Content-Type': 'multipart/form-data' },
    onUploadProgress: (progressEvent) => {
      // Calculate progress for each file
    }
  });
  
  return response.file_ids; // Server returns IDs for each uploaded file
}
```

---

## Part 2: Backend - File Handling

### New Endpoint: File Upload
**File**: Add to `backend/main.py`

```python
from fastapi import UploadFile, File
from typing import List
import tempfile
import os

@app.post("/api/files/upload")
async def upload_files(
    files: List[UploadFile] = File(..., description="Up to 6 files"),
    session_id: Optional[str] = None
):
    """
    Upload files for requirements processing.
    Stores files temporarily and extracts content for context.
    
    Returns:
        file_ids: List of unique file IDs for retrieval
        extracted_content: Preview of extracted text (first 500 chars)
    """
    if len(files) > 6:
        raise HTTPException(400, "Maximum 6 files allowed")
    
    results = []
    for file in files:
        # Validate file size (10MB max)
        content = await file.read()
        if len(content) > 10 * 1024 * 1024:
            raise HTTPException(400, f"{file.filename} exceeds 10MB limit")
        
        # Generate file ID
        file_id = f"file_{session_id or 'temp'}_{uuid.uuid4().hex[:8]}"
        
        # Save to temp storage
        temp_path = f"/tmp/brick_uploads/{file_id}"
        os.makedirs(os.path.dirname(temp_path), exist_ok=True)
        with open(temp_path, "wb") as f:
            f.write(content)
        
        # Extract content based on file type
        extracted = await extract_file_content(temp_path, file.content_type)
        
        results.append({
            "file_id": file_id,
            "filename": file.filename,
            "content_type": file.content_type,
            "size": len(content),
            "extracted_preview": extracted[:500] + "..." if len(extracted) > 500 else extracted
        })
    
    return {
        "file_ids": [r["file_id"] for r in results],
        "files": results
    }
```

### File Content Extraction Service
**New File**: `backend/services/file_extractor.py`

```python
"""
File content extraction service.
Supports: PDF, images (OCR), TXT, DOCX, XLSX, CSV, JSON, MD
"""

import os
from typing import Dict, Any
import logging

logger = logging.getLogger(__name__)

async def extract_file_content(file_path: str, content_type: str) -> str:
    """Extract text content from various file types."""
    
    ext = os.path.splitext(file_path)[1].lower()
    
    try:
        # PDF
        if ext == '.pdf' or 'pdf' in content_type:
            return await extract_pdf(file_path)
        
        # Images - OCR
        elif ext in ['.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff', '.webp']:
            return await extract_image_ocr(file_path)
        
        # Word documents
        elif ext == '.docx':
            return await extract_docx(file_path)
        
        # Excel/CSV
        elif ext in ['.xlsx', '.xls', '.csv']:
            return await extract_spreadsheet(file_path)
        
        # Text files (code, markdown, txt, json)
        elif ext in ['.txt', '.md', '.json', '.yaml', '.yml', '.xml', 
                     '.py', '.js', '.ts', '.jsx', '.tsx', '.c', '.cpp', 
                     '.h', '.hpp', '.java', '.go', '.rs', '.swift']:
            return await extract_text(file_path)
        
        else:
            return f"[Binary file: {os.path.basename(file_path)}]"
            
    except Exception as e:
        logger.error(f"Failed to extract {file_path}: {e}")
        return f"[Error extracting file: {str(e)}]"

async def extract_pdf(file_path: str) -> str:
    """Extract text from PDF using pdfplumber or PyPDF2."""
    try:
        import pdfplumber
        text = ""
        with pdfplumber.open(file_path) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
        return text or "[PDF: No text content found]"
    except ImportError:
        # Fallback to PyPDF2
        import PyPDF2
        text = ""
        with open(file_path, 'rb') as f:
            reader = PyPDF2.PdfReader(f)
            for page in reader.pages:
                text += page.extract_text() + "\n"
        return text

async def extract_image_ocr(file_path: str) -> str:
    """Extract text from image using OCR."""
    try:
        from PIL import Image
        import pytesseract
        
        image = Image.open(file_path)
        
        # Extract text
        text = pytesseract.image_to_string(image)
        
        # Also get image description for context
        description = f"[Image: {image.size[0]}x{image.size[1]} pixels]"
        
        return f"{description}\n\nExtracted text:\n{text}" if text.strip() else description
        
    except ImportError:
        return "[Image: OCR not available. Install pytesseract.]"
    except Exception as e:
        return f"[Image: {str(e)}]"

async def extract_docx(file_path: str) -> str:
    """Extract text from Word document."""
    try:
        from docx import Document
        doc = Document(file_path)
        return "\n".join([para.text for para in doc.paragraphs])
    except ImportError:
        return "[DOCX: python-docx not installed]"

async def extract_spreadsheet(file_path: str) -> str:
    """Extract text from Excel/CSV."""
    import pandas as pd
    
    if file_path.endswith('.csv'):
        df = pd.read_csv(file_path)
    else:
        df = pd.read_excel(file_path)
    
    # Convert to markdown table format
    return df.to_markdown(index=False)

async def extract_text(file_path: str) -> str:
    """Extract text from plain text files."""
    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
        return f.read()
```

---

## Part 3: Requirements Gathering Page - Agent Integration

### Current Issues
1. Hardcoded parameters (mass=5kg, complexity="moderate")
2. SafetyAgent not called
3. No file content context

### Updated Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    REQUIREMENTS GATHERING                        │
│                     /api/chat/requirements                       │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  1. Receive Request                                              │
│     ├── message: string (user input)                             │
│     ├── file_ids: string[] (uploaded files)                      │
│     ├── conversation_history                                     │
│     └── session_id                                               │
│                                                                  │
│  2. Load File Contents                                           │
│     └── For each file_id: extract content from temp storage      │
│                                                                  │
│  3. Run Agents (in parallel)                                     │
│     ├─► ConversationalAgent ──► chat response                    │
│     ├─► EnvironmentAgent ─────► environment detection            │
│     ├─► GeometryEstimator ────► feasibility (with params)        │
│     ├─► CostAgent ────────────► cost estimate (with params)      │
│     └─► SafetyAgent ──────────► safety screening                 │
│                                                                  │
│  4. Extract Parameters from Context                              │
│     └── Use LLM to parse mass, material, complexity from:        │
│         - User message                                           │
│         - File contents                                          │
│         - Conversation history                                   │
│                                                                  │
│  5. Return Response                                              │
│     ├── response: string (chat reply)                            │
│     ├── feasibility: {geometry, cost, environment, safety}       │
│     ├── extracted_params: {mass, material, complexity}           │
│     ├── file_context: {files_processed, total_chars}             │
│     └── requirements_complete: boolean                           │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### Updated Backend Endpoint

**File**: `backend/main.py` - `/api/chat/requirements`

```python
class ChatRequirementsRequest(BaseModel):
    message: str
    conversation_history: List[str] = []
    user_intent: str = ""
    mode: str = "requirements_gathering"
    ai_model: str = "groq"
    session_id: Optional[str] = None
    file_ids: List[str] = []  # NEW: Uploaded file IDs

@app.post("/api/chat/requirements")
async def chat_requirements_endpoint(req: ChatRequirementsRequest):
    """
    Intelligent Chat Endpoint for Requirements Gathering.
    Now with file context integration.
    """
    
    # 1. Load file contents
    file_contexts = []
    total_file_chars = 0
    for file_id in req.file_ids:
        content = await load_file_content(file_id)
        if content:
            file_contexts.append({
                "file_id": file_id,
                "filename": get_filename(file_id),
                "content": content[:5000]  # Limit to 5k chars per file
            })
            total_file_chars += len(content)
    
    # 2. Build augmented context
    augmented_message = req.message
    if file_contexts:
        file_summary = "\n\n[Attached Files Context]:\n"
        for fc in file_contexts:
            file_summary += f"\n--- {fc['filename']} ---\n{fc['content'][:1000]}...\n"
        augmented_message += file_summary
    
    # 3. Run ConversationalAgent with augmented context
    response_text = await conversational_agent.chat(
        user_input=augmented_message,  # Include file context
        history=req.conversation_history,
        current_intent=req.user_intent,
        session_id=session_id
    )
    
    # 4. Extract parameters from combined context
    # Use LLM to parse: mass, material, complexity, dimensions
    extracted_params = await extract_design_params(
        message=req.message,
        file_contents=[fc["content"] for fc in file_contexts],
        conversation_history=req.conversation_history
    )
    
    # 5. Run agents with EXTRACTED parameters (not hardcoded)
    geom_estimator = GeometryEstimator()
    cost_agent = CostAgent()
    env_agent = EnvironmentAgent()
    safety_agent = SafetyAgent()
    
    # Use extracted params or defaults
    mass_kg = extracted_params.get("mass_kg", 5.0)
    material = extracted_params.get("material", "aluminum")
    complexity = extracted_params.get("complexity", "moderate")
    max_dim = extracted_params.get("max_dim_m", 1.0)
    
    # Parallel execution
    env_result = env_agent.detect_environment(req.user_intent + " " + req.message)
    
    geom_result = geom_estimator.estimate(
        req.user_intent, 
        {"max_dim": max_dim, "mass_kg": mass_kg}
    )
    
    cost_result = await cost_agent.quick_estimate({
        "mass_kg": mass_kg,
        "complexity": complexity,
        "material_name": material
    })
    
    # NEW: Safety check
    safety_result = await safety_agent.run({
        "materials": [material],
        "application_type": env_result.get("type", "industrial"),
        "physics_results": {}  # No physics yet in requirements phase
    })
    
    # 6. Check completeness
    requirements_complete = await conversational_agent.is_requirements_complete(session_id)
    
    return {
        "response": response_text,
        "feasibility": {
            "geometry": geom_result,
            "cost": cost_result,
            "environment": env_result,
            "safety": safety_result  # NEW
        },
        "extracted_params": extracted_params,  # NEW
        "file_context": {
            "files_processed": len(file_contexts),
            "total_chars": total_file_chars
        },
        "session_id": session_id,
        "requirements_complete": requirements_complete,
        "requirements": final_requirements if requirements_complete else {}
    }

async def extract_design_params(
    message: str,
    file_contents: List[str],
    conversation_history: List[str]
) -> Dict[str, Any]:
    """
    Use LLM to extract design parameters from all context sources.
    """
    context = f"""
User message: {message}

File contents:
{"".join(file_contents)}

Conversation history:
{"".join(conversation_history[-5:])}  # Last 5 messages

Extract these parameters if mentioned:
- mass_kg: numeric weight in kg (or convert from lbs/g)
- material: material name (e.g., "aluminum 6061", "titanium", "steel")
- complexity: "simple", "moderate", or "complex"
- max_dim_m: maximum dimension in meters
- application: "aerospace", "automotive", "medical", "industrial", etc.
- quantity: number of units

Return as JSON.
"""
    
    # Use lightweight LLM call for extraction
    try:
        llm = get_llm_provider("groq")
        response = llm.generate(
            prompt=context,
            system_prompt="Extract design parameters. Return valid JSON only."
        )
        import json
        return json.loads(response)
    except Exception as e:
        logger.warning(f"Parameter extraction failed: {e}")
        return {}  # Return empty, will use defaults
```

---

## Part 4: Frontend Requirements Page Updates

### Enhanced Agent Status Panel

```jsx
// 4-box status panel (was 3)
<div className="grid grid-cols-4 gap-3 mb-4">
  {/* Environment */}
  <AgentStatusBox 
    label="Environment" 
    value={requirements.environment?.type || "DETECTING..."}
    color={theme.colors.accent.primary}
  />
  
  {/* Feasibility */}
  <AgentStatusBox 
    label="Feasibility"
    value={requirements.feasibility?.geometry?.feasible ? "Possible" : "Impossible"}
    indicator={requirements.feasibility?.geometry?.feasible ? "green" : "red"}
  />
  
  {/* Cost */}
  <AgentStatusBox 
    label="Est. Cost"
    value={`$${requirements.feasibility?.cost?.estimated_cost_usd || "0"}`}
  />
  
  {/* NEW: Safety */}
  <AgentStatusBox 
    label="Safety"
    value={requirements.feasibility?.safety?.status || "Checking..."}
    indicator={
      requirements.feasibility?.safety?.status === "safe" ? "green" :
      requirements.feasibility?.safety?.status === "hazards_detected" ? "yellow" : "gray"
    }
  />
</div>

// Show extracted parameters (collapsible)
{requirements.extracted_params && (
  <div className="mt-2 p-2 rounded bg-opacity-20">
    <p className="text-xs text-muted">Detected from your input:</p>
    <div className="flex gap-2 flex-wrap mt-1">
      {requirements.extracted_params.mass_kg && (
        <Badge>Mass: {requirements.extracted_params.mass_kg}kg</Badge>
      )}
      {requirements.extracted_params.material && (
        <Badge>Material: {requirements.extracted_params.material}</Badge>
      )}
      {requirements.extracted_params.complexity && (
        <Badge>Complexity: {requirements.extracted_params.complexity}</Badge>
      )}
    </div>
  </div>
)}

// Show file context indicator
{requirements.file_context?.files_processed > 0 && (
  <div className="flex items-center gap-2 text-xs text-muted">
    <FileText size={14} />
    <span>
      {requirements.file_context.files_processed} file(s) analyzed 
      ({requirements.file_context.total_chars} characters)
    </span>
  </div>
)}
```

### File Attachment Display in Chat

```jsx
// Show attached files in the first message
{attachedFiles.length > 0 && (
  <div className="mb-4 p-3 rounded bg-tertiary">
    <p className="text-xs font-semibold mb-2">Attached Files:</p>
    <div className="flex flex-wrap gap-2">
      {attachedFiles.map((file, idx) => (
        <FileBadge key={idx} filename={file.name} type={file.type} />
      ))}
    </div>
  </div>
)}
```

---

## Part 5: Database & Storage

### Option A: Temporary Storage (Recommended for MVP)
- Files stored in `/tmp/brick_uploads/` with session_id prefix
- Auto-cleaned after 24 hours
- No database schema changes needed

### Option B: Persistent Storage (Future)
**New Table**: `project_files`

```sql
CREATE TABLE project_files (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    project_id UUID REFERENCES projects(id),
    session_id TEXT,
    filename TEXT NOT NULL,
    file_path TEXT NOT NULL,
    content_type TEXT,
    file_size INTEGER,
    extracted_content TEXT,
    created_at TIMESTAMP DEFAULT NOW(),
    expires_at TIMESTAMP DEFAULT (NOW() + INTERVAL '7 days')
);

CREATE INDEX idx_project_files_session ON project_files(session_id);
```

---

## Part 6: Implementation Phases

### Phase 1: Core File Upload (Day 1-2)
1. ✅ Create `FileUploadZone` component
2. ✅ Update Landing page with file upload UI
3. ✅ Create `/api/files/upload` endpoint
4. ✅ Create `file_extractor.py` service (basic: text, PDF)

### Phase 2: Requirements Integration (Day 3-4)
1. ✅ Update `/api/chat/requirements` to accept `file_ids`
2. ✅ Add file content loading to endpoint
3. ✅ Integrate extracted context into ConversationalAgent
4. ✅ Add SafetyAgent to requirements flow
5. ✅ Create parameter extraction function

### Phase 3: Enhanced Extraction (Day 5)
1. ✅ Add OCR for images (pytesseract)
2. ✅ Add DOCX support
3. ✅ Add spreadsheet (Excel/CSV) support
4. ✅ Add progress indicators for file processing

### Phase 4: UI Polish (Day 6)
1. ✅ Update Requirements page with 4-box status panel
2. ✅ Show extracted parameters as badges
3. ✅ Show file processing status
4. ✅ Error handling for failed extractions

---

## Part 7: Dependencies to Add

### Backend (`requirements.txt`)
```
# File extraction
pdfplumber>=0.10.0
PyPDF2>=3.0.0
python-docx>=0.8.11
openpyxl>=3.1.0
pandas>=2.0.0
pillow>=10.0.0
pytesseract>=0.3.10

# Temp file cleanup
schedule>=1.2.0
```

### Frontend (`package.json`)
```json
{
  "dependencies": {
    "react-dropzone": "^14.2.3",
    "react-pdf": "^7.5.1"
  }
}
```

---

## Part 8: API Reference

### New Endpoints

```
POST /api/files/upload
├── Content-Type: multipart/form-data
├── Parameters:
│   ├── files: File[] (max 6, max 10MB each)
│   └── session_id: string (optional)
└── Response:
    ├── file_ids: string[]
    ├── files: [{file_id, filename, content_type, size, extracted_preview}]
    └── error: string (if any)

GET /api/files/{file_id}/content
├── Returns extracted text content
└── For debugging/verification

DELETE /api/files/{file_id}
└── Remove uploaded file
```

### Updated Endpoints

```
POST /api/chat/requirements
├── NEW Parameters:
│   └── file_ids: string[] (uploaded file IDs)
└── NEW Response Fields:
    ├── feasibility.safety: {status, safety_score, hazards}
    ├── extracted_params: {mass_kg, material, complexity, max_dim_m}
    └── file_context: {files_processed, total_chars}
```

---

## Summary

| Feature | Status | Effort |
|---------|--------|--------|
| Landing file upload (6 files) | 🆕 New | 2 days |
| File content extraction | 🆕 New | 2 days |
| Requirements page integration | 🔧 Update | 2 days |
| SafetyAgent integration | 🔧 Update | 1 day |
| Parameter extraction from context | 🆕 New | 1 day |
| UI updates (badges, indicators) | 🔧 Update | 1 day |

**Total: 6-7 days for full implementation**

**MVP (Core flow): 3 days**
- File upload + basic text/PDF extraction
- Requirements endpoint integration
- Basic UI updates
