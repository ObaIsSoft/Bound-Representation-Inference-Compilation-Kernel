
# Add to the end of backend/main.py

# --- Neural SDF Training API ---

class NeuralSDFTrainingRequest(BaseModel):
    design: Dict[str, Any]
    region: Optional[Dict[str, Any]] = None  # {min: [x,y,z], max: [x,y,z]}

@app.post("/api/neural_sdf/train")
async def train_neural_sdf(request: NeuralSDFTrainingRequest):
    """
    Trains a SIREN neural network on the provided geometry.
    
    Args:
        design: Design object with .content field (JSON geometry description)
        region: Optional bounding box for localized training
        
    Returns:
        {
            "status": "success",
            "weights": [...],  # Layer weights/biases
            "metadata": {shape, dims, training_time}
        }
    """
    from scripts.train_siren import train_from_design
    import time
    
    try:
        start_time = time.time()
        
        # Extract geometry from design
        content = request.design.get("content")
        if not content:
            raise HTTPException(status_code=400, detail="No design content provided")
            
        if isinstance(content, str):
            import json
            content = json.loads(content)
        
        # Train network
        logger.info(f"Training Neural SDF for {content.get('geometry', 'unknown')} geometry")
        weights, transform = train_from_design(content, region=request.region)
        
        training_time = time.time() - start_time
        
        return {
            "status": "success",
            "weights": weights,
            "metadata": {
                "shape": content.get("geometry", "custom"),
                "dims": content.get("args", []),
                "training_time": round(training_time, 2),
                "region": request.region,
                "transform": transform
            }
        }
        
    except Exception as e:
        logger.error(f"Neural SDF training failed: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Training failed: {str(e)}")
