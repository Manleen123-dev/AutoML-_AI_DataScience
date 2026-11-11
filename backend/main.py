from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from services.eda import get_basic_summary
from database import db
import os
import uuid
from pydantic import BaseModel
from dotenv import load_dotenv
load_dotenv()

from contextlib import asynccontextmanager
import shutil

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Cleanup uploads and db on startup
    if os.path.exists("uploads"):
        shutil.rmtree("uploads")
    os.makedirs("uploads", exist_ok=True)
    try:
        await db.datasets.drop()
    except Exception:
        pass
    yield

app = FastAPI(title="AI Data Science Copilot API", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

os.makedirs("uploads", exist_ok=True)

class TargetUpdateRequest(BaseModel):
    target_column: str

@app.get("/")
def read_root():
    return {"message": "AI Data Science Copilot API is running"}

@app.post("/upload")
async def upload_dataset(file: UploadFile = File(...)):
    if not file.filename.endswith('.csv'):
        raise HTTPException(status_code=400, detail="Only CSV files are supported")
        
    contents = await file.read()
    try:
        summary = get_basic_summary(contents, file.filename)
        
        # Save to disk
        dataset_id = str(uuid.uuid4())
        file_path = os.path.join("uploads", f"{dataset_id}_{file.filename}")
        with open(file_path, "wb") as f:
            f.write(contents)
            
        summary["id"] = dataset_id
        summary["file_path"] = file_path
        summary["target_column"] = None
        
        # Save to MongoDB
        await db.datasets.insert_one(summary.copy()) # copy since motor might mutate with _id
        
        if "_id" in summary:
            summary.pop("_id")
            
        return summary
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.put("/dataset/{dataset_id}/target")
async def update_target(dataset_id: str, request: TargetUpdateRequest):
    result = await db.datasets.update_one(
        {"id": dataset_id},
        {"$set": {"target_column": request.target_column}}
    )
    if result.matched_count == 0:
        raise HTTPException(status_code=404, detail="Dataset not found")
    return {"message": "Target updated successfully"}

from services.eda import get_feature_details

@app.get("/dataset/{dataset_id}/feature/{column_name}")
async def get_feature_analysis(dataset_id: str, column_name: str):
    dataset = await db.datasets.find_one({"id": dataset_id})
    if not dataset:
        raise HTTPException(status_code=404, detail="Dataset not found")
        
    file_path = dataset.get("file_path")
    if not file_path or not os.path.exists(file_path):
         raise HTTPException(status_code=404, detail="Dataset file not found on disk")
         
    try:
        details = get_feature_details(file_path, column_name)
        return details
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

from typing import Optional

class PlotRequest(BaseModel):
    plot_type: str
    x_column: str
    y_column: Optional[str] = None

from services.eda import generate_plot_data

@app.post("/dataset/{dataset_id}/plot")
async def get_plot_data(dataset_id: str, request: PlotRequest):
    dataset = await db.datasets.find_one({"id": dataset_id})
    if not dataset:
        raise HTTPException(status_code=404, detail="Dataset not found")
        
    file_path = dataset.get("file_path")
    if not file_path or not os.path.exists(file_path):
         raise HTTPException(status_code=404, detail="Dataset file not found on disk")
         
    try:
        plot_data = generate_plot_data(file_path, request.plot_type, request.x_column, request.y_column)
        return {"data": [plot_data]}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

from services.eda import get_missing_values_report

@app.get("/dataset/{dataset_id}/missing")
async def get_missing_report(dataset_id: str):
    dataset = await db.datasets.find_one({"id": dataset_id})
    if not dataset:
        raise HTTPException(status_code=404, detail="Dataset not found")
        
    file_path = dataset.get("file_path")
    if not file_path or not os.path.exists(file_path):
         raise HTTPException(status_code=404, detail="Dataset file not found on disk")
         
    try:
        report = get_missing_values_report(file_path)
        return report
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

from services.eda import get_correlation_matrix

@app.get("/dataset/{dataset_id}/correlation")
async def get_correlation(dataset_id: str, method: str = "pearson", features: str = None):
    dataset = await db.datasets.find_one({"id": dataset_id})
    if not dataset:
        raise HTTPException(status_code=404, detail="Dataset not found")
        
    file_path = dataset.get("file_path")
    if not file_path or not os.path.exists(file_path):
         raise HTTPException(status_code=404, detail="Dataset file not found on disk")
         
    try:
        if method not in ["pearson", "spearman"]:
            method = "pearson"
        features_list = [f.strip() for f in features.split(",")] if features else None
        report = get_correlation_matrix(file_path, method, features_list)
        return report
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

from services.eda import get_outliers_report

def extract_summary(dataset: dict) -> dict:
    if "summary" in dataset:
        s = dataset["summary"]
        if "_id" in s:
            s.pop("_id")
        return s
    
    s = dataset.copy()
    if "_id" in s:
        s.pop("_id")
    for k in ["can_undo", "can_redo", "prev_summary", "prev_file_path"]:
        if k in s:
            s.pop(k)
    return s

@app.get("/dataset/{dataset_id}")
async def get_dataset(dataset_id: str):
    dataset = await db.datasets.find_one({"id": dataset_id})
    if not dataset:
        raise HTTPException(status_code=404, detail="Dataset not found")
        
    summary = extract_summary(dataset)
        
    summary["can_undo"] = dataset.get("can_undo", False)
    summary["can_redo"] = dataset.get("can_redo", False)
        
    return summary

@app.get("/dataset/{dataset_id}/outliers")
async def get_outliers(dataset_id: str):
    dataset = await db.datasets.find_one({"id": dataset_id})
    if not dataset:
        raise HTTPException(status_code=404, detail="Dataset not found")
        
    file_path = dataset.get("file_path")
    if not file_path or not os.path.exists(file_path):
         raise HTTPException(status_code=404, detail="Dataset file not found on disk")
         
    try:
        report = get_outliers_report(file_path)
        return report
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

from services.feature_engineering import apply_feature_engineering
from services.preprocessing import apply_preprocessing
from pydantic import BaseModel
from typing import List, Optional

class PreprocessOperation(BaseModel):
    type: str
    columns: List[str]

class PreprocessRequest(BaseModel):
    operations: List[PreprocessOperation]

class FEOperation(BaseModel):
    type: str
    column: Optional[str] = None
    column1: Optional[str] = None
    column2: Optional[str] = None
    operation: Optional[str] = None
    new_column_name: Optional[str] = None
    bins: Optional[int] = None
    strategy: Optional[str] = None
    degree: Optional[int] = None

class FERequest(BaseModel):
    operations: List[FEOperation]

def get_buffer_paths(dataset: dict):
    base_id = dataset["id"]
    current_path = dataset.get("file_path", f"uploads/{base_id}.csv")
    
    if current_path.endswith("_v1.csv"):
        output_path = f"uploads/{base_id}_v2.csv"
    elif current_path.endswith("_v2.csv"):
        output_path = f"uploads/{base_id}_v1.csv"
    else:
        output_path = f"uploads/{base_id}_v2.csv"
        
    return current_path, output_path

@app.post("/dataset/{dataset_id}/preprocess")
async def preprocess(dataset_id: str, request: PreprocessRequest):
    dataset = await db.datasets.find_one({"id": dataset_id})
    if not dataset:
        raise HTTPException(status_code=404, detail="Dataset not found")
        
    file_path = dataset.get("file_path")
    if not file_path or not os.path.exists(file_path):
         raise HTTPException(status_code=404, detail="Dataset file not found on disk")
         
    try:
        input_path, output_path = get_buffer_paths(dataset)
        ops_dict = [op.dict(exclude_none=True) for op in request.operations]
        
        apply_preprocessing(input_path, output_path, ops_dict)
        
        with open(output_path, "rb") as f:
            contents = f.read()
        new_summary = get_basic_summary(contents, dataset.get("filename"))
        new_summary["id"] = dataset_id
        new_summary["file_path"] = output_path
        new_summary["target_column"] = dataset.get("target_column")
        
        await db.datasets.update_one(
            {"id": dataset_id},
            {"$set": {
                "summary": new_summary,
                "file_path": output_path,
                "prev_summary": extract_summary(dataset),
                "prev_file_path": input_path,
                "can_undo": True,
                "can_redo": False
            }}
        )
        
        if "_id" in new_summary:
            new_summary.pop("_id")
            
        new_summary["can_undo"] = True
        new_summary["can_redo"] = False
        return new_summary
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/dataset/{dataset_id}/feature-engineering")
async def feature_engineering(dataset_id: str, request: FERequest):
    dataset = await db.datasets.find_one({"id": dataset_id})
    if not dataset:
        raise HTTPException(status_code=404, detail="Dataset not found")
        
    file_path = dataset.get("file_path")
    if not file_path or not os.path.exists(file_path):
         raise HTTPException(status_code=404, detail="Dataset file not found on disk")
         
    try:
        input_path, output_path = get_buffer_paths(dataset)
        ops_dict = [op.dict(exclude_none=True) for op in request.operations]
        
        apply_feature_engineering(input_path, output_path, ops_dict)
        
        with open(output_path, "rb") as f:
            contents = f.read()
        new_summary = get_basic_summary(contents, dataset.get("filename"))
        new_summary["id"] = dataset_id
        new_summary["file_path"] = output_path
        new_summary["target_column"] = dataset.get("target_column")
        
        await db.datasets.update_one(
            {"id": dataset_id},
            {"$set": {
                "summary": new_summary,
                "file_path": output_path,
                "prev_summary": extract_summary(dataset),
                "prev_file_path": input_path,
                "can_undo": True,
                "can_redo": False
            }}
        )
        
        if "_id" in new_summary:
            new_summary.pop("_id")
            
        new_summary["can_undo"] = True
        new_summary["can_redo"] = False
        return new_summary
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/dataset/{dataset_id}/undo")
async def undo_operation(dataset_id: str):
    dataset = await db.datasets.find_one({"id": dataset_id})
    if not dataset or not dataset.get("can_undo"):
        raise HTTPException(status_code=400, detail="Nothing to undo.")
        
    current_path = dataset.get("file_path")
    current_summary = extract_summary(dataset)
    prev_path = dataset.get("prev_file_path")
    prev_summary = dataset.get("prev_summary")
    
    await db.datasets.update_one(
        {"id": dataset_id},
        {"$set": {
            "summary": prev_summary,
            "file_path": prev_path,
            "prev_summary": current_summary,
            "prev_file_path": current_path,
            "can_undo": False,
            "can_redo": True
        }}
    )
    
    if "_id" in prev_summary:
        prev_summary.pop("_id")
    prev_summary["can_undo"] = False
    prev_summary["can_redo"] = True
    return prev_summary

@app.post("/dataset/{dataset_id}/redo")
async def redo_operation(dataset_id: str):
    dataset = await db.datasets.find_one({"id": dataset_id})
    if not dataset or not dataset.get("can_redo"):
        raise HTTPException(status_code=400, detail="Nothing to redo.")
        
    current_path = dataset.get("file_path")
    current_summary = extract_summary(dataset)
    prev_path = dataset.get("prev_file_path")
    prev_summary = dataset.get("prev_summary")
    
    await db.datasets.update_one(
        {"id": dataset_id},
        {"$set": {
            "summary": prev_summary,
            "file_path": prev_path,
            "prev_summary": current_summary,
            "prev_file_path": current_path,
            "can_undo": True,
            "can_redo": False
        }}
    )
    
    if "_id" in prev_summary:
        prev_summary.pop("_id")
    prev_summary["can_undo"] = True
    prev_summary["can_redo"] = False
    return prev_summary

@app.delete("/dataset/{dataset_id}")
async def delete_dataset(dataset_id: str):
    dataset = await db.datasets.find_one({"id": dataset_id})
    if not dataset:
        raise HTTPException(status_code=404, detail="Dataset not found")
        
    await db.datasets.delete_one({"id": dataset_id})
    
    # Delete associated files
    import glob
    for file_path in glob.glob(f"uploads/{dataset_id}*"):
        try:
            os.remove(file_path)
        except Exception:
            pass
            
    return {"message": "Dataset ejected successfully"}

from fastapi.responses import StreamingResponse, FileResponse
from services.llm import stream_chat_response
from typing import List

class ChatMessage(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    message: str
    history: List[ChatMessage] = []
    password: Optional[str] = None

class PasswordRequest(BaseModel):
    password: str

@app.post("/verify-assistant")
async def verify_assistant(request: PasswordRequest):
    expected_password = os.getenv("ASSISTANT_PASSWORD")
    if not expected_password or request.password == expected_password:
        return {"success": True}
    return {"success": False}

@app.post("/dataset/{dataset_id}/chat")
async def chat_with_dataset(dataset_id: str, request: ChatRequest):
    expected_password = os.getenv("ASSISTANT_PASSWORD")
    if expected_password and request.password != expected_password:
        raise HTTPException(status_code=401, detail="Unauthorized: Invalid password")
        
    dataset = await db.datasets.find_one({"id": dataset_id})
    if not dataset:
        raise HTTPException(status_code=404, detail="Dataset not found")
        
    summary = extract_summary(dataset)
    
    # We use stream_chat_response as an async generator
    return StreamingResponse(
        stream_chat_response(summary, request.history, request.message),
        media_type="text/plain"
    )

@app.get("/dataset/{dataset_id}/export")
async def export_dataset(dataset_id: str):
    dataset = await db.datasets.find_one({"id": dataset_id})
    if not dataset:
        raise HTTPException(status_code=404, detail="Dataset not found")
        
    file_path = dataset["file_path"]
    original_filename = dataset["filename"]
    
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="Underlying file not found")
        
    return FileResponse(
        path=file_path,
        media_type="text/csv",
        filename=f"processed_{original_filename}"
    )
