from fastapi import APIRouter

router = APIRouter()

@router.post("/")
def predict(sequence: str):
    return {
        "message": "Protein Stability model will be connected here."
    }