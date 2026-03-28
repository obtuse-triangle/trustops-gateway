from app.app_factory import create_app
from fastapi.middleware.cors import CORSMiddleware

app = create_app()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
