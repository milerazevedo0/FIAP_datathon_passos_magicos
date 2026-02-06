from fastapi import FastAPI, Request
from fastapi.templating import Jinja2Templates
from app.routes import router as predict_router
from app.monitoring_routes import router as monitoring_router

app = FastAPI(title="Defasagem Escolar API")
templates = Jinja2Templates(directory="app/templates")

app.include_router(predict_router)
app.include_router(monitoring_router)


@app.get("/")
def home(request: Request):
    """
    Página inicial com descrição do projeto.
    """
    return templates.TemplateResponse(
        "index.html",
        {"request": request}
    )
