from fastapi import FastAPI
from routers import router

app = FastAPI()

app.title = 'BILL'

app.include_router(router)
