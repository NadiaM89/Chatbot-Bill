from fastapi import FastAPI
from routers import router

# Crear una instancia de la aplicación FastAPI
app = FastAPI()

# Establecer el título de la aplicación
app.title = 'BILL'

# Incluir el router definido en el módulo routers
app.include_router(router)
