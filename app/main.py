from typing import List
from fastapi import FastAPI, HTTPException
import pandas as pd
from pydantic import BaseModel
from pipeline import churnRF, realizar_pipeline
from models.models import PrevisaoList
import uvicorn

app = FastAPI()

@app.get("/treinamento")
def executar_pipeline(
    vizualizar_distribuicao_classes: bool = False,
    melhorModelo: bool = False,
    gerar_baseline: bool = False,
):
    return realizar_pipeline(
        vizualizar_distribuicao_classes=vizualizar_distribuicao_classes,
        melhorModelo=melhorModelo,
        gerar_baseline=gerar_baseline,
    )

@app.post("/realizar_previsao")
async def realizar_previsao(previsao_list: PrevisaoList):
    try: 
        dados_brutos = [p.dict() for p in previsao_list.previsoes]
        previsoes = churnRF(dados_brutos)
        return {"previsoes": previsoes.tolist()}  
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)