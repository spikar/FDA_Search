import uvicorn
from fastapi import FastAPI, Query
from typing import Annotated
from fastapi.middleware.cors import CORSMiddleware
from starlette.responses import JSONResponse
import contextlib
from collections.abc import AsyncIterator
from functools import cache
from dotenv import load_dotenv
from langchain.memory import ConversationBufferWindowMemory
from core import run_chain
import traceback
import warnings

warnings.filterwarnings("ignore")

@contextlib.asynccontextmanager
async def lifecycle(app: FastAPI) -> AsyncIterator[None]:
    app.state.state = State()
    yield

app = FastAPI(lifespan=lifecycle)
app.add_middleware(CORSMiddleware, allow_origins=['*'], allow_methods=['*'], allow_headers=['*'])
load_dotenv()

class State:
    def __init__(self) -> None:
        self._memory: dict[str, ConversationBufferWindowMemory] = {}
    
    @cache
    def memory_qa(self, session_id: str) -> ConversationBufferWindowMemory:
        try:
            return self._memory[session_id]
        except KeyError:
            self._memory[session_id] = memory = ConversationBufferWindowMemory(
                input_key='question', output_key='answer', k=3, return_messages=True
            )
            return memory

    @cache
    async def call_qa(self, *, query: str, session_id: str) -> dict:
        try:
            inputs = {"question": query}
            memory = self.memory_qa(session_id)
            answer, result_json = await run_chain(inputs=inputs, memory=memory)
            memory.save_context({"question": query}, {"answer": answer})
            return result_json
        except Exception as e:
            traceback.print_exception(e)
            return {"error": str(e)}

@app.post("/query")
async def process_query(
    query: Annotated[str, Query()]
):
    try:
        session_id = "393811a0-41d0-4598-980d-4fead9c2925b"
        result = await app.state.state.call_qa(query=query, session_id=session_id)
        return JSONResponse(content=result)
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": str(e)}
        )

@app.delete('/user_session')
async def invalidate(session_id: Annotated[str, Query()]):
    success = False
    try:
        del app.state.state._memory[session_id]
        success = True
    except KeyError:
        pass
    return JSONResponse({'status': success})

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)