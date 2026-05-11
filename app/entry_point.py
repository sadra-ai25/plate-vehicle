from settings import HOST, PORT_NUMBER
from uvicorn import run
from endpoints import app

if __name__ == '__main__':
    run(app, host=HOST, port=PORT_NUMBER)
