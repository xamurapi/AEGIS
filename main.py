"""AEGIS -- Entry point. Launches the autonomous runtime with dashboard."""
import uvicorn
from aegis.config import API_HOST, API_PORT

if __name__ == "__main__":
    print(f"""
    === A E G I S ===
    Autonomous Evolving General Intelligence System v1.0
    Dashboard: http://localhost:{API_PORT}
    API:       http://localhost:{API_PORT}/api/status
    WebSocket: ws://localhost:{API_PORT}/ws
    """)
    uvicorn.run("aegis.api.server:app", host=API_HOST, port=API_PORT, reload=False)
