"""
app_main.py — AItzik AI Car Agent
===================================
Entry point for the API server.

Usage:
    python app_main.py              # default: port 8000
    python app_main.py --port 9000  # custom port
    python app_main.py --reload     # auto-reload on code changes (dev mode)
"""

import argparse
import uvicorn


def main():
    parser = argparse.ArgumentParser(description="AItzik — AI Car Agent API Server")
    parser.add_argument("--host",   default="0.0.0.0",  help="Host to bind (default: 0.0.0.0)")
    parser.add_argument("--port",   default=8001, type=int, help="Port to listen on (default: 8000)")
    parser.add_argument("--reload", action="store_true",   help="Enable auto-reload for development")
    args = parser.parse_args()

    print(f"""
╔══════════════════════════════════════════════╗
║          🚗  AItzik — AI Car Agent           ║
╠══════════════════════════════════════════════╣
║  Web UI  →  http://localhost:{args.port:<5}          ║
╚══════════════════════════════════════════════╝
""")

    uvicorn.run(
        "api.server:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
    )


if __name__ == "__main__":
    main()
