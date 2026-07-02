from backend.app import app

if __name__ == "__main__":
    import os
    port = int(os.getenv("PORT", "10000"))
    app.run(host="0.0.0.0", port=port)
