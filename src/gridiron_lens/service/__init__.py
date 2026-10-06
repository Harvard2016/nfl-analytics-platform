"""Local inference service for uploads: a small FastAPI app, one background worker, SQLite job state, private file storage.

Runs on the owner's machine and binds to loopback by default. It is not deployed with the website.
"""
