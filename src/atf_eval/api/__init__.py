"""atf-eval as a service: a FastAPI app that scores submitted agent trajectories.

    atf-eval serve                       # start the server (needs the [server] extra)
    atf-eval submit --adapter ... ...    # agent side: run the agent, submit, fetch results

See app.py for the endpoints and client.py for the agent-side client.
"""
