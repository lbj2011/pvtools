# NOTE: When deploying via `heroku container:push web` (Docker-based),
# Heroku IGNORES this Procfile and uses the CMD in the Dockerfile instead.
# Keep this file in sync with the Dockerfile's gunicorn command so that
# buildpack-based deploys (`git push heroku`) and container deploys behave
# the same way.  Entry-point note: this file references `app:app` (the
# Flask app named `app` in app.py), while the Dockerfile uses
# `index:server` -- those should resolve to the same Flask server object.
web: gunicorn app:app \
--worker-class gthread \
--workers 1 \
--threads 8 \
--timeout 600