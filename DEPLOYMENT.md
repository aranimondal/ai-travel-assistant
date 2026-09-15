# Public deployment

The recommended deployment is a Render Web Service using the Dockerfile in this repository. Render provides a public `onrender.com` URL for the service and can automatically redeploy the linked `main` branch. With `autoDeployTrigger: checksPass` in `render.yaml`, Render waits for GitHub CI checks before rebuilding and deploying the service.

## One-time setup

1. Create or sign in to a Render account.
2. Connect the GitHub account that owns `aranimondal/ai-travel-assistant`.
3. Create a Render Blueprint/Web Service from this repository and use the committed `render.yaml`.
4. In the service secrets/environment settings, set:

```text
OPENAI_API_KEY=<your OpenAI API key>
```

`render.yaml` already sets the application to use the OpenAI provider and `gpt-5.6-luna` for the hosted demo. The secret itself is not stored in GitHub.

## Deployment flow

```text
GitHub push to main
        |
        v
GitHub Actions
  - install dependencies
  - run pytest
  - build linux/amd64 Docker image
  - publish image to GHCR
        |
        v
Render (linked to main)
  - waits for CI checks to pass
  - rebuilds Dockerfile from that commit
  - deploys the new container
        |
        v
Public Streamlit URL
```

Render's Git-backed Docker service is the deployment source of truth for the running application. The GitHub Actions image build provides an additional reproducible container artifact for the same commit.

## Application container

The Dockerfile:

- uses Python 3.11
- installs `requirements.txt`
- copies the source and knowledge-base documents
- builds the FAISS knowledge-base index during the image build
- starts Streamlit on `0.0.0.0` using Render's `$PORT`

This makes the deployed service self-contained instead of requiring a separate vector database or MCP server process outside the application.

## Automatic updates

After the one-time Render setup:

```text
Developer change
      -> git push main
      -> GitHub Actions
      -> tests pass
      -> Docker image build succeeds
      -> Render detects the successful commit
      -> Render rebuilds/deploys
      -> public UI updates
```

No manual redeploy is required for normal code changes.

## Public access

Render web services expose a public `onrender.com` subdomain. Keep the service public so evaluators can open the URL from Chrome, Edge, Safari, mobile browsers, tablets, or desktop browsers.

## Important secret handling

Never commit `.env` or an OpenAI API key. The repository contains only `.env.example`. Use Render's environment/secret settings for the hosted application.
