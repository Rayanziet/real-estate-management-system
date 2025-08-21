import asyncio
import functools
import logging
import os

import click
import sqlalchemy
import sqlalchemy.ext.asyncio
import uvicorn

from a2a.server.apps import A2AStarletteApplication
from a2a.server.request_handlers import DefaultRequestHandler
from a2a.server.tasks import InMemoryTaskStore
from a2a.types import (
    AgentCapabilities,
    AgentCard,
    AgentSkill,
)
from root_agent import root_agent
from agent_executor import ADKAgentExecutor
from dotenv import load_dotenv
from starlette.applications import Starlette

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MissingAPIKeyError(Exception):
    """Exception for missing API key."""


def make_sync(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        return asyncio.run(func(*args, **kwargs))

    return wrapper


@click.command()
@click.option('--host', default='localhost')
@click.option('--port', default=10002)
@make_sync
async def main(host, port):
    agent_card = AgentCard(
        name=root_agent.name,
        description=root_agent.description,
        version='1.0.0',
        url=os.environ.get('APP_URL', 'http://localhost:10002'),
        defaultInputModes=['text', 'text/plain'],
        defaultOutputModes=['text', 'text/plain'],
        capabilities=AgentCapabilities(streaming=True),
        skills=[
            AgentSkill(
                id='document_analysis',
                name='Document Analysis',
                description='Analyzes real estate documents using RAG pipeline.',
                tags=['real_estate', 'document', 'analysis', 'rag'],
                examples=[
                    'Analyze this property tax document for me.',
                    'What are the key insights from this market report?',
                ],
            ),
            AgentSkill(
                id='communication',
                name='Email Communication',
                description='Creates professional email drafts for real estate communications.',
                tags=['real_estate', 'email', 'communication', 'client'],
                examples=[
                    'Create an email draft for a property inquiry.',
                    'Draft a follow-up email to a client.',
                ],
            ),
            AgentSkill(
                id='nearby_places',
                name='Nearby Places Search',
                description='Finds nearby amenities and places around properties.',
                tags=['real_estate', 'location', 'amenities', 'neighborhood'],
                examples=[
                    'What schools are near this property?',
                    'Find restaurants within walking distance.',
                ],
            ),
            AgentSkill(
                id='distance_calculation',
                name='Distance & Travel Time',
                description='Calculates travel times and distances between locations.',
                tags=['real_estate', 'travel', 'commute', 'accessibility'],
                examples=[
                    'How long does it take to get to downtown from this property?',
                    'Calculate the distance to the nearest hospital.',
                ],
            ),
            AgentSkill(
                id='general_real_estate',
                name='General Real Estate',
                description='Provides general real estate guidance and information.',
                tags=['real_estate', 'general', 'guidance', 'information'],
                examples=[
                    'What should I consider when buying a first home?',
                    'Explain the current market trends in this area.',
                ],
            )
        ],
    )

    # Use in-memory task store for simplicity
    task_store = InMemoryTaskStore()

    request_handler = DefaultRequestHandler(
        agent_executor=ADKAgentExecutor(
            agent=root_agent,
            status_message='Processing your real estate request...',
            artifact_name='real_estate_response',
        ),
        task_store=task_store,
    )

    a2a_app = A2AStarletteApplication(
        agent_card=agent_card, http_handler=request_handler
    )
    routes = a2a_app.routes()
    app = Starlette(
        routes=routes,
        middleware=[],
    )

    config = uvicorn.Config(app, host=host, port=port, log_level='info')
    server = uvicorn.Server(config)
    await server.serve()


if __name__ == '__main__':
    main()
