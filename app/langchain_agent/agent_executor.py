import logging
from a2a.server.agent_execution import AgentExecutor, RequestContext
from a2a.server.events import EventQueue
from a2a.server.tasks import TaskUpdater
from a2a.types import (
    InternalError,
    Part,
    TaskState,
    TextPart,
    UnsupportedOperationError,
)
from a2a.utils.errors import ServerError
from property_search import RealEstateAgent 

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RealEstateAgentExecutor(AgentExecutor):
    """Real Estate Agent Executor for A2A communication."""
    
    def __init__(self):
        self.agent = RealEstateAgent()
    
    async def execute(
        self,
        context: RequestContext,
        event_queue: EventQueue,
    ) -> None:
        """Execute the real estate agent with A2A protocol"""
        
        if not context.task_id or not context.context_id:
            raise ValueError("RequestContext must have task_id and context_id")
        if not context.message:
            raise ValueError("RequestContext must have a message")
        
        updater = TaskUpdater(event_queue, context.task_id, context.context_id)
        
        if not context.current_task:
            await updater.submit()
        
        await updater.start_work()
        
        query = context.get_user_input()
        
        try:
            async for item in self.agent.stream(query, context.context_id):
                try:
                    is_task_complete = item["is_task_complete"]
                    require_user_input = item["require_user_input"]
                    parts = [Part(root=TextPart(text=item["content"]))]
                    
                    if not is_task_complete and not require_user_input:
                        # Agent is working - send intermediate update
                        await updater.update_status(
                            TaskState.working,
                            message=updater.new_agent_message(parts),
                        )
                    elif require_user_input:
                        # Agent needs clarification from user
                        await updater.update_status(
                            TaskState.input_required,
                            message=updater.new_agent_message(parts),
                        )
                        break
                    else:
                        # Task complete - add final result as artifact
                        await updater.add_artifact(
                            parts,
                            name="property_search_results",
                        )
                        await updater.complete()
                        break
                except Exception as e:
                    logger.error(f"Error processing stream item: {e}")
                    # Continue processing other items
                    continue
                    
        except Exception as e:
            logger.error(f"An error occurred while streaming the response: {e}")
            # Try to send error status before raising
            try:
                await updater.update_status(
                    TaskState.failed,
                    message=updater.new_agent_message([Part(root=TextPart(text=f"Error: {str(e)}"))]),
                )
            except:
                pass  # If we can't send error status, just continue
            raise ServerError(error=InternalError()) from e
        finally:
            # Always try to cleanup
            try:
                await self.agent.cleanup()
            except Exception as cleanup_error:
                logger.warning(f"Cleanup error: {cleanup_error}")
    
    async def cancel(self, context: RequestContext, event_queue: EventQueue) -> None:
        """Cancel the current task - not supported for this agent"""
        raise ServerError(error=UnsupportedOperationError())