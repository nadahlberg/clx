from clx.app.models import Message
from clx.app.tools import (
    AddTrainingExamples,
    Annotate,
    AskUser,
    Search,
    UpdateLabelInstructions,
    UpdateProjectInstructions,
)
from clx.llm.agent import Agent

SYSTEM_PROMPT_TEMPLATE = """\
You are an assistant for the project "{project_name}".
You are working on the label "{label_name}".

You have access to tools for searching project documents and updating
instructions. Use them when the user asks you to find data, refine
instructions, or configure the project.

{project_instructions_block}\
{label_instructions_block}\
"""

PROJECT_INSTRUCTIONS_BLOCK = """\
## Project Instructions
{project_instructions}

"""

LABEL_INSTRUCTIONS_BLOCK = """\
## Label Instructions
{label_instructions}

"""


class CLXAgent(Agent):
    """A thread-backed agent that persists messages to the DB."""

    default_tools = [
        Search,
        AddTrainingExamples,
        Annotate,
        UpdateLabelInstructions,
        UpdateProjectInstructions,
        AskUser,
    ]

    def __init__(self, thread, **kwargs):
        self.thread = thread
        label = thread.label
        project = label.project

        # Build system prompt dynamically (never saved to DB).
        project_instructions = project.instructions.strip()
        label_instructions = label.instructions.strip()

        project_block = (
            PROJECT_INSTRUCTIONS_BLOCK.format(
                project_instructions=project_instructions
            )
            if project_instructions
            else ""
        )
        label_block = (
            LABEL_INSTRUCTIONS_BLOCK.format(
                label_instructions=label_instructions
            )
            if label_instructions
            else ""
        )

        system_prompt = SYSTEM_PROMPT_TEMPLATE.format(
            project_name=project.name,
            label_name=label.name,
            project_instructions_block=project_block,
            label_instructions_block=label_block,
        )

        # Load persisted messages and prepend system prompt.
        db_messages = list(
            thread.messages.order_by("created_at").values_list(
                "data", flat=True
            )
        )
        messages = [{"role": "system", "content": system_prompt}] + db_messages

        # Track how many messages are already persisted so on_step
        # only saves new ones.  The +1 accounts for the system prompt
        # which is in self.messages but not in the DB.
        self._persisted_count = len(db_messages) + 1

        super().__init__(
            model=thread.model,
            messages=messages,
            state=thread.state or {},
            **kwargs,
        )

    def on_step(self, response_message):
        """Save any new messages to the database."""
        new_messages = self.messages[self._persisted_count :]
        if not new_messages:
            return
        objects = []
        for msg in new_messages:
            usage = {}
            if self.r and self.r.usage:
                usage = dict(self.r.usage)
            num_tokens = usage.get("total_tokens", 0)
            objects.append(
                Message(
                    thread=self.thread,
                    data=msg,
                    num_tokens=num_tokens
                    if msg.get("role") == "assistant"
                    else 0,
                )
            )
        Message.objects.bulk_create(objects)
        self._persisted_count = len(self.messages)

        # Persist agent state back to the thread.
        self.thread.state = self.state
        self.thread.save(update_fields=["state", "updated_at"])
