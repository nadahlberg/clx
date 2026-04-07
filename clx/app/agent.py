import litellm

from clx.app.models import Message
from clx.app.tools import (
    AddTrainingExamples,
    Annotate,
    AskUser,
    CompactMemory,
    Search,
    UpdateLabelInstructions,
    UpdateProjectInstructions,
)
from clx.llm.agent import Agent, message_tokens

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
        CompactMemory,
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

        # Find the last compact message (if any) and load from there.
        compact_msg = (
            thread.messages.filter(is_compact=True)
            .order_by("-created_at")
            .first()
        )

        if compact_msg:
            summary = compact_msg.data.get("content", "")
            db_messages = list(
                thread.messages.filter(created_at__gt=compact_msg.created_at)
                .order_by("created_at")
                .values_list("data", flat=True)
            )
            messages = [
                {"role": "system", "content": system_prompt},
                {
                    "role": "user",
                    "content": f"[Prior conversation summary]\n{summary}",
                },
            ] + db_messages
            # +2 for system prompt and synthetic summary message
            self._persisted_count = len(db_messages) + 2
        else:
            db_messages = list(
                thread.messages.order_by("created_at").values_list(
                    "data", flat=True
                )
            )
            messages = [
                {"role": "system", "content": system_prompt}
            ] + db_messages
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
        objects = [
            Message(
                thread=self.thread,
                data=msg,
                num_tokens=message_tokens(msg),
                is_compact=(
                    msg.get("role") == "tool"
                    and msg.get("name") == "CompactMemory"
                ),
            )
            for msg in new_messages
        ]
        Message.objects.bulk_create(objects)
        self._persisted_count = len(self.messages)

        # Accumulate cost on the thread.
        if self.r and self.r.usage:
            try:
                self.thread.total_cost += litellm.completion_cost(
                    completion_response=self.r
                )
            except Exception:
                pass

        # Persist agent state and cost back to the thread.
        self.thread.state = self.state
        self.thread.save(
            update_fields=[
                "state",
                "total_cost",
                "updated_at",
            ]
        )
