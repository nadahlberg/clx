from clx.app.models import Message
from clx.llm.agent import Agent

SYSTEM_PROMPT_TEMPLATE = """\
You are an assistant for the project "{project_name}".
You are working on the label "{label_name}".

{instructions}\
"""


class AnnoAgent(Agent):
    """A thread-backed annotation agent that persists messages to the DB."""

    def __init__(self, thread, **kwargs):
        self.thread = thread
        label = thread.label
        project = label.project

        # Build system prompt dynamically (never saved to DB).
        instructions = label.instructions.strip()
        system_prompt = SYSTEM_PROMPT_TEMPLATE.format(
            project_name=project.name,
            label_name=label.name,
            instructions=instructions,
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
                    num_tokens=num_tokens if msg.get("role") == "assistant" else 0,
                )
            )
        Message.objects.bulk_create(objects)
        self._persisted_count = len(self.messages)
