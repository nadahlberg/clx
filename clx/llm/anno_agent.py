import simplejson as json
from pydantic import BaseModel

from clx.llm.agent import Agent

PROJECT_INSTRUCTIONS_TEMPLATE = """
You are an annotation assistant providing single-label classification
annotations for the following label: {label_name}.

When annotating you will be provided a text example. You should respond
with a boolean `value` indicating whether the label "{label_name}" applies to
the text, and a brief, one-sentence `reason` explaining how your decision
aligns with the guidelines below.

Here are some guidelines you should follow when annotating:

Consider these project-level instructions. These are general, project-wide
instructions that apply to all labels in the project. They may include examples
of labels other than the one that you are annotating, just remember that you are
currently annotating for the label "{label_name}" specifically.

```
{project_instructions}
```
"""

LABEL_INSTRUCTIONS_TEMPLATE = """
The user has also provided some label-specific instructions. These should take precedence
over the project-level instructions if they are in conflict.

```
{label_instructions}
```
"""


class Annotation(BaseModel):
    """An annotation value with a reason."""

    value: bool
    reason: str


class AnnoAgent(Agent):
    """An annotation agent for single-label classification."""

    default_model = "gemini/gemini-2.5-flash-lite"
    default_completion_args = {
        "response_format": Annotation,
    }
    on_init_args = [
        "label_name",
        "label_instructions",
        "project_instructions",
        "decisions",
    ]

    def on_init(
        self, label_name, label_instructions, project_instructions, decisions
    ):
        system_prompt = PROJECT_INSTRUCTIONS_TEMPLATE.format(
            label_name=label_name,
            project_instructions=project_instructions,
        )
        if label_instructions is not None:
            system_prompt += "\n\n" + LABEL_INSTRUCTIONS_TEMPLATE.format(
                label_name=label_name,
                label_instructions=label_instructions,
            )
        messages = [{"role": "system", "content": system_prompt}]
        for decision in decisions:
            messages.append({"role": "user", "content": decision["text"]})
            json_content = Annotation(
                value=decision["value"], reason=decision["reason"]
            ).model_dump_json()
            messages.append({"role": "assistant", "content": json_content})
        self.state["prefix_messages"] = messages

    def __call__(self, text: str) -> Annotation:
        self.messages = [*self.state["prefix_messages"]]
        response = self.step(messages=[{"role": "user", "content": text}])
        return Annotation(**json.loads(response["content"]))
